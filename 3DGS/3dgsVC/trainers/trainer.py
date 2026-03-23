"""Training loop for paper-faithful and legacy 3DGSMR modes."""

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from scipy.io import savemat
except Exception:  # pragma: no cover
    savemat = None

from data import MRIDataset
from data.transforms import fft3c, ifft3c
from gaussian import GaussianModel3D, TileVoxelizer, Voxelizer
from losses import CombinedLoss
from losses.losses import TVLoss
from metrics import MetricPlateauStopper, evaluate_reconstruction


class GaussianTrainer:
    def __init__(self, config: Dict[str, Any], device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = self.config.setdefault("mode", {})
        self.paper_faithful = self.mode.get("paper_faithful", False)
        self.experimental_reproduction = self.mode.get("experimental_reproduction", True)
        self.self_supervised_deploy = self.mode.get("self_supervised_deploy", False)
        self.strict_cuda = self.mode.get("strict_cuda", False)
        self.require_lpips = self.mode.get("require_lpips", False)
        self.metric_config = self.config.get("metrics", {})
        print(f"Using device: {self.device}")
        print(
            f"Modes: paper_faithful={self.paper_faithful}, experimental_reproduction={self.experimental_reproduction}, "
            f"self_supervised_deploy={self.self_supervised_deploy}, strict_cuda={self.strict_cuda}"
        )

        self.current_iteration = 0
        self._position_grads_for_densify = None
        self.best_metrics: Dict[str, float] = {}
        self.best_psnr = float("-inf")
        self.best_ssim = float("-inf")
        self.best_lpips = float("inf")
        self.metric_plateau_stopper: Optional[MetricPlateauStopper] = None
        self.loss_plateau_best = float("inf")
        self.loss_plateau_bad_count = 0
        self._skip_scheduler_step_once = False

        self._resolve_strategy()
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_output()
        self._setup_stopping()

    def _resolve_strategy(self):
        adaptive_config = self.config.setdefault("adaptive_control", {})
        gaussian_config = self.config.setdefault("gaussian", {})
        acc = self.config["data"]["acceleration_factor"]
        strategy = adaptive_config.get("strategy", "auto")
        threshold = adaptive_config.get("high_acceleration_threshold", 10)

        if strategy == "auto":
            if acc >= threshold:
                self._resolved_use_long_axis = True
                self._resolved_use_cloning = False
                if gaussian_config.get("initial_num_points", 500) > 1000:
                    gaussian_config["initial_num_points"] = 500
            else:
                self._resolved_use_long_axis = False
                self._resolved_use_cloning = True
        elif strategy == "long_axis":
            self._resolved_use_long_axis = True
            self._resolved_use_cloning = False
        else:
            self._resolved_use_long_axis = False
            self._resolved_use_cloning = True

        if self.paper_faithful and acc < threshold and gaussian_config.get("initial_num_points") != 200000:
            print("[Strategy] Paper-faithful low-mid mode allows override, but the paper default remains M=200k + original split/clone.")
        print(
            f"[Strategy] resolved long_axis={self._resolved_use_long_axis}, cloning={self._resolved_use_cloning}, "
            f"M={gaussian_config.get('initial_num_points')}"
        )

    def _setup_data(self):
        data_config = self.config["data"]
        self.dataset = MRIDataset(
            data_path=data_config["data_path"],
            acceleration_factor=data_config["acceleration_factor"],
            mask_type=data_config.get("mask_type", "stacked_2d_gaussian"),
            use_acs=data_config.get("use_acs", True),
            acs_lines=data_config.get("acs_lines", 24),
            csm_path=data_config.get("csm_path"),
            mask_path=data_config.get("mask_path"),
            readout_axis=data_config.get("readout_axis"),
            phase_axes=tuple(data_config["phase_axes"]) if data_config.get("phase_axes") is not None else None,
            normalize_kspace=data_config.get("normalize", data_config.get("normalize_kspace", True)),
            normalization_percentile=data_config.get("normalization_percentile", 99.9),
            device=str(self.device),
        )
        data = self.dataset.get_data()
        self.kspace_full_cpu = data["kspace_full"]
        self.kspace_undersampled_complex = data.get("kspace_undersampled_complex")
        if self.kspace_undersampled_complex is None:
            kspace_ri = data["kspace_undersampled"]
            num_coils = kspace_ri.shape[0] // 2
            self.kspace_undersampled_complex = torch.complex(kspace_ri[:num_coils], kspace_ri[num_coils:])
        self.kspace_undersampled_complex = self.kspace_undersampled_complex.to(self.device)
        self.mask = data["mask"].to(self.device)
        self.mask_coils = self.mask.unsqueeze(0)
        self.volume_shape = tuple(data["volume_shape"])
        self.target_image = data["ground_truth"].to(self.device)
        self._zero_filled_for_init = data.get("zero_filled_complex", data["zero_filled"]).to(self.device)
        self.sensitivity_maps = data["sensitivity_maps"].to(self.device)
        self.normalization_scale = float(data.get("normalization_scale", 1.0))
        self.num_coils = self.sensitivity_maps.shape[0]
        print(
            f"Volume shape: {self.volume_shape}, coils: {self.num_coils}, acc: {data_config['acceleration_factor']}x, "
            f"norm_scale: {self.normalization_scale:.6f}"
        )
        del self.dataset

    def _setup_model(self):
        gaussian_config = self.config["gaussian"]
        init_method = gaussian_config.get("init_method", "from_image")
        init_mode = gaussian_config.get("init_mode", "random" if self.paper_faithful else "importance")
        if init_method == "from_image":
            self.gaussian_model = GaussianModel3D.from_image(
                image=self._zero_filled_for_init,
                num_points=gaussian_config["initial_num_points"],
                density_scale_k=gaussian_config.get("density_scale_k", 0.2),
                init_mode=init_mode,
                device=str(self.device),
            )
        else:
            self.gaussian_model = GaussianModel3D(
                num_points=gaussian_config["initial_num_points"],
                volume_shape=tuple(self.volume_shape),
                device=str(self.device),
            )
        del self._zero_filled_for_init
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        vox_config = self.config.get("voxelizer", {})
        vox_type = vox_config.get("type", "chunk")
        if vox_type in {"tile_cuda", "tile"}:
            self.voxelizer = TileVoxelizer(
                tuple(self.volume_shape),
                tile_size=vox_config.get("tile_size", 8),
                max_radius=vox_config.get("max_radius", 20),
                use_cuda=(vox_type == "tile_cuda"),
                strict_cuda=self.strict_cuda and vox_type == "tile_cuda",
                device=str(self.device),
            )
        else:
            self.voxelizer = Voxelizer(volume_shape=tuple(self.volume_shape), device=str(self.device))
        print(f"Initialized with {self.gaussian_model.num_points} Gaussians")

    def _setup_loss(self):
        loss_config = self.config["loss"]
        image_weight = loss_config.get("image_weight", 0.0)
        if self.paper_faithful and image_weight > 0:
            raise ValueError("Paper-faithful mode does not allow image-domain auxiliary loss.")
        self.criterion = CombinedLoss(
            kspace_weight=loss_config.get("kspace_weight", 1.0),
            image_weight=image_weight,
            tv_weight=loss_config.get("tv_weight", 0.0),
            loss_type=loss_config.get("loss_type", "l2"),
        ).to(self.device)

    def _setup_optimizer(self):
        train_config = self.config["training"]
        gaussian_config = self.config["gaussian"]
        params = self.gaussian_model.get_optimizable_params(
            lr_position=gaussian_config.get("position_lr", 1e-4),
            lr_density=gaussian_config.get("density_lr", 1e-3),
            lr_scale=gaussian_config.get("scale_lr", 5e-4),
            lr_rotation=gaussian_config.get("rotation_lr", 1e-4),
        )
        self.optimizer = optim.Adam(params)
        scheduler_config = train_config.get("lr_scheduler", {})
        if scheduler_config.get("type", "exponential") == "exponential":
            self.scheduler = ExponentialLR(self.optimizer, gamma=scheduler_config.get("gamma", 0.999))
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=train_config["max_iterations"])

    def _setup_output(self):
        output_config = self.config["output"]
        self.output_dir = output_config["output_dir"]
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.result_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "logs")) if output_config.get("use_tensorboard", True) else None
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as handle:
            yaml.dump(self.config, handle, default_flow_style=False)

    def _setup_stopping(self):
        stopping_config = self.config.get("stopping", {})
        if self.experimental_reproduction and stopping_config.get("enable_metric_plateau", False):
            self.metric_plateau_stopper = MetricPlateauStopper(
                patience_evals=stopping_config.get("patience_evals", 4),
                min_iterations_before_stop=stopping_config.get("min_iterations_before_stop", 0),
                metric_min_delta=stopping_config.get("metric_min_delta", {}),
                best_metric=stopping_config.get("best_metric", "psnr"),
            )
        self.enable_loss_plateau = stopping_config.get("enable_loss_plateau", False) and not self.experimental_reproduction
        self.loss_min_delta = stopping_config.get("loss_min_delta", 0.0)
        self.loss_patience = stopping_config.get("patience_evals", 0)
        self.loss_min_iteration = stopping_config.get("min_iterations_before_stop", 0)
        self.best_metric_name = stopping_config.get("best_metric", "psnr")

    def _get_volume(self):
        return self.voxelizer(
            positions=self.gaussian_model.positions,
            scales=self.gaussian_model.get_scale_values(),
            rotations=self.gaussian_model.rotations,
            density=self.gaussian_model.density,
        )

    @staticmethod
    def _as_complex_volume(vol_result):
        if isinstance(vol_result, tuple):
            return torch.complex(vol_result[0], vol_result[1])
        return vol_result

    def _apply_hard_data_consistency(self, volume: torch.Tensor) -> torch.Tensor:
        if not self.mode.get("apply_hard_data_consistency", True):
            return volume
        pred_kspace = fft3c(volume.unsqueeze(0) * self.sensitivity_maps)
        kspace_dc = pred_kspace * (1.0 - self.mask_coils) + self.kspace_undersampled_complex
        coil_images_dc = ifft3c(kspace_dc)
        return torch.sum(torch.conj(self.sensitivity_maps) * coil_images_dc, dim=0)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        volume = self._as_complex_volume(self._get_volume())
        kspace_pred = fft3c(volume.unsqueeze(0) * self.sensitivity_maps)
        return volume, kspace_pred

    def _kspace_loss(self, pred_kspace: torch.Tensor, loss_type: str) -> torch.Tensor:
        diff = pred_kspace * self.mask_coils - self.kspace_undersampled_complex
        if loss_type == "l1":
            return torch.abs(diff).sum()
        return diff.real.square().sum() + diff.imag.square().sum()

    def forward_with_loss(self):
        vol_result = self._get_volume()
        volume = self._as_complex_volume(vol_result)
        volume_for_eval = volume.detach()

        tv_weight = self.config["loss"].get("tv_weight", 0.0)
        tv_loss = None
        if tv_weight > 0:
            tv_loss = TVLoss()(volume)

        loss_type = self.config["loss"].get("loss_type", "l2")
        pred_kspace = fft3c(volume.unsqueeze(0) * self.sensitivity_maps)
        kspace_loss = self._kspace_loss(pred_kspace, loss_type)

        losses = {"kspace_loss": kspace_loss}
        total_loss = self.config["loss"].get("kspace_weight", 1.0) * kspace_loss
        if tv_weight > 0 and tv_loss is not None:
            losses["tv_loss"] = tv_loss
            total_loss = total_loss + tv_weight * tv_loss

        image_weight = self.config["loss"].get("image_weight", 0.0)
        if image_weight > 0 and self.target_image is not None:
            image_loss = self.criterion.image_loss(volume, self.target_image)
            losses["image_loss"] = image_loss
            total_loss = total_loss + image_weight * image_loss

        losses["total_loss"] = total_loss
        return volume_for_eval, losses

    def compute_gradient_stats(self):
        grads = self._position_grads_for_densify
        if grads is None:
            if self.gaussian_model.positions.grad is None:
                return {}
            grads = self.gaussian_model.positions.grad
        grad_norm = torch.norm(grads, dim=-1)
        return {"grad_norm": grad_norm, "mean_grad": grad_norm.mean(), "max_grad": grad_norm.max()}

    def adaptive_density_control(self, iteration: int) -> Dict[str, int]:
        adaptive_config = self.config["adaptive_control"]
        if not adaptive_config.get("enable", True):
            return {"split": 0, "clone": 0, "prune": 0}
        start_iter = adaptive_config.get("densify_from_iter", 100)
        end_iter = adaptive_config.get("densify_until_iter", 2500)
        interval = adaptive_config.get("densify_every", 100)
        max_num_points = self.config["gaussian"].get("max_num_points", 400000)
        if self.gaussian_model.num_points >= max_num_points:
            return {"split": 0, "clone": 0, "prune": 0}
        if iteration < start_iter or iteration > end_iter or iteration % interval != 0:
            return {"split": 0, "clone": 0, "prune": 0}

        grad_stats = self.compute_gradient_stats()
        if not grad_stats:
            return {"split": 0, "clone": 0, "prune": 0}

        grad_norm = grad_stats["grad_norm"]
        grad_threshold = adaptive_config.get("grad_threshold", 0.01)
        scale_threshold = adaptive_config.get("scale_threshold", 0.01)
        max_scale_limit = adaptive_config.get("max_scale", 0.5)
        opacity_threshold = adaptive_config.get("opacity_threshold", 0.01)
        long_axis_offset_factor = self.config["gaussian"].get("long_axis_offset_factor", 1.0)
        stats = {"split": 0, "clone": 0, "prune": 0}

        high_grad_mask = grad_norm > grad_threshold
        scales = self.gaussian_model.get_scale_values()
        max_scale = scales.max(dim=-1)[0]
        split_mask = high_grad_mask & (max_scale > scale_threshold)
        if split_mask.sum() > 0 and self.gaussian_model.num_points + int(split_mask.sum().item()) <= max_num_points:
            stats["split"] = self.gaussian_model.densify_and_split(
                grads=grad_norm,
                grad_threshold=grad_threshold,
                scale_threshold=scale_threshold,
                use_long_axis_splitting=self._resolved_use_long_axis,
                long_axis_offset_factor=long_axis_offset_factor,
            )
            high_grad_mask = None
            if stats["split"] > 0:
                self._rebuild_optimizer()

        if self._resolved_use_cloning and high_grad_mask is not None:
            clone_mask = high_grad_mask & (max_scale <= scale_threshold)
            if clone_mask.sum() > 0 and self.gaussian_model.num_points + int(clone_mask.sum().item()) <= max_num_points:
                stats["clone"] = self.gaussian_model.densify_and_clone(grad_norm, grad_threshold, scale_threshold)
                if stats["clone"] > 0:
                    self._rebuild_optimizer()

        scales = self.gaussian_model.get_scale_values()
        max_scale = scales.max(dim=-1)[0]
        densities = torch.abs(self.gaussian_model.density)
        prune_mask = (densities < opacity_threshold) | (max_scale > max_scale_limit)
        keep_mask = ~prune_mask
        if keep_mask.sum() >= 100 and prune_mask.sum() > 0:
            self.gaussian_model._update_params(keep_mask, is_prune=True)
            stats["prune"] = int(prune_mask.sum().item())
            self._rebuild_optimizer()

        return stats

    @staticmethod
    def _transplant_adam_state(
        old_optimizer: optim.Adam,
        old_params_map: dict,
        keep_mask: torch.Tensor,
        is_prune: bool,
        new_optimizer: optim.Adam,
    ):
        """Copy Adam exp_avg / exp_avg_sq from old optimizer to new one,
        selecting only the kept rows (via *keep_mask*) and zero-padding any
        newly appended rows (from split / clone)."""
        for new_group, old_group in zip(
            new_optimizer.param_groups, old_optimizer.param_groups
        ):
            for new_p, old_p in zip(new_group["params"], old_group["params"]):
                if old_p not in old_optimizer.state:
                    continue
                old_s = old_optimizer.state[old_p]
                if "exp_avg" not in old_s:
                    continue
                new_s = {}
                for key in ("exp_avg", "exp_avg_sq"):
                    old_buf = old_s[key]  # shape (N_old, ...)
                    kept = old_buf[keep_mask]  # shape (N_kept, ...)
                    if is_prune or kept.shape[0] == new_p.shape[0]:
                        new_s[key] = kept
                    else:
                        # split / clone appended rows; pad with zeros
                        pad_shape = list(kept.shape)
                        pad_shape[0] = new_p.shape[0] - kept.shape[0]
                        new_s[key] = torch.cat(
                            [kept, torch.zeros(pad_shape, device=kept.device, dtype=kept.dtype)],
                            dim=0,
                        )
                new_s["step"] = old_s.get("step", torch.tensor(0.0))
                new_optimizer.state[new_p] = new_s

    def _rebuild_optimizer(self):
        gaussian_config = self.config["gaussian"]
        params = self.gaussian_model.get_optimizable_params(
            lr_position=gaussian_config.get("position_lr", 1e-4),
            lr_density=gaussian_config.get("density_lr", 1e-3),
            lr_scale=gaussian_config.get("scale_lr", 5e-4),
            lr_rotation=gaussian_config.get("rotation_lr", 1e-4),
        )
        # Capture current LR so the new optimizer continues at the decayed LR.
        current_lrs = [group["lr"] for group in self.optimizer.param_groups]
        old_optimizer = self.optimizer
        self.optimizer = optim.Adam(params)
        for group, lr in zip(self.optimizer.param_groups, current_lrs):
            group["lr"] = lr
            group["initial_lr"] = lr

        # Transplant Adam momentum from surviving Gaussian rows.
        old_params_map = getattr(self.gaussian_model, "_last_densify_old_params", None)
        keep_mask = getattr(self.gaussian_model, "_last_densify_keep_mask", None)
        is_prune = getattr(self.gaussian_model, "_last_densify_is_prune", False)
        if old_params_map is not None and keep_mask is not None:
            self._transplant_adam_state(
                old_optimizer, old_params_map, keep_mask, is_prune, self.optimizer
            )
            self.gaussian_model._last_densify_old_params = None
            self.gaussian_model._last_densify_keep_mask = None

        scheduler_config = self.config["training"].get("lr_scheduler", {})
        if scheduler_config.get("type", "exponential") == "exponential":
            self.scheduler = ExponentialLR(self.optimizer, gamma=scheduler_config.get("gamma", 0.999))
        else:
            remaining = max(self.config["training"]["max_iterations"] - self.current_iteration, 1)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=remaining)
        self._skip_scheduler_step_once = True

    def train_step(self):
        self.gaussian_model.train()
        self.optimizer.zero_grad()
        _, loss_dict = self.forward_with_loss()
        loss_dict["total_loss"].backward()
        if self.gaussian_model.positions.grad is not None:
            self._position_grads_for_densify = self.gaussian_model.positions.grad.detach().clone()
        else:
            self._position_grads_for_densify = None
        max_grad_norm = self.config["training"].get("max_grad_norm", 1.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.gaussian_model.parameters(), max_grad_norm)
        self.optimizer.step()
        return {key: float(value.item()) for key, value in loss_dict.items()}

    def evaluate(self) -> Dict[str, float]:
        if self.self_supervised_deploy or not self.experimental_reproduction:
            return {}
        self.gaussian_model.eval()
        with torch.no_grad():
            volume = self._apply_hard_data_consistency(self._as_complex_volume(self._get_volume()))
            return evaluate_reconstruction(
                pred=volume,
                target=self.target_image,
                compute_3d_ssim=True,
                compute_lpips_metric=self.metric_config.get("compute_lpips_during_train", False),
                lpips_device=self.device,
                require_lpips=self.require_lpips,
            )

    def _metric_value_for_best(self, metrics: Dict[str, float]) -> float:
        value = metrics.get(self.best_metric_name)
        if value is None:
            return float("-inf")
        if self.best_metric_name == "lpips":
            return -float(value)
        return float(value)

    def _update_best_metrics(self, metrics: Dict[str, float]) -> bool:
        if not metrics:
            return False
        score = self._metric_value_for_best(metrics)
        best_score = self._metric_value_for_best(self.best_metrics) if self.best_metrics else float("-inf")
        is_best = score > best_score
        if is_best:
            self.best_metrics = dict(metrics)
            self.best_psnr = metrics.get("psnr", self.best_psnr)
            self.best_ssim = metrics.get("ssim", self.best_ssim)
            self.best_lpips = metrics.get("lpips", self.best_lpips)
        return is_best

    def _check_loss_plateau(self, iteration: int, loss_value: float) -> bool:
        if not self.enable_loss_plateau:
            return False
        if loss_value < self.loss_plateau_best - self.loss_min_delta:
            self.loss_plateau_best = loss_value
            self.loss_plateau_bad_count = 0
        else:
            self.loss_plateau_bad_count += 1
        return iteration >= self.loss_min_iteration and self.loss_plateau_bad_count >= self.loss_patience

    def save_checkpoint(self, iteration: int, is_best: bool = False):
        checkpoint = {
            "iteration": iteration,
            "gaussian_state": self.gaussian_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_metrics": self.best_metrics,
            "config": self.config,
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, "latest.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, "best.pth"))
        save_every = self.config["training"].get("save_every", 500)
        if iteration % save_every == 0:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_{iteration:06d}.pth"))

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.gaussian_model.load_state_dict(checkpoint["gaussian_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.current_iteration = checkpoint["iteration"]
        self.best_metrics = checkpoint.get("best_metrics", {})
        self.best_psnr = self.best_metrics.get("psnr", self.best_psnr)
        self.best_ssim = self.best_metrics.get("ssim", self.best_ssim)
        self.best_lpips = self.best_metrics.get("lpips", self.best_lpips)
        print(f"Loaded checkpoint from iteration {self.current_iteration}")

    def save_reconstruction(self, iteration: int):
        if savemat is None:
            raise ImportError("Saving .mat requires scipy")
        self.gaussian_model.eval()
        with torch.no_grad():
            volume_raw = self._as_complex_volume(self._get_volume())
            volume = self._apply_hard_data_consistency(volume_raw)
            savemat(os.path.join(self.result_dir, f"reconstruction_raw_{iteration:06d}.mat"), {"reconstruction_raw": volume_raw.detach().cpu().numpy()}, do_compression=True)
            volume_np = volume.detach().cpu().numpy()
            savemat(os.path.join(self.result_dir, f"reconstruction_{iteration:06d}.mat"), {"reconstruction": volume_np}, do_compression=True)
            savemat(os.path.join(self.result_dir, "reconstruction_final.mat"), {"reconstruction": volume_np}, do_compression=True)

    def train(self, resume_from: Optional[str] = None):
        train_config = self.config["training"]
        max_iterations = train_config["max_iterations"]
        eval_every = train_config.get("eval_every", 100)
        log_every = train_config.get("log_every", 50)
        save_every = train_config.get("save_every", 500)

        if resume_from is not None:
            self.load_checkpoint(resume_from)

        print(f"\nStarting training from iteration {self.current_iteration}")
        print(f"Total iterations: {max_iterations}")
        print("-" * 60)

        should_stop = False
        stop_reason = None
        last_iteration = self.current_iteration
        progress = tqdm(range(self.current_iteration, max_iterations), desc="Training", dynamic_ncols=True)
        for iteration in progress:
            self.current_iteration = iteration
            last_iteration = iteration
            loss_dict = self.train_step()
            adaptive_stats = self.adaptive_density_control(iteration)
            if self._skip_scheduler_step_once:
                self._skip_scheduler_step_once = False
            else:
                self.scheduler.step()

            if iteration % log_every == 0:
                grad_stats = self.compute_gradient_stats()
                mean_grad = grad_stats.get("mean_grad", torch.tensor(0.0)).item() if grad_stats else 0.0
                progress.set_postfix({"loss": f"{loss_dict['total_loss']:.2e}", "grad": f"{mean_grad:.2e}", "n_pts": self.gaussian_model.num_points})
                if self.writer:
                    self.writer.add_scalar("Loss/total_loss", loss_dict["total_loss"], iteration)
                    if "kspace_loss" in loss_dict:
                        self.writer.add_scalar("Loss/kspace_loss", loss_dict["kspace_loss"], iteration)
                    if "tv_loss" in loss_dict:
                        self.writer.add_scalar("Loss/tv_loss", loss_dict["tv_loss"], iteration)
                    self.writer.add_scalar("Stats/num_points", self.gaussian_model.num_points, iteration)
                    self.writer.add_scalar("Stats/mean_grad", mean_grad, iteration)

            metrics = {}
            if iteration % eval_every == 0 or iteration == max_iterations - 1:
                if self.experimental_reproduction and not self.self_supervised_deploy:
                    metrics = self.evaluate()
                    is_best = self._update_best_metrics(metrics)
                    if self.writer:
                        for key, value in metrics.items():
                            self.writer.add_scalar(f"Metrics/{key.upper()}", value, iteration)
                    print(f"\n[Iter {iteration}] metrics: {metrics}")
                    if any(value > 0 for value in adaptive_stats.values()):
                        print(f"  Density control: {adaptive_stats}")
                    print(f"  Num Gaussians: {self.gaussian_model.num_points}")
                    self.save_checkpoint(iteration, is_best=is_best)
                    if self.metric_plateau_stopper is not None:
                        should_stop = self.metric_plateau_stopper.update(
                            iteration,
                            {name: metrics[name] for name in ("psnr", "ssim", "lpips") if name in metrics},
                        )
                        if should_stop:
                            stop_reason = "GT metrics plateau"
                else:
                    if self._check_loss_plateau(iteration, loss_dict["total_loss"]):
                        should_stop = True
                        stop_reason = "loss plateau"
                    if iteration % save_every == 0 or iteration == max_iterations - 1:
                        self.save_checkpoint(iteration, is_best=False)

            if should_stop:
                print(f"\nEarly stop at iteration {iteration}: {stop_reason}")
                break

        print("\n" + "=" * 60)
        print("Training completed")
        if self.best_metrics:
            print(f"Best metrics: {self.best_metrics}")
        self.save_reconstruction(last_iteration)
        return self.best_metrics
