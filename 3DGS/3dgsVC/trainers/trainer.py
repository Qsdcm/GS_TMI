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

from torch.utils.checkpoint import checkpoint as grad_checkpoint

from data import MRIDataset
from data.transforms import fft3c
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
            readout_axis=data_config.get("readout_axis", 0),
            phase_axes=tuple(data_config.get("phase_axes", [1, 2])),
            device=str(self.device),
        )
        data = self.dataset.get_data()
        self.kspace_full_cpu = data["kspace_full"]
        self.kspace_undersampled = data["kspace_undersampled"].to(self.device)
        self.mask = data["mask"].to(self.device)
        self.volume_shape = tuple(data["volume_shape"])
        self.target_image = data["ground_truth"].to(self.device)
        self._zero_filled_for_init = data["zero_filled"].to(self.device)
        self.sensitivity_maps = data["sensitivity_maps"].to(self.device)
        self.num_coils = self.sensitivity_maps.shape[0]
        print(f"Volume shape: {self.volume_shape}, coils: {self.num_coils}, acc: {data_config['acceleration_factor']}x")
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

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        volume = self._as_complex_volume(self._get_volume())
        kspace_real_list = []
        kspace_imag_list = []
        for coil_idx in range(self.num_coils):
            coil_kspace = fft3c((volume * self.sensitivity_maps[coil_idx]).unsqueeze(0)).squeeze(0)
            kspace_real_list.append(coil_kspace.real)
            kspace_imag_list.append(coil_kspace.imag)
        kspace_pred = torch.cat([torch.stack(kspace_real_list, 0), torch.stack(kspace_imag_list, 0)], 0)
        return volume, kspace_pred

    def _coil_loss(self, volume, coil_csm, target_real, target_imag, mask, is_l1):
        coil_kspace = fft3c((volume * coil_csm).unsqueeze(0)).squeeze(0)
        pred_real = coil_kspace.real * mask
        pred_imag = coil_kspace.imag * mask
        target_real = target_real * mask
        target_imag = target_imag * mask
        if bool(is_l1.item()):
            return torch.abs(pred_real - target_real).sum() + torch.abs(pred_imag - target_imag).sum()
        return ((pred_real - target_real) ** 2).sum() + ((pred_imag - target_imag) ** 2).sum()

    def _coil_loss_ri(self, vol_r, vol_i, csm_r, csm_i, target_real, target_imag, mask, is_l1):
        img_r = vol_r * csm_r - vol_i * csm_i
        img_i = vol_r * csm_i + vol_i * csm_r
        coil_kspace = fft3c(torch.complex(img_r, img_i).unsqueeze(0)).squeeze(0)
        pred_real = coil_kspace.real * mask
        pred_imag = coil_kspace.imag * mask
        target_real = target_real * mask
        target_imag = target_imag * mask
        if bool(is_l1.item()):
            return torch.abs(pred_real - target_real).sum() + torch.abs(pred_imag - target_imag).sum()
        return ((pred_real - target_real) ** 2).sum() + ((pred_imag - target_imag) ** 2).sum()

    def forward_with_loss(self):
        vol_result = self._get_volume()
        is_ri = isinstance(vol_result, tuple)
        if is_ri:
            vol_r, vol_i = vol_result
            volume_for_eval = torch.complex(vol_r.detach(), vol_i.detach())
            device = vol_r.device
        else:
            volume = vol_result
            volume_for_eval = volume
            device = volume.device

        tv_weight = self.config["loss"].get("tv_weight", 0.0)
        tv_loss = None
        if tv_weight > 0:
            if is_ri:
                tv_loss = TVLoss()(torch.sqrt(vol_r ** 2 + vol_i ** 2 + 1e-12))
            else:
                tv_loss = TVLoss()(volume)

        loss_type = self.config["loss"].get("loss_type", "l2")
        is_l1 = torch.tensor(loss_type == "l1", device=device)
        kspace_loss = torch.tensor(0.0, device=device)
        for coil_idx in range(self.num_coils):
            if is_ri:
                coil_loss = grad_checkpoint(
                    self._coil_loss_ri,
                    vol_r,
                    vol_i,
                    self.sensitivity_maps[coil_idx].real.contiguous(),
                    self.sensitivity_maps[coil_idx].imag.contiguous(),
                    self.kspace_undersampled[coil_idx],
                    self.kspace_undersampled[coil_idx + self.num_coils],
                    self.mask,
                    is_l1,
                    use_reentrant=True,
                )
            else:
                coil_loss = grad_checkpoint(
                    self._coil_loss,
                    volume,
                    self.sensitivity_maps[coil_idx],
                    self.kspace_undersampled[coil_idx],
                    self.kspace_undersampled[coil_idx + self.num_coils],
                    self.mask,
                    is_l1,
                    use_reentrant=True,
                )
            kspace_loss = kspace_loss + coil_loss

        losses = {"kspace_loss": kspace_loss}
        total_loss = self.config["loss"].get("kspace_weight", 1.0) * kspace_loss
        if tv_weight > 0 and tv_loss is not None:
            losses["tv_loss"] = tv_loss
            total_loss = total_loss + tv_weight * tv_loss

        image_weight = self.config["loss"].get("image_weight", 0.0)
        if image_weight > 0 and self.target_image is not None:
            image_loss = self.criterion.image_loss(volume_for_eval, self.target_image)
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
        model_changed = False
        if split_mask.sum() > 0 and self.gaussian_model.num_points + int(split_mask.sum().item()) <= max_num_points:
            stats["split"] = self.gaussian_model.densify_and_split(
                grads=grad_norm,
                grad_threshold=grad_threshold,
                scale_threshold=scale_threshold,
                use_long_axis_splitting=self._resolved_use_long_axis,
                long_axis_offset_factor=long_axis_offset_factor,
            )
            high_grad_mask = None
            model_changed = stats["split"] > 0

        if self._resolved_use_cloning and high_grad_mask is not None:
            clone_mask = high_grad_mask & (max_scale <= scale_threshold)
            if clone_mask.sum() > 0 and self.gaussian_model.num_points + int(clone_mask.sum().item()) <= max_num_points:
                stats["clone"] = self.gaussian_model.densify_and_clone(grad_norm, grad_threshold, scale_threshold)
                model_changed = model_changed or stats["clone"] > 0

        if model_changed:
            scales = self.gaussian_model.get_scale_values()
            max_scale = scales.max(dim=-1)[0]

        densities = torch.abs(self.gaussian_model.density)
        prune_mask = (densities < opacity_threshold) | (max_scale > max_scale_limit)
        keep_mask = ~prune_mask
        if keep_mask.sum() >= 100 and prune_mask.sum() > 0:
            self.gaussian_model._update_params(keep_mask, is_prune=True)
            stats["prune"] = int(prune_mask.sum().item())

        if any(value > 0 for value in stats.values()):
            self._rebuild_optimizer()
        return stats

    def _rebuild_optimizer(self):
        gaussian_config = self.config["gaussian"]
        params = self.gaussian_model.get_optimizable_params(
            lr_position=gaussian_config.get("position_lr", 1e-4),
            lr_density=gaussian_config.get("density_lr", 1e-3),
            lr_scale=gaussian_config.get("scale_lr", 5e-4),
            lr_rotation=gaussian_config.get("rotation_lr", 1e-4),
        )
        self.optimizer = optim.Adam(params)
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
            volume = self._as_complex_volume(self._get_volume())
            return evaluate_reconstruction(
                pred=volume,
                target=self.target_image,
                compute_3d_ssim=True,
                compute_lpips_metric=True,
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
            volume = self._as_complex_volume(self._get_volume())
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
