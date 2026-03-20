"""
3DGSMR Trainer — Aligned with Paper (TMI 2025)

Paper: "Three-Dimensional MRI Reconstruction with 3D Gaussian Representations"

Key alignment points:
1. Loss uses Sum reduction → proper gradient magnitude for densification
2. Acceleration-aware adaptive control:
   - Low-mid (2-8x): original clone+split, M=200k, s=0.01
   - High (>=10x): long-axis splitting, no cloning, M=500
3. Densification every 100 iters, up to 2500 iters, max 400k points
4. Pre-clip gradient saving for accurate densification threshold
5. Scheduler rebuilt after densification to avoid stale optimizer reference
6. tile_cuda: CUDA forward for evaluation, chunk-based for training backward
"""

import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm

try:
    from scipy.io import savemat
except Exception:  # pragma: no cover
    savemat = None

from torch.utils.checkpoint import checkpoint as grad_checkpoint

from data import MRIDataset
from data.transforms import fft3c, ifft3c
from gaussian import GaussianModel3D, Voxelizer, TileVoxelizer
from losses import CombinedLoss
from metrics import evaluate_reconstruction

class GaussianTrainer:
    """3D Gaussian MRI Reconstruction Trainer"""

    def __init__(self, config: Dict[str, Any], device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self._resolve_strategy()
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_output()

        self.current_iteration = 0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self._position_grads_for_densify = None

    def _resolve_strategy(self):
        """Paper: Original for low-mid acc, Long-axis for high acc or sparse M."""
        adaptive_config = self.config.setdefault('adaptive_control', {})
        gaussian_config = self.config.setdefault('gaussian', {})
        acc = self.config['data']['acceleration_factor']
        strategy = adaptive_config.get('strategy', 'auto')

        if strategy == 'auto':
            threshold = adaptive_config.get('high_acceleration_threshold', 10)
            current_m = gaussian_config.get('initial_num_points', 200000)
            if acc >= threshold or current_m <= 1000:
                self._resolved_use_long_axis = True
                self._resolved_use_cloning = False
                if acc >= threshold and current_m > 1000:
                    gaussian_config['initial_num_points'] = 500
                print(f"[Strategy:auto] acc={acc}x, M={gaussian_config['initial_num_points']} → long-axis, no clone")
            else:
                self._resolved_use_long_axis = False
                self._resolved_use_cloning = True
                print(f"[Strategy:auto] acc={acc}x, M={current_m} → original clone+split")
        elif strategy == 'long_axis':
            self._resolved_use_long_axis = True
            self._resolved_use_cloning = False
        elif strategy == 'original':
            self._resolved_use_long_axis = False
            self._resolved_use_cloning = True
        else:
            self._resolved_use_long_axis = adaptive_config.get('use_long_axis_splitting', True)
            self._resolved_use_cloning = adaptive_config.get('use_cloning', False)

    def _setup_data(self):
        data_config = self.config['data']
        self.dataset = MRIDataset(
            data_path=data_config['data_path'],
            acceleration_factor=data_config['acceleration_factor'],
            mask_type=data_config.get('mask_type', 'gaussian'),
            use_acs=data_config.get('use_acs', True),
            acs_lines=data_config.get('acs_lines', int(data_config.get('center_fraction', 0.08) * 256)),
            csm_path=data_config.get('csm_path', None),
            device=self.device
        )
        data = self.dataset.get_data()
        self.kspace_full_cpu = data['kspace_full']
        self.kspace_undersampled = data['kspace_undersampled'].to(self.device)
        self.mask = data['mask'].to(self.device)
        self.volume_shape = data['volume_shape']
        self.target_image = data['ground_truth'].to(self.device)
        self._zero_filled_for_init = data['zero_filled'].to(self.device)
        self.sensitivity_maps = data['sensitivity_maps'].to(self.device)
        self.num_coils = self.sensitivity_maps.shape[0]
        print(f"Volume shape: {self.volume_shape}, Coils: {self.num_coils}, Acc: {data_config['acceleration_factor']}x")
        del self.dataset

    def _setup_model(self):
        gaussian_config = self.config['gaussian']
        init_method = gaussian_config.get('init_method', 'from_image')
        if init_method == 'from_image':
            init_mode = gaussian_config.get('init_mode', 'importance')
            self.gaussian_model = GaussianModel3D.from_image(
                image=self._zero_filled_for_init,
                num_points=gaussian_config['initial_num_points'],
                initial_scale=gaussian_config.get('initial_scale', 2.0),
                density_scale_k=gaussian_config.get('density_scale_k', 0.2),
                init_mode=init_mode,
                device=str(self.device)
            )
        else:
            self.gaussian_model = GaussianModel3D(
                num_points=gaussian_config['initial_num_points'],
                volume_shape=tuple(self.volume_shape),
                device=str(self.device)
            )
        del self._zero_filled_for_init
        torch.cuda.empty_cache()

        # Voxelizer: tile_cuda for fast training, chunk as fallback
        vox_config = self.config.get('voxelizer', {})
        vox_type = vox_config.get('type', 'chunk')
        if vox_type in ('tile_cuda', 'tile'):
            tile_size = vox_config.get('tile_size', 8)
            max_radius = vox_config.get('max_radius', 20)
            use_cuda = (vox_type == 'tile_cuda')
            self.voxelizer = TileVoxelizer(
                tuple(self.volume_shape), tile_size, max_radius,
                use_cuda=use_cuda, device=str(self.device)
            )
        else:
            self.voxelizer = Voxelizer(volume_shape=tuple(self.volume_shape), device=str(self.device))

        print(f"Initialized with {self.gaussian_model.num_points} Gaussians")

    def _setup_loss(self):
        loss_config = self.config['loss']
        self.criterion = CombinedLoss(
            kspace_weight=loss_config.get('kspace_weight', 1.0),
            image_weight=loss_config.get('image_weight', 0.0),
            tv_weight=loss_config.get('tv_weight', 0.0),
            loss_type=loss_config.get('loss_type', 'l2')
        ).to(self.device)

    def _setup_optimizer(self):
        train_config = self.config['training']
        gaussian_config = self.config['gaussian']
        params = self.gaussian_model.get_optimizable_params(
            lr_position=gaussian_config.get('position_lr', 1e-4),
            lr_density=gaussian_config.get('density_lr', 1e-3),
            lr_scale=gaussian_config.get('scale_lr', 5e-4),
            lr_rotation=gaussian_config.get('rotation_lr', 1e-4)
        )
        self.optimizer = optim.Adam(params)
        scheduler_config = train_config.get('lr_scheduler', {})
        scheduler_type = scheduler_config.get('type', 'exponential')
        if scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(self.optimizer, gamma=scheduler_config.get('gamma', 0.999))
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=train_config['max_iterations'])

    def _setup_output(self):
        output_config = self.config['output']
        self.output_dir = output_config['output_dir']
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.result_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        if self.config['output'].get('use_tensorboard', True):
            self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'logs'))
        else:
            self.writer = None
        config_path = os.path.join(self.output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def _get_volume(self) -> torch.Tensor:
        """GS -> Image Domain (Complex), returns [D, H, W] complex or tuple (vol_r, vol_i)."""
        positions = self.gaussian_model.positions
        scales = self.gaussian_model.get_scale_values()
        rotations = self.gaussian_model.rotations
        density = self.gaussian_model.density

        result = self.voxelizer(positions=positions, scales=scales, rotations=rotations, density=density)
        return result

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Original forward for test/export."""
        volume = self._get_volume()
        num_coils = self.sensitivity_maps.shape[0]
        kspace_real_list, kspace_imag_list = [], []
        for c in range(num_coils):
            coil_kspace = fft3c((volume * self.sensitivity_maps[c]).unsqueeze(0)).squeeze(0)
            kspace_real_list.append(coil_kspace.real)
            kspace_imag_list.append(coil_kspace.imag)
        kspace_pred = torch.cat([torch.stack(kspace_real_list, 0), torch.stack(kspace_imag_list, 0)], 0)
        return volume, kspace_pred

    def _coil_loss(self, volume, coil_csm, target_real, target_imag, mask, loss_type_is_l1):
        """Compute loss for a single coil (complex volume input)."""
        coil_kspace = fft3c((volume * coil_csm).unsqueeze(0)).squeeze(0)
        pr, pi = coil_kspace.real * mask, coil_kspace.imag * mask
        tr, ti = target_real * mask, target_imag * mask
        if loss_type_is_l1:
            return torch.abs(pr - tr).sum() + torch.abs(pi - ti).sum()
        return ((pr - tr) ** 2).sum() + ((pi - ti) ** 2).sum()

    def _coil_loss_ri(self, vol_r, vol_i, csm_r, csm_i, target_real, target_imag, mask, loss_type_is_l1):
        """Compute loss for a single coil (separate real/imag inputs for CUDA path)."""
        img_r = vol_r * csm_r - vol_i * csm_i
        img_i = vol_r * csm_i + vol_i * csm_r
        coil_image = torch.complex(img_r, img_i)
        coil_kspace = fft3c(coil_image.unsqueeze(0)).squeeze(0)
        pred_real = coil_kspace.real * mask
        pred_imag = coil_kspace.imag * mask
        t_real = target_real * mask
        t_imag = target_imag * mask
        if loss_type_is_l1:
            loss = torch.abs(pred_real - t_real).sum() + torch.abs(pred_imag - t_imag).sum()
        else:
            loss = ((pred_real - t_real) ** 2).sum() + ((pred_imag - t_imag) ** 2).sum()
        return loss

    def forward_with_loss(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Memory-efficient forward + loss fusion.
        Handles both complex volume (chunk voxelizer) and (vol_r, vol_i) tuple (CUDA tile voxelizer).
        """
        vol_result = self._get_volume()

        # CUDA tile path returns tuple (vol_r, vol_i) to avoid PyTorch 1.13 complex+autograd bug
        if isinstance(vol_result, tuple):
            vol_r, vol_i = vol_result
            is_ri = True
            dev = vol_r.device
            # Detached complex for evaluation only (not in loss computation graph)
            volume_for_eval = torch.complex(vol_r.detach(), vol_i.detach())
        else:
            volume = vol_result
            is_ri = False
            dev = volume.device
            volume_for_eval = volume

        # TV loss on magnitude (pure float ops for CUDA path, no complex needed)
        tv_weight = self.config['loss'].get('tv_weight', 0.0)
        tv_l = None
        if tv_weight > 0:
            from losses.losses import TVLoss
            if is_ri:
                mag = torch.sqrt(vol_r ** 2 + vol_i ** 2 + 1e-12)
                tv_l = TVLoss()(mag)
            else:
                tv_l = TVLoss()(volume)

        num_coils = self.sensitivity_maps.shape[0]
        kspace_loss = torch.tensor(0.0, device=dev)
        loss_type = self.config['loss'].get('loss_type', 'l2')
        is_l1 = torch.tensor(loss_type == "l1", device=dev)

        for c in range(num_coils):
            if is_ri:
                coil_loss = grad_checkpoint(
                    self._coil_loss_ri,
                    vol_r, vol_i,
                    self.sensitivity_maps[c].real.contiguous(),
                    self.sensitivity_maps[c].imag.contiguous(),
                    self.kspace_undersampled[c],
                    self.kspace_undersampled[c + num_coils],
                    self.mask, is_l1,
                    use_reentrant=True,
                )
            else:
                coil_loss = grad_checkpoint(
                    self._coil_loss,
                    volume,
                    self.sensitivity_maps[c],
                    self.kspace_undersampled[c],
                    self.kspace_undersampled[c + num_coils],
                    self.mask, is_l1,
                    use_reentrant=True,
                )
            kspace_loss = kspace_loss + coil_loss

        losses = {}
        total_loss = self.config['loss'].get('kspace_weight', 1.0) * kspace_loss
        losses['kspace_loss'] = kspace_loss

        image_weight = self.config['loss'].get('image_weight', 0.0)
        if image_weight > 0 and self.target_image is not None:
            from losses import ImageLoss
            image_l = ImageLoss(loss_type)(volume_for_eval, self.target_image)
            losses['image_loss'] = image_l
            total_loss = total_loss + image_weight * image_l

        if tv_weight > 0 and tv_l is not None:
            losses['tv_loss'] = tv_l
            total_loss = total_loss + tv_weight * tv_l

        losses['total_loss'] = total_loss
        return volume_for_eval, losses

    def compute_gradient_stats(self) -> Dict[str, torch.Tensor]:
        grads = self._position_grads_for_densify
        if grads is None:
            if self.gaussian_model.positions.grad is None:
                return {}
            grads = self.gaussian_model.positions.grad
        grad_norm = torch.norm(grads, dim=-1)
        return {'grad_norm': grad_norm, 'mean_grad': grad_norm.mean(), 'max_grad': grad_norm.max()}

    def adaptive_density_control(self, iteration: int) -> Dict[str, int]:
        """Paper Section D: Adaptive density control."""
        adaptive_config = self.config['adaptive_control']
        if not adaptive_config.get('enable', True):
            return {'split': 0, 'clone': 0, 'prune': 0}

        stats = {'split': 0, 'clone': 0, 'prune': 0}
        start_iter = adaptive_config.get('densify_from_iter', 100)
        end_iter = adaptive_config.get('densify_until_iter', 2500)
        interval = adaptive_config.get('densify_every', 100)
        max_num_points = self.config['gaussian'].get('max_num_points', 400000)

        if self.gaussian_model.num_points >= max_num_points:
            return stats
        if iteration < start_iter or iteration > end_iter:
            return stats
        if iteration % interval != 0:
            return stats

        grad_stats = self.compute_gradient_stats()
        if not grad_stats:
            return stats

        grad_norm = grad_stats['grad_norm']
        grad_threshold = adaptive_config.get('grad_threshold', 0.005)
        scale_threshold = adaptive_config.get('scale_threshold', 0.01)
        max_scale_limit = adaptive_config.get('max_scale', 0.5)
        high_grad_mask = grad_norm > grad_threshold

        # Split
        use_long_axis = self._resolved_use_long_axis
        if high_grad_mask.shape[0] == self.gaussian_model.num_points:
            scales = self.gaussian_model.get_scale_values()
            max_scale = scales.max(dim=-1)[0]
            split_mask = high_grad_mask & (max_scale > scale_threshold)
            if split_mask.sum() > 0 and self.gaussian_model.num_points + split_mask.sum() <= max_num_points:
                self.gaussian_model.densify_and_split(
                    grads=grad_norm, grad_threshold=grad_threshold,
                    scale_threshold=scale_threshold, use_long_axis_splitting=use_long_axis
                )
                stats['split'] = split_mask.sum().item()
                high_grad_mask = None

        # Clone (only for original strategy)
        if self._resolved_use_cloning and high_grad_mask is not None:
            if high_grad_mask.shape[0] == self.gaussian_model.num_points:
                scales = self.gaussian_model.get_scale_values()
                max_scale = scales.max(dim=-1)[0]
                clone_mask = high_grad_mask & (max_scale <= scale_threshold)
                if clone_mask.sum() > 0 and self.gaussian_model.num_points + clone_mask.sum() <= max_num_points:
                    self.gaussian_model.densify_and_clone(grad_norm, grad_threshold, scale_threshold)
                    stats['clone'] = clone_mask.sum().item()

        # Prune
        opacity_threshold = adaptive_config.get('opacity_threshold', 0.01)
        densities = torch.abs(self.gaussian_model.density)
        if grad_stats and 'grad_norm' in grad_stats:
            gn = grad_stats['grad_norm']
            if gn.shape[0] == self.gaussian_model.num_points:
                paper_prune = (densities < opacity_threshold) & (gn > grad_threshold)
            else:
                paper_prune = densities < opacity_threshold
        else:
            paper_prune = densities < opacity_threshold
        scales = self.gaussian_model.get_scale_values()
        max_scale = scales.max(dim=-1)[0]
        prune_mask = paper_prune | (max_scale > max_scale_limit)
        keep_mask = ~prune_mask
        if keep_mask.sum() >= 100 and prune_mask.sum() > 0:
            self.gaussian_model._update_params(keep_mask, is_prune=True)
            stats['prune'] = prune_mask.sum().item()

        if stats['split'] > 0 or stats['clone'] > 0 or stats['prune'] > 0:
            self._rebuild_optimizer()
        return stats

    def _rebuild_optimizer(self):
        gaussian_config = self.config['gaussian']
        params = self.gaussian_model.get_optimizable_params(
            lr_position=gaussian_config.get('position_lr', 1e-4),
            lr_density=gaussian_config.get('density_lr', 1e-3),
            lr_scale=gaussian_config.get('scale_lr', 5e-4),
            lr_rotation=gaussian_config.get('rotation_lr', 1e-4)
        )
        self.optimizer = optim.Adam(params)
        scheduler_config = self.config['training'].get('lr_scheduler', {})
        if scheduler_config.get('type', 'exponential') == 'exponential':
            self.scheduler = ExponentialLR(self.optimizer, gamma=scheduler_config.get('gamma', 0.999))
        else:
            remaining = self.config['training']['max_iterations'] - self.current_iteration
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(remaining, 1))

    def train_step(self) -> Dict[str, float]:
        self.gaussian_model.train()
        self.optimizer.zero_grad()
        volume, loss_dict = self.forward_with_loss()
        loss_dict['total_loss'].backward()
        if self.gaussian_model.positions.grad is not None:
            self._position_grads_for_densify = self.gaussian_model.positions.grad.detach().clone()
        else:
            self._position_grads_for_densify = None
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.gaussian_model.parameters(), max_grad_norm)
        self.optimizer.step()
        return {k: v.item() for k, v in loss_dict.items()}

    def evaluate(self) -> Dict[str, float]:
        self.gaussian_model.eval()
        with torch.no_grad():
            vol_result = self._get_volume()
            # Handle tuple (vol_r, vol_i) from CUDA path
            if isinstance(vol_result, tuple):
                vol_r, vol_i = vol_result
                volume = torch.complex(vol_r, vol_i)
            else:
                volume = vol_result
            metrics = evaluate_reconstruction(pred=volume, target=self.target_image, compute_3d_ssim=True)
        return metrics

    def save_checkpoint(self, iteration: int, is_best: bool = False):
        checkpoint = {
            'iteration': iteration,
            'gaussian_state': self.gaussian_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr, 'best_ssim': self.best_ssim,
            'config': self.config
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best.pth'))
        save_every = self.config['training'].get('save_every', 500)
        custom_checkpoints = self.config['training'].get('custom_checkpoints', [10, 50, 100, 300, 600, 800, 1000, 1200])
        if iteration % save_every == 0 or iteration in custom_checkpoints:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'checkpoint_{iteration:06d}.pth'))

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.gaussian_model.load_state_dict(checkpoint['gaussian_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.current_iteration = checkpoint['iteration']
        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        self.best_ssim = checkpoint.get('best_ssim', 0.0)
        print(f"Loaded checkpoint from iteration {self.current_iteration}")

    def save_reconstruction(self, iteration: int):
        self.gaussian_model.eval()
        with torch.no_grad():
            vol_result = self._get_volume()
            # Handle tuple (vol_r, vol_i) from CUDA path
            if isinstance(vol_result, tuple):
                vol_r, vol_i = vol_result
                volume = torch.complex(vol_r, vol_i)
            else:
                volume = vol_result
            volume_np = volume.detach().cpu().numpy()
            if savemat is None:
                raise ImportError("Saving .mat requires scipy")
            savemat(os.path.join(self.result_dir, f'reconstruction_{iteration:06d}.mat'),
                    {'reconstruction': volume_np}, do_compression=True)
            savemat(os.path.join(self.result_dir, 'reconstruction_final.mat'),
                    {'reconstruction': volume_np}, do_compression=True)

    def train(self, resume_from: Optional[str] = None):
        train_config = self.config['training']
        max_iterations = train_config['max_iterations']
        eval_every = train_config.get('eval_every', 100)
        log_every = train_config.get('log_every', 50)
        save_every = train_config.get('save_every', 500)
        custom_checkpoints = train_config.get('custom_checkpoints', [10, 50, 100, 300, 600, 800, 1000, 1200])

        if resume_from is not None:
            self.load_checkpoint(resume_from)

        start_iter = self.current_iteration
        print(f"\nStarting training from iteration {start_iter}")
        print(f"Total iterations: {max_iterations}")
        print("-" * 50)

        pbar = tqdm(range(start_iter, max_iterations), desc="Training", dynamic_ncols=True)
        for iteration in pbar:
            self.current_iteration = iteration
            loss_dict = self.train_step()
            adaptive_stats = self.adaptive_density_control(iteration)
            self.scheduler.step()

            if iteration % log_every == 0:
                grad_stats = self.compute_gradient_stats()
                mean_grad = grad_stats.get('mean_grad', 0.0)
                if isinstance(mean_grad, torch.Tensor):
                    mean_grad = mean_grad.item()
                pbar.set_postfix({'loss': f"{loss_dict['total_loss']:.2e}", 'grad': f"{mean_grad:.2e}", 'n_pts': self.gaussian_model.num_points})
                if self.writer:
                    self.writer.add_scalar('Loss/total_loss', loss_dict['total_loss'], iteration)
                    self.writer.add_scalar('Stats/num_points', self.gaussian_model.num_points, iteration)
                    self.writer.add_scalar('Stats/mean_grad', mean_grad, iteration)

            if iteration % eval_every == 0 or iteration == max_iterations - 1:
                metrics = self.evaluate()
                is_best = metrics['psnr'] > self.best_psnr
                if is_best:
                    self.best_psnr = metrics['psnr']
                    self.best_ssim = metrics['ssim']
                if self.writer:
                    self.writer.add_scalar('Metrics/PSNR', metrics['psnr'], iteration)
                    self.writer.add_scalar('Metrics/SSIM', metrics['ssim'], iteration)
                print(f"\n[Iter {iteration}] PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
                if adaptive_stats['split'] > 0 or adaptive_stats['clone'] > 0 or adaptive_stats['prune'] > 0:
                    print(f"  Density: split={adaptive_stats['split']}, clone={adaptive_stats['clone']}, prune={adaptive_stats['prune']}")
                print(f"  Num Gaussians: {self.gaussian_model.num_points}")
                self.save_checkpoint(iteration, is_best)
            else:
                if (iteration % save_every == 0) or (iteration in custom_checkpoints):
                    self.save_checkpoint(iteration, is_best=False)

        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        self.save_reconstruction(max_iterations)
        return self.best_psnr, self.best_ssim
