"""
3DGSMR Trainer (Final Revised Version)

完全对齐论文:
1. Loss使用 Sum reduction (配合losses.py修改) -> 梯度量级正常
2. 传递极小的 scale_threshold (0.0005) -> 允许细微结构分裂
3. Densification频率=100iter, 持续到2500iter -> 确保长到400k点
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
from gaussian import GaussianModel3D, Voxelizer
from losses import CombinedLoss
from metrics import evaluate_reconstruction

class GaussianTrainer:
    """
    3D Gaussian MRI重建训练器
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device = None
    ):
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"Using device: {self.device}")
        
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_output()
        
        self.current_iteration = 0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        
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

        # kspace_full is NOT used during training/eval - keep on CPU for save_reconstruction only
        self.kspace_full_cpu = data['kspace_full']
        self.kspace_undersampled = data['kspace_undersampled'].to(self.device)
        self.mask = data['mask'].to(self.device)
        self.volume_shape = data['volume_shape']
        self.target_image = data['ground_truth'].to(self.device)
        # zero_filled is only needed for model init - will be freed after _setup_model
        self._zero_filled_for_init = data['zero_filled'].to(self.device)
        self.sensitivity_maps = data['sensitivity_maps'].to(self.device)

        # Store num_coils for convenience
        self.num_coils = self.sensitivity_maps.shape[0]

        print(f"Volume shape: {self.volume_shape}")
        print(f"K-space shape (undersampled): {self.kspace_undersampled.shape}")
        print(f"Number of coils: {self.num_coils}")
        print(f"Acceleration factor: {data_config['acceleration_factor']}")

        # Free dataset's internal copies to save CPU memory
        del self.dataset
        
    def _setup_model(self):
        gaussian_config = self.config['gaussian']
        init_method = gaussian_config.get('init_method', 'from_image')

        if init_method == 'from_image':
            self.gaussian_model = GaussianModel3D.from_image(
                image=self._zero_filled_for_init,
                num_points=gaussian_config['initial_num_points'],
                initial_scale=gaussian_config.get('initial_scale', 2.0),
                device=str(self.device)
            )
        else:
            self.gaussian_model = GaussianModel3D(
                num_points=gaussian_config['initial_num_points'],
                volume_shape=tuple(self.volume_shape),
                initial_scale=gaussian_config.get('initial_scale', 2.0),
                device=str(self.device)
            )

        # Free zero_filled - no longer needed
        del self._zero_filled_for_init
        torch.cuda.empty_cache()

        self.voxelizer = Voxelizer(
            volume_shape=tuple(self.volume_shape),
            device=str(self.device)
        )
        print(f"Initialized with {self.gaussian_model.num_points} Gaussians")
        
    def _setup_loss(self):
        loss_config = self.config['loss']
        self.criterion = CombinedLoss(
            kspace_weight=loss_config.get('kspace_weight', 1.0),
            image_weight=loss_config.get('image_weight', 0.0),
            tv_weight=loss_config.get('tv_weight', 0.0),
            loss_type=loss_config.get('loss_type', 'l2') # 论文倾向 L2
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
            self.scheduler = ExponentialLR(
                self.optimizer,
                gamma=scheduler_config.get('gamma', 0.999)
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['max_iterations']
            )
            
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
        """GS -> Image Domain (Complex), returns [D, H, W] complex"""
        positions = self.gaussian_model.positions
        if hasattr(self.gaussian_model, 'get_scales'):
            scales = self.gaussian_model.get_scales()
        else:
            scales = self.gaussian_model.get_scale_values()

        rotations = self.gaussian_model.rotations
        if hasattr(self.gaussian_model, 'get_densities'):
            density = self.gaussian_model.get_densities()
        else:
            density = self.gaussian_model.density

        volume = self.voxelizer(
            positions=positions,
            scales=scales,
            rotations=rotations,
            density=density
        )
        return volume

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Original forward (kept for evaluate/test). Processes coil-by-coil to save memory."""
        volume = self._get_volume()

        num_coils = self.sensitivity_maps.shape[0]
        kspace_real_list = []
        kspace_imag_list = []

        for c in range(num_coils):
            coil_image = volume * self.sensitivity_maps[c]  # [D, H, W] complex
            coil_kspace = fft3c(coil_image.unsqueeze(0)).squeeze(0)  # [D, H, W] complex
            kspace_real_list.append(coil_kspace.real)
            kspace_imag_list.append(coil_kspace.imag)

        kspace_pred = torch.cat([torch.stack(kspace_real_list, dim=0),
                                  torch.stack(kspace_imag_list, dim=0)], dim=0)
        return volume, kspace_pred

    def _coil_loss(self, volume, coil_csm, target_real, target_imag, mask, loss_type_is_l1):
        """Compute loss for a single coil. Designed for gradient checkpointing."""
        coil_image = volume * coil_csm
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
        Processes one coil at a time with gradient checkpointing,
        accumulating loss without materializing the full [2*Coils, D, H, W] tensor.
        Peak memory: O(D*H*W) instead of O(Coils*D*H*W).
        """
        volume = self._get_volume()  # [D, H, W] complex

        num_coils = self.sensitivity_maps.shape[0]
        kspace_loss = torch.tensor(0.0, device=volume.device)

        loss_type = self.config['loss'].get('loss_type', 'l2')
        is_l1 = torch.tensor(loss_type == "l1", device=volume.device)

        for c in range(num_coils):
            coil_loss = grad_checkpoint(
                self._coil_loss,
                volume,
                self.sensitivity_maps[c],
                self.kspace_undersampled[c],
                self.kspace_undersampled[c + num_coils],
                self.mask,
                is_l1,
                use_reentrant=False,
            )
            kspace_loss = kspace_loss + coil_loss

        # Build loss dict
        losses = {}
        total_loss = self.config['loss'].get('kspace_weight', 1.0) * kspace_loss
        losses['kspace_loss'] = kspace_loss

        # Image loss (if enabled)
        image_weight = self.config['loss'].get('image_weight', 0.0)
        if image_weight > 0 and self.target_image is not None:
            from losses import ImageLoss
            img_loss_fn = ImageLoss(loss_type)
            image_l = img_loss_fn(volume, self.target_image)
            losses['image_loss'] = image_l
            total_loss = total_loss + image_weight * image_l

        # TV loss (if enabled)
        tv_weight = self.config['loss'].get('tv_weight', 0.0)
        if tv_weight > 0:
            from losses.losses import TVLoss
            tv_l = TVLoss()(volume)
            losses['tv_loss'] = tv_l
            total_loss = total_loss + tv_weight * tv_l

        losses['total_loss'] = total_loss
        return volume, losses
    
    def compute_gradient_stats(self) -> Dict[str, torch.Tensor]:
        """计算梯度统计信息，增加 mean_grad 用于调试"""
        if self.gaussian_model.positions.grad is None:
            return {}
        
        grad_norm = torch.norm(self.gaussian_model.positions.grad, dim=-1)
        return {
            'grad_norm': grad_norm,
            'mean_grad': grad_norm.mean(), # 关键调试指标
            'max_grad': grad_norm.max()
        }
    
    def adaptive_density_control(self, iteration: int) -> Dict[str, int]:
        """
        自适应密度控制 (Paper Section IV)
        """
        adaptive_config = self.config['adaptive_control']
        if not adaptive_config.get('enable', True):
            return {'split': 0, 'clone': 0, 'prune': 0}

        stats = {'split': 0, 'clone': 0, 'prune': 0}
        
        # 参数读取
        start_iter = adaptive_config.get('densify_from_iter', 100)
        end_iter = adaptive_config.get('densify_until_iter', 2500) # 延长到2500
        interval = adaptive_config.get('densify_every', 100)
        max_num_points = self.config['gaussian'].get('max_num_points', 400000)

        # 硬性上限检查
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
        # 梯度阈值: 如果Loss是Sum, 梯度会很大, 0.0002很容易满足
        grad_threshold = adaptive_config.get('grad_threshold', 0.0002)
        
        if hasattr(self.gaussian_model, 'get_scale_values'):
            scales = self.gaussian_model.get_scale_values()
        else:
            scales = self.gaussian_model.get_scales()
        max_scale = scales.max(dim=-1)[0]
        
        # 关键: 读取极小的 scale_threshold (例如 0.0005)
        # 必须确保 config 中设置了这个值，否则分裂会被默认值 0.01 拦截
        scale_threshold = adaptive_config.get('scale_threshold', 0.0005)
        max_scale_limit = adaptive_config.get('max_scale', 0.5)
        
        high_grad_mask = grad_norm > grad_threshold
        
        # --- Split (Long-axis) ---
        if adaptive_config.get('use_long_axis_splitting', True):
            if high_grad_mask.shape[0] == self.gaussian_model.num_points:
                # 只有梯度大且尺度也够大的点才分裂
                # 如果 scale_threshold 太大 (如 0.01)，小点就永远无法分裂了
                split_mask = high_grad_mask & (max_scale > scale_threshold)
                
                if split_mask.sum() > 0:
                    # 预测分裂后的数量
                    if self.gaussian_model.num_points + split_mask.sum() <= max_num_points:
                        self.gaussian_model.densify_and_split(
                            grads=grad_norm,
                            grad_threshold=grad_threshold,
                            scale_threshold=scale_threshold, # 传递更小的阈值
                            use_long_axis_splitting=True
                        )
                        stats['split'] = split_mask.sum().item()
                        high_grad_mask = None # 消耗掉mask
        
        # --- Clone (High Accel usually disabled) ---
        if adaptive_config.get('use_cloning', False) and high_grad_mask is not None:
             if hasattr(self.gaussian_model, 'get_scale_values'):
                scales = self.gaussian_model.get_scale_values()
             else:
                scales = self.gaussian_model.get_scales()
             max_scale = scales.max(dim=-1)[0]
            
             if high_grad_mask.shape[0] == self.gaussian_model.num_points:
                clone_mask = high_grad_mask & (max_scale <= scale_threshold)
                if clone_mask.sum() > 0:
                    if self.gaussian_model.num_points + clone_mask.sum() <= max_num_points:
                        self.gaussian_model.densify_and_clone(grad_norm, grad_threshold, scale_threshold)
                        stats['clone'] = clone_mask.sum().item()
        
        # --- Prune ---
        opacity_threshold = adaptive_config.get('opacity_threshold', 0.01)
        
        if hasattr(self.gaussian_model, 'get_densities'):
            densities = torch.abs(self.gaussian_model.get_densities())
        else:
            densities = torch.abs(self.gaussian_model.density)
            
        if hasattr(self.gaussian_model, 'get_scale_values'):
            scales = self.gaussian_model.get_scale_values()
        else:
            scales = self.gaussian_model.get_scales()
        max_scale = scales.max(dim=-1)[0]
        
        prune_mask = (densities < opacity_threshold) | (max_scale > max_scale_limit)
        
        if (self.gaussian_model.num_points - prune_mask.sum()) >= 100:
            if prune_mask.sum() > 0:
                self.gaussian_model.prune(opacity_threshold)
                stats['prune'] = prune_mask.sum().item()
        
        # 只要结构变了，就必须重建优化器
        if stats['split'] > 0 or stats['clone'] > 0 or stats['prune'] > 0:
            train_config = self.config['training']
            gaussian_config = self.config['gaussian']
            params = self.gaussian_model.get_optimizable_params(
                lr_position=gaussian_config.get('position_lr', 1e-4),
                lr_density=gaussian_config.get('density_lr', 1e-3),
                lr_scale=gaussian_config.get('scale_lr', 5e-4),
                lr_rotation=gaussian_config.get('rotation_lr', 1e-4)
            )
            self.optimizer = optim.Adam(params)
        
        return stats
    
    def train_step(self) -> Dict[str, float]:
        self.gaussian_model.train()
        self.optimizer.zero_grad()

        # Use memory-efficient coil-by-coil forward + loss fusion
        volume, loss_dict = self.forward_with_loss()

        loss_dict['total_loss'].backward()

        # 梯度裁剪
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.gaussian_model.parameters(),
                max_grad_norm
            )

        self.optimizer.step()
        return {k: v.item() for k, v in loss_dict.items()}
    
    def evaluate(self) -> Dict[str, float]:
        self.gaussian_model.eval()
        with torch.no_grad():
            volume = self._get_volume()
            metrics = evaluate_reconstruction(
                pred=volume,
                target=self.target_image,
                compute_3d_ssim=True
            )
        return metrics
    
    def save_checkpoint(self, iteration: int, is_best: bool = False):
        checkpoint = {
            'iteration': iteration,
            'gaussian_state': self.gaussian_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'config': self.config
        }
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
        
        save_every = self.config['training'].get('save_every', 500)
        custom_checkpoints = self.config['training'].get(
            'custom_checkpoints',
            [10, 50, 100, 300, 600, 800, 1000, 1200]
        )
        
        if iteration % save_every == 0 or iteration in custom_checkpoints:
            iter_path = os.path.join(self.checkpoint_dir, f'checkpoint_{iteration:06d}.pth')
            torch.save(checkpoint, iter_path)
    
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
            volume = self._get_volume()
            volume_np = volume.detach().cpu().numpy()
            if savemat is None:
                raise ImportError(
                    "Saving .mat requires scipy. Please install it via: pip install scipy"
                )

            result_path = os.path.join(self.result_dir, f'reconstruction_{iteration:06d}.mat')
            savemat(result_path, {'reconstruction': volume_np}, do_compression=True)

            final_path = os.path.join(self.result_dir, 'reconstruction_final.mat')
            savemat(final_path, {'reconstruction': volume_np}, do_compression=True)
    
    def train(self, resume_from: Optional[str] = None):
        """完整训练流程"""
        train_config = self.config['training']
        max_iterations = train_config['max_iterations']
        eval_every = train_config.get('eval_every', 100)
        log_every = train_config.get('log_every', 50)
        save_every = train_config.get('save_every', 500)
        custom_checkpoints = train_config.get(
            'custom_checkpoints',
            [10, 50, 100, 300, 600, 800, 1000, 1200]
        )
        
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
            
            # 增强的Log信息
            if iteration % log_every == 0:
                grad_stats = self.compute_gradient_stats()
                mean_grad = grad_stats.get('mean_grad', 0.0)
                if isinstance(mean_grad, torch.Tensor):
                    mean_grad = mean_grad.item()
                    
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.2e}", # 科学计数法看大数
                    'grad': f"{mean_grad:.2e}",             # 监控梯度是否 > 0.0002
                    'n_pts': self.gaussian_model.num_points
                })
                
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
                
                # 打印分裂详情
                if adaptive_stats['split'] > 0 or adaptive_stats['clone'] > 0 or adaptive_stats['prune'] > 0:
                    print(f"  Density: split={adaptive_stats['split']}, clone={adaptive_stats['clone']}, prune={adaptive_stats['prune']}")
                    
                print(f"  Num Gaussians: {self.gaussian_model.num_points}")
                
                self.save_checkpoint(iteration, is_best)
            else:
                # 确保自定义检查点/固定间隔检查点一定会被保存
                if (iteration % save_every == 0) or (iteration in custom_checkpoints):
                    self.save_checkpoint(iteration, is_best=False)
        
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        self.save_reconstruction(max_iterations)
        return self.best_psnr, self.best_ssim