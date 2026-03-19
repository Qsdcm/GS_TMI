import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import math 

def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """将四元数转换为旋转矩阵"""
    norm = quaternion.norm(dim=-1, keepdim=True)
    quaternion = quaternion / (norm + 1e-8)
    w, x, y, z = quaternion.unbind(-1)
    row0 = torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1)
    row1 = torch.stack([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w], dim=-1)
    row2 = torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

class GaussianModel3D(nn.Module):
    def __init__(
        self,
        num_points: int,
        volume_shape: Tuple[int, int, int],
        initial_positions: Optional[torch.Tensor] = None,
        initial_densities: Optional[torch.Tensor] = None,
        initial_scales: Optional[torch.Tensor] = None,
        device: str = "cuda:0"
    ):
        super().__init__()
        self.num_points = num_points
        self.volume_shape = volume_shape
        self.device = device
        self._init_parameters(initial_positions, initial_densities, initial_scales)
        
    def _init_parameters(
        self,
        positions: Optional[torch.Tensor],
        densities: Optional[torch.Tensor],
        scales: Optional[torch.Tensor]
    ):
        N = self.num_points
        
        # 1. 位置
        if positions is None:
            positions = torch.rand(N, 3, device=self.device) * 2 - 1
        self.positions = nn.Parameter(positions)
        
        # 2. 尺度 (Log空间)
        if scales is None:
            # 初始化为约1个体素大小 (2.0 / 150 ≈ 0.013)
            # 确保初始尺度大于 scale_threshold (0.0005) 以允许分裂
            scales = torch.ones(N, 3, device=self.device) * (2.0 / self.volume_shape[0])
        self.scales = nn.Parameter(torch.log(torch.clamp(scales, min=1e-8)))
        
        # 3. 旋转 (四元数 [1,0,0,0])
        rotations = torch.zeros(N, 4, device=self.device)
        rotations[:, 0] = 1.0
        self.rotations = nn.Parameter(rotations)
        
        # 4. 密度
        if densities is None:
            densities = torch.randn(N, dtype=torch.complex64, device=self.device) * 0.1
        self.density_real = nn.Parameter(densities.real)
        self.density_imag = nn.Parameter(densities.imag)
        
    @property
    def density(self) -> torch.Tensor:
        return torch.complex(self.density_real, self.density_imag)
    
    def get_scale_values(self) -> torch.Tensor:
        return torch.exp(self.scales)

    # 兼容接口
    def get_scales(self) -> torch.Tensor:
        return self.get_scale_values()

    def get_densities(self) -> torch.Tensor:
        return self.density
    
    def get_optimizable_params(self, lr_position=1e-4, lr_density=1e-3, lr_scale=5e-4, lr_rotation=1e-4):
        return [
            {'params': [self.positions], 'lr': lr_position},
            {'params': [self.scales], 'lr': lr_scale},
            {'params': [self.rotations], 'lr': lr_rotation},
            {'params': [self.density_real], 'lr': lr_density},
            {'params': [self.density_imag], 'lr': lr_density},
        ]

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float = 0.0002,
        scale_threshold: float = 0.0005, # 默认值调低
        use_long_axis_splitting: bool = True
    ) -> int:
        with torch.no_grad():
            scales = self.get_scale_values()
            max_scales = scales.max(dim=-1)[0]
            
            mask = (grads > grad_threshold) & (max_scales > scale_threshold)
            if mask.sum() == 0: return 0
            
            p_pos = self.positions[mask]
            p_scale = scales[mask]
            p_rot = self.rotations[mask]
            p_den_r = self.density_real[mask]
            p_den_i = self.density_imag[mask]
            
            K = p_pos.shape[0]
            
            if use_long_axis_splitting:
                # 论文策略: 沿长轴分裂
                longest_axis = p_scale.argmax(dim=-1)
                offset_val = p_scale[torch.arange(K), longest_axis]
                
                local_shift = torch.zeros_like(p_pos)
                local_shift[torch.arange(K), longest_axis] = offset_val * 1.0 
                
                R = quaternion_to_rotation_matrix(p_rot)
                global_shift = torch.bmm(R, local_shift.unsqueeze(-1)).squeeze(-1)
                
                new_pos_1 = p_pos + global_shift
                new_pos_2 = p_pos - global_shift
                
                new_scale = p_scale.clone()
                # 论文: 其他两轴缩放 0.85
                mask_other = torch.ones_like(new_scale, dtype=torch.bool)
                mask_other[torch.arange(K), longest_axis] = False
                new_scale[mask_other] *= 0.85
                
                # 长轴本身也需要变短以避免重叠太大，这里保持0.6
                new_scale[torch.arange(K), longest_axis] *= 0.6
                
                new_positions = torch.cat([new_pos_1, new_pos_2], dim=0)
                new_scales = torch.cat([new_scale, new_scale], dim=0)
                new_rotations = torch.cat([p_rot, p_rot], dim=0)
                
                # 修正: 论文 Page 3 "central values are scaled down by a factor of 0.6"
                new_den_r = torch.cat([p_den_r * 0.6, p_den_r * 0.6], dim=0)
                new_den_i = torch.cat([p_den_i * 0.6, p_den_i * 0.6], dim=0)
                
            else:
                # 原始策略
                std = p_scale
                offset = torch.randn_like(p_pos) * std * 0.5
                new_positions = torch.cat([p_pos - offset, p_pos + offset], dim=0)
                new_scales = torch.cat([p_scale/1.6, p_scale/1.6], dim=0)
                new_rotations = torch.cat([p_rot, p_rot], dim=0)
                new_den_r = torch.cat([p_den_r * 0.6, p_den_r * 0.6], dim=0)
                new_den_i = torch.cat([p_den_i * 0.6, p_den_i * 0.6], dim=0)

            keep_mask = ~mask
            self._update_params(keep_mask, new_positions, new_scales, new_rotations, new_den_r, new_den_i)
            return K

    def densify_and_clone(self, grads, grad_threshold, scale_threshold):
        with torch.no_grad():
             scales = self.get_scale_values()
             max_scales = scales.max(dim=-1)[0]
             mask = (grads > grad_threshold) & (max_scales <= scale_threshold)
             if mask.sum() == 0: return 0
             
             new_pos = self.positions[mask]
             new_scale = scales[mask]
             new_rot = self.rotations[mask]
             new_dr = self.density_real[mask]
             new_di = self.density_imag[mask]
             
             self.positions = nn.Parameter(torch.cat([self.positions, new_pos], dim=0))
             self.scales = nn.Parameter(torch.cat([self.scales, torch.log(new_scale+1e-8)], dim=0))
             self.rotations = nn.Parameter(torch.cat([self.rotations, new_rot], dim=0))
             self.density_real = nn.Parameter(torch.cat([self.density_real, new_dr], dim=0))
             self.density_imag = nn.Parameter(torch.cat([self.density_imag, new_di], dim=0))
             self.num_points = self.positions.shape[0]
             return mask.sum().item()

    def prune(self, opacity_threshold):
        with torch.no_grad():
            density_mag = torch.abs(self.density)
            mask = density_mag > opacity_threshold
            if mask.sum() == self.num_points: return 0
            self._update_params(mask, None, None, None, None, None, is_prune=True)
            return (len(mask) - mask.sum().item())

    def _update_params(self, mask, new_pos=None, new_scale=None, new_rot=None, new_dr=None, new_di=None, is_prune=False):
        if is_prune:
            self.positions = nn.Parameter(self.positions[mask])
            self.scales = nn.Parameter(self.scales[mask])
            self.rotations = nn.Parameter(self.rotations[mask])
            self.density_real = nn.Parameter(self.density_real[mask])
            self.density_imag = nn.Parameter(self.density_imag[mask])
        else:
            self.positions = nn.Parameter(torch.cat([self.positions[mask], new_pos], dim=0))
            self.scales = nn.Parameter(torch.cat([self.scales[mask], torch.log(new_scale + 1e-8)], dim=0))
            self.rotations = nn.Parameter(torch.cat([self.rotations[mask], new_rot], dim=0))
            self.density_real = nn.Parameter(torch.cat([self.density_real[mask], new_dr], dim=0))
            self.density_imag = nn.Parameter(torch.cat([self.density_imag[mask], new_di], dim=0))
        self.num_points = self.positions.shape[0]

    @classmethod
    def from_image(cls, image: torch.Tensor, num_points: int, initial_scale: float = 2.0, device: str = "cuda:0"):
        # Handle [2, D, H, W] input
        if image.ndim == 4 and image.shape[0] == 2:
            C, D, H, W = image.shape
            # Compute magnitude for initialization probability
            mag = torch.sqrt(image[0]**2 + image[1]**2)
            # Flatten for sampling
            mag_flat = mag.flatten()
            
            # Densities from real/imag parts
            densities_real = image[0]
            densities_imag = image[1]
        else:
            D, H, W = image.shape
            mag = torch.abs(image)
            mag_flat = mag.flatten()
            
            if torch.is_complex(image):
                densities_real = image.real
                densities_imag = image.imag
            else:
                densities_real = image
                densities_imag = torch.zeros_like(image)

        threshold = torch.quantile(mag_flat, 0.90)
        candidates = torch.nonzero(mag_flat > threshold).squeeze()
        
        if candidates.numel() < num_points:
            candidates = torch.arange(mag_flat.numel(), device=device)
        
        indices_idx = torch.randperm(candidates.numel(), device=device)[:num_points]
        indices = candidates[indices_idx]
        
        z = indices // (H * W)
        rem = indices % (H * W)
        y = rem // W
        x = rem % W
        
        positions = torch.stack([
            z.float() / D * 2 - 1,
            y.float() / H * 2 - 1,
            x.float() / W * 2 - 1
        ], dim=-1)
        
        if num_points > 3:
            dist_mat = torch.cdist(positions, positions)
            dist_mat.fill_diagonal_(float('inf'))
            vals, _ = dist_mat.topk(3, largest=False, dim=1)
            mean_dist = vals.mean(dim=1, keepdim=True)
            scales_init = mean_dist.repeat(1, 3)
        else:
            scales_init = torch.ones(num_points, 3, device=device) * (2.0/D)
        
        # 密度初始化缩放因子 (Paper Table 1: k=0.15)
        init_den_r = densities_real.flatten()[indices] * 0.15
        init_den_i = densities_imag.flatten()[indices] * 0.15
        
        initial_densities = torch.complex(init_den_r, init_den_i)
        
        return cls(
            num_points=num_points,
            volume_shape=(D, H, W),
            initial_positions=positions,
            initial_densities=initial_densities,
            initial_scales=scales_init,
            device=device
        )