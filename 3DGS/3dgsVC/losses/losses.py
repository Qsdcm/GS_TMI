"""
Loss Functions for 3DGSMR (Revised)

Correction:
论文 Fig 5 显示 Loss 值在 10^5 - 10^6 量级，且 Table 1 使用 grad_threshold=0.02。
这意味着 Loss 必须是 Sum Reduction (求和) 而非 Mean Reduction (平均)。
使用 Mean 会导致梯度被除以像素数 (N~4e6)，使得梯度远小于 0.02,导致无法分裂。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

class KSpaceLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(
        self,
        kspace_pred: torch.Tensor,
        kspace_target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        # 只在采样位置计算
        kspace_pred_masked = kspace_pred * mask
        kspace_target_masked = kspace_target * mask
        
        diff = kspace_pred_masked - kspace_target_masked
        
        # 修正: 使用 sum() 而非 mean()
        if self.loss_type == "l1":
            loss = torch.abs(diff).sum()
        else:  # l2
            loss = (torch.abs(diff) ** 2).sum()
            
        return loss

class ImageLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(
        self,
        image_pred: torch.Tensor,
        image_target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        pred_mag = torch.abs(image_pred)
        target_mag = torch.abs(image_target)
        
        diff = pred_mag - target_mag
        
        if weight_map is not None:
            diff = diff * weight_map
        
        # 修正: 使用 sum() 而非 mean()
        if self.loss_type == "l1":
            loss = torch.abs(diff).sum()
        else:  # l2
            loss = (diff ** 2).sum()
            
        return loss

class CombinedLoss(nn.Module):
    def __init__(
        self,
        kspace_weight: float = 1.0,
        image_weight: float = 0.1,
        tv_weight: float = 0.0,
        loss_type: str = "l1"
    ):
        super().__init__()
        self.kspace_weight = kspace_weight
        self.image_weight = image_weight
        self.tv_weight = tv_weight
        
        # 论文主要关注 L2 data consistency (公式 2)
        # 如果 config 传进来 l1，这里会尊重 config，但建议使用 l2 配合 sum
        self.kspace_loss = KSpaceLoss(loss_type) 
        self.image_loss = ImageLoss(loss_type)
        # TV loss 通常保持 mean 或者根据需要调整权重，但在论文中 TV 权重为 0
        self.tv_loss = TVLoss()
        
    def forward(
        self,
        kspace_pred: torch.Tensor,
        kspace_target: torch.Tensor,
        mask: torch.Tensor,
        image_pred: Optional[torch.Tensor] = None,
        image_target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        losses = {}
        total_loss = torch.tensor(0.0, device=kspace_pred.device)
        
        # K-space loss
        kspace_l = self.kspace_loss(kspace_pred, kspace_target, mask)
        losses['kspace_loss'] = kspace_l
        total_loss = total_loss + self.kspace_weight * kspace_l
        
        # Image loss
        if self.image_weight > 0 and image_pred is not None and image_target is not None:
            image_l = self.image_loss(image_pred, image_target)
            losses['image_loss'] = image_l
            total_loss = total_loss + self.image_weight * image_l
        
        # TV loss (TV通常计算的是平均变分，如果这里不改sum，可能需要很大权重)
        if self.tv_weight > 0 and image_pred is not None:
            tv_l = self.tv_loss(image_pred)
            losses['tv_loss'] = tv_l
            total_loss = total_loss + self.tv_weight * tv_l
            
        losses['total_loss'] = total_loss
        return losses

# 保持 TVLoss 和 DataConsistencyLayer 不变或按需微调
class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.is_complex():
            image = torch.abs(image)
        diff_d = torch.abs(image[1:, :, :] - image[:-1, :, :])
        diff_h = torch.abs(image[:, 1:, :] - image[:, :-1, :])
        diff_w = torch.abs(image[:, :, 1:] - image[:, :, :-1])
        # TV 也可以改为 sum，保持量级一致
        tv_loss = diff_d.sum() + diff_h.sum() + diff_w.sum()
        return tv_loss

class DataConsistencyLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, kspace_pred, kspace_measured, mask):
        return kspace_pred * (1 - mask) + kspace_measured * mask