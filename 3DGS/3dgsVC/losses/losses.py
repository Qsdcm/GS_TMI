"""Loss definitions for 3DGSMR."""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .kspace_loss import KSpaceConsistencyLoss


def _to_magnitude(image: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(image):
        return torch.abs(image)
    if image.ndim >= 1 and image.shape[0] == 2:
        return torch.sqrt(image[0] ** 2 + image[1] ** 2 + 1e-12)
    return image


class MagnitudeTVLoss(nn.Module):
    """TV(|X|) exactly as used in the paper."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mag = _to_magnitude(image)
        diff_d = torch.abs(mag[1:, :, :] - mag[:-1, :, :])
        diff_h = torch.abs(mag[:, 1:, :] - mag[:, :-1, :])
        diff_w = torch.abs(mag[:, :, 1:] - mag[:, :, :-1])
        return diff_d.sum() + diff_h.sum() + diff_w.sum()


class ImageLoss(nn.Module):
    """Legacy non-paper image-domain auxiliary loss."""

    def __init__(self, loss_type: str = "l2"):
        super().__init__()
        if loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.loss_type = loss_type

    def forward(
        self,
        image_pred: torch.Tensor,
        image_target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = _to_magnitude(image_pred) - _to_magnitude(image_target)
        if weight_map is not None:
            diff = diff * weight_map
        if self.loss_type == "l1":
            return torch.abs(diff).sum()
        return (diff ** 2).sum()


class PaperLoss(nn.Module):
    """Paper objective: ||A(X)-b|| + lambda * TV(|X|)."""

    def __init__(
        self,
        kspace_weight: float = 1.0,
        tv_weight: float = 0.1,
        loss_type: str = "l2",
        image_weight: float = 0.0,
    ):
        super().__init__()
        self.kspace_weight = kspace_weight
        self.tv_weight = tv_weight
        self.image_weight = image_weight
        self.kspace_loss = KSpaceConsistencyLoss(loss_type)
        self.tv_loss = MagnitudeTVLoss()
        self.image_loss = ImageLoss(loss_type)

    def forward(
        self,
        kspace_pred: torch.Tensor,
        kspace_target: torch.Tensor,
        mask: torch.Tensor,
        image_pred: Optional[torch.Tensor] = None,
        image_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=kspace_pred.device)
        losses: Dict[str, torch.Tensor] = {}

        kspace_l = self.kspace_loss(kspace_pred, kspace_target, mask)
        total_loss = total_loss + self.kspace_weight * kspace_l
        losses["kspace_loss"] = kspace_l

        if self.tv_weight > 0 and image_pred is not None:
            tv_l = self.tv_loss(image_pred)
            total_loss = total_loss + self.tv_weight * tv_l
            losses["tv_loss"] = tv_l

        if self.image_weight > 0 and image_pred is not None and image_target is not None:
            image_l = self.image_loss(image_pred, image_target)
            total_loss = total_loss + self.image_weight * image_l
            losses["image_loss"] = image_l

        losses["total_loss"] = total_loss
        return losses


class DataConsistencyLayer(nn.Module):
    def forward(self, kspace_pred, kspace_measured, mask):
        return kspace_pred * (1 - mask) + kspace_measured * mask


KSpaceLoss = KSpaceConsistencyLoss
TVLoss = MagnitudeTVLoss
CombinedLoss = PaperLoss
