import torch
import torch.nn as nn


class KSpaceConsistencyLoss(nn.Module):
    """Paper loss term: data consistency in measured k-space only."""

    def __init__(self, loss_type: str = "l2"):
        super().__init__()
        if loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.loss_type = loss_type

    def forward(
        self,
        kspace_pred: torch.Tensor,
        kspace_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        diff = (kspace_pred - kspace_target) * mask
        if self.loss_type == "l1":
            return torch.abs(diff).sum()
        return (torch.abs(diff) ** 2).sum()
