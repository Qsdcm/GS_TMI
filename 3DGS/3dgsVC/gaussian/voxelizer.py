import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Tuple
import math

class Voxelizer(nn.Module):
    def __init__(self, volume_shape: Tuple[int, int, int], device: str = "cuda:0"):
        super().__init__()
        self.volume_shape = volume_shape
        self.device = device

    def _process_chunk(
        self,
        c_center: torch.Tensor,
        c_cov_inv: torch.Tensor,
        c_density: torch.Tensor,
        c_radius: torch.Tensor,
        shape_tensor: torch.Tensor,
        D: int, H: int, W: int,
    ) -> torch.Tensor:
        """Process a single chunk, returning a flat volume contribution."""
        device = c_center.device

        # Dynamic kernel size
        max_r = torch.ceil(c_radius.max()).int().item()
        max_r = min(max_r, 20)  # Hardware cap

        if max_r == 0:
            return torch.zeros(D * H * W, dtype=torch.complex64, device=device)

        # Generate local grid
        k_rng = torch.arange(-max_r, max_r + 1, device=device)
        kz, ky, kx = torch.meshgrid(k_rng, k_rng, k_rng, indexing='ij')
        kernel_offsets = torch.stack([kz, ky, kx], dim=-1).reshape(-1, 3).float()

        # Batch expand: (B, K^3, 3)
        c_center_round = torch.round(c_center)
        global_coords = c_center_round.unsqueeze(1) + kernel_offsets.unsqueeze(0)

        # Valid coordinate mask
        gz, gy, gx = global_coords[..., 0], global_coords[..., 1], global_coords[..., 2]
        valid_mask = (gz >= 0) & (gz < D) & (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W)

        # Mahalanobis distance
        diff_vox = global_coords - c_center.unsqueeze(1)
        diff_norm = diff_vox / (shape_tensor * 0.5)

        diff_emb = diff_norm @ c_cov_inv
        mahal = (diff_emb * diff_norm).sum(dim=-1)

        # 3-sigma mask
        mask_final = valid_mask & (mahal <= 9.0)

        # Sparse accumulation
        valid_indices = torch.nonzero(mask_final, as_tuple=True)
        if valid_indices[0].numel() == 0:
            return torch.zeros(D * H * W, dtype=torch.complex64, device=device)

        b_idx, p_idx = valid_indices

        weights = torch.exp(-0.5 * mahal[b_idx, p_idx])
        val_to_add = weights * c_density[b_idx]

        z_idx = gz[b_idx, p_idx].long()
        y_idx = gy[b_idx, p_idx].long()
        x_idx = gx[b_idx, p_idx].long()
        flat_indices = z_idx * (H * W) + y_idx * W + x_idx

        chunk_vol = torch.zeros(D * H * W, dtype=torch.complex64, device=device)
        chunk_vol.scatter_add_(0, flat_indices, val_to_add)
        return chunk_vol

    def forward(
        self,
        positions: torch.Tensor, # (N, 3)
        scales: torch.Tensor,    # (N, 3)
        rotations: torch.Tensor, # (N, 4)
        density: torch.Tensor,   # (N,)
        chunk_size: int = 4096
    ) -> torch.Tensor:
        D, H, W = self.volume_shape
        device = self.device

        from .gaussian_model import quaternion_to_rotation_matrix

        # Convert to voxel coordinates
        shape_tensor = torch.tensor([D, H, W], device=device).float()
        center_vox = (positions + 1.0) * 0.5 * shape_tensor - 0.5
        scale_vox = scales * shape_tensor * 0.5

        # 3-sigma radius
        radius_vox = scale_vox.max(dim=-1)[0] * 3.0

        # Sort by radius (small first) so each chunk has similar-sized points
        sort_indices = torch.argsort(radius_vox)

        center_sorted = center_vox[sort_indices]
        radius_sorted = radius_vox[sort_indices]
        density_sorted = density[sort_indices]
        scales_sorted = scales[sort_indices]
        rotations_sorted = rotations[sort_indices]

        # Covariance inverse
        R = quaternion_to_rotation_matrix(rotations_sorted)
        S_inv = torch.reciprocal(scales_sorted + 1e-8)
        L = R * S_inv.unsqueeze(1)
        cov_inv_sorted = L @ L.transpose(1, 2)

        volume_flat = torch.zeros(D * H * W, dtype=torch.complex64, device=device)

        N = positions.shape[0]

        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)

            c_center = center_sorted[i:end]
            c_cov_inv = cov_inv_sorted[i:end]
            c_density = density_sorted[i:end]
            c_radius = radius_sorted[i:end]

            # Dynamic chunk_size: if kernel is large, process fewer points at a time
            max_r_est = torch.ceil(c_radius.max()).int().item()
            max_r_est = min(max_r_est, 10)
            kernel_size = (2 * max_r_est + 1) ** 3

            if kernel_size > 1000:
                # Large kernel: split this chunk into sub-chunks
                # Target ~500MB peak: each element needs ~5 tensors * 3 floats * 4 bytes = 60 bytes per kernel point
                target_bytes = 500 * 1024 * 1024
                sub_chunk = max(32, target_bytes // (kernel_size * 60))
                sub_chunk = min(sub_chunk, end - i)
            else:
                sub_chunk = end - i

            for j in range(0, end - i, sub_chunk):
                j_end = min(j + sub_chunk, end - i)
                chunk_contribution = grad_checkpoint(
                    self._process_chunk,
                    c_center[j:j_end],
                    c_cov_inv[j:j_end],
                    c_density[j:j_end],
                    c_radius[j:j_end],
                    shape_tensor,
                    D, H, W,
                    use_reentrant=False,
                )
                volume_flat = volume_flat + chunk_contribution

        return volume_flat.view(D, H, W)