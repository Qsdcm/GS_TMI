import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    norm = quaternion.norm(dim=-1, keepdim=True)
    quaternion = quaternion / (norm + 1e-8)
    w, x, y, z = quaternion.unbind(-1)
    row0 = torch.stack([1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w], dim=-1)
    row1 = torch.stack([2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w], dim=-1)
    row2 = torch.stack([2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


class GaussianModel3D(nn.Module):
    def __init__(
        self,
        num_points: int,
        volume_shape: Tuple[int, int, int],
        initial_positions: Optional[torch.Tensor] = None,
        initial_densities: Optional[torch.Tensor] = None,
        initial_scales: Optional[torch.Tensor] = None,
        device: str = "cuda:0",
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
        scales: Optional[torch.Tensor],
    ):
        n_points = self.num_points
        if positions is None:
            positions = torch.rand(n_points, 3, device=self.device) * 2 - 1
        self.positions = nn.Parameter(positions)

        if scales is None:
            base_scale = torch.ones(n_points, 3, device=self.device) * (2.0 / max(self.volume_shape[0], 1))
            scales = base_scale
        self.scales = nn.Parameter(torch.log(torch.clamp(scales, min=1e-8)))

        rotations = torch.zeros(n_points, 4, device=self.device)
        rotations[:, 0] = 1.0
        self.rotations = nn.Parameter(rotations)

        if densities is None:
            densities = torch.randn(n_points, dtype=torch.complex64, device=self.device) * 0.1
        self.density_real = nn.Parameter(densities.real)
        self.density_imag = nn.Parameter(densities.imag)

    @property
    def density(self) -> torch.Tensor:
        return torch.complex(self.density_real, self.density_imag)

    def get_scale_values(self) -> torch.Tensor:
        return torch.exp(self.scales)

    def get_scales(self) -> torch.Tensor:
        return self.get_scale_values()

    def get_densities(self) -> torch.Tensor:
        return self.density

    def get_optimizable_params(self, lr_position=1e-4, lr_density=1e-3, lr_scale=5e-4, lr_rotation=1e-4):
        return [
            {"params": [self.positions], "lr": lr_position},
            {"params": [self.scales], "lr": lr_scale},
            {"params": [self.rotations], "lr": lr_rotation},
            {"params": [self.density_real], "lr": lr_density},
            {"params": [self.density_imag], "lr": lr_density},
        ]

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float = 0.0002,
        scale_threshold: float = 0.0005,
        use_long_axis_splitting: bool = True,
        long_axis_offset_factor: float = 1.0,
    ) -> int:
        with torch.no_grad():
            scales = self.get_scale_values()
            max_scales = scales.max(dim=-1)[0]
            mask = (grads > grad_threshold) & (max_scales > scale_threshold)
            if mask.sum() == 0:
                return 0

            parent_positions = self.positions[mask]
            parent_scales = scales[mask]
            parent_rotations = self.rotations[mask]
            parent_den_r = self.density_real[mask]
            parent_den_i = self.density_imag[mask]
            split_count = parent_positions.shape[0]

            if use_long_axis_splitting:
                longest_axis = parent_scales.argmax(dim=-1)
                child_scale = parent_scales.clone()
                child_scale[torch.arange(split_count), longest_axis] *= 0.5
                other_mask = torch.ones_like(child_scale, dtype=torch.bool)
                other_mask[torch.arange(split_count), longest_axis] = False
                child_scale[other_mask] *= 0.85

                local_shift = torch.zeros_like(parent_positions)
                local_shift[torch.arange(split_count), longest_axis] = (
                    child_scale[torch.arange(split_count), longest_axis] * long_axis_offset_factor
                )
                rotation_matrix = quaternion_to_rotation_matrix(parent_rotations)
                global_shift = torch.bmm(rotation_matrix, local_shift.unsqueeze(-1)).squeeze(-1)

                new_positions = torch.cat([parent_positions + global_shift, parent_positions - global_shift], dim=0)
                new_scales = torch.cat([child_scale, child_scale], dim=0)
                new_rotations = torch.cat([parent_rotations, parent_rotations], dim=0)
                new_den_r = torch.cat([parent_den_r * 0.6, parent_den_r * 0.6], dim=0)
                new_den_i = torch.cat([parent_den_i * 0.6, parent_den_i * 0.6], dim=0)
            else:
                # Paper specifies Gaussian children follow a normal distribution centered at the parent.
                noise_a = torch.randn_like(parent_positions) * parent_scales * 0.5
                noise_b = torch.randn_like(parent_positions) * parent_scales * 0.5
                new_positions = torch.cat([parent_positions + noise_a, parent_positions + noise_b], dim=0)
                new_scales = torch.cat([parent_scales / 1.6, parent_scales / 1.6], dim=0)
                new_rotations = torch.cat([parent_rotations, parent_rotations], dim=0)
                new_den_r = torch.cat([parent_den_r * 0.5, parent_den_r * 0.5], dim=0)
                new_den_i = torch.cat([parent_den_i * 0.5, parent_den_i * 0.5], dim=0)

            self._update_params(~mask, new_positions, new_scales, new_rotations, new_den_r, new_den_i)
            return split_count

    def densify_and_clone(self, grads: torch.Tensor, grad_threshold: float, scale_threshold: float) -> int:
        with torch.no_grad():
            scales = self.get_scale_values()
            max_scales = scales.max(dim=-1)[0]
            mask = (grads > grad_threshold) & (max_scales <= scale_threshold)
            if mask.sum() == 0:
                return 0

            self.density_real.data[mask] *= 0.5
            self.density_imag.data[mask] *= 0.5

            # Store old param refs for optimizer state transplant.
            # Clone keeps ALL old rows and appends new ones.
            n_old = self.positions.shape[0]
            keep_all = torch.ones(n_old, dtype=torch.bool, device=self.positions.device)
            self._last_densify_old_params = {
                id(self.positions): self.positions,
                id(self.scales): self.scales,
                id(self.rotations): self.rotations,
                id(self.density_real): self.density_real,
                id(self.density_imag): self.density_imag,
            }
            self._last_densify_keep_mask = keep_all
            self._last_densify_is_prune = False

            new_pos = self.positions[mask]
            new_scale = scales[mask]
            new_rot = self.rotations[mask]
            new_den_r = self.density_real[mask].clone()
            new_den_i = self.density_imag[mask].clone()

            self.positions = nn.Parameter(torch.cat([self.positions, new_pos], dim=0))
            self.scales = nn.Parameter(torch.cat([self.scales, torch.log(new_scale + 1e-8)], dim=0))
            self.rotations = nn.Parameter(torch.cat([self.rotations, new_rot], dim=0))
            self.density_real = nn.Parameter(torch.cat([self.density_real, new_den_r], dim=0))
            self.density_imag = nn.Parameter(torch.cat([self.density_imag, new_den_i], dim=0))
            self.num_points = self.positions.shape[0]
            return int(mask.sum().item())

    def prune(self, opacity_threshold: float) -> int:
        with torch.no_grad():
            density_mag = torch.abs(self.density)
            keep_mask = density_mag > opacity_threshold
            if keep_mask.sum() == self.num_points:
                return 0
            self._update_params(keep_mask, None, None, None, None, None, is_prune=True)
            return int((~keep_mask).sum().item())

    def _update_params(
        self,
        mask: torch.Tensor,
        new_pos: Optional[torch.Tensor] = None,
        new_scale: Optional[torch.Tensor] = None,
        new_rot: Optional[torch.Tensor] = None,
        new_dr: Optional[torch.Tensor] = None,
        new_di: Optional[torch.Tensor] = None,
        is_prune: bool = False,
    ):
        # Store old param references and the mask so the optimizer can
        # transplant Adam state for the surviving rows.
        self._last_densify_old_params = {
            id(self.positions): self.positions,
            id(self.scales): self.scales,
            id(self.rotations): self.rotations,
            id(self.density_real): self.density_real,
            id(self.density_imag): self.density_imag,
        }
        self._last_densify_keep_mask = mask
        self._last_densify_is_prune = is_prune

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

    @staticmethod
    def _extract_complex_image(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim == 4 and image.shape[0] == 2:
            mag = torch.sqrt(image[0] ** 2 + image[1] ** 2 + 1e-12)
            return mag, image[0], image[1]
        if torch.is_complex(image):
            return torch.abs(image), image.real, image.imag
        return torch.abs(image), image, torch.zeros_like(image)

    @staticmethod
    def _grid_positions_from_indices(indices: torch.Tensor, shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        d_size, h_size, w_size = shape
        z = indices // (h_size * w_size)
        rem = indices % (h_size * w_size)
        y = rem // w_size
        x = rem % w_size
        voxel_coords = torch.stack([z, y, x], dim=-1)
        positions = torch.stack(
            [
                ((z.float() + 0.5) / d_size) * 2 - 1,
                ((y.float() + 0.5) / h_size) * 2 - 1,
                ((x.float() + 0.5) / w_size) * 2 - 1,
            ],
            dim=-1,
        )
        return voxel_coords, positions

    @staticmethod
    def _exact_grid_scale_init(voxel_coords: torch.Tensor, shape: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
        d_size, h_size, w_size = shape
        step_d = 2.0 / max(d_size, 1)
        step_h = 2.0 / max(h_size, 1)
        step_w = 2.0 / max(w_size, 1)
        distances = []
        z = voxel_coords[:, 0]
        y = voxel_coords[:, 1]
        x = voxel_coords[:, 2]
        if d_size > 1:
            distances.append(torch.where(z > 0, torch.full_like(z, step_d, dtype=torch.float32), torch.full_like(z, float("inf"), dtype=torch.float32)))
            distances.append(torch.where(z < d_size - 1, torch.full_like(z, step_d, dtype=torch.float32), torch.full_like(z, float("inf"), dtype=torch.float32)))
        if h_size > 1:
            distances.append(torch.where(y > 0, torch.full_like(y, step_h, dtype=torch.float32), torch.full_like(y, float("inf"), dtype=torch.float32)))
            distances.append(torch.where(y < h_size - 1, torch.full_like(y, step_h, dtype=torch.float32), torch.full_like(y, float("inf"), dtype=torch.float32)))
        if w_size > 1:
            distances.append(torch.where(x > 0, torch.full_like(x, step_w, dtype=torch.float32), torch.full_like(x, float("inf"), dtype=torch.float32)))
            distances.append(torch.where(x < w_size - 1, torch.full_like(x, step_w, dtype=torch.float32), torch.full_like(x, float("inf"), dtype=torch.float32)))

        if not distances:
            base = torch.ones(voxel_coords.shape[0], 1, device=device) * 1e-3
            return base.repeat(1, 3)

        dist_tensor = torch.stack([dist.to(device) for dist in distances], dim=1)
        nearest_three = torch.topk(dist_tensor, k=min(3, dist_tensor.shape[1]), largest=False, dim=1).values
        if nearest_three.shape[1] < 3:
            nearest_three = torch.cat([nearest_three, nearest_three[:, -1:].repeat(1, 3 - nearest_three.shape[1])], dim=1)
        mean_dist = nearest_three.mean(dim=1, keepdim=True)
        return mean_dist.repeat(1, 3)

    @staticmethod
    def _sampled_3nn_scale_init(
        voxel_coords: torch.Tensor,
        positions: torch.Tensor,
        shape: Tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if positions.shape[0] <= 1 or cKDTree is None:
            return GaussianModel3D._exact_grid_scale_init(voxel_coords, shape, device)

        pos_np = positions.detach().cpu().numpy().astype(np.float32)
        tree = cKDTree(pos_np)
        k = min(4, pos_np.shape[0])
        distances, _ = tree.query(pos_np, k=k, workers=-1)
        if k == 1:
            mean_dist = np.full((pos_np.shape[0], 1), 1e-3, dtype=np.float32)
        else:
            neighbor_dist = np.asarray(distances[:, 1:], dtype=np.float32)
            if neighbor_dist.ndim == 1:
                neighbor_dist = neighbor_dist[:, None]
            if neighbor_dist.shape[1] < 3:
                pad = np.repeat(neighbor_dist[:, -1:], 3 - neighbor_dist.shape[1], axis=1)
                neighbor_dist = np.concatenate([neighbor_dist, pad], axis=1)
            mean_dist = np.maximum(neighbor_dist[:, :3].mean(axis=1, keepdims=True), 1e-6)
        return torch.from_numpy(np.repeat(mean_dist, 3, axis=1)).to(device=device, dtype=torch.float32)

    @classmethod
    def from_image(
        cls,
        image: torch.Tensor,
        num_points: int,
        initial_scale: float = 2.0,
        density_scale_k: float = 0.2,
        init_mode: str = "random",
        device: str = "cuda:0",
    ):
        del initial_scale
        dev = torch.device(device)
        mag, densities_real, densities_imag = cls._extract_complex_image(image)
        d_size, h_size, w_size = mag.shape
        total_voxels = d_size * h_size * w_size
        if num_points >= total_voxels:
            indices = torch.arange(total_voxels, device=dev)
            num_points = total_voxels
        elif init_mode == "importance":
            mag_flat = mag.flatten()
            threshold = torch.quantile(mag_flat, 0.90)
            candidates = torch.nonzero(mag_flat > threshold, as_tuple=False).squeeze(-1)
            if candidates.numel() < num_points:
                candidates = torch.arange(total_voxels, device=dev)
            choice = torch.randperm(candidates.numel(), device=dev)[:num_points]
            indices = candidates[choice]
        else:
            indices = torch.randperm(total_voxels, device=dev)[:num_points]

        voxel_coords, positions = cls._grid_positions_from_indices(indices, (d_size, h_size, w_size))
        scales_init = cls._sampled_3nn_scale_init(voxel_coords, positions, (d_size, h_size, w_size), dev)
        init_den_r = densities_real.flatten()[indices] * density_scale_k
        init_den_i = densities_imag.flatten()[indices] * density_scale_k
        initial_densities = torch.complex(init_den_r, init_den_i)

        print(
            f"[Init] mode={init_mode}, M={num_points}, k={density_scale_k}, "
            f"scale_range=[{scales_init.min().item():.6f}, {scales_init.max().item():.6f}]"
        )

        return cls(
            num_points=num_points,
            volume_shape=(d_size, h_size, w_size),
            initial_positions=positions,
            initial_densities=initial_densities,
            initial_scales=scales_init,
            device=device,
        )
