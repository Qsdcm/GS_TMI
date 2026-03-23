"""Tile-based voxelization for paper-faithful 3DGSMR."""

import hashlib
import math
import os
from typing import Tuple

import torch
import torch.nn as nn

_cuda_ext = None
_cuda_load_error = None


def _extension_sources() -> Tuple[str, str]:
    base_dir = os.path.join(os.path.dirname(__file__), "cuda_ext")
    return (
        os.path.join(base_dir, "tile_voxelize_cuda.cpp"),
        os.path.join(base_dir, "tile_voxelize_cuda_kernel.cu"),
    )


def _load_cuda_ext(strict: bool = False):
    global _cuda_ext, _cuda_load_error
    if _cuda_ext is not None:
        return _cuda_ext
    if _cuda_load_error is not None:
        if strict:
            raise RuntimeError(_cuda_load_error)
        return None
    try:
        from torch.utils.cpp_extension import load

        cpp_src, cuda_src = _extension_sources()
        with open(cpp_src, "rb") as handle:
            cpp_hash = hashlib.md5(handle.read()).hexdigest()[:8]
        with open(cuda_src, "rb") as handle:
            cu_hash = hashlib.md5(handle.read()).hexdigest()[:8]
        build_name = f"tile_voxelize_cuda_{cpp_hash}_{cu_hash}"
        build_dir = os.path.join(os.path.dirname(cuda_src), "build")
        os.makedirs(build_dir, exist_ok=True)

        extra_ldflags = []
        search_roots = []
        for env_name in ("CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH"):
            value = os.environ.get(env_name)
            if value:
                search_roots.append(value)
        search_roots.append("/usr/local/cuda")
        for root_dir in search_roots:
            for lib_dir in (os.path.join(root_dir, "lib64"), os.path.join(root_dir, "lib")):
                if os.path.isfile(os.path.join(lib_dir, "libcudart.so")):
                    extra_ldflags = [f"-L{lib_dir}"]
                    break
            if extra_ldflags:
                break

        _cuda_ext = load(
            name=build_name,
            sources=[cpp_src, cuda_src],
            build_directory=build_dir,
            verbose=False,
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_ldflags=extra_ldflags,
        )
        return _cuda_ext
    except Exception as exc:  # pragma: no cover
        _cuda_load_error = f"Failed to build tile_cuda extension: {exc}"
        if strict:
            raise RuntimeError(_cuda_load_error) from exc
        print(f"[TileVoxelizer] {_cuda_load_error}")
        print("[TileVoxelizer] Falling back to the PyTorch tile reference backend.")
        return None


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    norm = q.norm(dim=-1, keepdim=True)
    q = q / (norm + 1e-8)
    w, x, y, z = q.unbind(-1)
    r0 = torch.stack([1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w], dim=-1)
    r1 = torch.stack([2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w], dim=-1)
    r2 = torch.stack([2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y], dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


def build_tile_pairs(center_vox, radius_vox, volume_shape, tile_size):
    d_size, h_size, w_size = volume_shape
    device = center_vox.device
    n_points = center_vox.shape[0]

    n_td = math.ceil(d_size / tile_size)
    n_th = math.ceil(h_size / tile_size)
    n_tw = math.ceil(w_size / tile_size)
    num_tiles = n_td * n_th * n_tw

    r3 = radius_vox.unsqueeze(1).expand(-1, 3)
    bbox_min_vox = (center_vox - r3).floor().int()
    bbox_max_vox = (center_vox + r3).ceil().int()
    bbox_min_vox[:, 0].clamp_(0, d_size - 1)
    bbox_min_vox[:, 1].clamp_(0, h_size - 1)
    bbox_min_vox[:, 2].clamp_(0, w_size - 1)
    bbox_max_vox[:, 0].clamp_(0, d_size - 1)
    bbox_max_vox[:, 1].clamp_(0, h_size - 1)
    bbox_max_vox[:, 2].clamp_(0, w_size - 1)

    tile_min = bbox_min_vox // tile_size
    tile_max = bbox_max_vox // tile_size
    tile_min[:, 0].clamp_(0, n_td - 1)
    tile_min[:, 1].clamp_(0, n_th - 1)
    tile_min[:, 2].clamp_(0, n_tw - 1)
    tile_max[:, 0].clamp_(0, n_td - 1)
    tile_max[:, 1].clamp_(0, n_th - 1)
    tile_max[:, 2].clamp_(0, n_tw - 1)

    n_per = tile_max - tile_min + 1
    n_tiles_per = n_per[:, 0] * n_per[:, 1] * n_per[:, 2]
    total_pairs = int(n_tiles_per.sum().item())

    if total_pairs == 0:
        return (
            torch.zeros(num_tiles, 2, dtype=torch.int32, device=device),
            torch.zeros(0, dtype=torch.int32, device=device),
        )

    gauss_ids = torch.repeat_interleave(torch.arange(n_points, device=device, dtype=torch.int32), n_tiles_per)
    cumulative = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int64), n_tiles_per.cumsum(0)[:-1]],
        dim=0,
    )
    local_idx = torch.arange(total_pairs, device=device, dtype=torch.int64) - torch.repeat_interleave(cumulative, n_tiles_per)
    rng_h = torch.repeat_interleave(n_per[:, 1], n_tiles_per)
    rng_w = torch.repeat_interleave(n_per[:, 2], n_tiles_per)
    ld = local_idx // (rng_h * rng_w)
    lr = local_idx % (rng_h * rng_w)
    lh = lr // rng_w
    lw = lr % rng_w

    tmin_d = torch.repeat_interleave(tile_min[:, 0], n_tiles_per)
    tmin_h = torch.repeat_interleave(tile_min[:, 1], n_tiles_per)
    tmin_w = torch.repeat_interleave(tile_min[:, 2], n_tiles_per)
    pair_tile_d = tmin_d + ld.int()
    pair_tile_h = tmin_h + lh.int()
    pair_tile_w = tmin_w + lw.int()
    pair_tile_ids = (pair_tile_d * n_th * n_tw + pair_tile_h * n_tw + pair_tile_w).int()

    sort_idx = torch.argsort(pair_tile_ids)
    pair_tile_ids_sorted = pair_tile_ids[sort_idx]
    pair_ids_sorted = gauss_ids[sort_idx]

    tile_ranges = torch.zeros(num_tiles, 2, dtype=torch.int32, device=device)
    changes = torch.cat(
        [torch.tensor([True], device=device), pair_tile_ids_sorted[1:] != pair_tile_ids_sorted[:-1]],
        dim=0,
    )
    boundary_pos = torch.nonzero(changes, as_tuple=True)[0]
    boundary_tiles = pair_tile_ids_sorted[boundary_pos]
    starts = boundary_pos
    ends = torch.cat([boundary_pos[1:], torch.tensor([total_pairs], device=device, dtype=torch.int64)], dim=0)
    tile_ranges[boundary_tiles.long(), 0] = starts.int()
    tile_ranges[boundary_tiles.long(), 1] = ends.int()
    return tile_ranges, pair_ids_sorted


class _TileVoxelizeCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, center_vox, cov_inv, density_r, density_i, tile_ranges, pair_ids, half_shape, tile_size, d_size, h_size, w_size):
        cuda = _load_cuda_ext(strict=True)
        vol_r, vol_i = cuda.forward(
            center_vox.contiguous().float(),
            cov_inv.contiguous().float(),
            density_r.contiguous().float(),
            density_i.contiguous().float(),
            tile_ranges.contiguous().int(),
            pair_ids.contiguous().int(),
            half_shape.contiguous().float(),
            tile_size,
            d_size,
            h_size,
            w_size,
        )
        ctx.save_for_backward(center_vox, cov_inv, density_r, density_i, tile_ranges, pair_ids, half_shape)
        ctx.tile_size = tile_size
        ctx.dims = (d_size, h_size, w_size)
        return vol_r, vol_i

    @staticmethod
    def backward(ctx, grad_vol_r, grad_vol_i):
        center_vox, cov_inv, density_r, density_i, tile_ranges, pair_ids, half_shape = ctx.saved_tensors
        d_size, h_size, w_size = ctx.dims
        cuda = _load_cuda_ext(strict=True)
        grad_centers, grad_cov_inv, grad_den_r, grad_den_i = cuda.backward(
            center_vox.contiguous().float(),
            cov_inv.contiguous().float(),
            density_r.contiguous().float(),
            density_i.contiguous().float(),
            tile_ranges.contiguous().int(),
            pair_ids.contiguous().int(),
            half_shape.contiguous().float(),
            grad_vol_r.contiguous().float(),
            grad_vol_i.contiguous().float(),
            ctx.tile_size,
            d_size,
            h_size,
            w_size,
        )
        return grad_centers, grad_cov_inv, grad_den_r, grad_den_i, None, None, None, None, None, None, None


def _tile_forward_pytorch(center_vox, cov_inv, density, half_shape, tile_ranges, pair_ids, d_size, h_size, w_size, tile_size):
    num_tiles = tile_ranges.shape[0]
    n_th = math.ceil(h_size / tile_size)
    n_tw = math.ceil(w_size / tile_size)
    volume_flat = torch.zeros(d_size * h_size * w_size, dtype=torch.complex64, device=center_vox.device)

    for tile_id in range(num_tiles):
        g_start = int(tile_ranges[tile_id, 0].item())
        g_end = int(tile_ranges[tile_id, 1].item())
        if g_start >= g_end:
            continue
        gids = pair_ids[g_start:g_end].long()
        td = tile_id // (n_th * n_tw)
        rem = tile_id % (n_th * n_tw)
        th = rem // n_tw
        tw = rem % n_tw
        vd_s, vd_e = td * tile_size, min(td * tile_size + tile_size, d_size)
        vh_s, vh_e = th * tile_size, min(th * tile_size + tile_size, h_size)
        vw_s, vw_e = tw * tile_size, min(tw * tile_size + tile_size, w_size)
        vd = torch.arange(vd_s, vd_e, device=center_vox.device, dtype=torch.float32)
        vh = torch.arange(vh_s, vh_e, device=center_vox.device, dtype=torch.float32)
        vw = torch.arange(vw_s, vw_e, device=center_vox.device, dtype=torch.float32)
        gd, gh, gw = torch.meshgrid(vd, vh, vw, indexing="ij")
        voxels = torch.stack([gd, gh, gw], dim=-1).reshape(-1, 3)
        centers = center_vox[gids]
        mats = cov_inv[gids]
        rho = density[gids]
        diff_vox = voxels.unsqueeze(1) - centers.unsqueeze(0)
        diff_norm = diff_vox / half_shape.unsqueeze(0).unsqueeze(0)
        mahal = torch.einsum("vgd,gde,vge->vg", diff_norm, mats, diff_norm)
        weights = torch.exp(-0.5 * mahal) * (mahal <= 9.0).float()
        contrib = (weights * rho.unsqueeze(0)).sum(dim=1)
        flat_idx = gd.reshape(-1).long() * h_size * w_size + gh.reshape(-1).long() * w_size + gw.reshape(-1).long()
        volume_flat.scatter_add_(0, flat_idx, contrib)
    return volume_flat.view(d_size, h_size, w_size)


class TileVoxelizer(nn.Module):
    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        tile_size: int = 8,
        max_radius: int = 20,
        use_cuda: bool = True,
        strict_cuda: bool = False,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.volume_shape = volume_shape
        self.tile_size = tile_size
        self.max_radius = max_radius
        self.device = device
        self.strict_cuda = strict_cuda
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.backend = "tile"
        if self.use_cuda:
            ext = _load_cuda_ext(strict=strict_cuda)
            if ext is None:
                self.use_cuda = False
            else:
                self.backend = "tile_cuda"
        if not self.use_cuda:
            if strict_cuda:
                raise RuntimeError("tile_cuda was requested in strict mode, but the CUDA extension is unavailable.")
            self.backend = "tile"
        print(
            f"[TileVoxelizer] volume={volume_shape}, tile_size={tile_size}, max_radius={max_radius}, backend={self.backend}"
        )

    def forward(self, positions: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, density: torch.Tensor, chunk_size: int = 4096):
        del chunk_size
        d_size, h_size, w_size = self.volume_shape
        shape_tensor = torch.tensor([d_size, h_size, w_size], device=positions.device, dtype=torch.float32)
        half_shape = shape_tensor * 0.5
        center_vox = (positions + 1.0) * 0.5 * shape_tensor - 0.5
        scale_vox = scales * half_shape
        radius_vox = scale_vox.max(dim=-1)[0] * 3.0
        rotation_matrix = _quaternion_to_rotation_matrix(rotations)
        s_inv = torch.reciprocal(scales + 1e-8)
        transform = rotation_matrix * s_inv.unsqueeze(1)
        cov_inv = transform @ transform.transpose(1, 2)

        with torch.no_grad():
            tile_ranges, pair_ids = build_tile_pairs(center_vox.detach(), radius_vox.detach(), self.volume_shape, self.tile_size)

        if self.use_cuda:
            vol_r, vol_i = _TileVoxelizeCUDA.apply(
                center_vox,
                cov_inv,
                density.real.contiguous(),
                density.imag.contiguous(),
                tile_ranges,
                pair_ids,
                half_shape,
                self.tile_size,
                d_size,
                h_size,
                w_size,
            )
            return vol_r.view(d_size, h_size, w_size), vol_i.view(d_size, h_size, w_size)

        return _tile_forward_pytorch(
            center_vox,
            cov_inv,
            density,
            half_shape,
            tile_ranges,
            pair_ids,
            d_size,
            h_size,
            w_size,
            self.tile_size,
        )
