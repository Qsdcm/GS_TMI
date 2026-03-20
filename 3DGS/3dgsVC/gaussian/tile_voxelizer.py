"""
Tile-Based 3D Gaussian Voxelizer — Paper Section E

Paper: "the tile-based design, which divides the image volume into multiple
non-overlapping patches. Each tile contains a subset of the voxels."

Algorithm (Paper Section E):
1. Compute 3σ bounding box for each Gaussian (radius = max σ * 3)
2. Determine which tiles each Gaussian's bbox intersects → emit tile-Gaussian pairs
3. Sort pairs by tile ID → contiguous ranges
4. For each tile, gather assigned Gaussians via start/end indices
5. Accumulate Gaussian contributions to voxels within the tile (shared memory batching)

Two backends:
  - "tile_cuda": CUDA kernels with shared-memory per-tile processing (fast, requires compilation)
  - "tile": Pure-PyTorch fallback using same tile organization (portable, auto-differentiable)

Config: voxelizer_type = "tile_cuda" | "tile" | "chunk" (legacy)
"""

import os
import math
import torch
import torch.nn as nn
from typing import Tuple

# Try to JIT-compile the CUDA extension
_cuda_ext = None

def _load_cuda_ext():
    global _cuda_ext
    if _cuda_ext is not None:
        return _cuda_ext
    try:
        from torch.utils.cpp_extension import load
        import hashlib

        cuda_src = os.path.join(os.path.dirname(__file__),
                                'cuda_ext', 'tile_voxelize_cuda_kernel.cu')

        # Version the build name by source hash to auto-invalidate stale caches
        with open(cuda_src, 'rb') as f:
            src_hash = hashlib.md5(f.read()).hexdigest()[:8]
        build_name = f'tile_voxelize_cuda_{src_hash}'

        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        extra_ldflags = []
        for lib_dir in [os.path.join(conda_prefix, 'lib'),
                        os.path.join(conda_prefix, 'lib64'),
                        '/usr/local/cuda/lib64']:
            if os.path.isfile(os.path.join(lib_dir, 'libcudart.so')):
                extra_ldflags.append(f'-L{lib_dir}')
                break

        _cuda_ext = load(
            name=build_name,
            sources=[cuda_src],
            verbose=False,
            extra_cuda_cflags=['-O2'],
            extra_ldflags=extra_ldflags,
        )
        return _cuda_ext
    except Exception as e:
        print(f"[TileVoxelizer] CUDA extension compilation failed: {e}")
        print("[TileVoxelizer] Falling back to PyTorch tile implementation.")
        return None


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix."""
    norm = q.norm(dim=-1, keepdim=True)
    q = q / (norm + 1e-8)
    w, x, y, z = q.unbind(-1)
    r0 = torch.stack([1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w], dim=-1)
    r1 = torch.stack([2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w], dim=-1)
    r2 = torch.stack([2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y], dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


# ======================== Tile Pair Builder ========================

def build_tile_pairs(center_vox, radius_vox, volume_shape, tile_size):
    """
    Paper: "identify the Gaussian points that fall into the tile and emit
    tile-Gaussian pairs (tile's ID and Gaussian's ID). Then the pairs are
    sorted by the tiles' IDs."

    Returns:
        tile_ranges: [num_tiles, 2] int tensor — (start, end) into pair_ids
        pair_ids:    [total_pairs] int tensor — sorted Gaussian IDs
    """
    D, H, W = volume_shape
    ts = tile_size
    device = center_vox.device
    N = center_vox.shape[0]

    n_td = math.ceil(D / ts)
    n_th = math.ceil(H / ts)
    n_tw = math.ceil(W / ts)
    num_tiles = n_td * n_th * n_tw

    # Bounding box in tile coordinates
    r3 = radius_vox.unsqueeze(1).expand(-1, 3)
    bbox_min_vox = (center_vox - r3).floor().int()
    bbox_max_vox = (center_vox + r3).ceil().int()
    bbox_min_vox[:, 0].clamp_(0, D - 1)
    bbox_min_vox[:, 1].clamp_(0, H - 1)
    bbox_min_vox[:, 2].clamp_(0, W - 1)
    bbox_max_vox[:, 0].clamp_(0, D - 1)
    bbox_max_vox[:, 1].clamp_(0, H - 1)
    bbox_max_vox[:, 2].clamp_(0, W - 1)

    tile_min = bbox_min_vox // ts  # [N, 3]
    tile_max = bbox_max_vox // ts  # [N, 3]
    tile_min[:, 0].clamp_(0, n_td - 1)
    tile_min[:, 1].clamp_(0, n_th - 1)
    tile_min[:, 2].clamp_(0, n_tw - 1)
    tile_max[:, 0].clamp_(0, n_td - 1)
    tile_max[:, 1].clamp_(0, n_th - 1)
    tile_max[:, 2].clamp_(0, n_tw - 1)

    # Count tiles per Gaussian and total pairs
    n_per = (tile_max - tile_min + 1)  # [N, 3]
    n_tiles_per = n_per[:, 0] * n_per[:, 1] * n_per[:, 2]  # [N]
    total_pairs = n_tiles_per.sum().item()

    if total_pairs == 0:
        tile_ranges = torch.zeros(num_tiles, 2, dtype=torch.int32, device=device)
        pair_ids = torch.zeros(0, dtype=torch.int32, device=device)
        return tile_ranges, pair_ids

    # Enumerate all pairs — Gaussian IDs repeated per their tile count
    gauss_ids = torch.repeat_interleave(
        torch.arange(N, device=device, dtype=torch.int32), n_tiles_per
    )

    # Compute the tile ID for each pair
    # For each Gaussian, enumerate tiles in its [tile_min, tile_max] range
    offsets = torch.zeros(total_pairs, dtype=torch.int64, device=device)
    cumulative = torch.cat([torch.zeros(1, device=device, dtype=torch.int64),
                            n_tiles_per.cumsum(0)[:-1]])
    local_idx = torch.arange(total_pairs, device=device) - \
                torch.repeat_interleave(cumulative, n_tiles_per)

    # Decode local_idx into (ld, lh, lw) within each Gaussian's tile range
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

    # Sort by tile ID (Paper: "pairs are sorted by the tiles' IDs")
    sort_idx = torch.argsort(pair_tile_ids)
    pair_tile_ids_sorted = pair_tile_ids[sort_idx]
    pair_ids_sorted = gauss_ids[sort_idx]

    # Build contiguous ranges per tile
    # Paper: "each tile's pairs are contiguous ... record a starting and ending index"
    tile_ranges = torch.zeros(num_tiles, 2, dtype=torch.int32, device=device)

    if total_pairs > 0:
        # Find boundaries
        changes = torch.cat([
            torch.tensor([True], device=device),
            pair_tile_ids_sorted[1:] != pair_tile_ids_sorted[:-1]
        ])
        boundary_pos = torch.nonzero(changes, as_tuple=True)[0]
        boundary_tiles = pair_tile_ids_sorted[boundary_pos]

        starts = boundary_pos
        ends = torch.cat([boundary_pos[1:],
                          torch.tensor([total_pairs], device=device, dtype=torch.int64)])

        for i in range(boundary_tiles.shape[0]):
            tid = boundary_tiles[i].item()
            if 0 <= tid < num_tiles:
                tile_ranges[tid, 0] = starts[i].int()
                tile_ranges[tid, 1] = ends[i].int()

    return tile_ranges, pair_ids_sorted


# ======================== CUDA Autograd Function ========================

class _TileVoxelizeCUDA(torch.autograd.Function):
    """
    Custom autograd: forward/backward via CUDA per-Gaussian scatter/gather.
    PyTorch autograd handles gradients through cov_inv→(scale, rotation)
    and center_vox→positions.
    """

    @staticmethod
    def forward(ctx, center_vox, cov_inv, density_r, density_i,
                radius_vox, half_shape, D, H, W):
        cuda = _load_cuda_ext()

        vol_r, vol_i = cuda.forward(
            center_vox.contiguous().float(),
            cov_inv.contiguous().float(),
            density_r.contiguous().float(),
            density_i.contiguous().float(),
            radius_vox.contiguous().float(),
            half_shape.contiguous().float(),
            D, H, W
        )

        ctx.save_for_backward(center_vox, cov_inv, density_r, density_i,
                              radius_vox, half_shape)
        ctx.dims = (D, H, W)
        return vol_r, vol_i

    @staticmethod
    def backward(ctx, grad_vol_r, grad_vol_i):
        center_vox, cov_inv, density_r, density_i, radius_vox, half_shape = \
            ctx.saved_tensors
        D, H, W = ctx.dims
        cuda = _load_cuda_ext()

        grad_centers, grad_cov_inv, grad_den_r, grad_den_i = cuda.backward(
            center_vox.contiguous().float(),
            cov_inv.contiguous().float(),
            density_r.contiguous().float(),
            density_i.contiguous().float(),
            radius_vox.contiguous().float(),
            half_shape.contiguous().float(),
            grad_vol_r.contiguous().float(),
            grad_vol_i.contiguous().float(),
            D, H, W
        )

        return (grad_centers, grad_cov_inv, grad_den_r, grad_den_i,
                None, None, None, None, None)


# ======================== PyTorch Fallback Tile Processing ========================

def _tile_forward_pytorch(center_vox, cov_inv, density, radius_vox,
                          half_shape, tile_ranges, pair_ids,
                          D, H, W, tile_size):
    """
    Pure-PyTorch tile-based forward. Processes tiles sequentially but with
    vectorized per-tile ops. Fully differentiable via autograd.
    """
    num_tiles = tile_ranges.shape[0]
    ts = tile_size
    n_td = math.ceil(D / ts)
    n_th = math.ceil(H / ts)
    n_tw = math.ceil(W / ts)

    volume_flat = torch.zeros(D * H * W, dtype=torch.complex64, device=center_vox.device)

    for tile_id in range(num_tiles):
        g_start = tile_ranges[tile_id, 0].item()
        g_end = tile_ranges[tile_id, 1].item()
        if g_start >= g_end:
            continue

        gids = pair_ids[g_start:g_end].long()

        # Tile 3D index
        td = tile_id // (n_th * n_tw)
        rem = tile_id % (n_th * n_tw)
        th = rem // n_tw
        tw = rem % n_tw

        vd_s, vd_e = td * ts, min(td * ts + ts, D)
        vh_s, vh_e = th * ts, min(th * ts + ts, H)
        vw_s, vw_e = tw * ts, min(tw * ts + ts, W)

        # Build local voxel grid for this tile
        vd = torch.arange(vd_s, vd_e, device=center_vox.device, dtype=torch.float32)
        vh = torch.arange(vh_s, vh_e, device=center_vox.device, dtype=torch.float32)
        vw = torch.arange(vw_s, vw_e, device=center_vox.device, dtype=torch.float32)
        gd, gh, gw = torch.meshgrid(vd, vh, vw, indexing='ij')
        voxels = torch.stack([gd, gh, gw], dim=-1).reshape(-1, 3)  # [V, 3]

        # Gather Gaussian params
        c = center_vox[gids]       # [G, 3]
        M = cov_inv[gids]          # [G, 3, 3]
        rho = density[gids]        # [G] complex

        # Diff: [V, G, 3]
        diff_vox = voxels.unsqueeze(1) - c.unsqueeze(0)
        diff_norm = diff_vox / half_shape.unsqueeze(0).unsqueeze(0)

        # Mahalanobis: [V, G]
        Md = torch.einsum('vgd,gde,vge->vg', diff_norm, M, diff_norm)

        # 3σ mask and weights
        mask = Md <= 9.0
        weights = torch.exp(-0.5 * Md) * mask.float()  # [V, G]

        # Weighted accumulation: [V] complex
        contrib = (weights * rho.unsqueeze(0)).sum(dim=1)

        # Write to volume
        flat_d = gd.reshape(-1).long()
        flat_h = gh.reshape(-1).long()
        flat_w = gw.reshape(-1).long()
        flat_idx = flat_d * H * W + flat_h * W + flat_w
        volume_flat.scatter_add_(0, flat_idx, contrib)

    return volume_flat.view(D, H, W)


# ======================== TileVoxelizer Module ========================

class TileVoxelizer(nn.Module):
    """
    Paper Section E: Tile-based 3D Gaussian voxelization.

    Args:
        volume_shape: (D, H, W) output volume dimensions
        tile_size: side length of each cubic tile (default 8)
        use_cuda: True to use CUDA kernels, False for PyTorch fallback
        device: compute device
    """

    def __init__(self, volume_shape: Tuple[int, int, int],
                 tile_size: int = 8,
                 max_radius: int = 20,
                 use_cuda: bool = True,
                 device: str = "cuda:0"):
        super().__init__()
        self.volume_shape = volume_shape
        self.tile_size = tile_size
        self.max_radius = max_radius
        self.device = device

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            ext = _load_cuda_ext()
            if ext is None:
                self.use_cuda = False

        D, H, W = volume_shape
        n_td = math.ceil(D / tile_size)
        n_th = math.ceil(H / tile_size)
        n_tw = math.ceil(W / tile_size)
        self.tile_grid = (n_td, n_th, n_tw)
        self.num_tiles = n_td * n_th * n_tw
        print(f"[TileVoxelizer] volume={volume_shape}, tile_size={tile_size}, "
              f"max_radius={max_radius}, grid={self.tile_grid} ({self.num_tiles} tiles), "
              f"cuda={self.use_cuda}")

    def forward(
        self,
        positions: torch.Tensor,   # [N, 3] normalized [-1, 1]
        scales: torch.Tensor,      # [N, 3] actual scale values
        rotations: torch.Tensor,   # [N, 4] quaternions
        density: torch.Tensor,     # [N] complex
        chunk_size: int = 4096     # unused, kept for API compat
    ) -> torch.Tensor:
        D, H, W = self.volume_shape
        device = positions.device

        shape_tensor = torch.tensor([D, H, W], device=device, dtype=torch.float32)
        half_shape = shape_tensor * 0.5

        # Convert to voxel coordinates (differentiable)
        center_vox = (positions + 1.0) * 0.5 * shape_tensor - 0.5
        scale_vox = scales * half_shape

        # 3σ radius in voxels, capped for efficiency (same as chunk-based max_r)
        # Paper: "limit its influence to within three standard deviations"
        radius_vox = scale_vox.max(dim=-1)[0] * 3.0
        radius_vox = radius_vox.clamp(max=float(self.max_radius))

        # Covariance inverse (differentiable through scales and rotations)
        R = _quaternion_to_rotation_matrix(rotations)
        S_inv = torch.reciprocal(scales + 1e-8)
        L = R * S_inv.unsqueeze(1)
        cov_inv = L @ L.transpose(1, 2)  # [N, 3, 3]

        # Dispatch to CUDA or PyTorch backend
        if self.use_cuda:
            # CUDA path: fast forward + backward via custom kernels
            vol_r, vol_i = _TileVoxelizeCUDA.apply(
                center_vox, cov_inv,
                density.real.contiguous(), density.imag.contiguous(),
                radius_vox, half_shape,
                D, H, W
            )
            # Return separate real/imag to avoid PyTorch 1.13 CUDA driver bug
            # with torch.complex() on custom autograd Function outputs.
            # Callers must handle the tuple (vol_r, vol_i) case.
            return vol_r.view(D, H, W), vol_i.view(D, H, W)
        else:
            # PyTorch tile-based fallback (uses pair building)
            with torch.no_grad():
                tile_ranges, pair_ids = build_tile_pairs(
                    center_vox.detach(), radius_vox.detach(),
                    self.volume_shape, self.tile_size
                )
            return _tile_forward_pytorch(
                center_vox, cov_inv, density, radius_vox,
                half_shape, tile_ranges, pair_ids,
                D, H, W, self.tile_size
            )
