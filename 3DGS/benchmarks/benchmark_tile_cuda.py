import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '3dgsVC'))

from gaussian.tile_voxelizer import TileVoxelizer
from gaussian.voxelizer import Voxelizer


def make_args():
    parser = argparse.ArgumentParser(description='Benchmark tile_cuda against reference voxelizers')
    parser.add_argument('--shape', type=int, nargs=3, default=[16, 16, 16])
    parser.add_argument('--num_points', type=int, default=64)
    parser.add_argument('--tile_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def random_inputs(shape, num_points, device, seed):
    torch.manual_seed(seed)
    positions = (torch.rand(num_points, 3, device=device) * 2.0 - 1.0).requires_grad_(True)
    scales = (0.05 + torch.rand(num_points, 3, device=device) * 0.15).requires_grad_(True)
    rotations = torch.zeros(num_points, 4, device=device)
    rotations[:, 0] = 1.0
    rotations.requires_grad_(True)
    density = torch.complex(torch.randn(num_points, device=device), torch.randn(num_points, device=device)).requires_grad_(True)
    return positions, scales, rotations, density


def measure(voxelizer, positions, scales, rotations, density):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA benchmark requires a CUDA device')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    out = voxelizer(positions, scales, rotations, density)
    if isinstance(out, tuple):
        out = torch.complex(out[0], out[1])
    loss = out.abs().sum()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated()
    return out.detach(), elapsed, peak_mem


def main():
    args = make_args()
    device = torch.device(args.device)
    positions, scales, rotations, density = random_inputs(tuple(args.shape), args.num_points, device, args.seed)

    ref_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (positions, scales, rotations, density)]
    cuda_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (positions, scales, rotations, density)]

    reference = Voxelizer(tuple(args.shape), device=str(device))
    tile_cuda = TileVoxelizer(tuple(args.shape), tile_size=args.tile_size, max_radius=max(args.shape), use_cuda=True, strict_cuda=True, device=str(device))

    ref_out, ref_time, ref_mem = measure(reference, *ref_inputs)
    cuda_out, cuda_time, cuda_mem = measure(tile_cuda, *cuda_inputs)

    report = {
        'shape': args.shape,
        'num_points': args.num_points,
        'max_abs_error': float(torch.max(torch.abs(ref_out - cuda_out)).item()),
        'reference_seconds': ref_time,
        'tile_cuda_seconds': cuda_time,
        'reference_peak_memory_bytes': int(ref_mem),
        'tile_cuda_peak_memory_bytes': int(cuda_mem),
        'speedup_vs_reference': ref_time / max(cuda_time, 1e-12),
        'memory_ratio_vs_reference': cuda_mem / max(ref_mem, 1),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
