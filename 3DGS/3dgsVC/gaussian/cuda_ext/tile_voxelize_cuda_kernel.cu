/*
 * 3D Gaussian Voxelization — CUDA Kernels
 *
 * Per-Gaussian scatter approach:
 *   Forward: one thread block per Gaussian, scatter contributions to its 3σ bbox
 *   Backward: one thread block per Gaussian, gather gradients from its 3σ bbox
 *
 * This avoids the costly Python-level tile-pair building while still using
 * the paper's core ideas (3σ truncation, Mahalanobis-based weighting).
 * For the paper's tile-based shared-memory optimization, a future version
 * could pre-sort Gaussians by spatial locality on the CUDA side.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// ====================== Forward: Per-Gaussian Scatter ======================

__global__ void voxelize_forward_kernel(
    const float* __restrict__ centers,    // [N, 3] voxel coords
    const float* __restrict__ cov_inv,    // [N, 9] flattened 3x3
    const float* __restrict__ density_r,  // [N]
    const float* __restrict__ density_i,  // [N]
    const float* __restrict__ radius,     // [N] 3σ radius in voxels
    const float* __restrict__ half_shape, // [3]
    float* __restrict__ vol_r,            // [D*H*W]
    float* __restrict__ vol_i,            // [D*H*W]
    int D, int H, int W, int N
) {
    const int gid = blockIdx.x;
    if (gid >= N) return;

    const float hs0 = half_shape[0], hs1 = half_shape[1], hs2 = half_shape[2];
    const float cx = centers[gid*3+0], cy = centers[gid*3+1], cz = centers[gid*3+2];
    const float r = radius[gid];
    const float rho_r = density_r[gid], rho_i = density_i[gid];

    // Guard against NaN/Inf from numerical issues after densification
    if (isnan(cx) || isnan(cy) || isnan(cz) || isnan(r) || isinf(cx) || isinf(cy) || isinf(cz))
        return;

    // Load covariance inverse into registers
    float M[9];
    #pragma unroll
    for (int m = 0; m < 9; m++) M[m] = cov_inv[gid*9+m];

    // 3σ bounding box (clamped to volume)
    const int d0 = max(0, (int)floorf(cx - r));
    const int d1 = min(D-1, (int)ceilf(cx + r));
    const int h0 = max(0, (int)floorf(cy - r));
    const int h1 = min(H-1, (int)ceilf(cy + r));
    const int w0 = max(0, (int)floorf(cz - r));
    const int w1 = min(W-1, (int)ceilf(cz + r));

    const int rd = d1-d0+1, rh = h1-h0+1, rw = w1-w0+1;
    const int total = rd * rh * rw;

    // Each thread scatters to a subset of voxels in the bbox
    for (int vid = threadIdx.x; vid < total; vid += blockDim.x) {
        const int ld = vid / (rh * rw);
        const int lr = vid % (rh * rw);
        const int lh = lr / rw;
        const int lw = lr % rw;
        const int gd = d0+ld, gh = h0+lh, gw = w0+lw;

        const float dx = ((float)gd - cx) / hs0;
        const float dy = ((float)gh - cy) / hs1;
        const float dz = ((float)gw - cz) / hs2;

        const float Md0 = M[0]*dx + M[1]*dy + M[2]*dz;
        const float Md1 = M[3]*dx + M[4]*dy + M[5]*dz;
        const float Md2 = M[6]*dx + M[7]*dy + M[8]*dz;
        const float mahal = dx*Md0 + dy*Md1 + dz*Md2;

        if (mahal <= 9.0f) {
            const float w = expf(-0.5f * mahal);
            const int flat = gd*H*W + gh*W + gw;
            atomicAdd(&vol_r[flat], w * rho_r);
            atomicAdd(&vol_i[flat], w * rho_i);
        }
    }
}

// ====================== Backward: Per-Gaussian Gather ======================
// One THREAD per Gaussian (simple, no shared memory, no reduction needed)

__global__ void voxelize_backward_kernel(
    const float* __restrict__ centers,
    const float* __restrict__ cov_inv,
    const float* __restrict__ density_r,
    const float* __restrict__ density_i,
    const float* __restrict__ radius,
    const float* __restrict__ half_shape,
    const float* __restrict__ grad_vol_r,
    const float* __restrict__ grad_vol_i,
    float* __restrict__ grad_centers,     // [N, 3]
    float* __restrict__ grad_cov_inv,     // [N, 9]
    float* __restrict__ grad_den_r,       // [N]
    float* __restrict__ grad_den_i,       // [N]
    int D, int H, int W, int N
) {
    // One thread per Gaussian — no shared memory, no block reduction
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    const float hs0 = half_shape[0], hs1 = half_shape[1], hs2 = half_shape[2];
    const float cx = centers[gid*3+0], cy = centers[gid*3+1], cz = centers[gid*3+2];
    const float r = radius[gid];
    const float rho_r = density_r[gid], rho_i = density_i[gid];

    float M[9];
    #pragma unroll
    for (int m = 0; m < 9; m++) M[m] = cov_inv[gid*9+m];

    const int d0 = max(0, (int)floorf(cx-r)), d1 = min(D-1, (int)ceilf(cx+r));
    const int h0 = max(0, (int)floorf(cy-r)), h1 = min(H-1, (int)ceilf(cy+r));
    const int w0 = max(0, (int)floorf(cz-r)), w1 = min(W-1, (int)ceilf(cz+r));

    float g_dr=0, g_di=0, g_cx=0, g_cy=0, g_cz=0;
    float g_M[9] = {0,0,0,0,0,0,0,0,0};

    for (int vd=d0; vd<=d1; vd++) {
        for (int vh=h0; vh<=h1; vh++) {
            for (int vw=w0; vw<=w1; vw++) {
                const float dx = ((float)vd-cx)/hs0;
                const float dy = ((float)vh-cy)/hs1;
                const float dz = ((float)vw-cz)/hs2;
                const float Md0 = M[0]*dx+M[1]*dy+M[2]*dz;
                const float Md1 = M[3]*dx+M[4]*dy+M[5]*dz;
                const float Md2 = M[6]*dx+M[7]*dy+M[8]*dz;
                const float mahal = dx*Md0+dy*Md1+dz*Md2;
                if (mahal <= 9.0f) {
                    const float w = expf(-0.5f * mahal);
                    const int flat = vd*H*W + vh*W + vw;
                    const float gvr = grad_vol_r[flat], gvi = grad_vol_i[flat];
                    g_dr += w*gvr;
                    g_di += w*gvi;
                    const float dL_dw = rho_r*gvr + rho_i*gvi;
                    const float dL_dd = dL_dw * (-0.5f) * w;
                    g_cx += dL_dd * (-2.0f*Md0/hs0);
                    g_cy += dL_dd * (-2.0f*Md1/hs1);
                    g_cz += dL_dd * (-2.0f*Md2/hs2);
                    g_M[0]+=dL_dd*dx*dx; g_M[1]+=dL_dd*dx*dy; g_M[2]+=dL_dd*dx*dz;
                    g_M[3]+=dL_dd*dy*dx; g_M[4]+=dL_dd*dy*dy; g_M[5]+=dL_dd*dy*dz;
                    g_M[6]+=dL_dd*dz*dx; g_M[7]+=dL_dd*dz*dy; g_M[8]+=dL_dd*dz*dz;
                }
            }
        }
    }

    // Each gid is unique — direct write, no atomics needed
    grad_den_r[gid] = g_dr;
    grad_den_i[gid] = g_di;
    grad_centers[gid*3+0] = g_cx;
    grad_centers[gid*3+1] = g_cy;
    grad_centers[gid*3+2] = g_cz;
    #pragma unroll
    for (int m=0; m<9; m++) grad_cov_inv[gid*9+m] = g_M[m];
}

// ====================== C++ Wrappers ======================

std::vector<torch::Tensor> voxelize_forward_cuda(
    torch::Tensor centers, torch::Tensor cov_inv,
    torch::Tensor density_r, torch::Tensor density_i,
    torch::Tensor radius, torch::Tensor half_shape,
    int D, int H, int W
) {
    const at::cuda::CUDAGuard device_guard(centers.device());
    int N = centers.size(0);
    auto vol_r = torch::zeros({D*H*W}, centers.options());
    auto vol_i = torch::zeros({D*H*W}, centers.options());
    auto cov_flat = cov_inv.reshape({-1, 9}).contiguous();

    if (N > 0) {
        // Launch on PyTorch's current CUDA stream (critical for correct ordering
        // with grad_checkpoint and coil-by-coil FFT operations)
        auto stream = at::cuda::getCurrentCUDAStream(centers.device().index());
        voxelize_forward_kernel<<<N, 256, 0, stream.stream()>>>(
            centers.data_ptr<float>(), cov_flat.data_ptr<float>(),
            density_r.data_ptr<float>(), density_i.data_ptr<float>(),
            radius.data_ptr<float>(), half_shape.data_ptr<float>(),
            vol_r.data_ptr<float>(), vol_i.data_ptr<float>(),
            D, H, W, N);
        // Check for launch errors
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {vol_r, vol_i};
}

std::vector<torch::Tensor> voxelize_backward_cuda(
    torch::Tensor centers, torch::Tensor cov_inv,
    torch::Tensor density_r, torch::Tensor density_i,
    torch::Tensor radius, torch::Tensor half_shape,
    torch::Tensor grad_vol_r, torch::Tensor grad_vol_i,
    int D, int H, int W
) {
    const at::cuda::CUDAGuard device_guard(centers.device());
    int N = centers.size(0);
    auto cov_flat = cov_inv.reshape({-1,9}).contiguous();
    auto gc = torch::zeros({N,3}, centers.options());
    auto gm = torch::zeros({N,9}, centers.options());
    auto gdr = torch::zeros({N}, centers.options());
    auto gdi = torch::zeros({N}, centers.options());

    if (N > 0) {
        int bk_threads = 256;
        int bk_blocks = (N + bk_threads - 1) / bk_threads;
        auto stream = at::cuda::getCurrentCUDAStream(centers.device().index());
        voxelize_backward_kernel<<<bk_blocks, bk_threads, 0, stream.stream()>>>(
            centers.data_ptr<float>(), cov_flat.data_ptr<float>(),
            density_r.data_ptr<float>(), density_i.data_ptr<float>(),
            radius.data_ptr<float>(), half_shape.data_ptr<float>(),
            grad_vol_r.data_ptr<float>(), grad_vol_i.data_ptr<float>(),
            gc.data_ptr<float>(), gm.data_ptr<float>(),
            gdr.data_ptr<float>(), gdi.data_ptr<float>(),
            D, H, W, N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {gc, gm.reshape({N,3,3}), gdr, gdi};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &voxelize_forward_cuda,  "Gaussian voxelization forward (CUDA)");
    m.def("backward", &voxelize_backward_cuda, "Gaussian voxelization backward (CUDA)");
}
