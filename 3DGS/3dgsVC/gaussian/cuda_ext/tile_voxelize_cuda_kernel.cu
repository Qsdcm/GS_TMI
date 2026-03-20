#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kStageGaussians = 32;

__device__ inline void decode_tile(
    int tile_id,
    int tile_size,
    int d_size,
    int h_size,
    int w_size,
    int n_th,
    int n_tw,
    int* d0,
    int* d1,
    int* h0,
    int* h1,
    int* w0,
    int* w1,
    int* rh,
    int* rw,
    int* total_voxels) {
    const int td = tile_id / (n_th * n_tw);
    const int rem = tile_id % (n_th * n_tw);
    const int th = rem / n_tw;
    const int tw = rem % n_tw;
    *d0 = td * tile_size;
    *d1 = min(*d0 + tile_size, d_size);
    *h0 = th * tile_size;
    *h1 = min(*h0 + tile_size, h_size);
    *w0 = tw * tile_size;
    *w1 = min(*w0 + tile_size, w_size);
    const int rd = *d1 - *d0;
    *rh = *h1 - *h0;
    *rw = *w1 - *w0;
    *total_voxels = rd * (*rh) * (*rw);
}

__global__ void tile_voxelize_forward_kernel(
    const float* __restrict__ centers,
    const float* __restrict__ cov_inv,
    const float* __restrict__ density_r,
    const float* __restrict__ density_i,
    const int32_t* __restrict__ tile_ranges,
    const int32_t* __restrict__ pair_ids,
    const float* __restrict__ half_shape,
    float* __restrict__ vol_r,
    float* __restrict__ vol_i,
    int tile_size,
    int d_size,
    int h_size,
    int w_size,
    int num_tiles) {
    const int tile_id = blockIdx.x;
    if (tile_id >= num_tiles) {
        return;
    }

    const int pair_start = tile_ranges[tile_id * 2 + 0];
    const int pair_end = tile_ranges[tile_id * 2 + 1];
    if (pair_start >= pair_end) {
        return;
    }

    const int n_th = (h_size + tile_size - 1) / tile_size;
    const int n_tw = (w_size + tile_size - 1) / tile_size;

    int d0, d1, h0, h1, w0, w1, rh, rw, total_voxels;
    decode_tile(tile_id, tile_size, d_size, h_size, w_size, n_th, n_tw, &d0, &d1, &h0, &h1, &w0, &w1, &rh, &rw, &total_voxels);

    __shared__ int sh_gid[kStageGaussians];
    __shared__ float sh_centers[kStageGaussians * 3];
    __shared__ float sh_cov[kStageGaussians * 9];
    __shared__ float sh_den_r[kStageGaussians];
    __shared__ float sh_den_i[kStageGaussians];

    const float hs0 = half_shape[0];
    const float hs1 = half_shape[1];
    const float hs2 = half_shape[2];

    for (int vid = threadIdx.x; vid < total_voxels; vid += blockDim.x) {
        const int ld = vid / (rh * rw);
        const int lr = vid % (rh * rw);
        const int lh = lr / rw;
        const int lw = lr % rw;
        const int gd = d0 + ld;
        const int gh = h0 + lh;
        const int gw = w0 + lw;
        float acc_r = 0.0f;
        float acc_i = 0.0f;

        for (int batch_start = pair_start; batch_start < pair_end; batch_start += kStageGaussians) {
            const int stage_count = min(kStageGaussians, pair_end - batch_start);
            if (threadIdx.x < stage_count) {
                const int gid = pair_ids[batch_start + threadIdx.x];
                sh_gid[threadIdx.x] = gid;
                sh_centers[threadIdx.x * 3 + 0] = centers[gid * 3 + 0];
                sh_centers[threadIdx.x * 3 + 1] = centers[gid * 3 + 1];
                sh_centers[threadIdx.x * 3 + 2] = centers[gid * 3 + 2];
                sh_den_r[threadIdx.x] = density_r[gid];
                sh_den_i[threadIdx.x] = density_i[gid];
                #pragma unroll
                for (int m = 0; m < 9; ++m) {
                    sh_cov[threadIdx.x * 9 + m] = cov_inv[gid * 9 + m];
                }
            }
            __syncthreads();

            for (int local_g = 0; local_g < stage_count; ++local_g) {
                const float cx = sh_centers[local_g * 3 + 0];
                const float cy = sh_centers[local_g * 3 + 1];
                const float cz = sh_centers[local_g * 3 + 2];
                const float dx = (static_cast<float>(gd) - cx) / hs0;
                const float dy = (static_cast<float>(gh) - cy) / hs1;
                const float dz = (static_cast<float>(gw) - cz) / hs2;
                const float* m = &sh_cov[local_g * 9];
                const float md0 = m[0] * dx + m[1] * dy + m[2] * dz;
                const float md1 = m[3] * dx + m[4] * dy + m[5] * dz;
                const float md2 = m[6] * dx + m[7] * dy + m[8] * dz;
                const float mahal = dx * md0 + dy * md1 + dz * md2;
                if (mahal <= 9.0f) {
                    const float weight = expf(-0.5f * mahal);
                    acc_r += weight * sh_den_r[local_g];
                    acc_i += weight * sh_den_i[local_g];
                }
            }
            __syncthreads();
        }

        const int flat = gd * h_size * w_size + gh * w_size + gw;
        vol_r[flat] = acc_r;
        vol_i[flat] = acc_i;
    }
}

__global__ void tile_voxelize_backward_kernel(
    const float* __restrict__ centers,
    const float* __restrict__ cov_inv,
    const float* __restrict__ density_r,
    const float* __restrict__ density_i,
    const int32_t* __restrict__ tile_ranges,
    const int32_t* __restrict__ pair_ids,
    const float* __restrict__ half_shape,
    const float* __restrict__ grad_vol_r,
    const float* __restrict__ grad_vol_i,
    float* __restrict__ grad_centers,
    float* __restrict__ grad_cov_inv,
    float* __restrict__ grad_den_r,
    float* __restrict__ grad_den_i,
    int tile_size,
    int d_size,
    int h_size,
    int w_size,
    int num_tiles) {
    const int tile_id = blockIdx.x;
    if (tile_id >= num_tiles) {
        return;
    }

    const int pair_start = tile_ranges[tile_id * 2 + 0];
    const int pair_end = tile_ranges[tile_id * 2 + 1];
    if (pair_start >= pair_end) {
        return;
    }

    const int n_th = (h_size + tile_size - 1) / tile_size;
    const int n_tw = (w_size + tile_size - 1) / tile_size;

    int d0, d1, h0, h1, w0, w1, rh, rw, total_voxels;
    decode_tile(tile_id, tile_size, d_size, h_size, w_size, n_th, n_tw, &d0, &d1, &h0, &h1, &w0, &w1, &rh, &rw, &total_voxels);

    __shared__ int sh_gid[kStageGaussians];
    __shared__ float sh_centers[kStageGaussians * 3];
    __shared__ float sh_cov[kStageGaussians * 9];
    __shared__ float sh_den_r[kStageGaussians];
    __shared__ float sh_den_i[kStageGaussians];

    const float hs0 = half_shape[0];
    const float hs1 = half_shape[1];
    const float hs2 = half_shape[2];

    for (int batch_start = pair_start; batch_start < pair_end; batch_start += kStageGaussians) {
        const int stage_count = min(kStageGaussians, pair_end - batch_start);
        if (threadIdx.x < stage_count) {
            const int gid = pair_ids[batch_start + threadIdx.x];
            sh_gid[threadIdx.x] = gid;
            sh_centers[threadIdx.x * 3 + 0] = centers[gid * 3 + 0];
            sh_centers[threadIdx.x * 3 + 1] = centers[gid * 3 + 1];
            sh_centers[threadIdx.x * 3 + 2] = centers[gid * 3 + 2];
            sh_den_r[threadIdx.x] = density_r[gid];
            sh_den_i[threadIdx.x] = density_i[gid];
            #pragma unroll
            for (int m = 0; m < 9; ++m) {
                sh_cov[threadIdx.x * 9 + m] = cov_inv[gid * 9 + m];
            }
        }
        __syncthreads();

        for (int vid = threadIdx.x; vid < total_voxels; vid += blockDim.x) {
            const int ld = vid / (rh * rw);
            const int lr = vid % (rh * rw);
            const int lh = lr / rw;
            const int lw = lr % rw;
            const int gd = d0 + ld;
            const int gh = h0 + lh;
            const int gw = w0 + lw;
            const int flat = gd * h_size * w_size + gh * w_size + gw;
            const float gvr = grad_vol_r[flat];
            const float gvi = grad_vol_i[flat];

            for (int local_g = 0; local_g < stage_count; ++local_g) {
                const int gid = sh_gid[local_g];
                const float cx = sh_centers[local_g * 3 + 0];
                const float cy = sh_centers[local_g * 3 + 1];
                const float cz = sh_centers[local_g * 3 + 2];
                const float dx = (static_cast<float>(gd) - cx) / hs0;
                const float dy = (static_cast<float>(gh) - cy) / hs1;
                const float dz = (static_cast<float>(gw) - cz) / hs2;
                const float* m = &sh_cov[local_g * 9];
                const float md0 = m[0] * dx + m[1] * dy + m[2] * dz;
                const float md1 = m[3] * dx + m[4] * dy + m[5] * dz;
                const float md2 = m[6] * dx + m[7] * dy + m[8] * dz;
                const float mahal = dx * md0 + dy * md1 + dz * md2;
                if (mahal <= 9.0f) {
                    const float weight = expf(-0.5f * mahal);
                    const float rho_r = sh_den_r[local_g];
                    const float rho_i = sh_den_i[local_g];
                    const float dL_dw = rho_r * gvr + rho_i * gvi;
                    const float dL_dmahal = dL_dw * (-0.5f) * weight;
                    atomicAdd(&grad_den_r[gid], weight * gvr);
                    atomicAdd(&grad_den_i[gid], weight * gvi);
                    atomicAdd(&grad_centers[gid * 3 + 0], dL_dmahal * (-2.0f * md0 / hs0));
                    atomicAdd(&grad_centers[gid * 3 + 1], dL_dmahal * (-2.0f * md1 / hs1));
                    atomicAdd(&grad_centers[gid * 3 + 2], dL_dmahal * (-2.0f * md2 / hs2));
                    atomicAdd(&grad_cov_inv[gid * 9 + 0], dL_dmahal * dx * dx);
                    atomicAdd(&grad_cov_inv[gid * 9 + 1], dL_dmahal * dx * dy);
                    atomicAdd(&grad_cov_inv[gid * 9 + 2], dL_dmahal * dx * dz);
                    atomicAdd(&grad_cov_inv[gid * 9 + 3], dL_dmahal * dy * dx);
                    atomicAdd(&grad_cov_inv[gid * 9 + 4], dL_dmahal * dy * dy);
                    atomicAdd(&grad_cov_inv[gid * 9 + 5], dL_dmahal * dy * dz);
                    atomicAdd(&grad_cov_inv[gid * 9 + 6], dL_dmahal * dz * dx);
                    atomicAdd(&grad_cov_inv[gid * 9 + 7], dL_dmahal * dz * dy);
                    atomicAdd(&grad_cov_inv[gid * 9 + 8], dL_dmahal * dz * dz);
                }
            }
        }
        __syncthreads();
    }
}

}  // namespace

std::vector<torch::Tensor> tile_voxelize_forward_cuda(
    torch::Tensor centers,
    torch::Tensor cov_inv,
    torch::Tensor density_r,
    torch::Tensor density_i,
    torch::Tensor tile_ranges,
    torch::Tensor pair_ids,
    torch::Tensor half_shape,
    int tile_size,
    int d_size,
    int h_size,
    int w_size) {
    const at::cuda::CUDAGuard device_guard(centers.device());
    const int num_tiles = tile_ranges.size(0);
    auto vol_r = torch::zeros({d_size * h_size * w_size}, centers.options());
    auto vol_i = torch::zeros({d_size * h_size * w_size}, centers.options());
    auto cov_flat = cov_inv.reshape({-1, 9}).contiguous();

    if (num_tiles > 0) {
        auto stream = at::cuda::getCurrentCUDAStream(centers.device().index());
        tile_voxelize_forward_kernel<<<num_tiles, kThreads, 0, stream.stream()>>>(
            centers.data_ptr<float>(),
            cov_flat.data_ptr<float>(),
            density_r.data_ptr<float>(),
            density_i.data_ptr<float>(),
            tile_ranges.data_ptr<int32_t>(),
            pair_ids.data_ptr<int32_t>(),
            half_shape.data_ptr<float>(),
            vol_r.data_ptr<float>(),
            vol_i.data_ptr<float>(),
            tile_size,
            d_size,
            h_size,
            w_size,
            num_tiles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {vol_r, vol_i};
}

std::vector<torch::Tensor> tile_voxelize_backward_cuda(
    torch::Tensor centers,
    torch::Tensor cov_inv,
    torch::Tensor density_r,
    torch::Tensor density_i,
    torch::Tensor tile_ranges,
    torch::Tensor pair_ids,
    torch::Tensor half_shape,
    torch::Tensor grad_vol_r,
    torch::Tensor grad_vol_i,
    int tile_size,
    int d_size,
    int h_size,
    int w_size) {
    const at::cuda::CUDAGuard device_guard(centers.device());
    const int num_tiles = tile_ranges.size(0);
    auto cov_flat = cov_inv.reshape({-1, 9}).contiguous();
    auto grad_centers = torch::zeros_like(centers);
    auto grad_cov_inv = torch::zeros({cov_flat.size(0), 9}, centers.options());
    auto grad_den_r = torch::zeros_like(density_r);
    auto grad_den_i = torch::zeros_like(density_i);

    if (num_tiles > 0) {
        auto stream = at::cuda::getCurrentCUDAStream(centers.device().index());
        tile_voxelize_backward_kernel<<<num_tiles, kThreads, 0, stream.stream()>>>(
            centers.data_ptr<float>(),
            cov_flat.data_ptr<float>(),
            density_r.data_ptr<float>(),
            density_i.data_ptr<float>(),
            tile_ranges.data_ptr<int32_t>(),
            pair_ids.data_ptr<int32_t>(),
            half_shape.data_ptr<float>(),
            grad_vol_r.data_ptr<float>(),
            grad_vol_i.data_ptr<float>(),
            grad_centers.data_ptr<float>(),
            grad_cov_inv.data_ptr<float>(),
            grad_den_r.data_ptr<float>(),
            grad_den_i.data_ptr<float>(),
            tile_size,
            d_size,
            h_size,
            w_size,
            num_tiles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {grad_centers, grad_cov_inv.reshape({-1, 3, 3}), grad_den_r, grad_den_i};
}
