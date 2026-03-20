#include <torch/extension.h>

#include <vector>

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
    int w_size);

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
    int w_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tile_voxelize_forward_cuda, "Tile voxelization forward (CUDA)");
    m.def("backward", &tile_voxelize_backward_cuda, "Tile voxelization backward (CUDA)");
}
