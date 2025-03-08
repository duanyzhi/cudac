#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// WMMA matrix dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_kernel(half *a, half *b, float *c, int m, int n, int k) {
    // Declare matrix fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load matrix fragments
    wmma::load_matrix_sync(a_frag, a, k);  // k = leading dimension for row-major
    wmma::load_matrix_sync(b_frag, b, k);  // k = leading dimension for column-major

    // Matrix multiply-accumulate
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store result
    wmma::store_matrix_sync(c, c_frag, n, wmma::mem_row_major);
}


torch::Tensor mma_forward(torch::Tensor a, torch::Tensor b) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    std::cout << "[M, K, N]: " << M << ", " << K << ", " << N << "\n";
    std::cout << "[M/WMMA_M, K/WMMA_K, N/WMMA_N]: " << M / WMMA_M << ", " << K << ", " << N << "\n";

    torch::Tensor c = torch::empty(
        {M, N}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
    );

    // 启动核函数
    const dim3 gridDim(1);  // 分块网格维度
    //const dim3 gridDim(N / WMMA_N, M / WMMA_M);  // 分块网格维度
    const dim3 blockDim(WMMA_M, WMMA_N);         // 每个块的线程数

    wmma_kernel<<<gridDim, blockDim>>>(
        reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<float*>(c.data_ptr<float>()),
        M, N, K
    );

    cudaStreamSynchronize(stream);
    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mma", &mma_forward, "MMA forward (CUDA)",
          py::arg("a"), py::arg("b"));
}
