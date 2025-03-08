#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


__global__ void fma_kernel_v0(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ o,
    const int size) {

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int thread_stride = blockDim.x * blockDim.y;
    const int thread_id = tidy * blockDim.x + tidx;

    #pragma unroll 8
    for (int idx = thread_id; idx < size; idx += thread_stride) {
        o[idx] = a[idx] * b[idx] + c[idx];
    }

    // do not sync for diff thread
    // __syncthreads();
}

// 25ms
__global__ void fma_kernel_v1(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ o,
    const int size) {

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int thread_stride = blockDim.x * blockDim.y;
    const int thread_id = tidy * blockDim.x + tidx;

    #pragma unroll 8
    for (int idx = thread_id; idx < size; idx += thread_stride) {
        // o[idx] = a[idx] * b[idx] + c[idx];
        o[idx] = fma(a[idx], b[idx], c[idx]);
        // o[idx] = __fmaf_rn(a[idx], b[idx], c[idx]);
    }

    // do not sync for diff thread
    // __syncthreads();
}


torch::Tensor fma_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto o = torch::empty_like(a);

    const int size = a.numel();

    const dim3 blocks(1);
    const dim3 threads(32, 32);
    fma_kernel_v1<<<blocks, threads>>>(
        reinterpret_cast<float*>(a.data_ptr<float>()),
        reinterpret_cast<float*>(b.data_ptr<float>()),
        reinterpret_cast<float*>(c.data_ptr<float>()),
        reinterpret_cast<float*>(o.data_ptr<float>()),
        size
    );

    cudaStreamSynchronize(stream);
    return o;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fma", &fma_forward, "FMA forward (CUDA)",
          py::arg("a"), py::arg("b"), py::arg("c"));
}
