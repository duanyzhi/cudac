#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <math.h> 
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

// if N <= 16
template <
    const int ROW_PER_WARP
    > 
__global__ void Sgemv_v2( 
    float * __restrict__ A,
    float * __restrict__ x,
    float * __restrict__ y, 
    const int M,
    const int N) {
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size=32;
    int laneId= tx % warp_size;
    int current_warp_row = (blockDim.y * bx + ty) * ROW_PER_WARP;
    const int kWarp_size = warp_size / ROW_PER_WARP;
    int kLaneId = laneId % kWarp_size;
    int current_thread_row = current_warp_row + laneId / kWarp_size;

    if(current_thread_row < M){
        float res=0;
        int current_col = kLaneId;
        res += A[current_thread_row * N + current_col] * x[current_col];
        res = warpReduceSum<kWarp_size>(res);
        if(kLaneId==0) y[current_thread_row]=res;
    }
}

at::Tensor gemv_forward(at::Tensor a, at::Tensor b) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(0);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_forward, "Gemm forward (CUDA)",
          py::arg("a"), py::arg("b"));
}