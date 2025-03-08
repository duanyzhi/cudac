#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


#include <cuda_fp16.h>

struct __align__(8) half4 {  // 8字节对齐（half 占2字节，4个共8字节）
    __half x, y, z, w;
};

__global__ void vecadd_kernel_v0(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    const int add_size) {

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int thread_stride = blockDim.x * blockDim.y;
    const int thread_id = tidy * blockDim.x + tidx;

    #pragma unroll 8
    for (int idx = thread_id; idx < add_size; idx += thread_stride) {
        c[idx] = __hadd(a[idx], b[idx]);
    }

    // do not sync for diff thread
    // __syncthreads();
}

__global__ void print_half2_kernel(half2 value) {
    // 将 half2 分解为两个 half 并转换为 float
    float x = __half2float(value.x);
    float y = __half2float(value.y);
    
    // 打印结果
    printf("half2: (%f, %f)\n", x, y);
}

// use half2 add
__global__ void vecadd_kernel_v1(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    const int add_size) {

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int thread_stride = blockDim.x * blockDim.y;
    const int thread_id = tidy * blockDim.x + tidx;

    const int vec_size = add_size / 2;
    const half2* a_vec = reinterpret_cast<const half2*>(a);
    const half2* b_vec = reinterpret_cast<const half2*>(b);
    half2* c_vec = reinterpret_cast<half2*>(c);

    #pragma unroll
    for (int i = thread_id; i < vec_size; i += thread_stride) {
        half2 a_val = a_vec[i];
        half2 b_val = b_vec[i];
        c_vec[i] = __hadd2(a_val, b_val);

        // float x = __half2float(c_vec[i].x);
        // float y = __half2float(c_vec[i].y);
        // 
        // // 打印结果
        // printf("half2: (%f, %f)\n", x, y);
    }

    // #pragma unroll 8
    // for (int idx = thread_id; idx < add_size; idx += thread_stride) {
    //     c[idx] = __hadd(a[idx], b[idx]);
    // }
    // __syncthreads();
}



torch::Tensor vec_add_forward(torch::Tensor a, torch::Tensor b) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto c = torch::empty_like(a);

    const int add_size = a.numel();

    // create __half4

    const dim3 blocks(1);
    const dim3 threads(32, 32);
    vecadd_kernel_v0<<<blocks, threads>>>(
        reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
        add_size
    );

    cudaStreamSynchronize(stream);
    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vec_add", &vec_add_forward, "VecAdd forward (CUDA)",
          py::arg("a"), py::arg("b"));
}
