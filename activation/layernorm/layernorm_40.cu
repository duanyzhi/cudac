#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <iostream>
#include "pybind.h"

static const int NUM_STREAMS = 8;
static cudaStream_t streams[NUM_STREAMS];
static bool streams_created = false;

// Custom error-checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

void create_streams() {
    if (!streams_created) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
        }
        streams_created = true;
    }
}

void destroy_streams() {
    if (streams_created) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
        streams_created = false;
    }
}

template <typename scalar_t>
__global__ void layernorm_streamed_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size,
    const int chunk_size,
    const int chunk_offset) {
    // printf("debug 0\n");

    using accscalar_t = at::acc_type<scalar_t, true>;  // double

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int instance_idx = blockIdx.x + chunk_offset;
    // printf("tidx %d, tidy %d, instance_idx %d , chunk offset %d: ", tidx, tidy, instance_idx, chunk_offset);

    extern __shared__ char smem[];
    accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
    accscalar_t* s_sum_sq = s_sum + blockDim.x * blockDim.y;

    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    scalar_t* out_ptr = output + instance_idx * normalized_size;

    const int thread_stride = blockDim.x * blockDim.y;
    const int thread_id = tidy * blockDim.x + tidx;

    accscalar_t local_sum = 0;
    accscalar_t local_sum_sq = 0;

    //printf("thread id: %d normalized_size: %d ;;;; ", thread_id, normalized_size);

    #pragma unroll 8
    for (int idx = thread_id; idx < normalized_size; idx += thread_stride) {
        // printf("debug for idx: %d thread_stride: %d ", idx, thread_stride);
        //printf("%d ", idx);
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        // printf("idx %d val %f ", idx, val);
        local_sum += val;
        local_sum_sq += val * val;
        //printf("sum % f;\n", local_sum_sq);
    }

    s_sum[thread_id] = local_sum;
    s_sum_sq[thread_id] = local_sum_sq;
    __syncthreads();

    if (thread_id < 32) {
        accscalar_t warp_sum = 0;
        accscalar_t warp_sum_sq = 0;

        #pragma unroll
        for (int i = thread_id; i < thread_stride; i += 32) {
            warp_sum += s_sum[i];
            warp_sum_sq += s_sum_sq[i];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            warp_sum_sq += __shfl_down_sync(0xffffffff, warp_sum_sq, offset);
        }

        if (thread_id == 0) {
            s_sum[0] = warp_sum;
            s_sum_sq[0] = warp_sum_sq;
        }
    }
    __syncthreads();

    // printf("debug 3\n");

    __shared__ accscalar_t mean, inv_std;
    if (thread_id == 0) {
        mean = s_sum[0] / normalized_size;
        accscalar_t variance = (s_sum_sq[0] / normalized_size) - (mean * mean);
        inv_std = rsqrt(variance + static_cast<accscalar_t>(eps));
    }
    __syncthreads();

    // printf("debug 4\n");
    #pragma unroll 8
    for (int idx = thread_id; idx < normalized_size; idx += thread_stride) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        accscalar_t normalized = (val - mean) * inv_std;
        out_ptr[idx] = static_cast<scalar_t>(
            normalized * static_cast<accscalar_t>(weight[idx]) +
            static_cast<accscalar_t>(bias[idx]));
        //printf("output ptr and value: %d -> %f", idx, out_ptr[idx]);
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
    create_streams();

    auto output = torch::empty_like(x);

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);

    // printf("Shared Memory per Block (default): %zu bytes\n", prop.sharedMemPerBlock);
    // printf("Max Shared Memory per Block (configurable): %zu bytes\n", prop.sharedMemPerBlockOptin);

    // printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    // printf("Max grid dimensions: (%d, %d, %d)\n",
    //        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    //
    // printf("Max registers per block: %d\n", prop.regsPerBlock);
    // printf("Max registers per thread: %d\n", prop.regsPerMultiprocessor / prop.maxThreadsPerMultiProcessor);


    const int normalized_size = weight.numel();
    const int outer_size = x.numel() / normalized_size;
    const int chunk_size = (outer_size + NUM_STREAMS - 1) / NUM_STREAMS;

    const dim3 threads(32, 32);
    const int shared_mem_size = threads.x * threads.y * 2 * sizeof(float);
    // std::cout << "x numel and normalized size: " << x.numel() << " " << normalized_size << "\n";

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        for (int i = 0; i < NUM_STREAMS; i++) {
            int stream_chunk_size = std::min(chunk_size, outer_size - i * chunk_size);
            if (stream_chunk_size <= 0) {
                std::cout << "stream chunk size invalid.\n";
                break;
            }

            const dim3 blocks(stream_chunk_size);

            //  std::cout << "stream chunk size: " << stream_chunk_size << " th.x" << threads.x << " th.y" << threads.y << " " << streams[i] << "\n";
            //  std::cout << "Alloc shared mem: " << shared_mem_size / 1024 << " KB \n";
            //  std::cout << x.scalar_type() << "?\n";
            //  std::cout << "debug input ptr: " << x.data_ptr<scalar_t>() << "\n";

            layernorm_streamed_kernel<scalar_t><<<blocks, threads, shared_mem_size, streams[i]>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                static_cast<float>(eps),
                output.data_ptr<scalar_t>(),
                normalized_size,
                chunk_size,
                i * chunk_size);
        }
    }));

   // cudaError_t err = cudaGetLastError();
   // printf("CUDA error: %s\n", cudaGetErrorString(err));

    // Synchronize all streams before returning
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    return output;
}
