#include <cuda.h>
#include <cuda_runtime.h> 
#include <cuda_fp16.h>
#include <iostream>


const unsigned int WARP_REDUCE_MASK = 0xffffffff;

__global__ void warp_sum(float *data) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float pval = data[tid];

  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 16, 32);
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 8, 32);
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 4, 32);
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 2, 32);
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 1, 32);
  __syncthreads();
  data[tid] = pval;
}


int main() {
    const int N = 32 * 2;
    float* cpu_data = (float*)malloc(sizeof(float) * N);
    float* out_data = (float*)malloc(sizeof(float) * N);
    std::cout << "In:\n";
    for (int i = 0; i < N; ++i) {
      cpu_data[i] = i;
      std::cout << cpu_data[i] << " ";
    }
    std::cout << "\n";

    float* device_data;
    cudaMalloc(&device_data, sizeof(float) * N);

    cudaMemcpy(device_data, cpu_data, sizeof(float) *N, cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(64);
    warp_sum<<<grid, block>>>(device_data);
    cudaMemcpy(out_data, device_data, sizeof(float) *N, cudaMemcpyDeviceToHost);

    std::cout << "warp_sum(should be 496):\n";
    for (int i = 0; i < N; ++i) {
      std::cout << out_data[i] << " ";
    }
    std::cout << "\n";

    free(cpu_data);
    free(out_data);
    cudaFree(device_data);
}

