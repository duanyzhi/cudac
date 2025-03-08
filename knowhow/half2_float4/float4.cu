#include <cuda.h>
#include <cuda_runtime.h> 
#include <cuda_fp16.h>
#include <iostream>


const unsigned int WARP_REDUCE_MASK = 0xffffffff;

__global__ void float_sum(float *data, int step) {
  __shared__ double sum[32];
 
  for (uint idx = threadIdx.x * step; idx < threadIdx.x * step + step; idx += 1) {
    sum[threadIdx.x] += data[idx];
  }
  __syncthreads();
  double pval = sum[threadIdx.x];

  // warp_sum
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 16, 32);
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 8, 32);
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 4, 32);
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 2, 32);
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 1, 32);

  __syncthreads();
  // 536854528
  // printf("result tid %d num %f", threadIdx.x, pval);
}

__global__ void float4_sum(const float4 *data, int step) {
  __shared__ double sum[32];
 
  for (uint idx = threadIdx.x * step; idx < threadIdx.x * step + step; idx += 1) {
    float4 val = data[idx];  // global memory to register
    sum[threadIdx.x] += (val.x + val.y + val.z + val.w);
  }
  __syncthreads();
  double pval = sum[threadIdx.x];

  // warp_sum
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 16, 32);
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 8, 32);
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 4, 32);
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 2, 32);
  pval += __shfl_xor_sync(WARP_REDUCE_MASK, pval, 1, 32);

  __syncthreads();
  // 536854528
  // printf("result tid %d num %f", threadIdx.x, pval);
}

int main() {
    const int N = 32 * 1024;
    float* cpu_data = (float*)malloc(sizeof(float) * N);
    float* out_data = (float*)malloc(sizeof(float) * N);
    // std::cout << "In:\n";
    for (int i = 0; i < N; ++i) {
      cpu_data[i] = i;
      // std::cout << cpu_data[i] << " ";
    }
    // std::cout << "\n";

    float* device_data;
    cudaMalloc(&device_data, sizeof(float) * N);

    cudaMemcpy(device_data, cpu_data, sizeof(float) *N, cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(32);

    //float sum
    // float_sum<<<grid, block>>>(device_data, 1024);
    // 536854528
    // cudaMemcpy(out_data, device_data, sizeof(float) *N, cudaMemcpyDeviceToHost);

    // float4 sum
    const float4* data_f4 = reinterpret_cast<const float4 *>(device_data);
    float4_sum<<<grid, block>>>(data_f4, 1024 / 4);
    

    // std::cout << "warp_sum(should be 496):\n";
    // for (int i = 0; i < N; ++i) {
    //   std::cout << out_data[i] << " ";
    // }
    // std::cout << "\n";

    free(cpu_data);
    free(out_data);
    cudaFree(device_data);
}

