#include <cuda.h>
#include <cuda_runtime.h> 
#include <cuda_fp16.h>
#include <iostream>

__global__ void shfl_xor_example(float* data, int laneMask=1) {
    int tid = threadIdx.x;
    float value = data[tid];

    // 每个线层和+1线程交换数据
    float shuffled_value = __shfl_xor_sync(0xFFFFFFFF, value, laneMask, 32);

    data[tid] = shuffled_value;
}

int main() {
    const int N = 32;
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
    dim3 block(32);
    int laneMask = 1;
    shfl_xor_example<<<grid, block>>>(device_data, laneMask);

    cudaMemcpy(out_data, device_data, sizeof(float) *N, cudaMemcpyDeviceToHost);

    std::cout << "__shfl_xor_sync:\n";
    for (int i = 0; i < N; ++i) {
      std::cout << out_data[i] << " ";
    }
    std::cout << "\n";

    free(cpu_data);
    free(out_data);
}

