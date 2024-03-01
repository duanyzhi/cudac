#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ __forceinline__ void relu_kernel(float* __restrict__ a, float* __restrict__ b, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x; 
  printf("%d %d %d \n", blockDim.x, blockIdx.x, i);
  if (i >= N) return;
  b[i] = a[i] > 0 ? a[i] : 0;
}

int main() {
  int N = 1000;
  size_t size = N * sizeof(float);
  float* ha = (float*)malloc(size);
  float* hb = (float*)malloc(size);
  for (int i = 0; i < N; ++i) {
    ha[i] = std::pow(-1, i) * i;
    std::cout << ha[i] << " ";
  }

  float* da;
  float* db;
  cudaMalloc((void**)&da, size);
  cudaMalloc((void**)&db, size);

  // cudaMemcpy
  cudaMemcpy((void*)da, (void*)ha, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid =
    (N + threadsPerBlock - 1) / threadsPerBlock;
 
  relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(da, db, N);

  // wait for synchronize
  cudaDeviceSynchronize();

  // copy cuda to cpu
  cudaMemcpy((void*)hb, (void*)db, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    std::cout << hb[i] << " ";
  }
  std::cout << "\n";
}
