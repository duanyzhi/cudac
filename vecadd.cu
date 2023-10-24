#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// device code
__global__ void vecadd(float* A, float* B, float* C, int N) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  printf("index %d\n", index);
  if (index < N) {
    C[index] = A[index] + B[index];
  }
}

int main() {
  int N = 1024;
  size_t size = N * sizeof(float);

  // allocate cpu buffer
  float* hA = (float*)malloc(size);
  float* hB = (float*)malloc(size);
  float* hC = (float*)malloc(size);
  for (int i = 0; i < N; ++i) {
    hA[i] = 1.0;
    hB[i] = 2.0;
  }

  // allocate gpu buffer
  float* dA;
  float* dB;
  float* dC;
  cudaMalloc((void**)&dA, size);
  cudaMalloc((void**)&dB, size);
  cudaMalloc((void**)&dC, size);

  // cudaMemcpy
  cudaMemcpy((void*)dA, (void*)hA, size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)dB, (void*)hB, size, cudaMemcpyHostToDevice);

  // run kernel
  int threadsPerBlock = 256;
  int blocksPerGrid =
    (N + threadsPerBlock - 1) / threadsPerBlock;
  vecadd<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);

  // wait for synchronize
  cudaDeviceSynchronize();

  // copy cuda to cpu
  cudaMemcpy((void*)hC, (void*)dC, size, cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < N; ++i) {
    std::cout << hC[i] << " ";
  }


  // free all memory
  free(hA);
  free(hB);
  free(hC);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}
