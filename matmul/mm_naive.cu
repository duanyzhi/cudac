#include "fun.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void mm_cuda_naive(float* A, float* B, float* C, int M, int K, int N) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= M) return;
  if (col >= N) return;

  float sum_c = 0;
  for (int i = 0; i < K; ++i) {
    sum_c += A[row * K + i] *
             B[col + i * N];
  }
  C[row * N + col] = sum_c;
}


void run_mm_cuda_naive(float* hA, float* hB, float* hC, int M, int K, int N) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  float* dA;
  float* dB;
  float* dC;
  cudaMalloc((void**)&dA, size_A);
  cudaMalloc((void**)&dB, size_B);
  cudaMalloc((void**)&dC, size_C);

  // cudaMemcpy
  cudaMemcpy((void*)dA, (void*)hA, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)dB, (void*)hB, size_B, cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);
  //// run kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x,
               (N + dimBlock.y - 1) / dimBlock.y);

  // 1 thread for 1 A row multipy 1 B col
  mm_cuda_naive<<<dimGrid, dimBlock>>>(dA, dB, dC, M, K, N);

  // wait for synchronize
  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize( stop );

  // copy cuda to cpu
  cudaMemcpy((void*)hC, (void*)dC, size_C, cudaMemcpyDeviceToHost);
  
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("cuda mm naive time for [%d, %d, %d] is %f ms.\n", M, K, N, milliseconds);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}
