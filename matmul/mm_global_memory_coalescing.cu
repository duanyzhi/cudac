#include "fun.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 16

__global__ void mm_cuda_memory_coalescing(float* A, float* B, float* C, int M, int K, int N) {
  const int row = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
  const int col = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

  if (row >= M) return;
  if (col >= N) return;

  //printf("bidx %d, tx %d, bidy %d, ty %d, row: %d, col: %d.\n", blockIdx.x, threadIdx.x, blockIdx.y, threadIdx.y, row, col);
  float sum_c = 0;
  for (int i = 0; i < K; ++i) {
    sum_c += A[row * K + i] *
             B[col + i * N];
  }
  C[row * N + col] = sum_c;
}


void run_mm_cuda_memory_coalescing(float* hA, float* hB, float* hC, int M, int K, int N) {
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

  dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
  dim3 dimGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
  mm_cuda_memory_coalescing<<<dimGrid, dimBlock>>>(dA, dB, dC, M, K, N);
  std::cout << "grid x " << dimGrid.x << " " << dimGrid.y << " " << dimBlock.x << " " << dimBlock.y << "\n";

  // wait for synchronize
  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize( stop );

  // copy cuda to cpu
  cudaMemcpy((void*)hC, (void*)dC, size_C, cudaMemcpyDeviceToHost);
  
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("cuda mm gpu memory coalescing time for [%d, %d, %d] is %f ms.\n", M, K, N, milliseconds);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}
