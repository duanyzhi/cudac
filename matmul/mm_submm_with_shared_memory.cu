#include "fun.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16

// only work for M(or K, or N) // BLOCK_SIZE == 0
// rule: one thread for one row or col
__global__ void mm_cuda_sub_shared_memory(float* A, float* B, float* C, int M, int K, int N) {
  // block
  int bx = blockIdx.x;
  int by = blockIdx.y;
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row_c = by * BLOCK_SIZE + ty;  
  int col_c = bx * BLOCK_SIZE + tx;  

  if (row_c >= M) return;
  if (col_c >= N) return;

  // row begin is for begin indx for current blcok y
  int row_begin = by * BLOCK_SIZE * K;
  int row_end = row_begin + K - 1; // last element for this row
  // step to next ele for this thread, one thread compute multi ele by step BLOCK_SIZE
  int row_step = BLOCK_SIZE;

  int col_begin = bx * BLOCK_SIZE;
  int col_end = col_begin + (K - 1) * N;
  int col_step = BLOCK_SIZE * N;

  float sum = 0;  // compute one thread for one row or col
  for (int row = row_begin, col = col_begin; row <= row_end;
    row += row_step, col += col_step) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[row + K * ty + tx];
    Bs[ty][tx] = B[col + N * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      sum += As[ty][k] * Bs[k][tx];
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }
  C[row_c * N + col_c] = sum;
}

/***
 *  -------------> x(col)
 *  |
 *  |      * (row, col)
 *  |
 *  y(row)
 ***/
void run_mm_cuda_sub_mm_shared_memory(float* hA, float* hB, float* hC, int M, int K, int N) {
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
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
               (M + dimBlock.y - 1) / dimBlock.y);

  mm_cuda_sub_shared_memory<<<dimGrid, dimBlock>>>(dA, dB, dC, M, K, N);

  // wait for synchronize
  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize( stop );

  // copy cuda to cpu
  cudaMemcpy((void*)hC, (void*)dC, size_C, cudaMemcpyDeviceToHost);
  
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("cuda submatrix for [%d, %d, %d] is %f ms.\n", M, K, N, milliseconds);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}
