#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "fun.h"

// cuBLAS SGEMM 
void run_mm_cuda_cublas(float* hA, float* hB, float* hC, int M, int N, int K) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  cudaError_t cudaStat;  // cudaMalloc status
  cublasStatus_t stat;   // cuBLAS functions status
  cublasHandle_t handle; // cuBLAS context

  size_t size_A = M * N * sizeof(float);
  size_t size_B = N * K * sizeof(float);
  size_t size_C = M * K * sizeof(float);

  float* dA;
  float* dB;
  float* dC;
  cudaMalloc((void**)&dA, size_A);
  cudaMalloc((void**)&dB, size_B);
  cudaMalloc((void**)&dC, size_C);

  // run kernel
  stat = cublasCreate(&handle); // initialize CUBLAS context
  float alpha = 1.0f;
  float beta = 0.0f;

  // cudaMemcpy
  cudaMemcpy((void*)dA, (void*)hA, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)dB, (void*)hB, size_B, cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);
  // m -> M, n -> K, k -> N
  // ref: https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmex
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, dB, K,
                     dA, N, &beta, dC, K);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize( stop );

  // copy cuda to cpu
  cudaMemcpy((void*)hC, (void*)dC, size_C, cudaMemcpyDeviceToHost);
  
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  cublasDestroy(handle); // destroy CUBLAS context

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("cuda mm cublas time for [%d, %d, %d] is %f ms.\n", M, N, K, milliseconds);
}
