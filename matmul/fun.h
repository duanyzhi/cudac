#include <cuda_runtime.h>

void mm_cpu(float* A, float* B, float* C, int M, int N, int K);

void run_mm_cuda_naive(float* A, float* B, float* C, int M, int N, int K);

void run_mm_cuda_cublas(float* A, float* B, float* C, int M, int N, int K);

void run_mm_cuda_memory_coalescing(float* hA, float* hB, float* hC, int M, int N, int K);
