#include "fun.h"

#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

#include <chrono>
using namespace std::chrono;

void printMatrix(int row, int col, float* ele) {
  std::cout << "Matrix (row, col) " << row << ", " << col << "\n";
  for (int r = 0; r < row; ++r) {
    for (int j = 0; j < col; ++j) {
      std::cout << ele[r * col + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

void init_matrix(float* data, int num) {
  float L = -5.0;
  float H = 5.0;
  for (int i = 0; i < num; ++i) {
    data[i] = L + static_cast <float> (std::rand()) /( static_cast <float> (RAND_MAX/(H - L)));
  }
}

void compute_error(float* base, float* dst, int num) {
   float sum = 0;
   for (int i = 0; i < num; ++i) {
     sum += (base[i] - dst[i]) * (base[i] - dst[i]);
   }
   sum /= num;
   std::cout << "mse error is " << sum << "\n";
}

// ./main M N K
// A: M * N, B: N * K, C: M * K
int main(int argc, char* argv[]) {
  std::cout << argc << "\n";
  int M{10}, N{10}, K{10};
  if (argc == 4) {
    M = std::stoi(argv[1]);
    N = std::stoi(argv[2]);
    K = std::stoi(argv[3]);
  }
  std::cout << "M, N, K: " << M << ", " << N << ", " << K << "\n";

  size_t size_A = M * N * sizeof(float);
  size_t size_B = N * K * sizeof(float);
  size_t size_C = M * K * sizeof(float);

  // allocate cpu buffer
  float* hA = (float*)malloc(size_A);
  float* hB = (float*)malloc(size_B);
  float* hC = (float*)malloc(size_C);
 
  init_matrix(hA, M * N);
  init_matrix(hB, N * K);

  if (M < 1000) {
    auto start = high_resolution_clock::now();
    mm_cpu(hA, hB, hC, M, N, K);
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "cpu time for matrix [" << M << ", " << N << ", " << K << "] is " << duration.count() << " ms." << std::endl;
  }

  // matmul cuda native
  float* hC_naive = (float*)malloc(size_C);
  run_mm_cuda_naive(hA, hB, hC_naive, M, N, K);
  compute_error(hC, hC_naive, M * K);

  // cuda matmul cublas
  float* hC_cublas = (float*)malloc(size_C);
  run_mm_cuda_cublas(hA, hB, hC_cublas, M, N, K);
  compute_error(hC, hC_cublas, M * K);

  // gpu memory coalescing
  float* hC_coalescing = (float*)malloc(size_C);
  run_mm_cuda_memory_coalescing(hA, hB, hC_coalescing, M, N, K);
  compute_error(hC, hC_coalescing, M * K);

  // check outputs
  // printMatrix(M, N, hA);
  // printMatrix(N, K, hB);
  // printMatrix(M, K, hC);
  // printMatrix(M, K, hC_naive);
  // printMatrix(M, K, hC_cublas);
 
  // free host malloc
  free(hA);
  free(hB);
  free(hC);
  free(hC_naive);
  free(hC_cublas);
  free(hC_coalescing);
}
