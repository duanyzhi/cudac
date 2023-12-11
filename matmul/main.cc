#include "fun.h"

#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

#include <chrono>
using namespace std::chrono;

void printMatrix(std::string info, int row, int col, float* ele) {
  std::cout << info << " Matrix (row, col) " << row << ", " << col << "\n";
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

void compute_error(std::string info, float* base, float* dst, int num) {
   float sum = 0;
   for (int i = 0; i < num; ++i) {
     sum += (base[i] - dst[i]) * (base[i] - dst[i]);
   }
   sum /= num;
   std::cout << info << " mse error is " << sum << "\n";
}

// ./main M N K
// A: M * N, B: N * K, C: M * K
// A: M * K, B: K * N, C: M * N
int main(int argc, char* argv[]) {
  std::cout << argc << "\n";
  int M{32}, K{32}, N{32};
  if (argc == 4) {
    M = std::stoi(argv[1]);
    K = std::stoi(argv[2]);
    N = std::stoi(argv[3]);
  }
  std::cout << "M, N, K: " << M << ", " << N << ", " << K << "\n";

  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  // allocate cpu buffer
  float* hA = (float*)malloc(size_A);
  float* hB = (float*)malloc(size_B);
  float* hC = (float*)malloc(size_C);
 
  init_matrix(hA, M * K);
  init_matrix(hB, K * N);

  if (M < 1000) {
    auto start = high_resolution_clock::now();
    mm_cpu(hA, hB, hC, M, K, N);
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "cpu time for matrix [" << M << ", " << K << ", " << N << "] is " << duration.count() << " ms." << std::endl;
  }

  // matmul cuda native
  float* hC_naive = (float*)malloc(size_C);
  run_mm_cuda_naive(hA, hB, hC_naive, M, K, N);
  compute_error("naive: ", hC, hC_naive, M * N);

  // cuda matmul cublas
  float* hC_cublas = (float*)malloc(size_C);
  run_mm_cuda_cublas(hA, hB, hC_cublas, M, K, N);
  compute_error("cublas: ", hC, hC_cublas, M * N);

  // cuda memory coalescing
  float* hC_coalescing = (float*)malloc(size_C);
  run_mm_cuda_memory_coalescing(hA, hB, hC_coalescing, M, K, N);
  compute_error("memory coalescing: ", hC, hC_coalescing, M * N);

  // sub matrix
  float* hC_subMatrix = (float*)malloc(size_C);
  run_mm_cuda_sub_mm_shared_memory(hA, hB, hC_subMatrix, M, K, N);
  compute_error("memory subMatrix: ", hC, hC_subMatrix, M * N);

  // check outputs
  if (M <= 50 && K <= 50 && N <= 50) {
    printMatrix("A: ", M, K, hA);
    printMatrix("B: ", K, N, hB);
    printMatrix("host C: ", M, N, hC);
    printMatrix("naive cuda: ", M, N, hC_naive);
    printMatrix("cublas: ", M, N, hC_cublas);
    printMatrix("global memory coalescing: ", M, N, hC_coalescing);
    printMatrix("sub Matrix: ", M, N, hC_subMatrix);
  }
 
  // free host malloc
  free(hA);
  free(hB);
  free(hC);
  free(hC_naive);
  free(hC_cublas);
  free(hC_coalescing);
  free(hC_subMatrix);
}
