#include "fun.h"

void mm_cpu(float* A, float* B, float* C, int M, int K, int N) {
  for (int row = 0; row < M; ++row) {
    for (int bw = 0; bw < N; ++bw) {
      float sum = 0;
      for (int col = 0; col < K; ++col) {
        sum += A[row * K + col] * B[col * N + bw];
      }
      C[row * N + bw] = sum;
    }
  }
}
