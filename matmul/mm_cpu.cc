#include "fun.h"

void mm_cpu(float* A, float* B, float* C, int M, int N, int K) {
  for (int row = 0; row < M; ++row) {
    for (int bw = 0; bw < K; ++bw) {
      float sum = 0;
      for (int col = 0; col < N; ++col) {
        sum += A[row * N + col] * B[col * K + bw];
      }
      C[row * K + bw] = sum;
    }
  }
}
