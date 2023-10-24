#include <cuda_runtime.h>
#include <vector>
#include <iostream>

typedef struct {
  int width;
  int height;
  float* elements;
} Matrix;

#define BLOCK_SIZE 16

// device code
__global__ void matmul(Matrix A, Matrix B, Matrix C) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  printf("(row, col): %d, %d\n", row, col);
  if (col >= B.width) return;
  if (row >= A.height) return;

  float sum_c = 0;
  for (int i = 0; i < A.width; ++i) {
    sum_c += A.elements[row * A.width + i] *
             B.elements[col + i * B.width];
  }
  C.elements[row * C.width + col] = sum_c;
}

void printMatrix(int row, int col, float* ele) {
  std::cout << "Matrix (row, col) " << row << ", " << col << "\n";
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      std::cout << ele[i * col + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

int main() {
  int W = 10;
  int H = 20;
  // A: [W, H], B: [H, W], C: [H, H]
  size_t size = W * H * sizeof(float);
  size_t size_C = H * H * sizeof(float);

  // allocate cpu buffer
  float* hA = (float*)malloc(size);
  float* hB = (float*)malloc(size);
  float* hC = (float*)malloc(size_C);
  for (int i = 0; i < W * H; ++i) {
    hA[i] = i;
    hB[i] = i;
  }

  // allocate gpu buffer
  float* dA;
  float* dB;
  float* dC;
  cudaMalloc((void**)&dA, size);
  cudaMalloc((void**)&dB, size);
  cudaMalloc((void**)&dC, size_C);

  // cudaMemcpy
  cudaMemcpy((void*)dA, (void*)hA, size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)dB, (void*)hB, size, cudaMemcpyHostToDevice);

  Matrix ma, mb, mc;
  ma.width = W;
  ma.height = H;
  ma.elements = dA;
  mb.width = H;
  mb.height = W;
  mb.elements = dB;
  mc.width = H;
  mc.height = H;
  mc.elements = dC;

  // run kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((mb.width + dimBlock.x - 1) / dimBlock.x,
               (ma.height + dimBlock.y - 1) / dimBlock.y);
  std::cout << dimGrid.x << " " << dimGrid.y << "\n";
  matmul<<<dimGrid, dimBlock>>>(ma, mb, mc);

  // wait for synchronize
  cudaDeviceSynchronize();

  // copy cuda to cpu
  cudaMemcpy((void*)hC, (void*)dC, size_C, cudaMemcpyDeviceToHost);
  
  printMatrix(ma.height, ma.width, hA);
  printMatrix(mb.height, mb.width, hB);
  printMatrix(mc.height, mc.width, hC);

  // free all memory
  free(hA);
  free(hB);
  free(hC);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}
