#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// WMMA matrix dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
const int wrap_size = 32;
const int BLOCK_WRAP_M = 8;  // one block process 8 wrap_m
const int BLOCK_WRAP_N = 4;  // one block process 4 wrap_n

//  dim3 gridDim(1);
//  dim3 blockDim(32);
__global__ void wmma_kernel(half *a, half *b, float *c, int m, int n, int k) {
    // Declare matrix fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load matrix fragments
    wmma::load_matrix_sync(a_frag, a, k);  // k = leading dimension for row-major
    wmma::load_matrix_sync(b_frag, b, n);  // k = leading dimension for column-major

    // Matrix multiply-accumulate
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store result
    wmma::store_matrix_sync(c, c_frag, n, wmma::mem_row_major);
}

   
//  dim3 gridDim(2, 2);
//  dim3 blockDim(32);
__global__ void wmma_kernel_v1(half *a, half *b, float *c, int M, int N, int K) {
   // Leading dimensions. Packed with no transpositions.
   int lda = K;
   int ldb = N;
   int ldc = N;

   float alpha = 1.0;
   float beta = 1.0;

   // Tile using a 2D grid
   int warpM = blockIdx.x; // (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = blockIdx.y; // (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      printf("aRow aCol %d %d bRow bCol %d %d\n", aRow, aCol, bRow, bCol);

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow * lda + aCol, lda);
         wmma::load_matrix_sync(b_frag, b + bRow * ldb + bCol, ldb);

         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

#pragma unroll
      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
   }
}

__global__ void wmma_kernel_v2(half *a, half *b, float *c, int M, int N, int K) {
   // Leading dimensions for row major. Packed with no transpositions.
   int lda = K;
   int ldb = N;
   int ldc = N;

   float alpha = 1.0;
   float beta = 1.0;

   // Tile using a 2D grid
   int warpM = blockIdx.x * BLOCK_WRAP_M + threadIdx.x / wrap_size;
   int warpN = blockIdx.y * BLOCK_WRAP_N + threadIdx.y;
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow * lda + aCol, lda);
         wmma::load_matrix_sync(b_frag, b + bRow * ldb + bCol, ldb);

         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

#pragma unroll
      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
   }
}

at::Tensor mma_forward(at::Tensor a, at::Tensor b) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    at::Tensor c = at::zeros(
        {M, N}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
    );

   // cudaDeviceProp prop;
   // cudaGetDeviceProperties(&prop, 0);
   // printf("Max grid size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // 启动核函数
    // const dim3 gridDim(1);  // 分块网格维度
    // const dim3 gridDim(N / WMMA_N, M / WMMA_M);  // 分块网格维度
    // const dim3 blockDim(WMMA_M, WMMA_N);         // 每个块的线程数

    // dim3 gridDim(N / WMMA_N, M / WMMA_M);  // 分块网格维度
    // dim3 blockDim(WMMA_N, WMMA_M);         // 每个块的线程数

    // dim3 gridDim(1);
    // dim3 blockDim(32 * 4);  // One warp

    // dim3 gridDim;
    // dim3 blockDim;
 
    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    // blockDim.x = 128;
    // blockDim.y = 4;

    // gridDim.x = (M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    // gridDim.y = (N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    int wrap_M = max((M + WMMA_M - 1) / WMMA_M / BLOCK_WRAP_M, 1);
    int wrap_N = max((N + WMMA_N - 1) / WMMA_N / BLOCK_WRAP_N, 1);

    //  printf("wrapM, wrapN: %d %d \n", wrap_M, wrap_N);

    // one block max has 1024 thread, which is 1024 / 32 = 32 blocks for wmma.
    
    // 32 * 32 -> (1, 1), (32, 32)
    // 512 * 512 ->  (1, 1), (512 / 16 * 32, 512 / 16 * 32) = (1024, 1024)
    // 1024 * 1024 ->
    // 4096 * 4096 -> 
   
    dim3 gridDim(wrap_M, wrap_N);
    dim3 blockDim(BLOCK_WRAP_M * wrap_size, BLOCK_WRAP_N);  // One block

    wmma_kernel_v2<<<gridDim, blockDim>>>(
        reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<float*>(c.data_ptr<float>()),
        M, N, K
    );

    cudaStreamSynchronize(stream);
    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mma", &mma_forward, "MMA forward (CUDA)",
          py::arg("a"), py::arg("b"));
}
