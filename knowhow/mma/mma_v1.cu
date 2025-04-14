#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

// WMMA matrix dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_kernel_v1(half *a, half *b, float *c, int M, int N, int K) {
    // 声明分片（Fragment）对象
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // 初始化累加器分片为 0
    wmma::fill_fragment(acc_frag, 0.0f); // M=16, N=16, K=16

    // 计算分块坐标
    int tile_row = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_col = blockIdx.x * blockDim.x + threadIdx.x;

    // 从全局内存加载矩阵分片到 Fragment
    wmma::load_matrix_sync(a_frag, a + tile_row * K, K); // A 矩阵布局为行主序
    wmma::load_matrix_sync(b_frag, b + tile_col, N);     // B 矩阵布局为列主序

    // 执行矩阵乘累加：acc_frag = a_frag * b_frag + acc_frag
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // 将累加器结果存储到全局内存
    wmma::store_matrix_sync(c + tile_row * N + tile_col, acc_frag, N, wmma::mem_row_major);
}


__global__ void wmma_kernel(half *a, half *b, float *c, int m, int n, int k) {
    // Declare matrix fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load matrix fragments
    wmma::load_matrix_sync(a_frag, a, k);  // k = leading dimension for row-major
    wmma::load_matrix_sync(b_frag, b, k);  // k = leading dimension for column-major

    // Matrix multiply-accumulate
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store result
    wmma::store_matrix_sync(c, c_frag, n, wmma::mem_row_major);
}

int main() {
    //const int M = 4096, N = 4096, K = 4096;
    const int M = 16, N = 16, K = 16;
    size_t a_size = M * K * sizeof(half);
    size_t b_size = K * N * sizeof(half);
    size_t c_size = M * N * sizeof(float);

    // Allocate host memory
    half *h_a = new half[M*K];
    half *h_b = new half[K*N];
    float *h_c = new float[M*N];

    // Initialize input matrices
    for(int i = 0; i < M*K; i++) h_a[i] = __float2half(1.0f);
    for(int i = 0; i < K*N; i++) h_b[i] = __float2half(1.0f);

    // Allocate device memory
    half *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, a_size);
    cudaMalloc(&d_b, b_size);
    cudaMalloc(&d_c, c_size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 grid(1);
    dim3 block(32);  // One warp
    wmma_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);

    // dim3 gridDim(N / WMMA_N, M / WMMA_M);  // 分块网格维度
    // dim3 blockDim(WMMA_N, WMMA_M);         // 每个块的线程数
    // wmma_kernel_v1<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);

    // Copy result back
    cudaMemcpy(h_c, d_c, c_size, cudaMemcpyDeviceToHost);

    // Verify result (should be 16.0 for all elements)
    bool valid = true;
    for(int i = 0; i < M*N; i++) {
        std::cout << h_c[i] << " \n";
        if(fabs(h_c[i] - 16.0f) > 1e-3) {
            valid = false;
            break;
        }
    }

    std::cout << "Verification: " << (valid ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
