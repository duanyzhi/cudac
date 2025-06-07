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

// // WMMA matrix dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
const int warpSize = 32;

// const int BLOCK_WRAP_M = 8;  // one block process 8 wrap_m
// const int BLOCK_WRAP_N = 4;  // one block process 4 wrap_n

// for (int cluster_m = 0; cluster_m < GemmM; cluster_m += ClusterTileM) {
//   for (int cluster_n = 0; cluster_n < GemmN; cluster_n += ClusterTileN) {

//     // cutlass::gemm::collective::CollectiveMma: mainloop that iterates over all k-tiles
//     // No loop unrolling is performed at this stage
//     for (int k_tile = 0; k_tile < size<2>(gmem_tensor_A); k_tile++) {

//       // loops inside cute::gemm(tiled_mma, a, b, c); Dispatch 5: (V,M,K) x (V,N,K) => (V,M,N)
//       // TiledMma uses the hardware instruction provided through its Mma_Atom
//       // TiledMma's atom layout, value layout, and permutations define the iteration order
//       for (int tiled_mma_k = 0; tiled_mma_k < size<2>(A); tiled_mma_k++) {
//         for (int tiled_mma_m = 0; tiled_mma_m < size<1>(A); tiled_mma_m++) {
//           for (int tiled_mma_n = 0; tiled_mma_n < size<1>(B); tiled_mma_n++) {

//             // TiledMma's vector mode dispatches to the underlying instruction.
//             mma.call(d, a, b, c);
//           } // tiled_mma_n
//         } // tiled_mma_m
//       } // tiled_mma_k
//     } // k_tile mainloop
//   } // cluster_m
// } // cluster_n

// const int BM = 128;  // m tile
// const int BN = 128;  // n tile
// const int BK = 32;
// const int wrap_size = 32;
// const int BlockThreadsNum = 1024; // for RTX4090

// __global__ void wmma_kernel_v2(half *a, half *b, float *c, int M, int N, int K) {
//    // Leading dimensions for row major. Packed with no transpositions.
//    int lda = K;
//    int ldb = N;
//    int ldc = N;

//    float alpha = 1.0;
//    float beta = 1.0;

//    // Tile using a 2D grid
//    int warpM = blockIdx.x * BLOCK_WRAP_M + threadIdx.x / wrap_size;
//    int warpN = blockIdx.y * BLOCK_WRAP_N + threadIdx.y;
 
//    // Declare the fragments
//    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
//    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
//    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
//    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

//    wmma::fill_fragment(acc_frag, 0.0f);

//    // Loop over k
//    for (int i = 0; i < K; i += WMMA_K) {
//       int aRow = warpM * WMMA_M;
//       int aCol = i;

//       int bRow = i;
//       int bCol = warpN * WMMA_N;

//       // Bounds checking
//       if (aRow < M && aCol < K && bRow < K && bCol < N) {
//          // Load the inputs
//          wmma::load_matrix_sync(a_frag, a + aRow * lda + aCol, lda);
//          wmma::load_matrix_sync(b_frag, b + bRow * ldb + bCol, ldb);

//          // Perform the matrix multiplication
//          wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

//       }
//    }

//    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
//    int cRow = warpM * WMMA_M;
//    int cCol = warpN * WMMA_N;

//    if (cRow < M && cCol < N) {
//       wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

// #pragma unroll
//       for(int i=0; i < c_frag.num_elements; i++) {
//          c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
//       }

//       // Store the output
//       wmma::store_matrix_sync(c + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
//    }
// }

__global__ void lanuch_gemm_kernel(half *a, half *b, half *c, 
                             int M, int N, int K) 
{
    // 定义wmma在各个方向上的维度（实际上就是单次计算的小矩阵（tile）的size）
    // Leading dimensions. Packed with no transpositions.
    // A[M, K] B[K, N] C[M, N]
    int lda = K;        // 矩阵 A 的行数
    int ldb = N;        // 矩阵 B 的行数
    int ldc = N;        // C 的行数(即 A 的行数)

    // 【注意！】A,B的 size不局限于16*16，因为 16*16 只是单次 wmma 可运算的矩阵大小；
    // 而通过多个 warp（ SM 中的基本执行代码单元，包含 32 个 thread ）和循环实现更大 size 的矩阵运算。
    
    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    printf("wrapM, %d, wrapN, %d.\n", warpM, warpN);

    // 声明 fragment 变量
    // Declare the fragments
    // fragement变量用于对应一次wmma操作中的小矩阵块，声明时所填的参数表明了该fragment在计算中的用途和角色
    
    // 声明规则如下。
    // wmma::fragment<wmma 矩阵类型, （WMMA_M, WMMA_N, WMMA_K）三个参数用于表述数据的维度形状 , 矩阵数据类型（精度）, wmma 存储类型（可选）> 变量名;

    // 用于乘法的 matrix_a 和 matrix_b
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    // matrix_a 或 matrix_b 还必须说明数据在内存中的存储形式（如：wmma::col_major）是按行还是按列以方便 GPU 在读取数据时循环计算一系列 tile 的乘积。

    // 附加项 和 输出矩阵 则定义为accumulator
   //  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // Loop over the K-dimension
    // 循环计算一系列 tile 的乘积
    for (int i = 0; i < K; i += WMMA_K) {

        // 利用 warp 的 ID 来决定这个 warp 所计算的 tile
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        printf("aRow: %d, aCol: %d; || ", aRow, aCol);
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // 计算之前需要使用在 wmma 命名空间下的 load_matrix_sync 函数加载数据，操作如下。
            // wmma::load_matrix_sync(fragment 变量, 起始地址, leading dimension);     其中 leading dimension 表示数据在 按行存储时矩阵的总行数 或是 按列存储时的总列数 。

            // 这样知道了输入矩阵的存储方式和 leading dimension 在加载数据时 GPU 可以知道如何在不连续的数据中取出我们需要的 tile 。

            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow * lda + aCol, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
 
            // 每次循环的结果都累加在 acc_fag 上。
            // Perform the matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            // 最后 acc_frag 中的结果也就是结果矩阵 c 中该 warp 对应的 tile 。
        }
    }
   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
//       wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

// #pragma unroll
//       for(int i=0; i < c_frag.num_elements; i++) {
//          c_frag.x[i] =  __float2half(acc_frag.x[i]) + c_frag.x[i];
//       }

      // Store the output
      wmma::store_matrix_sync(c + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
   }
}

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

template <int BM, int BN, int BK, int LDK, int LDN>
__global__ void flat_gemm_m8n32k256x8_bz1_kernel(const half *__restrict__ a, const half *__restrict__ b,
                                                 const half *__restrict__ bias, half *__restrict__ c, const int M,
                                                 const int N, const int K) {
    int bx = blockIdx.x;
    int bz = blockIdx.z;
    int k_start = K / gridDim.z * bz;

    int tid = threadIdx.x;
    int wid = tid >> 5;  // WARP id

    // bx for N, if bx is out of range, return
    if (bx >= N / BN) return;

    __shared__ half smem[(BM + BN) * LDK];
    half *s_a = smem;
    half *s_b = smem + BM * LDK;

    //                             M   N   K
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));

    int load_a_smem_m = (tid >> 5);       // 0 ~ 7    | 0 1  2 ...  7
    int load_a_smem_k = (tid & 31) << 3;  // 0 ~ 248  | 0 8 16 ... 248(32个数)
    int load_b_smem_n = (tid >> 5) << 2;  // 0 ~ 28   | 0 4  8 ... 28
    int load_b_smem_k = load_a_smem_k;

    // ptx address space conversion
    size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
    size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half);
    int load_b_smem_addrs[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
        load_b_smem_addrs[i] =
            s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);

    int load_a_gmem_m = load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = k_start + load_a_smem_k;
    int load_b_gmem_k = k_start + load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    for (int bk = 0; bk < (K / gridDim.z) / BK; bk++) {
        if (load_a_gmem_m < M) {
            asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :
                : "r"(load_a_smem_addr_0), "l"(&a[load_a_gmem_addr]));
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :
                : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(frag_a, &s_a[wid * 16 + 128 * i], LDK);
            wmma::load_matrix_sync(frag_b, &s_b[wid * 16 + 128 * i], LDK);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    wmma::store_matrix_sync(&smem[wid * 8 * LDN], frag_c, LDN, wmma::mem_row_major);

    __syncthreads();

    int shmem_c_m = tid >> 5;  // 0, 1, 2, 3, 4, 5, 6, 7
    int shmem_c_n = tid & 31;  // 0, 1, 2, 3, 4, ..., 31
    int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
    int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

#pragma unroll
    for (int s = 1; s < 8; s++) {
        smem[shmem_c_addr] = __hadd(smem[shmem_c_addr], smem[shmem_c_addr + s * 8 * LDN]);
    }

    if (shmem_c_m < M) {
        if (bias != nullptr)
            c[gmem_c_addr] = __hadd(smem[shmem_c_addr], bias[gmem_c_addr % N]);
        else
            c[gmem_c_addr] = smem[shmem_c_addr];
    }
}

// __global__ void lanuch_gemm_kernel(half *a, half *b, half *c, int M, int N, int K) {
//    printf("bx: %d, by: %d. || ", blockIdx.x, blockIdx.y);
     
//    int tx = threadIdx.x;  // 0-1024
//    // blockIdx.x -> M
//    // blockIdx.y -> N

//    int bx = (blockIdx.x * BM) * K;
//    int by = (blockIdx.y * BN); 
 
//   __shared__ float As[BK * BM];
//   __shared__ float Bs[BK * BN];
//   __shared__ float Cs[BM * BN];

// //    // Declare the fragments
//    // mma: 16, 8, 16
//    // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
//    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
//    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
//    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

//    wmma::fill_fragment(acc_frag, 0.0f);

//   const int ldg_a_num = BK * BM / BlockThreadsNum / 4;

//   // load submatrix to smem
//   for (int ik = 0; ik < K; ik+=BK) {
//     int sA_idx = tx + ik;
//     //  each thread loads one element of each matrix
//     As[ty][tx] = A[row + K * ty + tx];
//     Bs[ty][tx] = B[col + N * ty + tx];
//   }
// }

// #define OFFSET(row, col, ld) ((row) * (ld) + (col))

// template <int BM, int BN, int BK, int LDK, int LDN>
// __global__ void flat_gemm_m16n32k256x8_db_kernel(const half *__restrict__ a, const half *__restrict__ b,
//                                                  const half *__restrict__ bias, half *__restrict__ c, const int M,
//                                                  const int N, const int K) {
// #if __CUDA_ARCH__ < 800
//     return;
// #endif
//     int bx = blockIdx.x;
//     int bz = blockIdx.z;
//     int k_start = K / gridDim.z * bz;

//     int tid = threadIdx.x;
//     int wid = tid >> 5;  // WARP id

//     // bx for N, if bx is out of range, return
//     if (bx >= N / BN) return;

//     extern __shared__ half smem[];
//     half *s_a = smem;
//     half *s_b = smem + 2 * BM * LDK;
//     int s_a_offset = BM * LDK;
//     int s_b_offset = BN * LDK;

//     //                             M   N   K
//     wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
//     wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
//     wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

//     wmma::fill_fragment(frag_c, __float2half(0.0f));

//     int load_a_smem_m = (tid >> 5) << 1;  // 0 ~ 14   | 0 2  4 ... 14
//     int load_a_smem_k = (tid & 31) << 3;  // 0 ~ 248  | 0 8 16 ... 248
//     int load_b_smem_n = (tid >> 5) << 2;  // 0 ~ 28   | 0 4  8 ... 28
//     int load_b_smem_k = load_a_smem_k;

//     // ptx address space conversion
//     size_t s_a_base_addr = __cvta_generic_to_shared(s_a);
//     size_t s_b_base_addr = __cvta_generic_to_shared(s_b);

//     int load_a_smem_addrs[2];
// #pragma unroll
//     for (int i = 0; i < 2; i++) {
//         load_a_smem_addrs[i] =
//             s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
//     }
//     int load_b_smem_addrs[4];
// #pragma unroll
//     for (int i = 0; i < 4; i++) {
//         load_b_smem_addrs[i] =
//             s_b_base_addr + OFFSET(load_b_smem_n, load_b_smem_k, LDK) * sizeof(half) + i * (LDK) * sizeof(half);
//     }

//     int load_a_gmem_m = load_a_smem_m;
//     int load_b_gmem_n = bx * BN + load_b_smem_n;
//     int load_a_gmem_k = k_start + load_a_smem_k;
//     int load_b_gmem_k = k_start + load_b_smem_k;

//     int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
//     int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

//     {
// #pragma unroll
//         for (int i = 0; i < 2; i++) {
//             if ((load_a_gmem_m + i) < M) {
//                 asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
//                     :
//                     : "r"(load_a_smem_addrs[i]), "l"(&a[load_a_gmem_addr + i * K]));
//             }
//         }

// #pragma unroll
//         for (int i = 0; i < 4; i++) {
//             asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
//                 :
//                 : "r"(load_b_smem_addrs[i]), "l"(&b[load_b_gmem_addr + i * K]));
//         }

//         asm("cp.async.commit_group;\n" ::);
//         asm("cp.async.wait_group 0;\n" ::);
//         __syncthreads();
//     }

// #pragma unroll
//     for (int bk = 1; bk < (K / gridDim.z) / BK; bk++) {
//         int smem_sel = (bk & 1) ^ 1;
//         int smem_sel_next = ((bk - 1) & 1) ^ 1;

//         load_a_gmem_addr += BK;
//         load_b_gmem_addr += BK;

// #pragma unroll
//         for (int i = 0; i < 2; i++) {
//             if ((load_a_gmem_m + i) < M) {
//                 asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
//                     :
//                     : "r"(load_a_smem_addrs[i] + smem_sel_next * s_a_offset * (int)sizeof(half)),
//                       "l"(&a[load_a_gmem_addr + i * K]));
//             }
//         }

// #pragma unroll
//         for (int i = 0; i < 4; i++) {
//             asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
//                 :
//                 : "r"(load_b_smem_addrs[i] + smem_sel_next * s_b_offset * (int)sizeof(half)),
//                   "l"(&b[load_b_gmem_addr + i * K]));
//         }

//         for (int i = 0; i < 4; i++) {
//             wmma::load_matrix_sync(frag_a, &s_a[smem_sel * s_a_offset + (wid & 3) * 16 + 64 * i], LDK);
//             wmma::load_matrix_sync(frag_b,
//                                    &s_b[smem_sel * s_b_offset + (wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i], LDK);
//             wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
//         }

//         asm("cp.async.commit_group;\n" ::);
//         asm("cp.async.wait_group 0;\n" ::);
//         __syncthreads();
//     }

//     int smem_sel = (((K / gridDim.z) / BK) & 1) ^ 1;

//     for (int i = 0; i < 4; i++) {
//         wmma::load_matrix_sync(frag_a, &s_a[smem_sel * s_a_offset + (wid & 3) * 16 + 64 * i], LDK);
//         wmma::load_matrix_sync(frag_b, &s_b[smem_sel * s_b_offset + (wid >> 2) * 16 * LDK + (wid & 3) * 16 + 64 * i],
//                                LDK);
//         wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
//     }

//     __syncthreads();

//     wmma::store_matrix_sync(&smem[(wid & 3) * 16 * LDN + (wid >> 2) * 16], frag_c, LDN, wmma::mem_row_major);

//     __syncthreads();

//     int shmem_c_m = tid >> 4;         // 0, 1, 2, 3, 4, ..., 15
//     int shmem_c_n = (tid & 15) << 1;  // 0, 2, 4, 6, 8, ..., 30
//     int shmem_c_addr = OFFSET(shmem_c_m, shmem_c_n, LDN);
//     int gmem_c_addr = OFFSET(shmem_c_m, bx * BN + shmem_c_n, N);

// #pragma unroll
//     for (int s = 1; s < 4; s++) {
//         *(half2 *)(&smem[shmem_c_addr]) =
//             __hadd2(*(half2 *)(&smem[shmem_c_addr]), *(half2 *)(&smem[shmem_c_addr + s * 16 * LDN]));
//     }

//     if (shmem_c_m < M) {
//         if (bias != nullptr)
//             *(half2 *)(&c[gmem_c_addr]) = __hadd2(*(half2 *)(&smem[shmem_c_addr]), *(half2 *)(&bias[gmem_c_addr % N]));
//         else
//             *(half2 *)(&c[gmem_c_addr]) = *(half2 *)(&smem[shmem_c_addr]);
//     }
// }
const int BLOCK_WRAP_M = 8;  // one block process 8 wrap_m
const int BLOCK_WRAP_N = 4;  // one block process 4 wrap_n

at::Tensor gemm_forward(at::Tensor a, at::Tensor b) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(0);

    at::Tensor c = at::zeros(
        {M, N}, 
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)
    );


    int wrap_M = max((M + WMMA_M - 1) / WMMA_M / BLOCK_WRAP_M, 1);
    int wrap_N = max((N + WMMA_N - 1) / WMMA_N / BLOCK_WRAP_N, 1);


    dim3 gridDim(wrap_M, wrap_N);
    dim3 blockDim(BLOCK_WRAP_M * 32, BLOCK_WRAP_N);  // One block

   //  dim3 gridDim(M / WMMA_M, N / WMMA_N);
   //  dim3 blockDim(1024);

    lanuch_gemm_kernel<<<gridDim, blockDim>>>(
        reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
        M, N, K
    );

   //  flat_gemm_m8n32k256x8_bz1_kernel<16, 32, 256, 264, 40>
   //          <<<dim3(N / 32, 1, 1), dim3(256), 0, stream>>>(
   //             reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
   //             reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
   //             nullptr,
   //             reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
   //             M, N, K);

    cudaStreamSynchronize(stream);
    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_forward, "Gemm forward (CUDA)",
          py::arg("a"), py::arg("b"));
}
