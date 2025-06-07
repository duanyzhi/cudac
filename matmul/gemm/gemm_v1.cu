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

#define DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// template <int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
// __global__ void gemm_kernel(const half* __restrict__ a, const half* __restrict__ b,
//                           const half* __restrict__ bias, half* __restrict__ c, const int M,
//                           const int N, const int K) {
//   const int blockId = blockIdx.y * gridDim.x + blockIdx.x;

//   /***
//    *   ------> bx
//    *   |
//    *   |
//    *   V
//    *   by
//    * ***/
//   const int bx = blockIdx.x;
//   const int by = blockIdx.y;

//   int tid = threadIdx.x;
//   int wid = tid / 32;
//   int lane_id = tid % 32;

//   constexpr int LDN = BN;
//   constexpr int LDK = BK;
//   // D = A @ B + C
//   // BLOCK_A : [BM, K], BLOCK_B: [K, BN]
//   // iter_k = K / BK

//   constexpr int smem_num = BM * BK + BN * BK;
//   // constexpr int smem_num = ((BM + BN) * LDK) > (2 * BM * LDN) ? ((BM + BN) * LDK) : (2 * BM * LDN);
//   // Wrap A: [BM, BK], Wrap B: [BK, BN]
//   // mma: [M, N, K] = [16, 8, 16]
//   // BM * BN * 2 for storge smem for results
//   printf("debug 0");
//   __shared__ half smem[smem_num];
//   half* s_a = smem;
//   half* s_b = smem + BM * BK;
//   printf("debug 1");
//   constexpr int thread_num = 256;  // max thread for RTX4090
//   // Thread Block Tile things: prepare for load gmem to smem
//   constexpr int BK_load_step = 8;    // one float4 load 8 half data
//   constexpr int BK_step = BK / BK_load_step;  // 128 / 8 = 16
//   constexpr int BM_load_step = BM / (thread_num / BK_step); // 128 / （256 / 16）= 8

//   constexpr int load_a_smen_cycle = BM * BK / thread_num / 8;   // one thread load load_a_smem_cycle * 8 number.
  
//   int load_a_smem_m = (tid / BK_step) * BM_load_step;  // row for tid load a.
//   int load_a_smem_k = (tid % BK_step) * BK_load_step;  // col for tid load a.
//   printf("d2.");


//   // matrix B
//   constexpr int BN_load_step = BN / (thread_num / BK_step);
//   constexpr int load_b_smem_cycle = BN * BK / thread_num / 8;
//   int load_b_smem_n = (tid / BK_step) * BN_load_step;
//   int load_b_smem_k = load_a_smem_k;

//   // get smem addr for a in tid
//   half* load_a_smem_addrs[load_a_smen_cycle];  // one tid process load_a_smem_cycle data
//   for (int i = 0; i < load_a_smen_cycle; ++i) {
//     load_a_smem_addrs[i] = s_a + load_a_smem_m * BK + load_a_smem_k + i * BK;
//   }

//   half* load_b_smem_addrs[load_b_smem_cycle];
// #pragma unroll
//   for (int i = 0; i < load_b_smem_cycle; i++) {
//     load_b_smem_addrs[i] = s_b + load_b_smem_n * BK + load_b_smem_k + i * BK;
//   }

//   int load_block_a_gmem_addr = by * BM * K;  // start a matrix ptr in block (bx, by)
//   int load_block_b_gmem_addr = bx * BN;
//   int load_thread_a_gmem_addr = load_block_a_gmem_addr + load_a_smem_m * K + load_a_smem_k;
//   int load_thread_b_gmem_addr = load_block_b_gmem_addr + load_b_smem_n * K + load_b_smem_k;

//   printf("d1.");

//   // warp tile
//   constexpr int warp_size = 32;
//   constexpr int num_frag_a = BM / WMMA_M;
//   constexpr int num_frag_b = BN / WMMA_N;
//   constexpr int num_frag_c = num_frag_a * num_frag_b;

//   // wrap tile 
//   int load_a_smem_offset = WMMA_M * BK; // step for next num_frag_a
//   int load_b_smem_offset = WMMA_N * BK;

//   constexpr int warp_num = thread_num / warp_size;  // 256 / 32 = 8
//   constexpr int mma_offset = warp_num * WMMA_K;  // 8 * 16 = 128
//   constexpr int mma_cycle = BK / mma_offset;  // 128 / 128 = 1
//   assert(mma_cycle >= 0 && "mma_cycle must be non-negative");

//   nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a[num_frag_a];
//   nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_b[num_frag_b];
//   nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[num_frag_a][num_frag_b];

// #pragma unroll
//   for (int i = 0; i < num_frag_a; i++) {
//         for (int j = 0; j < num_frag_b; j++) {
//             nvcuda::wmma::fill_fragment(frag_c[i][j], 0.0f);
//         }
//   }
//   printf("prepare data done.");
//   // ---------------------------------------------------------------------
//   // run main loop with BK
//   for (int bk = 0; bk < K / BK; bk++) {
//     // printf("bk: %d wid: %d. ", bk, wid);
//     // load gmem to smem
//     for (int i = 0; i < load_a_smen_cycle; i++) {
//        *(float4*)(load_a_smem_addrs[i]) = *(float4*)(&a[load_thread_a_gmem_addr + i * K]);
//     }

// #pragma unroll
//     for (int i = 0; i < load_b_smem_cycle; i++) {
//         *(float4*)(load_b_smem_addrs[i]) = *(float4*)(&b[load_thread_b_gmem_addr + i * K]);
//     }

//     __syncthreads();  // wait for all thread load done. then the bk matrix a and matrix b all load to shared memory.
  
//     for (int i = 0; i < mma_cycle; i++) {
//       for (int m_index = 0; m_index < num_frag_a; m_index++) {
//         nvcuda::wmma::load_matrix_sync(frag_a[m_index],
//           &s_a[load_a_smem_offset * m_index + wid * WMMA_K + i * mma_offset], BK);

//         for (int n_index = 0; n_index < num_frag_b; n_index++) {   
//           printf("mma for m, n: %d %d. wid is % d. || ", m_index, n_index, wid);
//           nvcuda::wmma::load_matrix_sync(frag_b[n_index],
//             &s_b[load_b_smem_offset * n_index + wid * WMMA_K + i * mma_offset], BK);
 
//           nvcuda::wmma::mma_sync(frag_c[m_index][n_index], frag_a[m_index], frag_b[n_index],
//                                  frag_c[m_index][n_index]);
//         }
//       }
//     }
//     __syncthreads();
//     load_thread_a_gmem_addr += BK;
//     load_thread_b_gmem_addr += BK;
//   }

// //   // write back to smem
//   float* smem_float = (float*)(&smem[0]);
//   // 0, 1, 2, 3 -> 0; 4, 5, 6, 7 -> 1
//   unsigned int write_back_row_index = lane_id / 4; // 0, 1 -> 0; 2, 3 -> 1; 4, 5 -> 2
//   unsigned int write_back_col_index = lane_id % 16;  // 0 - 32 -> 0 - 16
//   if (wid == 0) {
// #pragma unroll
//         for (int m_index = 0; m_index < num_frag_a; m_index++) {
// #pragma unroll
//             for (int n_index = 0; n_index < num_frag_b; n_index++) {
// #pragma unroll
//                 for (int ele = 0; ele < 16 * 16; ele++) {
//                     smem_float[WMMA_M * LDN * m_index + n_index * WMMA_N + (write_back_row_index + ele) * LDN +
//                                write_back_col_index] = frag_c[m_index][n_index].x[ele];
//                 }
//             }
//         }
//   }
//   if (wid == 0) {
// #pragma unroll
//   for (int m_index = 0; m_index < num_frag_a; m_index++) {
// #pragma unroll
//       for (int n_index = 0; n_index < num_frag_b; n_index++) {
//         const int store_gmem_ptr = by * BM * N + bx * BN;
//         int smem_store_ptr = WMMA_M * m_index * N + n_index * WMMA_N;
//         // printf("gmemn_store_ptr: %d. || ", store_gmem_ptr + smem_store_ptr);
//         nvcuda::wmma::store_matrix_sync(c + store_gmem_ptr + smem_store_ptr, frag_c[m_index][n_index], N, nvcuda::wmma::mem_row_major);
//       }
//   }
// }

template <int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void gemm_kernel_v1(const half* __restrict__ a, const half* __restrict__ b,
                          const half* __restrict__ bias, float* __restrict__ c, const int M,
                          const int N, const int K) {
  const int blockId = blockIdx.y * gridDim.x + blockIdx.x;

  /***
   *   ------> bx
   *   |
   *   |
   *   V
   *   by
   * ***/
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // if (bx > 0) return;
  // if (by > 0) return;

  int tid = threadIdx.x;
  int wid = tid / 32;
  int lane_id = tid % 32;

  constexpr int LDN = BN;  // 128
  constexpr int LDK = BK;  // 128
  // D = A @ B + C
  // BLOCK_A : [BM, K], BLOCK_B: [K, BN]
  // iter_k = K / BK

  // constexpr int smem_num = BM * BK + BN * BK;
  constexpr int smem_num = ((BM + BN) * LDK) > (2 * BM * LDN) ? ((BM + BN) * LDK) : (2 * BM * LDN);

  constexpr int max_smem_block = 49152; // bytes
  assert(max_smem_block >= smem_num && "smem_num must be small 49152");
  // constexpr int smem_num = ((BM + BN) * LDK) > (2 * BM * LDN) ? ((BM + BN) * LDK) : (2 * BM * LDN);
  // Wrap A: [BM, BK], Wrap B: [BK, BN]
  // mma: [M, N, K] = [16, 8, 16]
  // BM * BN * 2 for storge smem for results
  // printf("smem_num: %d .", smem_num);
  __shared__ half smem[smem_num];
  half* s_a = smem;
  half* s_b = smem + BM * BK;

  constexpr int thread_num = 32;  // max thread for RTX4090
  // Thread Block Tile things: prepare for load gmem to smem
  constexpr int BK_load_step = 8;    // one float4 load 8 half data
  constexpr int BK_step = BK / BK_load_step;  // 64 / 8 = 8
  constexpr int BM_load_step = BM / (thread_num / BK_step); // 64 / （256 / 16）= 4

  constexpr int load_a_smen_cycle = BM * BK / thread_num / 8;   // 2, one thread load load_a_smem_cycle * 8 number.
  assert(load_a_smen_cycle >= 0 && "load_a_smen_cycle must be non-negative");

  int load_a_smem_m = (tid / BK_step) * BM_load_step;  // row for tid load a.
  int load_a_smem_k = (tid % BK_step) * BK_load_step;  // col for tid load a.
  // printf("load_a_smem_m %d: load_a_smem_k %d. ", load_a_smem_m, load_a_smem_k);


  // matrix B
  constexpr int BN_load_step = BN / (thread_num / BK_step);  // 64 / (256 / 8) = 2
  constexpr int load_b_smem_cycle = BN * BK / thread_num / 8; // 2
  int load_b_smem_n = (tid / BK_step) * BN_load_step;
  int load_b_smem_k = load_a_smem_k;

  // printf("BK_load_step: %d BK_step:%d BM_load_step:%d load_a_smen_cycle:%d\n", BK_load_step, BK_step, BM_load_step, load_a_smen_cycle);
  // BK_load_step: 8 BK_step:16 BM_load_step:4 load_a_smen_cycle:4

  // get smem addr for a in tid
  // printf("load_a_smen_cycle: %d. ", load_a_smen_cycle);
  half* load_a_smem_addrs[load_a_smen_cycle];  // 8 one tid process load_a_smem_cycle data
  for (int i = 0; i < load_a_smen_cycle; ++i) {
    // printf("i %d && offset %d ", i, load_a_smem_m * BK + load_a_smem_k + i * BK);
    // load_a_smem_addrs[i] = s_a;
    load_a_smem_addrs[i] = s_a + load_a_smem_m * BK + load_a_smem_k + i * BK;
  }
  // printf("load_a_smem_addrs: %p = ", load_a_smem_addrs[0]);
  // __syncthreads(); 
  // for (int i = 0; i < load_a_smen_cycle; ++i) {
  //   printf("load_a_smem_addrs[i] : %p = ", (void*)(load_a_smem_addrs[0]));
  // }
  // printf("sizeof half* %d\n", sizeof(half*));

  half* load_b_smem_addrs[load_b_smem_cycle];
#pragma unroll
  for (int i = 0; i < load_b_smem_cycle; i++) {
    load_b_smem_addrs[i] = s_b + load_b_smem_n * BK + load_b_smem_k + i * BK;
  }

  int load_block_a_gmem_addr = by * BM * K;  // start a matrix ptr in block (bx, by)
  int load_block_b_gmem_addr = bx * BN;
  int load_thread_a_gmem_addr = load_block_a_gmem_addr + load_a_smem_m * K + load_a_smem_k;
  int load_thread_b_gmem_addr = load_block_b_gmem_addr + load_b_smem_n * K + load_b_smem_k;

  // warp tile
  constexpr int warp_size = 32;
  constexpr int num_frag_a = BM / WMMA_M;  // 4
  constexpr int num_frag_b = BN / WMMA_N;  // 4
  constexpr int num_frag_c = num_frag_a * num_frag_b;  // 16

  // wrap tile 
  int load_a_smem_offset = WMMA_M * BK; // 1024, step for next num_frag_a
  int load_b_smem_offset = WMMA_N * BK;

  constexpr int warp_num = thread_num / warp_size;  // 256 / 32 = 8
  constexpr int mma_offset = warp_num * WMMA_K;  // 8 * 16 = 128
  constexpr int mma_cycle = BK / mma_offset >= 1 ? BK / mma_offset : 1;  // 128 / 128 = 1
  assert(mma_cycle >= 0 && "mma_cycle must be non-negative");
  // printf("mma_offset %d. ", mma_offset);

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a[num_frag_a];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_b[num_frag_b];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[num_frag_a][num_frag_b];

  // const int max_a_size = M * K;
#pragma unroll
  for (int i = 0; i < num_frag_a; i++) {
        for (int j = 0; j < num_frag_b; j++) {
            nvcuda::wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
  }

  for (int bk = 0; bk < K / BK; bk++) {
    // printf("bk: %d wid: %d. ", bk, wid);
    // load gmem to smem
    for (int i = 0; i < load_a_smen_cycle; i++) {
      // if (load_thread_a_gmem_addr + i * K > max_a_size) {
      //   printf("Error: load_thread_a_gmem_addr + i * K: %d || ", load_thread_a_gmem_addr + i * K);
      // }
      // printf("i: %d, bx %d, by %d, load_block_a_gmem_addr %d, load_a_smem_m %d, load_a_smem_k: %d load_thread_a_gmem_addr + i * K: %d, ptr: %d\n", i, bx, by, load_block_a_gmem_addr, load_a_smem_m, load_a_smem_k, load_thread_a_gmem_addr + i * K, &a[load_thread_a_gmem_addr + i * K]);
      // printf("load_a_smem_addrs[i]: %d ", load_a_smem_addrs[i]);
      // printf("gmem ptr: %p smem %p .\n", a[load_thread_a_gmem_addr + i * K], load_a_smem_addrs[i]);
      *(float4*)(load_a_smem_addrs[i]) = *(float4*)(&a[load_thread_a_gmem_addr + i * K]);
    }

#pragma unroll
    for (int i = 0; i < load_b_smem_cycle; i++) {
        *(float4*)(load_b_smem_addrs[i]) = *(float4*)(&b[load_thread_b_gmem_addr + i * K]);
    }

    __syncthreads();  // wait for all thread load done. then the bk matrix a and matrix b all load to shared memory.
  
    for (int i = 0; i < mma_cycle; i++) {
      for (int m_index = 0; m_index < num_frag_a; m_index++) {
        nvcuda::wmma::load_matrix_sync(frag_a[m_index],
          &s_a[load_a_smem_offset * m_index + wid * WMMA_K + i * mma_offset], BK);

        for (int n_index = 0; n_index < num_frag_b; n_index++) {   
          // printf("mma for m, n: %d %d. wid is % d. || ", m_index, n_index, wid);
          nvcuda::wmma::load_matrix_sync(frag_b[n_index],
            &s_b[load_b_smem_offset * n_index + wid * WMMA_K + i * mma_offset], BK);
 
          // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
          nvcuda::wmma::mma_sync(frag_c[m_index][n_index], frag_a[m_index], frag_b[n_index],
                                 frag_c[m_index][n_index]);
        }
      }
    }
    __syncthreads();
    load_thread_a_gmem_addr += BK;
    load_thread_b_gmem_addr += BK;
  }
  
  float* smem_float = (float*)(&smem[0]);
  // frag_c[0][0] = frag_a[0][0]*frag_b[0][0] + frag_a[0][1] * frag_b[1][0] + ... + frag_a[0][num_frag_a]* frag_b[num_frag_a][0]
  // for (int i = 0; i < warp_num; i++) {
  //       if (wid == i) {
// #pragma unroll
//             for (int m_index = 0; m_index < num_frag_a; m_index++) {
// #pragma unroll
//                 for (int n_index = 0; n_index < num_frag_b; n_index++) {
//                     // for (int ele = 0; ele < wmma_num; ele++) {
//                     //     smem_float[WMMA_M * LDN * m_index + n_index * WMMA_N + (write_back_row_index + ele) * LDN +
//                     //                write_back_col_index] += frag_c[m_index][n_index].x[ele];
//                     // }
//                 }
//             }
//         }

//         __syncthreads();
//     } 
  if (wid == 0) {
    for (int m_index = 0; m_index < num_frag_a; m_index++) {
      for (int n_index = 0; n_index < num_frag_b; n_index++) {
        int smem_store_ptr = m_index * WMMA_M * BN + n_index * WMMA_N;
        nvcuda::wmma::store_matrix_sync(smem_float + smem_store_ptr, frag_c[m_index][n_index], BN, nvcuda::wmma::mem_row_major);
        // __syncthreads();
        // printf("m_index: %d n_index %d", m_index, n_index);
        // for (int i = 0; i < BM; ++i) {
        //   for (int j = 0; j < BN; ++j) {
        //     printf("smem_store_ptr %d value: %f \n", smem_store_ptr, smem_float[smem_store_ptr + i * BN + j]);
        //   }
        // }
      }
    }
  }
  __syncthreads();
  if (tid == 0 && bx == 1  && by == 0) {
    printf("bx %d, by: %d\n", bx, by);
    for (int i = 0; i < BM * BN; ++i) {
      // printf(" %d %d %f, ",  i/BN, i%BN, smem_float[i]);

      printf("%f, ",smem_float[i]);
      if ((1+i) % BN == 0) {
        printf("\n");
      }
    }
  }
  // printf("\n");
  // to global
  int store_step = BM * BN / thread_num; // 16
  int row_thread_num = BN / store_step; // one row 4 thread with 16 * 64 num
  int store_row = tid / row_thread_num;
  int store_col = tid % row_thread_num;
  printf("tid %d, store_row %d, store_col: %d bx: %d, by %d\n", tid, store_row, store_col, bx, by);
  int gmem_store_ptr = (by * BM + store_row) * N + bx * BN + store_col * store_step;

  int smem2gmem_ptr = store_row * BN + store_col * store_step;
  for (int i = 0; i < store_step; ++i) {
    // printf("gptr: %d sptr: %d i: %d\n.", gmem_store_ptr, smem2gmem_ptr, i);
    c[gmem_store_ptr + i] = smem_float[smem2gmem_ptr + i];
  }
  
}


at::Tensor gemm_v1_forward(at::Tensor a, at::Tensor b) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(0);

    at::Tensor c = at::zeros(
        {M, N}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
    );
    // at::Tensor c = at::zeros({M, N}, a.options());

    constexpr int BM = 16;
    constexpr int BN = 16;
    constexpr int BK = 32;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    dim3 GridDim(DIV_UP(N, BN), DIV_UP(M, BM));  // 8 * 8
    dim3 BlockDim(64, 1);
    printf("grid: %d, %d: %d.", GridDim.x, GridDim.y, BlockDim.x);
    float* c_ptr = reinterpret_cast<float*>(c.data_ptr());

    gemm_kernel_v1<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>
        <<<GridDim, BlockDim, 0, stream>>>(
               reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
               reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
               nullptr,
               c_ptr,
               M, N, K);

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_v1", &gemm_v1_forward, "Gemm forward (CUDA)",
          py::arg("a"), py::arg("b"));
}
