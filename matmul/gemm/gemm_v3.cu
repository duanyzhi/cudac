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

template <int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int THREADNUM>
__global__ void gemm_kernel_v3(const half* __restrict__ a, const half* __restrict__ b,
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
  static_assert(max_smem_block >= smem_num && "smem_num must be small 49152");
  // constexpr int smem_num = ((BM + BN) * LDK) > (2 * BM * LDN) ? ((BM + BN) * LDK) : (2 * BM * LDN);
  // Wrap A: [BM, BK], Wrap B: [BK, BN]
  // mma: [M, N, K] = [16, 8, 16]
  // BM * BN * 2 for storge smem for results
  // printf("smem_num: %d .", smem_num);
  __shared__ half smem[smem_num];
  half* s_a = smem;
  half* s_b = smem + BM * BK;

  // constexpr int thread_num = 256;  // max thread for RTX4090
  // Thread Block Tile things: prepare for load gmem to smem
  constexpr int BK_load_step = 8;    // one float4 load 8 half data
  constexpr int BK_step = BK / BK_load_step;  // 32 / 8 = 4
  constexpr int BM_load_step = BM / (THREADNUM / BK_step); // 32 / (64 / 4) = 2

  constexpr int load_a_smen_cycle = BM * BK / THREADNUM / 8;   // 32 * 32 / 64 / 8 = 2, one thread load load_a_smem_cycle * 8 number.
  static_assert(load_a_smen_cycle >= 0 && "load_a_smen_cycle must be non-negative");

  int load_a_smem_m = (tid / BK_step) * BM_load_step;  // row for tid load a.
  int load_a_smem_k = (tid % BK_step) * BK_load_step;  // col for tid load a.
  // printf("tid %d load_a_smem_m %d: load_a_smem_k %d. \n", tid, load_a_smem_m, load_a_smem_k);

  // matrix B
  constexpr int BN_load_step = BN / (THREADNUM / BK_step);  // 32 / (64 / 4) = 2
  constexpr int load_b_smem_cycle = BN * BK / THREADNUM / 8; // 2
  int load_b_smem_n = (tid / BK_step) * BN_load_step;
  int load_b_smem_k = load_a_smem_k;
  // printf("tid: %d. load_a_smem_m: %d, load_a_smem_k %d . load_b_smem_k %d: load_b_smem_n %d. \n", tid, load_a_smem_m, load_a_smem_k, load_b_smem_k, load_b_smem_n);

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
  int load_block_b_gmem_addr = bx * BN * K;
  int load_thread_a_gmem_addr = load_block_a_gmem_addr + load_a_smem_m * K + load_a_smem_k;
  int load_thread_b_gmem_addr = load_block_b_gmem_addr + load_b_smem_n * K + load_b_smem_k;

  // warp tile
  constexpr int warp_size = 32;
  constexpr int num_frag_a = BM / WMMA_M;  // 2
  constexpr int num_frag_b = BN / WMMA_N;  // 2
  constexpr int num_frag_c = num_frag_a * num_frag_b;  // 4

  // wrap tile 
  int load_a_smem_offset = WMMA_M * BK; // 512, step for next num_frag_a
  int load_b_smem_offset = WMMA_N * BK;

  constexpr int warp_num = THREADNUM / warp_size;  // 64 / 32 = 2
  constexpr int mma_offset = warp_num * WMMA_K;  // 2 * 16 = 32
  constexpr int mma_cycle = BK / mma_offset >= 1 ? BK / mma_offset : 1;
  static_assert(mma_cycle >= 0 && "mma_cycle must be non-negative");
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

  // main for loop for K
  for (int bk = 0; bk < K / BK; bk++) {
    // load gmem to smem
    for (int i = 0; i < load_a_smen_cycle; i++) {
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
          nvcuda::wmma::load_matrix_sync(frag_b[n_index],
            &s_b[load_b_smem_offset * n_index + wid * WMMA_K + i * mma_offset], BK);
 
          // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
          nvcuda::wmma::mma_sync(frag_c[m_index][n_index], frag_a[m_index], frag_b[n_index],
                                 frag_c[m_index][n_index]);
        }
      }
    }
    __syncthreads();
    // printf
    // if (tid == 0 && bx == 0 && by == 0 && bk == 0) {
    //   for (int ii = 0; ii < BM; ++ii) {
    //     for (int jj = 0; jj < BK; ++jj) {
    //       float vf = __half2float(s_a[ii * BK + jj]);
    //       printf("%f, ", vf);
    //     }
    //     printf("\n");
    //   }
    // }
    // if (tid == 0 && bx == 1 && by == 1 && bk == 1) {
    //   for (int jj = 0; jj < BK; ++jj) {
    //     for (int ii = 0; ii < BN; ++ii) {
    //       float vf = __half2float(s_b[ii * BK + jj]);
    //       printf("%f, ", vf);
    //     }
    //     printf("\n");
    //   }
    // }

    load_thread_a_gmem_addr += BK;
    load_thread_b_gmem_addr += BK;
  }
  __syncthreads();
  
  float* smem_float = (float*)(&smem[0]);
  // store first warp
  if (wid == 0) {
#pragma unroll
        for (int m_index = 0; m_index < num_frag_a; m_index++) {
#pragma unroll
            for (int n_index = 0; n_index < num_frag_b; n_index++) {
#pragma unroll
                for (int i = 0; i < frag_c[m_index][n_index].num_elements; i++) {
                    int row{0}, col{0};
                    int groupID = lane_id >> 2;
                    int threadID_in_group = lane_id % 4;
                    if (i < 4) {
                      row = i < 2 ? groupID : groupID + 8;
                      col = (threadID_in_group * 2) + (i & 0x1);
                    } else {
                      int j = i - 4;
                      row = j < 2 ? groupID : groupID + 8;
                      col = (threadID_in_group * 2) + (j & 0x1) + 8; 
                    }
                    // printf("(%2d,%2d) ", row, col);  // in one frag

                    // float vf = frag_c[m_index][n_index].x[i];

                    int row_smem = m_index * WMMA_M + row;
                    int col_smem = n_index * WMMA_N + col;

                    smem_float[row_smem * BN + col_smem] = frag_c[m_index][n_index].x[i];
                }
            }
        }
  }

  __syncthreads();

  for (int i = 1; i < warp_num; i++) {
        if (wid == i) {
#pragma unroll
            for (int m_index = 0; m_index < num_frag_a; m_index++) {
#pragma unroll
                for (int n_index = 0; n_index < num_frag_b; n_index++) {
#pragma unroll
                  for (int i = 0; i < frag_c[m_index][n_index].num_elements; i++) {
                      int row{0}, col{0};
                      int groupID = lane_id >> 2;
                      int threadID_in_group = lane_id % 4;
                      if (i < 4) {
                        row = i < 2 ? groupID : groupID + 8;
                        col = (threadID_in_group * 2) + (i & 0x1);
                      } else {
                        int j = i - 4;
                        row = j < 2 ? groupID : groupID + 8;
                        col = (threadID_in_group * 2) + (j & 0x1) + 8; 
                      }
                      // printf("(%2d,%2d) ", row, col);  // in one frag

                      // float vf = frag_c[m_index][n_index].x[i];

                      int row_smem = m_index * WMMA_M + row;
                      int col_smem = n_index * WMMA_N + col;

                      smem_float[row_smem * BN + col_smem] += frag_c[m_index][n_index].x[i];  // sum all fragment for all wrap
                  }
                }
            }
        }

        __syncthreads();
  }


  // for (int id = 0; id < 64; ++id) {
  //   if (id == tid) {
  //     // printf("tid: %d. \n", tid);
  //   //   printf("\n");
  //     for (int m_index = 0; m_index < num_frag_a; m_index++) {
  //       for (int n_index = 0; n_index < num_frag_b; n_index++) {
  //           // printf("m_index: %d. n_index: %d. ", m_index, n_index);
  //           printf("\n");
  //           int fragem_c_ptr = m_index * WMMA_M * N + n_index * WMMA_N;
  //           for (int i = 0; i < frag_c[m_index][n_index].num_elements; i++) {
  //               int row{0}, col{0};
  //               int groupID = lane_id >> 2;
  //               int threadID_in_group = lane_id % 4;
  //               if (i < 4) {
  //                 row = i < 2 ? groupID : groupID + 8;
  //                 col = (threadID_in_group * 2) + (i & 0x1);
  //               } else {
  //                 int j = i - 4;
  //                 row = j < 2 ? groupID : groupID + 8;
  //                 col = (threadID_in_group * 2) + (j & 0x1) + 8; 
  //               }
  //               printf("(%2d,%2d) ", row, col);  // in one frag

  //               float vf = frag_c[m_index][n_index].x[i];

  //               int row_smem = m_index * WMMA_M + row;
  //               int clo_smem = n_index * WMMA_N + col;
  //               // printf("(%2d,%2d) ", row_smem, clo_smem);

  //               printf("%f ", vf);
  //               c[gmem_store_ptr + row_smem * N + clo_smem] = vf;
  //           }
  //       }
  //     }
  //   }
  // }
  __syncthreads();

  // shared to global memory
  int gmem_store_ptr = by * BM * N + bx * BN;
  if (tid == 0) {
    for (int m = 0; m < BM; m++) {
      for (int n = 0; n < BN; n++) {
        // printf("%f ", smem_float[m * BN + n]);
        c[gmem_store_ptr + m * N + n] = smem_float[m * BN + n];
      }
      // printf("\n");
    }
  }
}

at::Tensor gemm_forward(at::Tensor a, at::Tensor b) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(0);

    at::Tensor c = at::zeros(
        {M, N}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
    );
    // at::Tensor c = at::zeros({M, N}, a.options());

    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int thread_num = 256;

    dim3 GridDim(DIV_UP(N, BN), DIV_UP(M, BM));  // 8 * 8
    dim3 BlockDim(thread_num, 1);
    printf("grid: %d, %d: %d.\n", GridDim.x, GridDim.y, BlockDim.x);
    float* c_ptr = reinterpret_cast<float*>(c.data_ptr());

    gemm_kernel_v3<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, thread_num>
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
    m.def("gemm", &gemm_forward, "Gemm forward (CUDA)",
          py::arg("a"), py::arg("b"));
}
