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

template <int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void flat_gemm(const half* __restrict__ a, const half* __restrict__ b,
                                                    const half* __restrict__ bias, float* __restrict__ c, const int M,
                                                    const int N, const int K) {
    // int bx = blockIdx.x;
    // int by = blockIdx.y;

    int SWIZZLE = DIV_UP(M, BM);

    const int blockid = blockIdx.y * gridDim.x + blockIdx.x;
    const int bx = blockid / SWIZZLE % gridDim.x;
    const int by = blockid / SWIZZLE / gridDim.x * SWIZZLE + blockid % SWIZZLE;

    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane_id = tid % 32;

    // LDN, LDK for avoid bank conflict in smem
    constexpr int LDN = BN + 8;
    constexpr int LDK = BK + 8;

    constexpr int warp_size = 32;
    constexpr int num_frag_a = BM / WMMA_M;
    constexpr int num_frag_b = BN / WMMA_N;
    constexpr int num_frag_c = num_frag_a * num_frag_b;
    constexpr int wmma_num = WMMA_M * WMMA_N / warp_size;

    // blockDim.x  fixed to 256
    constexpr int thread_num = 256;

    // float4 fetch data
    constexpr int BK_step = BK / 8;  // 128 / 8

    constexpr int BM_load_step = BM / (thread_num / BK_step);  // BM / 256 / 8 = 64 1
    constexpr int BK_load_step = 8;
    constexpr int BN_load_step = BN / (thread_num / BK_step);

    constexpr int load_a_smem_cycle = BM * BK / thread_num / 8;
    constexpr int load_b_smem_cycle = BN * BK / thread_num / 8;

    constexpr int warp_num = thread_num / warp_size;
    constexpr int mma_offset = warp_num * WMMA_K;
    constexpr int mma_cycle = BK / mma_offset;

    constexpr int smem_num = ((BM + BN) * LDK) > (2 * BM * LDN) ? ((BM + BN) * LDK) : (2 * BM * LDN);

    __shared__ half smem[smem_num];
    half* s_a = smem;
    half* s_b = smem + BM * LDK;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a[num_frag_a];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_b[num_frag_b];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[num_frag_a][num_frag_b];

#pragma unroll
    for (int i = 0; i < num_frag_a; i++) {
        for (int j = 0; j < num_frag_b; j++) {
            nvcuda::wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    int load_a_smem_m = (tid / BK_step) * BM_load_step;
    int load_a_smem_k = (tid % BK_step) * BK_load_step;
    int load_b_smem_n = (tid / BK_step) * BN_load_step;
    int load_b_smem_k = load_a_smem_k;

    half* load_a_smem_addrs[load_a_smem_cycle];
#pragma unroll
    for (int i = 0; i < load_a_smem_cycle; i++) {
        load_a_smem_addrs[i] = s_a + OFFSET(load_a_smem_m, load_a_smem_k, LDK) + i * LDK;
    }

    half* load_b_smem_addrs[load_b_smem_cycle];
#pragma unroll
    for (int i = 0; i < load_b_smem_cycle; i++) {
        load_b_smem_addrs[i] = s_b + OFFSET(load_b_smem_n, load_b_smem_k, LDK) + i * LDK;
    }

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_k = load_a_smem_k;
    int load_b_gmem_k = load_b_smem_k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_n, load_b_gmem_k, K);

    int load_a_smem_offset = WMMA_M * LDK;
    int load_b_smem_offset = WMMA_N * LDK;

    for (int bk = 0; bk < K / BK; bk++) {
        if (load_a_gmem_m < M) {
#pragma unroll
            for (int i = 0; i < load_a_smem_cycle; i++) {
                *(float4*)(load_a_smem_addrs[i]) = *(float4*)(&a[load_a_gmem_addr + i * K]);
            }
        }

#pragma unroll
        for (int i = 0; i < load_b_smem_cycle; i++) {
            *(float4*)(load_b_smem_addrs[i]) = *(float4*)(&b[load_b_gmem_addr + i * K]);
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < mma_cycle; i++) {
            // first round to fetch frag_a[0] and frag_b[all]
            nvcuda::wmma::load_matrix_sync(frag_a[0], &s_a[wid * WMMA_K + i * mma_offset], LDK);
            for (int n_index = 0; n_index < num_frag_b; n_index++) {
                nvcuda::wmma::load_matrix_sync(frag_b[n_index],
                                          &s_b[load_b_smem_offset * n_index + wid * WMMA_K + i * mma_offset], LDK);
                nvcuda::wmma::mma_sync(frag_c[0][n_index], frag_a[0], frag_b[n_index], frag_c[0][n_index]);
            }

            for (int m_index = 1; m_index < num_frag_a; m_index++) {
                nvcuda::wmma::load_matrix_sync(frag_a[m_index],
                                          &s_a[load_a_smem_offset * m_index + wid * WMMA_K + i * mma_offset], LDK);

                for (int n_index = 0; n_index < num_frag_b; n_index++) {
                    nvcuda::wmma::mma_sync(frag_c[m_index][n_index], frag_a[m_index], frag_b[n_index],
                                      frag_c[m_index][n_index]);
                }
            }
        }

        __syncthreads();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK;
    }

    unsigned int write_back_row_index = lane_id / 16 * 4;
    unsigned int write_back_col_index = lane_id % 16;

    float* smem_float = (float*)(&smem[0]);

    // store back 1st warp result
    if (wid == 0) {
#pragma unroll
        for (int m_index = 0; m_index < num_frag_a; m_index++) {
#pragma unroll
            for (int n_index = 0; n_index < num_frag_b; n_index++) {
#pragma unroll
                for (int ele = 0; ele < wmma_num; ele++) {
                    smem_float[WMMA_M * LDN * m_index + n_index * WMMA_N + (write_back_row_index + ele) * LDN +
                               write_back_col_index] = frag_c[m_index][n_index].x[ele];
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
                    for (int ele = 0; ele < wmma_num; ele++) {
                        smem_float[WMMA_M * LDN * m_index + n_index * WMMA_N + (write_back_row_index + ele) * LDN +
                                   write_back_col_index] += frag_c[m_index][n_index].x[ele];
                    }
                }
            }
        }

        __syncthreads();
    }

    int smem_c_m = (tid / WMMA_M) * num_frag_a;
    int smem_c_n = (tid % WMMA_N) * num_frag_b;
    int smem_c_addr = OFFSET(smem_c_m, smem_c_n, LDN);

    int gmem_c_m = by * BM + smem_c_m;
    int gmem_c_n = bx * BN + smem_c_n;
    int gmem_c_addr = OFFSET(gmem_c_m, gmem_c_n, N);

    if (bias == nullptr) {
#pragma unroll
        for (int i = 0; i < num_frag_a; i++) {
            if (gmem_c_m + i < M) {
#pragma unroll
                for (int j = 0; j < num_frag_b; j++) {
                    c[gmem_c_addr + i * N + j] = __float2half(smem_float[smem_c_addr + i * LDN + j]);
                }
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < num_frag_a; i++) {
            if (gmem_c_m + i < M) {
#pragma unroll
                for (int j = 0; j < num_frag_b; j++) {
                    c[gmem_c_addr + i * N + j] =
                        __hadd(__float2half(smem_float[smem_c_addr + i * LDN + j]), bias[(gmem_c_addr + j) % N]);
                }
            }
        }
    }
}


at::Tensor gemm_v0_forward(at::Tensor a, at::Tensor b) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(0);

    at::Tensor c = at::zeros(
        {M, N}, 
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)
    );

    const int BM = 16;
    const int BN = 64;
    const int BK = 128;
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    flat_gemm<64, 64, 64, 16, 16, 16>
        <<<dim3(DIV_UP(N, BN), DIV_UP(M, BM)), dim3(256, 1), 0, stream>>>(
               reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
               reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
               nullptr,
               reinterpret_cast<float*>(c.data_ptr<>()),
               M, N, K);

    cudaStreamSynchronize(stream);
    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_v0", &gemm_v0_forward, "Gemm forward (CUDA)",
          py::arg("a"), py::arg("b"));
}
