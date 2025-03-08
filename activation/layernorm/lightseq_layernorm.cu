#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <iostream>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include "pybind.h"

const unsigned int WARP_REDUCE_MASK = 0xffffffff;
enum class ReduceType { kMax = 0, kSum };
const float LN_EPSILON = 1e-8f;
#define WARP_SIZE 32
#define MAX_THREADS 1024
const float CUDA_FLOAT_INF_NEG = -100000000.f;  // FIXME later
const float CUDA_FLOAT_INF_POS = 100000000.f;   // FIXME later
const int CUDA_INT_INF = 2147483647;
const float REDUCE_FLOAT_INF_NEG = -100000000.f;
const float REDUCE_FLOAT_INF_POS = 100000000.f;

template <ReduceType Rtype, int Num>
__inline__ __device__ void blockReduce(float *pval);

// use template to make code more concise
template <ReduceType Rtype, int Num>
__inline__ __device__ void warpReduce(float *pval);


template <>
__inline__ __device__ void warpReduce<ReduceType::kSum, 1>(float *pval) {
  *pval += __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 16, 32);
  *pval += __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 8, 32);
  *pval += __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 4, 32);
  *pval += __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 2, 32);
  *pval += __shfl_xor_sync(WARP_REDUCE_MASK, *pval, 1, 32);
}


template <>
__inline__ __device__ void warpReduce<ReduceType::kSum, 2>(float *pval) {
  float val0_tmp, val1_tmp;
#define WarpReduceSumOneStep(a, b)                                 \
  val0_tmp = __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 0), a, b); \
  val1_tmp = __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 1), a, b); \
  *(pval + 0) += val0_tmp;                                         \
  *(pval + 1) += val1_tmp

  WarpReduceSumOneStep(16, 32);
  WarpReduceSumOneStep(8, 32);
  WarpReduceSumOneStep(4, 32);
  WarpReduceSumOneStep(2, 32);
  WarpReduceSumOneStep(1, 32);

#undef WarpReduceSumOneStep
}


template <>
__inline__ __device__ void blockReduce<ReduceType::kSum, 2>(float *pval) {
  const int num = 2;
  static __shared__ float shared[num][32];
  int lane_id = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduce<ReduceType::kSum, num>(pval);

  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      shared[i][wid] = *(pval + i);
    }
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x >> 5)) {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = shared[i][lane_id];
    }
  } else {
#pragma unroll
    for (int i = 0; i < num; ++i) {
      *(pval + i) = 0.f;
    }
  }
  warpReduce<ReduceType::kSum, num>(pval);
}

template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  // step 0. compute local sum
  float l_sum = 0;
  float l_square_sum = 0;
  const float4 *inp_f4 =
      reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l_square_sum +=
        val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }

  // step 1. compute reduce sum
  float mean_dim = float(hidden_size) * 4.f;
  float reduce_val[2] = {l_sum, l_square_sum};
  blockReduce<ReduceType::kSum, 2>(reduce_val);
  __shared__ float s_mean, s_var;
  if (threadIdx.x == 0) {
    s_mean = reduce_val[0] / mean_dim;
    if (means != nullptr) {
      means[blockIdx.x] = s_mean;
    }
    s_var = reduce_val[1] / mean_dim - s_mean * s_mean + LN_EPSILON;
    vars[blockIdx.x] = s_var;
    s_var = rsqrtf(s_var);
  }
  __syncthreads();

  // step 2. layer norm result
  float4 *output_f4 =
      reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 vscale = __ldg(reinterpret_cast<const float4 *>(scale) + idx);
    float4 vbias = __ldg(reinterpret_cast<const float4 *>(bias) + idx);
    float4 val = inp_f4[idx];
    val.x = (val.x - s_mean) * s_var * vscale.x + vbias.x;
    val.y = (val.y - s_mean) * s_var * vscale.y + vbias.y;
    val.z = (val.z - s_mean) * s_var * vscale.z + vbias.z;
    val.w = (val.w - s_mean) * s_var * vscale.w + vbias.w;
    output_f4[idx] = val;
  }
}


template <>
__global__ void ker_layer_norm<__half>(__half *ln_res, __half *vars,
                                       __half *means, const __half *inp,
                                       const __half *scale, const __half *bias,
                                       int hidden_size) {
  // step 0. compute local sum
  float l_sum = 0;
  float l_square_sum = 0;
  const float4 *inp_f4 =
      reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val_f4 = inp_f4[idx];
    __half2 *val_h2 = (__half2 *)(&val_f4);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 val_f2 = __half22float2(val_h2[i]);
      l_sum += val_f2.x + val_f2.y;
      l_square_sum += val_f2.x * val_f2.x + val_f2.y * val_f2.y;
    }
  }

  // step 1. compute reduce sum
  float mean_dim = float(hidden_size) * 8.f;
  float reduce_val[2] = {l_sum, l_square_sum};
  blockReduce<ReduceType::kSum, 2>(reduce_val);
  __shared__ float s_mean, s_var;
  if (threadIdx.x == 0) {
    s_mean = reduce_val[0] / mean_dim;
    if (means != nullptr) {
      means[blockIdx.x] = __float2half(s_mean);
    }
    s_var = reduce_val[1] / mean_dim - s_mean * s_mean + LN_EPSILON;
    vars[blockIdx.x] = __float2half(s_var);
    s_var = rsqrtf(s_var);
  }
  __syncthreads();

  // step 2. layer norm result
  float4 *output_f4 =
      reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    // load scale, bias, input
    float4 scale_f4 = __ldg(reinterpret_cast<const float4 *>(scale) + idx);
    __half2 *scale_h2 = reinterpret_cast<__half2 *>(&scale_f4);
    float4 bias_f4 = __ldg(reinterpret_cast<const float4 *>(bias) + idx);
    __half2 *bias_h2 = reinterpret_cast<__half2 *>(&bias_f4);
    float4 val_f4 = inp_f4[idx];
    __half2 *val_h2 = reinterpret_cast<__half2 *>(&val_f4);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 scale_f2 = __half22float2(scale_h2[i]);
      float2 bias_f2 = __half22float2(bias_h2[i]);
      float2 val_f2 = __half22float2(val_h2[i]);
      val_f2.x = (val_f2.x - s_mean) * s_var * scale_f2.x + bias_f2.x;
      val_f2.y = (val_f2.y - s_mean) * s_var * scale_f2.y + bias_f2.y;
      val_h2[i] = __float22half2_rn(val_f2);
    }
    output_f4[idx] = val_f4;
  }
}

template <typename T>
void launch_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                       const T *scale, const T *bias, int batch_size,
                       int hidden_dim, cudaStream_t stream);


template <>
void launch_layer_norm<float>(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      ln_res, vars, means, inp, scale, bias, hidden_dim);
}

template <>
void launch_layer_norm<__half>(__half *ln_res, __half *vars, __half *means,
                               const __half *inp, const __half *scale,
                               const __half *bias, int batch_size,
                               int hidden_dim, cudaStream_t stream) {
  if (hidden_dim % 8 != 0) {
    throw std::runtime_error("violate hidden_dim % 8 = 0");
  }
  hidden_dim >>= 3;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<__half><<<grid_dim, block_dim, 0, stream>>>(
      ln_res, vars, means, inp, scale, bias, hidden_dim);
}

template <typename T>
const T *rptr(const torch::Tensor &tensor) {
  return reinterpret_cast<const T *>(tensor.data_ptr());
}

template <typename T>
T *rptr(torch::Tensor &tensor) {
  return reinterpret_cast<T *>(tensor.data_ptr());
}

template <typename T>
void torch_launch_layer_norm(torch::Tensor &ln_res, torch::Tensor &vars,
                             torch::Tensor &means, const torch::Tensor &inp,
                             const torch::Tensor &scale,
                             const torch::Tensor &bias, int batch_size,
                             int hidden_dim, bool with_mean) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (with_mean) {
    launch_layer_norm(rptr<T>(ln_res), rptr<T>(vars), rptr<T>(means),
                      rptr<T>(inp), rptr<T>(scale), rptr<T>(bias), batch_size,
                      hidden_dim, stream);
  } else {
    launch_layer_norm(rptr<T>(ln_res), rptr<T>(vars), (T *)nullptr,
                      rptr<T>(inp), rptr<T>(scale), rptr<T>(bias), batch_size,
                      hidden_dim, stream);
  }
}

void ls_layer_norm(torch::Tensor &ln_res, torch::Tensor &vars,
                   torch::Tensor &means, const torch::Tensor &inp,
                   const torch::Tensor &scale,
                   const torch::Tensor &bias, int batch_size,
                   int hidden_dim, bool with_mean) {
   torch_launch_layer_norm<float>(ln_res, vars,
                   means, inp,
                   scale,
                   bias, batch_size,
                   hidden_dim, with_mean);
}

