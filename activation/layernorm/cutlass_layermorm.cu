#include <cfloat>
#include <cuda_fp16.h>
#include <cfloat>
#define FINAL_MASK 0xffffffff
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <iostream>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include "pybind.h"



struct half4 {
    half x, y, z, w;
};

template<typename T, int NUM>
__inline__ __device__ T warpReduceSum(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++) {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
    return (T)(0.0f);
}



template<typename T, int NUM>
__inline__ __device__ T blockReduceSum(T* val)
{
    __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSum<T, NUM>(val);

    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < NUM; i++) {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++) {
        val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
    }
    warpReduceSum<T, NUM>(val);
    return (T)0.0f;
}

template<typename T, int NUM>
__inline__ __device__ T warpReduceMax(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++) {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] = max(val[i], __shfl_xor_sync(FINAL_MASK, val[i], mask, 32));
    }
    return (T)(0.0f);
}

template<typename T, int NUM>
__inline__ __device__ T blockReduceMax(T* val)
{
    static __shared__ T shared[32][NUM];
    int lane = threadIdx.x & 0x1f;  // in-warp idx
    int wid = threadIdx.x >> 5;     // warp idx

    warpReduceMax<T, NUM>(val);  // get maxx in each warp

    if (lane == 0)  // record in-warp maxx by warp Idx
    {
#pragma unroll
        for (int i = 0; i < NUM; i++) {
            shared[wid][i] = val[i];
        }
    }

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++) {
        val[i] = is_mask ? shared[lane][i] : (T)(-FLT_MAX);
    }
    warpReduceMax<T, NUM>(val);

    return (T)0.0f;
}

/** \brief interface to do layernorm on a device memory tensor with RowMajor layout.
 * \tparam T: data type
 */
// template <typename T>
// void layernorm(cutlass::MatrixCoord tensor_size,
//                TensorRef<T, layout::RowMajor> ref_output,
//                TensorRef<T, layout::RowMajor> ref_input,
//                TensorRef<T, layout::RowMajor> ref_gamma,
//                TensorRef<T, layout::RowMajor> ref_beta,
//                cudaStream_t stream);

/**
 * output [m, n] row-major
 * input [m, n] row-major
 * gamma [n]
 * beta [n]
 * grid(m)
 * block(block_size) -- each block deals with n elements ; each thread deals with ITEM_PER_THREAD elements
*/
template<typename T, int ITEM_PER_THREAD>
__global__ void layernorm_twoPassAlgo_stored_locally_e1(T* output, 
                                                        const T* input, 
                                                        const T* gamma, 
                                                        const T* beta, 
                                                        const int m, 
                                                        const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  T local_val[ITEM_PER_THREAD];
  float local_sums[1] = {0.0f};
  int offset = m_idx * n;
  input += offset;
  output += offset;

  const T zero = T(0.0f);
  #pragma unroll
  for (int i = 0 ; i < ITEM_PER_THREAD ; i++){ 
    int index = tid + i*bdimx;
    local_val[i] = index < n ? input[index] : zero;   
    local_sums[0] += static_cast<float>(local_val[i]); 
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  #pragma unroll
  for (int i = 0 ; i < ITEM_PER_THREAD ; i++){
    int index = tid + i*bdimx;
    if (index < n){
      const float tmp = static_cast<float>(local_val[i]) - s_mean;
      local_sums[0] += tmp * tmp;
    }
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + 1e-5);
  }
  __syncthreads();

  #pragma unroll
  for (int i = 0 ; i < ITEM_PER_THREAD ; i++){
    int index = tid + i*bdimx;
    if (index < n) {
      const T gamma_val = gamma[index];
      const T beta_val = beta[index];
      output[index] = T((static_cast<float>(local_val[i]) - s_mean) * s_variance * static_cast<float>(gamma_val) + static_cast<float>(beta_val));
    }
  }
}

/**
 * output [m, n] row-major
 * input [m, n] row-major
 * gamma [n]
 * beta [n]
 * grid(m)
 * block(block_size) -- each block deals with block_size*ITEM_PER_THREAD*2 elements;
*/
template<typename T2, typename T, int ITEM_PER_THREAD>
__global__ void layernorm_twoPassAlgo_stored_locally_e2(T2* output,
                                                        const T2* input,
                                                        const T2* gamma,
                                                        const T2* beta,
                                                        const int m,
                                                        const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  T2 local_val[ITEM_PER_THREAD];
  const int n_2 = n / 2;
  int offset = m_idx * n_2;
  input += offset;
  output += offset;

  const T2 zero = {T(0.0f), T(0.0f)};
  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    local_val[i] = index < n_2 ? input[index] : zero;
    local_sums[0] += static_cast<float>(local_val[i].x) + static_cast<float>(local_val[i].y);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    if (index < n_2){
      const float2 tmp = {static_cast<float>(local_val[i].x) - s_mean,
                          static_cast<float>(local_val[i].y) - s_mean};
      local_sums[0] += tmp.x * tmp.x + tmp.y * tmp.y;
    }
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + 1e-5);
  }
  __syncthreads();

  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    if (index < n_2){
      const T2 gamma_val = gamma[index];
      const T2 beta_val = beta[index];
      T2 tmp;
      tmp.x = T((static_cast<float>(local_val[i].x) - s_mean)*s_variance*static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x));
      tmp.y = T((static_cast<float>(local_val[i].y) - s_mean)*s_variance*static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y));
      output[index] = tmp;
    }
  }
}

/**
 * output [m, n] row-major
 * input [m, n] row-major
 * gamma [n]
 * beta [n]
 * grid(m)
 * block(block_size) -- each block deals with block_size*ITEM_PER_THREAD*4 elements;
*/
template<typename T4, typename T, int ITEM_PER_THREAD>
__global__ void layernorm_twoPassAlgo_stored_locally_e4(T4* output,
                                                        const T4* input,
                                                        const T4* gamma,
                                                        const T4* beta,
                                                        const int m,
                                                        const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  T4 local_val[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  int offset = m_idx * n_4;
  input += offset;
  output += offset;

  const T4 zero = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    local_val[i] = index < n_4 ? input[index] : zero;
    local_sums[0] += static_cast<float>(local_val[i].x) + static_cast<float>(local_val[i].y) +
                     static_cast<float>(local_val[i].z) + static_cast<float>(local_val[i].w);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    if (index < n_4){
      const float4 tmp = {static_cast<float>(local_val[i].x) - s_mean,
                          static_cast<float>(local_val[i].y) - s_mean,
                          static_cast<float>(local_val[i].z) - s_mean,
                          static_cast<float>(local_val[i].w) - s_mean};
      local_sums[0] += tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z + tmp.w * tmp.w;
    }
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + 1e-5);
  }
  __syncthreads();

  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    if (index < n_4){
      const T4 gamma_val = gamma[index];
      const T4 beta_val = beta[index];
      T4 tmp;
      tmp.x = T((static_cast<float>(local_val[i].x) - s_mean)*s_variance*static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x));
      tmp.y = T((static_cast<float>(local_val[i].y) - s_mean)*s_variance*static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y));
      tmp.z = T((static_cast<float>(local_val[i].z) - s_mean)*s_variance*static_cast<float>(gamma_val.z) + static_cast<float>(beta_val.z));
      tmp.w = T((static_cast<float>(local_val[i].w) - s_mean)*s_variance*static_cast<float>(gamma_val.w) + static_cast<float>(beta_val.w));
      output[index] = tmp;
    }
  }
}

/**
 * output [m, n] row-major
 * input [m, n] row-major
 * gamma [n]
 * beta [n]
 * grid(m)
 * block(block_size) -- each block deals with n elements ; each thread deals with ITEM_PER_THREAD elements
*/
template<typename T>
__global__ void layernorm_twoPassAlgo_e1(T* output,
                                         const T* input,
                                         const T* gamma,
                                         const T* beta,
                                         const int m,
                                         const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  int offset = m_idx * n;
  input += offset;
  output += offset;

  for (int index = tid ; index < n ; index += bdimx){
    float local_val = static_cast<float>(input[index]);
    local_sums[0] += local_val;
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  for (int index = tid ; index < n ; index += bdimx){
    float local_val = static_cast<float>(input[index]);
    local_val = local_val - s_mean;
    local_sums[0] += local_val * local_val;
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + 1e-5);
  }
  __syncthreads();

  for (int index = tid ; index < n ; index += bdimx){
    const T gamma_val = gamma[index];
    const T beta_val = beta[index];
    const T local_val = input[index];
    output[index] = T((static_cast<float>(local_val) - s_mean) * s_variance * static_cast<float>(gamma_val) + static_cast<float>(beta_val));
  }
}

/**
 * output [m, n] row-major
 * input [m, n] row-major
 * gamma [n]
 * beta [n]
 * grid(m)
 * block(block_size) -- each block deals with block_size*ITEM_PER_THREAD*2 elements;
*/
template<typename T2, typename T>
__global__ void layernorm_twoPassAlgo_e2(T2* output,
                                         const T2* input,
                                         const T2* gamma,
                                         const T2* beta,
                                         const int m,
                                         const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  const int n_2 = n / 2;
  int offset = m_idx * n_2;
  input += offset;
  output += offset;

  for (int index = tid; index < n_2; index += bdimx) {
    const T2 local_val = input[index];
    local_sums[0] += static_cast<float>(local_val.x) + static_cast<float>(local_val.y);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  for (int index = tid; index < n_2; index += bdimx) {
    const T2 local_val = input[index];
    const float2 tmp = {static_cast<float>(local_val.x) - s_mean,
                        static_cast<float>(local_val.y) - s_mean};
    local_sums[0] += tmp.x * tmp.x + tmp.y * tmp.y;
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + 1e-5);
  }
  __syncthreads();

  for (int index = tid; index < n_2; index += bdimx) {
    const T2 local_val = input[index];
    const T2 gamma_val = gamma[index];
    const T2 beta_val = beta[index];
    T2 tmp;
    tmp.x = T((static_cast<float>(local_val.x) - s_mean)*s_variance*static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x));
    tmp.y = T((static_cast<float>(local_val.y) - s_mean)*s_variance*static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y));
    output[index] = tmp;
  }
}

template <typename T>
void layernorm(int m, int n, at::Tensor ref_output,
               at::Tensor ref_input,
               at::Tensor ref_gamma,
               at::Tensor ref_beta,
               cudaStream_t stream){
  T* output = reinterpret_cast<float*>(ref_output.data_ptr<float>());
  const T* input = reinterpret_cast<float*>(ref_input.data_ptr<float>());
  const T* gamma = reinterpret_cast<float*>(ref_gamma.data_ptr<float>());
  const T* beta = reinterpret_cast<float*>(ref_beta.data_ptr<float>());
  dim3 grid(m);
  dim3 block((n + 31)/32*32);
  if (block.x > 1024){
    block.x = 1024;
  }
  // TODO : There should be better configs for different cases, we only use several samples to show how to use here
  // TODO : using registers to store values locally can reduce the loads from global memory and speedup the kernels.
  if ((n % 4 == 0) && (n >= 128) && (n <= 4096)) {
    block.x = (n/4 + 31)/32*32;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4<float4, float, 1><<<grid, block, 0, stream>>>(
        (float4*)output,
        (const float4*)input,
        (const float4*)gamma,
        (const float4*)beta,
        m,
        n);
    } // if (std::is_same<T, float>::value)
    else {
      layernorm_twoPassAlgo_stored_locally_e4<half4, half, 1><<<grid, block, 0, stream>>>(
        (half4*)output,
        (const half4*)input,
        (const half4*)gamma,
        (const half4*)beta,
        m,
        n);
    }
  } //if ((n % 4 == 0) && (n >= 128) && (n <= 4096))
  else if (n % 2 == 0) {
    if (n / 2 <= 1024) {
      block.x = (n/2 + 31)/32*32;
      if (std::is_same<T, float>::value) {
        layernorm_twoPassAlgo_stored_locally_e2<float2, float, 1><<<grid, block, 0, stream>>>(
          (float2*)output,
          (const float2*)input,
          (const float2*)gamma,
          (const float2*)beta,
          m,
          n);
      } //if (std::is_same<T, float>::value)
      else {
        layernorm_twoPassAlgo_stored_locally_e2<half2, half, 1><<<grid, block, 0, stream>>>(
          (half2*)output,
          (const half2*)input,
          (const half2*)gamma,
          (const half2*)beta,
          m,
          n);
      }
    } // if (n / 2 <= 1024)
    else if (n <= 8192) {
      block.x = ((n + 7)/8 + 31)/32*32;
      if (std::is_same<T, float>::value) {
        layernorm_twoPassAlgo_stored_locally_e2<float2, float, 4><<<grid, block, 0, stream>>>(
          (float2*)output,
          (const float2*)input,
          (const float2*)gamma,
          (const float2*)beta,
          m,
          n);
      } // if (std::is_same<T, float>::value)
      else {
        layernorm_twoPassAlgo_stored_locally_e2<half2, half, 4><<<grid, block, 0, stream>>>(
          (half2*)output,
          (const half2*)input,
          (const half2*)gamma,
          (const half2*)beta,
          m,
          n);
      }
    } // if (n <= 8192)
    else if (n <= 16384) {
      block.x = ((n + 15)/ 16 + 31)/32*32;
      if (std::is_same<T, float>::value) {
        layernorm_twoPassAlgo_stored_locally_e2<float2, float, 8><<<grid, block, 0, stream>>>(
          (float2*)output,
          (const float2*)input,
          (const float2*)gamma,
          (const float2*)beta,
          m,
          n);
      } // if (std::is_same<T, float>::value)
      else {
        layernorm_twoPassAlgo_stored_locally_e2<half2, half, 8><<<grid, block, 0, stream>>>(
          (half2*)output,
          (const half2*)input,
          (const half2*)gamma,
          (const half2*)beta,
          m,
          n);
      }
    } // if (n <= 16384)
    else if (n <= 32768) {
      block.x = ((n + 31)/32 + 31)/32*32;
      if (std::is_same<T, float>::value) {
        layernorm_twoPassAlgo_stored_locally_e2<float2, float, 16><<<grid, block, 0, stream>>>(
          (float2*)output,
          (const float2*)input,
          (const float2*)gamma,
          (const float2*)beta,
          m,
          n);
      } // if (std::is_same<T, float>::value)
      else {
        layernorm_twoPassAlgo_stored_locally_e2<half2, half, 16><<<grid, block, 0, stream>>>(
          (half2*)output,
          (const half2*)input,
          (const half2*)gamma,
          (const half2*)beta,
          m,
          n);
      }
    } // if (n <= 32768)
    else {
      if (block.x > 512)
        block.x = 512;
      if (std::is_same<T, float>::value) {
        layernorm_twoPassAlgo_e2<float2, float><<<grid, block, 0, stream>>>(
          (float2 *)output, 
          (const float2 *)input,
          (const float2 *)gamma, 
          (const float2 *)beta, 
          m, 
          n);
      } // if (std::is_same<T, float>::value)
      else {
        layernorm_twoPassAlgo_e2<half2, half><<<grid, block, 0, stream>>>(
          (half2 *)output,
          (const half2 *)input,
          (const half2 *)gamma,
          (const half2 *)beta,
          m,
          n);
      }
    }
  } // if (n % 2 == 0)
  else {
    if (n <= 1024) {
      layernorm_twoPassAlgo_stored_locally_e1<T, 1><<<grid, block, 0, stream>>>(
        output, 
        input, 
        gamma, 
        beta, 
        m, 
        n);
    } // if (n <= 1024)
    else if (n <= 8192) {
      block.x = ((n + 7)/8 + 31)/32*32;
      layernorm_twoPassAlgo_stored_locally_e1<T, 8><<<grid, block, 0, stream>>>(
        output,
        input,
        gamma,
        beta,
        m,
        n);
    } // if (n <= 8192)
    else if (n <= 16384) {
      block.x = ((n + 15)/16 + 32)/32*32;
      layernorm_twoPassAlgo_stored_locally_e1<T, 16><<<grid, block, 0, stream>>>(
        output,
        input,
        gamma,
        beta,
        m,
        n);
    } // if (n <= 16384)
    else if (n <= 32768) {
      block.x = ((n + 31)/32 + 31)/32*32;
      layernorm_twoPassAlgo_stored_locally_e1<T, 32><<<grid, block, 0, stream>>>(
        output,
        input,
        gamma,
        beta,
        m,
        n);
    } // if (n <= 32768)
    else{
      if (block.x > 512) {
        block.x = 512;
      }
      layernorm_twoPassAlgo_e1<<<grid, block, 0, stream>>>(
        output, 
        input, 
        gamma, 
        beta, 
        m, 
        n);
    }
  } 
}

at::Tensor cutlass_layernorm(at::Tensor x, at::Tensor weight,
    at::Tensor bias, double eps) {
  auto output = torch::empty_like(x);

  const int batch_size = x.size(0);
  const int hidden_dim = x.size(1);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // layernorm<float>(batch_size, hidden_dim, output, x, weight, bias, stream);
}
