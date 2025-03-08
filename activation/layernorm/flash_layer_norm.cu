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
#define MAX_THREADS 1024

__device__ void warp_sum(double* __restrict__ val) {
  *val += __shfl_xor_sync(WARP_REDUCE_MASK, *val, 16, 32);
  *val += __shfl_xor_sync(WARP_REDUCE_MASK, *val, 8, 32);
  *val += __shfl_xor_sync(WARP_REDUCE_MASK, *val, 4, 32);
  *val += __shfl_xor_sync(WARP_REDUCE_MASK, *val, 2, 32);
  *val += __shfl_xor_sync(WARP_REDUCE_MASK, *val, 1, 32);
}

__inline__ __device__ void warp_sum2(float *pval) {
  // float val0_tmp, val1_tmp;
  *(pval + 0) += __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 0), 16, 32);
  *(pval + 0) += __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 0), 8, 32);
  *(pval + 0) += __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 0), 4, 32);
  *(pval + 0) += __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 0), 2, 32);
  *(pval + 0) += __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 0), 1, 32);

  *(pval + 1) += __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 1), 16, 32);
  *(pval + 1) += __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 1), 8, 32);
  *(pval + 1) += __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 1), 4, 32);
  *(pval + 1) += __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 1), 2, 32);
  *(pval + 1) += __shfl_xor_sync(WARP_REDUCE_MASK, *(pval + 1), 1, 32);
}

__inline__ __global__ void layernorm(const float4* __restrict__ x, const float4* __restrict__ weight,
    const float4* __restrict__ bias, float eps, float4* __restrict__ output, int hidden_dim) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    const float4* x_block = x + blockIdx.x * hidden_dim;
    uint stride = blockDim.x;

    float sum_x = 0;
    float sum_xx = 0;

    float4 val_next = x_block[tid];  // 正确初始化：预取第一个数据块

    #pragma unroll
    for (uint id = tid; id < hidden_dim; id+=stride) {
      float4 val = val_next;       // 当前处理的数据
      if (id + stride < hidden_dim) {
          val_next = x_block[id + stride];  // 安全预取下一批数据
      }

      float sum_val = val.x + val.y + val.z + val.w;
      float sum_sq_val =
          val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
      sum_x += sum_val;
      sum_xx += sum_sq_val;
    }

    // __syncthreads();
    // warp sum to get 1024 / 32 sum

    float reduce_val[2] = {sum_x, sum_xx};
    warp_sum2(reduce_val);

    // __syncthreads();

    int lane_id = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    // static __shared__ double shared_mean[32];  // one warp for sum
    // static __shared__ double shared_var[32];  // one warp for sum
    static __shared__ float shared[2][32];
    if (lane_id == 0) {  // push all lane_id 0 to shared
      shared[0][wid] = reduce_val[0];
      shared[1][wid] = reduce_val[1];
      // printf("lane id %d wid  %d mean %f", lane_id, wid, mean);
      // printf("reduce %f %f", reduce_val[0], reduce_val[1]);
    }

    __syncthreads();
    if (threadIdx.x < (blockDim.x >> 5)) {  // < 32
      reduce_val[0] = shared[0][lane_id];  // get mean from shared memory
      reduce_val[1] = shared[1][lane_id];  // get mean from shared memory
    } else {
      reduce_val[0] = 0.0f;  // only keey warp data
      reduce_val[1] = 0.0f;  // only keey warp data
    }
    warp_sum2(reduce_val);
    // __syncthreads();

    // printf("reduce %d %d", reduce_val[0], reduce_val[1]);
    __shared__ float reduce_mean, reduce_var;
    if (tid == 0) {
       // printf("reduce %f %f", reduce_val[0], reduce_val[1]);
       reduce_mean = __ddiv_rn(reduce_val[0], hidden_dim * 4.0f);
       reduce_var = __ddiv_rn(reduce_val[1], hidden_dim * 4.0f) - __dmul_rn(reduce_mean, reduce_mean) + eps;
       reduce_var = rsqrtf(reduce_var);
    }
    __syncthreads(); // wait reduce_mean and var

    // compute results
    float4 *output_block = output + blockIdx.x * hidden_dim;
    // #pragma unroll
    // for (uint id = tid; id < hidden_dim; id+=blockDim.x) {
    //   float alpha  = __ldg(weight + tid);
    //   float beta  = __ldg(bias + tid);
    //   output_block[id] = __ddiv_rn((x_block[id] - reduce_mean), reduce_var) * alpha + beta;
    // }


    #pragma unroll
    for (uint idx = tid; idx < hidden_dim; idx += blockDim.x) {
      float4 vscale = __ldg(reinterpret_cast<const float4 *>(weight) + idx);
      float4 vbias = __ldg(reinterpret_cast<const float4 *>(bias) + idx);
      float4 val = x_block[idx];
      val.x = (val.x - reduce_mean) * reduce_var * vscale.x + vbias.x;
      val.y = (val.y - reduce_mean) * reduce_var * vscale.y + vbias.y;
      val.z = (val.z - reduce_mean) * reduce_var * vscale.z + vbias.z;
      val.w = (val.w - reduce_mean) * reduce_var * vscale.w + vbias.w;
      output_block[idx] = val;
    }
}

at::Tensor flash_layernorm(at::Tensor x, at::Tensor weight,
    at::Tensor bias, double eps) {

  auto output = torch::empty_like(x);
  // std::cout << "input tensor scaler type " << x.scalar_type() << "\n";
  // std::cout << "weight tensor scaler type " << weight.scalar_type() << "\n";

  const int batch_size = x.size(0);
  int hidden_dim = 1;

  auto sizes = x.sizes();
  for (int i = 1; i < sizes.size(); ++i) {
      hidden_dim = hidden_dim * sizes[i];
  }

  // for (int i = 1; i < x.size().size(); ++i) {
  //    hidden_dim *= x.size(i);
  // }
  // // const int hidden_dim = x.size(1);
  // std::cout << "hidden_dim " << hidden_dim << "\n";

  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);
  // std::cout << "grid dim " << batch_size << " block dim " << block_dim.x << ", " << block_dim.y << "\n";

  const float4* x_f4 = reinterpret_cast<const float4 *>(x.data_ptr()); 
  const float4* w_f4 = reinterpret_cast<const float4 *>(weight.data_ptr()); 
  const float4* b_f4 = reinterpret_cast<const float4 *>(bias.data_ptr()); 
  float4* o_f4 = reinterpret_cast<float4 *>(output.data_ptr()); 
  hidden_dim = hidden_dim / 4;

  layernorm<<<grid_dim, block_dim>>>(
      x_f4, w_f4, b_f4,
      static_cast<float>(eps),
      o_f4, hidden_dim);
  return output;
}
