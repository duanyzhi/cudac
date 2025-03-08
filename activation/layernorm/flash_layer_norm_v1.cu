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

__global__ void layernorm(const float4* __restrict__ x, const float4* __restrict__ weight,
    const float4* __restrict__ bias, float eps, float4* __restrict__ output, int hidden_dim) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    const float4* x_block = x + blockIdx.x * hidden_dim;

    double sum_x = 0;
    double sum_xx = 0;
    #pragma unroll
    for (uint id = tid; id < hidden_dim; id+=blockDim.x) {
      float4 val = x_block[id];
      sum_x += val.x + val.y + val.z + val.w;
      sum_xx +=
          val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    // __syncthreads();
    // warp sum to get 1024 / 32 sum
    warp_sum(&sum_x);
    warp_sum(&sum_xx);

    __syncthreads();

    int lane_id = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    static __shared__ double shared_mean[32];  // one warp for sum
    static __shared__ double shared_var[32];  // one warp for sum
    if (lane_id == 0) {  // push all lane_id 0 to shared
      shared_mean[wid] = sum_x;
      shared_var[wid] = sum_xx;
      // printf("lane id %d wid  %d mean %f", lane_id, wid, mean);
    }

    __syncthreads();
    if (threadIdx.x < (blockDim.x >> 5)) {  // < 32
      sum_x = shared_mean[lane_id];  // get mean from shared memory
      sum_xx = shared_var[lane_id];  // get mean from shared memory
    } else {
      sum_x = 0.0f;  // only keey warp data
      sum_xx = 0.0f;  // only keey warp data
    }
    warp_sum(&sum_x);
    warp_sum(&sum_xx);
    __syncthreads();

    __shared__ double reduce_mean, reduce_var;
    if (tid == 0) {
       reduce_mean = __ddiv_rn(sum_x, hidden_dim * 4.0f);
       reduce_var = __ddiv_rn(sum_xx, hidden_dim * 4.0f) - __dmul_rn(reduce_mean, reduce_mean) + eps;
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
