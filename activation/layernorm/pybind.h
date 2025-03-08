#pragma once

#include <torch/extension.h>
#include <ATen/AccumulateType.h>

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5);

// at::Tensor layer_norm(
//     at::Tensor input,
//     at::IntArrayRef normalized_shape,
//     double epsilon = 1e-5);
//  
// void ls_layer_norm(torch::Tensor &ln_res, torch::Tensor &vars,
//                    torch::Tensor &means, const torch::Tensor &inp,
//                    const torch::Tensor &scale,
//                    const torch::Tensor &bias, int batch_size,
//                    int hidden_dim, bool with_mean);
 

at::Tensor flash_layernorm(at::Tensor x, at::Tensor weight,
    at::Tensor bias, double eps = 1e-5);
at::Tensor cutlass_layernorm(at::Tensor x, at::Tensor weight,
    at::Tensor bias, double eps);
 
