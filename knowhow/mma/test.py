import torch

M = 4096
K = 4096
N = 4096
A = torch.randn([M, K], device="cuda").half()
B = torch.randn([K, N], device="cuda").half()

C = A * B

import flash_fusion


flash_C = flash_fusion.mma(A, B)

print(C, flash_C)
print(torch.allclose(C, flash_C.half()))
