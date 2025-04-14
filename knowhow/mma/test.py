import torch

S = 4
M = 1024 * S
K = 1024 * S
N = 1024 * S
A = torch.randn([M, K], device="cuda").half()
B = torch.randn([K, N], device="cuda").half()

C = A @ B

import flash_fusion


for _ in range(10):
   flash_C = flash_fusion.mma(A, B)

print(C, flash_C)
print(torch.allclose(C, flash_C.half(), atol=0.01, rtol=0.01))

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

for _ in range(100):
   flash_C = flash_fusion.mma(A, B)

end.record()
torch.cuda.synchronize()
e2e_time = start.elapsed_time(end) / 100 / 1000
print("avg pytorch linear sync time: ", start.elapsed_time(end) / 100, " ms.")

compute_flops = M * (K + (K - 1)) * N

flash_throughput = compute_flops / e2e_time / (10 ** 9)
print("Flash Throughput: ", flash_throughput, " GFLOPS")

max_bias = (abs(C - flash_C)).max()
print("max bias: ", max_bias)