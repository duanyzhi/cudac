import torch

torch.set_printoptions(sci_mode=False, threshold=float('inf'))

S = 1
M = 1024 * S
K = 1024 * S
N = 1024 * S
#A = torch.randn([M, K], device="cuda").bfloat16()
#B = torch.randn([K, N], device="cuda").bfloat16()

# C = A @ B

import flash_fusion

class Linear(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.ln = torch.nn.Linear(in_feature, out_feature, bias=False, device="cuda", dtype=torch.half)

    def forward(self, x):
       return self.ln(x)

model = Linear(128, 128)
x = torch.randn([128, 128], device="cuda", dtype=torch.half)

for _ in range(1):
   torch_C = model(x)
   flash_C = flash_fusion.gemm(x, model.ln.weight)

torch.cuda.synchronize()
# print(C, flash_C)
# print(torch.allclose(C, flash_C.half(), atol=0.01, rtol=0.01))

start_torch = torch.cuda.Event(enable_timing=True)
end_torch = torch.cuda.Event(enable_timing=True)
start_torch.record()

for _ in range(1):
   torch_C = model(x)
   # torch_C = torch.nn.functional.linear(A, B)

end_torch.record()
torch.cuda.synchronize()
e2e_time_torch = start_torch.elapsed_time(end_torch) / 100 / 1000
print("avg pytorch linear sync time: ", start_torch.elapsed_time(end_torch) / 100, " ms.")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# print(model.ln.weight)
# print(torch_C)
# print(x)
# print(x[16:, ...])
# print(model.ln.weight[16:, 16:])
# flash_C = flash_fusion.gemm(x, model.ln.weight)

for _ in range(1):
   flash_C = flash_fusion.gemm(x, model.ln.weight)

end.record()
torch.cuda.synchronize()
e2e_time = start.elapsed_time(end) / 100 / 1000
print("avg flash linear sync time: ", start.elapsed_time(end) / 100, " ms.")

compute_flops = M * (K + (K - 1)) * N

flash_throughput_torch = compute_flops / e2e_time_torch / (10 ** 9)
print("Torch Throughput: ", flash_throughput_torch, " GFLOPS")

flash_throughput = compute_flops / e2e_time / (10 ** 9)
print("Flash Throughput: ", flash_throughput, " GFLOPS")
print(torch_C, flash_C)
print(torch.allclose(torch_C.float(), flash_C, atol=0.01, rtol=0.01))

# max_bias = (abs(torch_C - flash_C)).max()
# print("max bias: ", max_bias)
