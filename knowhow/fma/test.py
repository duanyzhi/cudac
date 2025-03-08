import torch
import nvtx
import flash_fusion

# FMA
# o = a * b + c

a = torch.randn([4096, 4096], device="cuda").float()
b = torch.randn([4096, 4096], device="cuda").float()
c = torch.randn([4096, 4096], device="cuda").float()

for _ in range(10):
    o = a * b + c

with nvtx.annotate('torch fma'):
    for _ in range(10):
        o = a * b + c
    torch.cuda.synchronize()

for _ in range(10):
    flash_out = flash_fusion.fma(a, b, c)


with nvtx.annotate('flash fma'):
    for _ in range(10):
        flash_out = flash_fusion.fma(a, b, c)

print(o, flash_out)

print(torch.allclose(c, flash_out, atol=1e-3, rtol=1e-3))
# 
diff = torch.abs(flash_out - o)
# print(diff)
# 
max_abs_error = torch.max(diff)  # 最大绝对误差: 0.02
max_rel_error = torch.max(diff / torch.abs(o))  # 最大相对误差: 0.01
# 
print("max abs error: ", max_abs_error, max_rel_error)
# print(max_abs_error > 1e-3)  # 输出: True（默认 atol=1e-8 不满足）
