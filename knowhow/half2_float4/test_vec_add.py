import torch
import flash_fusion
import nvtx

a = torch.randn([4096], device="cuda").half()
b = torch.randn([4096], device="cuda").half()


# wareup
for _ in range(10):
    c = a + b

c = 0
with nvtx.annotate('torch add'):
    for _ in range(100):
        c = a + b
    torch.cuda.synchronize()

print(c)

for _ in range(10):
    fusion_out = flash_fusion.vec_add(a, b)

with nvtx.annotate('flashfusion add'):
    for _ in range(100):
        fusion_out = flash_fusion.vec_add(a, b)
print(fusion_out)

print(torch.allclose(c, fusion_out))
