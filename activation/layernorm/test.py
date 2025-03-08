from pytorch_module import Model
import flow
import torch
import nvtx
import flash_fusion

# 2, 9216, 640
# 2, 2304, 1280
# 4096, 4096

batch_size = 2
features = 9216 * 640
dim1 = 640
# dim2 = 256

def get_inputs():
    # x = torch.randn(batch_size, features, dim1, dim2).cuda()
    # x = torch.randn(batch_size, features, dim1).cuda()
    x = torch.randn(batch_size, features).float().cuda()
    return [x]

def get_init_inputs():
    # return [(features, dim1, dim2)]
    # return [(features, dim1,)]
    return [(features,)]
it = 100

# pytorch
xs = [get_inputs()[0] for _ in range(it)]
norm_shape = get_init_inputs()[0]
layer_norm = Model(norm_shape).float().cuda()

torch_os = []
for _ in range(10):
    torch_out = layer_norm(xs[0])

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
  with nvtx.annotate('pytorch norm'):
      torch_os.append(layer_norm(xs[_]))
      # torch.cuda.synchronize()
end.record()
torch.cuda.synchronize()

print("avg pytorch layernorm sync time: ", start.elapsed_time(end) / 100, " ms.")

torch_compile_os = []
# torch compile
layer_norm.compile_forward(xs[0])

compile_start = torch.cuda.Event(enable_timing=True)
compile_end = torch.cuda.Event(enable_timing=True)

compile_start.record()
for _ in range(100):
  with nvtx.annotate('pytorch norm'):
      torch_compile_os.append(layer_norm.compile_forward(xs[_]))
compile_end.record()
torch.cuda.synchronize()

print("avg pytorch compile layernorm sync time: ", compile_start.elapsed_time(compile_end) / 100, " ms.")

## for _ in range(it):
##   if not torch.allclose(torch_os[_], torch_compile_os[_]):
##      print("compile out error")
##      print(torch_os[_], torch_compile_os[_])

# print(x.size(), layer_norm.ln.weight.size(), layer_norm.ln.bias.size())
# torch.Size([16, 64, 256, 256]) torch.Size([64, 256, 256]) torch.Size([64, 256, 256])


# AI engine.cu
# ai_engine_out = flow.ai_engine_layer_norm(xs[0], layer_norm.ln.weight, layer_norm.ln.bias)
# print("AI enine: ", torch.allclose(ai_engine_out, flow_os[0]))

## # apx norm
## for _ in range(10):
##    apx_out = flow.apx_layernorm(xs[_], norm_shape)
## 
## apx_os = []
## astart = torch.cuda.Event(enable_timing=True)
## aend = torch.cuda.Event(enable_timing=True)
## 
## astart.record()
## for _ in range(it):
##   with nvtx.annotate('apx norm'):
##        apx_os.append(flow.apx_layernorm(xs[_], norm_shape))
## 
## aend.record()
## torch.cuda.synchronize()
## 
## print("avg apx layernorm sync time: ", astart.elapsed_time(aend) / 100, " ms.")


## # print(torch_out, apx_out)
## for _ in range(it):
##     if not torch.allclose(torch_os[_], apx_os[_]):
##         print("apx out error")

#### ------------
# lighset

## ln_res = torch.zeros_like(torch_os[0])
## vars = torch.rand(batch_size)
## means = torch.rand(batch_size)
## scale = layer_norm.ln.weight
## bias = layer_norm.ln.bias
## hidden_dim = 1
## for s in norm_shape:
##   hidden_dim = hidden_dim * s
## flow.ls_layer_norm_fp32(ln_res, vars, means, xs[0], scale, bias, batch_size, hidden_dim, False)
## 
## torch.cuda.synchronize()
## 
## print(ln_res, torch_os[0])

### flash layernorm
print("w:", layer_norm.ln.weight.size(), layer_norm.ln.weight.dtype)
print("sum and mean ", torch.sum(xs[0]), torch.mean(xs[0]))
for _ in range(10):
    flash_out = flow.flash_layernorm(xs[0], layer_norm.ln.weight, layer_norm.ln.bias)

flash_start = torch.cuda.Event(enable_timing=True)
flash_end = torch.cuda.Event(enable_timing=True)

flash_start.record()

for _ in range(100):
    flash_out = flow.flash_layernorm(xs[0], layer_norm.ln.weight, layer_norm.ln.bias)

flash_end.record()
torch.cuda.synchronize()

print("avg flash layernorm sync time: ", flash_start.elapsed_time(flash_end) / 100, " ms.")

print(flash_out, torch_os[0], flash_out.size(), torch_os[0].size())
print(torch.allclose(torch_os[0], flash_out, atol=1e-2, rtol=1e-2))

max_bias = (abs(torch_os[0] - flash_out)).max()
print("max bias: ", max_bias)

##  # 计算吞吐量
##  flop = features * 4  # 计算量（FLOP）
##  time_used = flash_start.elapsed_time(flash_end)
##  tflops = flop * 100 / time_used / 1e12  # TFLOPS
##  print(f"Throughput: {tflops:.4f} TFLOPS")

print(xs[0])
# infini flash fusion
for _ in range(10):
    infini_out = flash_fusion.layernorm(xs[0], layer_norm.ln.weight, layer_norm.ln.bias, 1e-5)

infini_start = torch.cuda.Event(enable_timing=True)
infini_end = torch.cuda.Event(enable_timing=True)

infini_start.record()
for _ in range(100):
    infini_out = flash_fusion.layernorm(xs[0], layer_norm.ln.weight, layer_norm.ln.bias, 1e-5)

infini_end.record()
torch.cuda.synchronize()

print("avg infini layernorm sync time: ", infini_start.elapsed_time(infini_end) / 100, " ms.")

print(infini_out, torch_os[0])
print(torch.allclose(torch_os[0], infini_out, atol=1e-2, rtol=1e-2))

max_bias = (abs(torch_os[0] - infini_out)).max()
print("max bias: ", max_bias)


