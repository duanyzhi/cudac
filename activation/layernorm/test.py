from pytorch_module import Model
import flow
import torch

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]

# pytorch
x = get_inputs()[0]
layer_norm = Model(get_init_inputs()[0]).cuda()
torch_out = layer_norm(x)

flow_out = flow.forward(x, layer_norm.ln.weight, layer_norm.ln.bias)
print(torch.allclose(torch_out, flow_out))
