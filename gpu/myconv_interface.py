import torch
import time
from torch.utils.cpp_extension import load


# Compile and load CUDA extension
conv_module = load(name="myconv",
                     sources=["myconv_kernel.cu"],
                     verbose=True)

# Input parameters
#N, C_in, H, W = 2, 3, 33, 33
N, C_in, H, W = 2, 3, 50, 50
C_out, KH, KW = 8, 7, 7
stride, pad = 1, 1

# Allocate tensors
x = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.float32)
w = torch.randn(C_out, C_in, KH, KW, device="cuda", dtype=torch.float32)

# Reference solution (PyTorch)
print('torch.nn.functional.conv2d: begin')
# print(x, w)
out_ref = torch.nn.functional.conv2d(x, w, stride=stride, padding=pad)
print('torch.nn.functional.conv2d: complete')

# Run o4 kernel
print('conv_cuda: begin')
start_time = time.perf_counter()
out_custom = conv_module.conv_cuda(x, w, stride, pad)
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f'conv_cuda: complete {execution_time:.6f}')

# Test shape and correctness
print("CUDA --- shape check:", out_custom.shape == out_ref.shape)
print("CUDA --- correctness check:", torch.allclose(out_custom, out_ref, atol=1e-4))
