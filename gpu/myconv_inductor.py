import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel

if __name__ == "__main__":
    torch.manual_seed(0)

    # Instantiate your PyTorch model
    N, C, H, W = 2, 3, 33, 33
    x = torch.randn(N, C, H, W).cuda()
    
    model = ConvModel(H, W, in_channels=C, out_channels=8, kernel_size=4, stride=1, padding=1).cuda().eval()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("pytorch"):
            # Torch-Inductor compilation
            scripted_model = torch.compile(model, backend="inductor")
            out = scripted_model(x)
    prof.export_chrome_trace("/tmp/trace_inductor.json")
    
    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("Inductor --- shape check:", out.shape == conv_ref.shape)
    print("Inductor --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))