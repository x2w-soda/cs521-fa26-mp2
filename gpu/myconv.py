import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

class ConvModel(nn.Module):
    def __init__(self, H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.H = H
        self.W = W

        # TO DO: Define static shapes here. 
        self.tile_size = 64
        # Precompute output size
        self.out_h = (H + 2 * padding - kernel_size) // stride + 1
        self.out_w = (W + 2 * padding - kernel_size) // stride + 1

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def im2col_manual(self, x):
        N = x.shape[0]
        C = self.in_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        out_h = self.out_h
        out_w = self.out_w

        # Pad input
        x_pad = F.pad(x, (P, P, P, P))
        H_pad = x_pad.shape[2]
        W_pad = x_pad.shape[3]

        # Starting row/column for each output location
        row_idx = torch.arange(out_h, device=x.device, dtype=torch.long) * S
        col_idx = torch.arange(out_w, device=x.device, dtype=torch.long) * S
        # Kernel offsets
        rr, cc = torch.meshgrid(torch.arange(KH, device=x.device), torch.arange(KW, device=x.device), indexing="ij")

        # (out_h, out_w, KH, KW)
        rows = (row_idx[:, None, None, None] + rr[None, None, :, :]).expand(out_h, out_w, KH, KW)
        # (out_h, out_w, KH, KW)
        cols = (col_idx[None, :, None, None] + cc[None, None, :, :] ).expand(out_h, out_w, KH, KW)

        # (out_h*out_w, KH*KW)
        linear_idx = rows * W_pad + cols
        linear_idx = linear_idx.reshape(out_h * out_w, KH * KW)

        # (N, C, H_pad*W_pad) input
        x_flat = x_pad.reshape(N, C, H_pad * W_pad)

        # (N, C, out_h*out_w, KH*KW) kernel
        patches = x_flat[:, :, linear_idx]
        #print(patches)

        # return (N, out_h*out_w, C*KH*KW) patch
        patches = patches.permute(0, 2, 1, 3)
        patches = patches.reshape(N, out_h * out_w, C * KH * KW)

        return patches

    def conv2d_manual(self, x):
        N = x.shape[0]
        C_out = self.out_channels
        C = self.in_channels
        KH = KW = self.kernel_size

        # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        cols = self.im2col_manual(x)          

        # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
        weight_flat = self.weight.reshape(C_out, C * KH * KW)

        # TO DO: 3) perform tiled matmul after required reshaping is done.
        # For each batch:
        #
        #   cols[n]       : (num_pixels, K)
        #   weight_flat.T : (K, C_out)
        #
        # We calculate:
        #
        #   cols[n] @ weight_flat.T
        num_pixels = self.out_h * self.out_w
        K = C * KH * KW

        output = torch.empty(N, num_pixels, C_out, device=x.device, dtype=x.dtype)
        tile = self.tile_size

        for start in range(0, C_out, tile):
            end = min(start + tile, C_out)

            # (tile, K) weight_tile
            weight_tile = weight_flat[start:end]
            weight_tile_t = weight_tile.t().unsqueeze(0)
            weight_tile_t = weight_tile_t.expand(N, K, end - start)
            output[:, :, start:end] = torch.bmm(cols, weight_tile_t)


        # TO DO: 4) Add bias.
        output = output + self.bias.reshape(1, 1, C_out)
        #print(output)


        # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
        output = output.reshape(N, self.out_h, self.out_w, C_out)
        return output.permute(0, 3, 1, 2) # (N, C_out, out_h, out_w)

    def forward(self, x):
        return self.conv2d_manual(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, H, W = 2, 3, 33, 33
    x = torch.randn(N, C, H, W)
    out_channels=8
    kernel_size=4

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("pytorch"):
            model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1)
            out = model(x)
    prof.export_chrome_trace("/tmp/trace_pytorch.json")
        
    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("PyTorch --- shape check:", out.shape == conv_ref.shape)
    print("PyTorch --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))