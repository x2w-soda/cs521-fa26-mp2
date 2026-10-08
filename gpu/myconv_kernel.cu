#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// example
#define TILE_H 8   
#define TILE_W 8   
#define TILE_C 16  

// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float* __restrict__ x,       // input: N x C x H x W
    const float* __restrict__ w,       // weights: C_out x C_in x KH x KW
    float* __restrict__ out,           // output: N x C x H x W
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int pad,
    int out_h, int out_w
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n = blockIdx.z;
    const int out_y = blockIdx.y * TILE_H + ty;
    const int out_x = blockIdx.x * TILE_W + tx;
    const int SH_H = TILE_H * stride + KH - 1;
    const int SH_W = TILE_W * stride + KW - 1;

    // There's no 'heap-allocate' on the GPU so
    // the shared mem is dynamically provided by host it seems.
    extern __shared__ float shmem[];

    const int threads_per_block = blockDim.x * blockDim.y;
    const int tid = ty * blockDim.x + tx;

    for (int co = 0; co < C_out; ++co) {
        float acc = 0.0f; // accumulate in each channel

        // tiling
        for (int c_base = 0; c_base < C_in; c_base += TILE_C) {
            const int channels_this_tile = min(TILE_C, C_in - c_base);
            const int num_elements = channels_this_tile * SH_H * SH_W;

            for (int idx = tid; idx < num_elements; idx += threads_per_block) {
                int c_local = idx / (SH_H * SH_W);
                int remainder = idx % (SH_H * SH_W);
                int local_y = remainder / SH_W;
                int local_x = remainder % SH_W;
                
                int global_c = c_base + c_local;
                int global_y = blockIdx.y * TILE_H * stride + local_y - pad;
                int global_x = blockIdx.x * TILE_W * stride + local_x - pad;

                float value = 0.0f;
                if (global_y >= 0 && global_y < H && global_x >= 0 && global_x < W) {
                    int input_idx = ((n * C_in + global_c) * H + global_y) * W + global_x;
                    value = x[input_idx];
                }

                shmem[(c_local * SH_H + local_y) * SH_W + local_x] = value;
            }

            // should synchronize all stores to shmem
            __syncthreads();

            if (out_y < out_h && out_x < out_w) {
                for (int c_local = 0; c_local < channels_this_tile; ++c_local) {
                    int global_c = c_base + c_local;

                    for (int ky = 0; ky < KH; ++ky) {
                        for (int kx = 0; kx < KW; ++kx) {
                            int sh_y = ty * stride + ky;
                            int sh_x = tx * stride + kx;

                            // float value = shmem[(c_local * SH_H + sh_y) * SH_W + sh_x];
                            float value = shmem[(c_local * SH_H + sh_y) * SH_W + sh_x];
                            int weight_idx = ((co * C_in + global_c) * KH + ky) * KW + kx;

                            acc += value * w[weight_idx];
                        }
                    }
                }
            }

            // should synchronize all loads from shmem
            __syncthreads();
        }

        if (out_y < out_h && out_x < out_w) {
            int output_idx = ((n * C_out + co) * out_h + out_y) * out_w + out_x;
            out[output_idx] = acc;
        }
    }
}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w, int stride, int pad) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = w.size(0);
    int KH = w.size(2);
    int KW = w.size(3);

    int out_h = (H + 2 * pad - KH) / stride + 1;
    int out_w = (W + 2 * pad - KW) / stride + 1;

    auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());

    dim3 block(8, 8);
    dim3 grid((out_w + block.x - 1)/block.x,
              (out_h + block.y - 1)/block.y,
              N);

    // allocate shared mem for tiling
    int shmemSize = TILE_C * (TILE_H + KH - 1) * (TILE_W + KW - 1);

    gemm_gpu_o4_kernel<<<grid, block, shmemSize>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, KH, KW,
        stride, pad,
        out_h, out_w);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}