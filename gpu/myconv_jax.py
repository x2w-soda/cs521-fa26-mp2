import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from myconv import ConvModel
import jax.profiler

# Create a log directory
logdir = "./jax_trace"

def im2col_manual_jax(x, KH, KW, S, P, out_h, out_w):
    ''' 
        Reimplement the same function (im2col_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
    '''
    # x: (N, C, H, W)
    N, C, H, W = x.shape

    # (N, C, H + 2P, W + 2P)
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (P, P), (P, P)))

    H_pad = H + 2 * P
    W_pad = W + 2 * P

    row_idx = jnp.arange(out_h) * S
    col_idx = jnp.arange(out_w) * S

    # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    # cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)
    rr, cc = jnp.meshgrid(jnp.arange(KH), jnp.arange(KW), indexing="ij")

    # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
    rows = (row_idx[:, None, None, None] + rr[None, None, :, :])
    rows = jnp.broadcast_to(rows, (out_h, out_w, KH, KW))

    # TO DO: 3) perform tiled matmul after required reshaping is done.
    cols = (col_idx[None, :, None, None] + cc[None, None, :, :])
    cols = jnp.broadcast_to(cols, (out_h, out_w, KH, KW))

    # TO DO: 4) Add bias.
    linear_idx = rows * W_pad + cols
    linear_idx = linear_idx.reshape(out_h * out_w, KH * KW)

    # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
    x_flat = x_pad.reshape(N, C, H_pad * W_pad)


    # (N, C, out_h*out_w, KH*KW)
    out = x_flat[:, :, linear_idx]
    #print(out)
    out = jnp.transpose(out, (0, 2, 1, 3))
    out = out.reshape(N, out_h * out_w, C * KH * KW)
    #print(out)

    return out


def conv2d_manual_jax(x, weight, bias, stride=1, padding=1):
    '''
        Reimplement the same function (conv2d_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
        Hint: Unlike PyTorch, JAX arrays are immutable, so you cannot do indexing like out[i:j, :] = ... inside a JIT. You may use .at[].set() instead.
    '''
    N, C, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    out_h = (H + 2 * padding - KH) // stride + 1
    out_w = (W + 2 * padding - KW) // stride + 1

    # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)

    # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
    weight_flat = weight.reshape(C_out, C * KH * KW)

    # TO DO: 3) perform tiled matmul after required reshaping is done.
    tile_size = 64
    outputs = []

    for start in range(0, C_out, tile_size):
        end = min(start + tile_size, C_out)

        weight_tile = weight_flat[start:end]
        tile_output = jnp.matmul(cols, weight_tile.T)

        outputs.append(tile_output)

    # TO DO: 4) Add bias.
    out = jnp.concatenate(outputs, axis=2)
    out = out + bias[None, None, :]
    #print(out)

    # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
    out = out.reshape(N, out_h, out_w, C_out)
    out = jnp.transpose(out, (0, 3, 1, 2))
    #print(out)

    return out

if __name__ == "__main__":
    # Instantiate PyTorch model
    H, W = 50, 50
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=7, stride=1, padding=1)
    model.eval()

    # Example input
    x_torch = torch.randn(2, 3, H, W)

    # Export weights and biases
    params = {
        "weight": model.weight.detach().cpu().numpy(),  # shape (out_channels, in_channels, KH, KW)
        "bias": model.bias.detach().cpu().numpy()       # shape (out_channels,)
    }

    # Convert model input, weights and bias into jax arrays
    x_jax = jnp.array(x_torch.numpy())
    weight_jax = jnp.array(params["weight"])
    bias_jax = jnp.array(params["bias"])

    jax.profiler.start_trace(f"/tmp/tensorboard", create_perfetto_trace=True)
    with jax.profiler.TraceAnnotation("myconv_jax"):
        # enable JIT compilation
        conv2d_manual_jax_jit = jit(conv2d_manual_jax)

        # call your JAX function
        out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)
    jax.profiler.stop_trace()
    
    # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=1, padding=1)
    print("JAX --- shape check:", out_jax.shape == conv_ref.shape)
    print("JAX --- correctness check:", torch.allclose(torch.from_numpy(np.array(out_jax)), conv_ref, atol=1e-1))
