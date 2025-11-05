import torch.nn as nn
import torch
import math
from einops import einsum, reduce, rearrange

def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, i: int):
    x_max = torch.max(x, keepdim=True, dim=i).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=i, keepdim=True)


# y = Wx
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        ### Construct a linear transformation module.
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        W = torch.empty((out_features , in_features), device=device, dtype=dtype)
        std = math.sqrt(2.0 / (in_features + out_features))

        min_cutoff = -3 * std
        max_cutoff = 3 * std
        
        # Normalize         N(mean=0, std = (2/ d_input_size + d_output_size))
        torch.nn.init.trunc_normal_(W, mean = 0, std = std, a = min_cutoff, b = max_cutoff)

        self.weight = nn.Parameter(W)

    def forward(self, x: torch.Tensor):
        y = einsum(self.weight, x, 'out_dim in_dim, ... in_dim -> ... out_dim')
        return y


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.num_embeddings = embedding_dim
        self.device = device
        self.dtype = dtype

        weight = torch.empty((num_embeddings , embedding_dim), dtype=dtype, device=device)

        std = math.sqrt(1)

        min_cutoff = -3
        max_cutoff = 3

        torch.nn.init.trunc_normal_(weight, mean = 0, std = std, a = min_cutoff, b = max_cutoff)

        self.weight = nn.Parameter(weight)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()

        self.eps = eps
        self.d_model = d_model

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(x ** 2, '... i -> ... ()', 'mean')) + self.eps

        result = x * self.weight / rms

        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device = None, dtype= None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)
        self.W3 = Linear(d_model, d_ff)

    def forward(self, x : torch.Tensor):
        l1 = self.W1(x)
        
        return self.W2(silu(l1) * self.W3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()

        self.d_k = d_k
        
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)
        freqs = torch.arange(0, d_k, 2, device=device) / d_k
        inv_freq = 1.0 / (theta ** freqs)

        angles = positions * inv_freq
        
        self.register_buffer('cos', angles.cos().to(dtype=None), persistent=False)
        self.register_buffer('sin', angles.sin().to(dtype=None), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_p = self.cos[token_positions]
        sin_p = self.sin[token_positions]

        x1,x2 = x[..., 0::2], x[..., 1::2]

        x_rot_even = x1 * cos_p - x2 * sin_p
        x_rot_odd = x1 * sin_p + x2 * cos_p

        x_rot = rearrange([x_rot_even, x_rot_odd], "b ... -> ... b")
        x_out = rearrange(x_rot, "... d1 d2 -> ... (d1 d2)")

        return x_out



if __name__ == '__main__':
    test = RotaryPositionalEmbedding(1000.0,10,10)
    l = test.forward(torch.arange(10), torch.arange(10))
    print(l)