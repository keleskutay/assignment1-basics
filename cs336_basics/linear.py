import torch.nn as nn
import torch
import math

# y = Wx
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        ### Construct a linear transformation module.
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        W = torch.empty((in_features + out_features), device=device dtype=dtype)
        std = math.sqrt(2.0 / (in_features + out_features))

        min_cutoff = -3 * std
        max_cutoff = 3 * std
        
        # Normalize         N(mean=0, std = (2/ d_input_size + d_output_size))
        torch.nn.init.trunc_normal_(W, mean = 0, std = std, a = min_cutoff, b = max_cutoff)

        self.weight = nn.Parameter(W)

    def forward(self, x: torch.Tensor):
        pass


if __name__ == '__main__':
    test = Linear(1,2)
    print(test.weight)