import torch
from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Construct a linear transformation module.
        This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__() # call the superclass constructor

        # 1. Build empty weight
        # W.shape = (out, in)
        W = torch.empty(
            out_features, in_features, device=device, dtype=dtype)

        # 2. Initialize
        # ~ N(0, 2 / (d_in + d_out)) → truncated to [-3σ, 3σ]
        # σ² = 2 / (d_in + d_out)
        # σ = sqrt(2 / (d_in + d_out))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(W, mean=0, std= std, a=-3 * std, b=3 * std)

        self.W = nn.Parameter(W)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        # y = x @ W.T
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


def build_linear(in_features, out_features, weights=None, device=None, dtype=None):

    linear = Linear(in_features, out_features, device=device, dtype=dtype)

    if weights is not None:
        linear.load_state_dict({
            "W": weights.to(device=device, dtype=dtype)
        })

    return linear