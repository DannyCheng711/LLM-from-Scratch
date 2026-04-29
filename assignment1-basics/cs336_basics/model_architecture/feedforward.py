import torch
from torch import nn
from .linear import Linear


class SwiGLU(nn.Module):
    """
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu. (hidden layer)

        FFN(x)=W2(SiLU(W1x)⊙(W3x))

        x              : (..., d_model)
        w1(x)          : (..., d_ff)
        silu(w1(x))    : (..., d_ff)
        w3(x)          : (..., d_ff)
        element-wise   : (..., d_ff)
        w2(...)        : (..., d_model)
    """
    def __init__(self, d_model, d_ff, device=None, dtype=None):

        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=None, dtype=None) # d_in, d_out
        self.w2 = Linear(d_ff, d_model, device=None, dtype=None)
        self.w3 = Linear(d_model, d_ff, device=None, dtype=None)

    # silu
    def _silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.w2(self._silu(self.w1(x)) * self.w3(x))


def build_SwiGLU(d_model, d_ff, w1_weight, w2_weight, w3_weight, device=None, dtype=None):
    ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    ffn.w1.load_state_dict({"weight": w1_weight.to(device=device, dtype=dtype)})
    ffn.w2.load_state_dict({"weight": w2_weight.to(device=device, dtype=dtype)})
    ffn.w3.load_state_dict({"weight": w3_weight.to(device=device, dtype=dtype)})

    return ffn
