import torch
from torch import nn
from einops import einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__() # call the superclass constructor


        self.eps = eps
        # learnable weight (g_i)
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype))



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS function
        """

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        return (x / rms) * self.weight

def build_rmsnorm(d_model, eps, weights=None, device=None, dtype=None):
    """
        d_model: int,
        eps: float,
        weights: Float[Tensor, " d_model"],
        in_features: Float[Tensor, " ... d_model"],
    """

    rmsnorm = RMSNorm(d_model, eps, device=device, dtype=dtype)

    if weights is not None:
        rmsnorm.load_state_dict({
            "weight": weights.to(device=device, dtype=dtype)
        })

    return rmsnorm