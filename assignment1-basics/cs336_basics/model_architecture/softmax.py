import torch
from torch import nn

class Softmax(nn.Module):
    """
    dim: dimension along which softmax is applied
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable softmax
        """
        # keepdim=True keeps the reduced dimension with size 1
        x_shifted = x - torch.max(x, dim = self.dim, keepdim=True).values
        exp_x = torch.exp(x_shifted)
        return exp_x / exp_x.sum(dim=self.dim, keepdim=True)