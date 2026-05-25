import torch
from torch import nn

class MultiheadSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass