import torch
from torch import nn
from einops import einsum, rearrange

class RoPE(nn.Module):
    """
    theta: float Θ value for the RoPE
    d_k: int dimension of query and key vectors
    max_seq_len: int Maximum sequence length that will be inputted
    device: torch.device | None = None Device to store the buffer on
    """
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()

        assert d_k % 2 == 0, "d_k must be divisible by 2"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # pair index: 0, 1, .... (d_k / 2 - 1)
        pair_idx = torch.arange(d_k // 2, device=device)
        # inverse frequency:  1 / theta^(2k / d_k)
        inv_freq = 1.0 / (theta ** (2 * pair_idx / d_k))
        # token position index: 0, 1, ..., max_seq_len - 1
        positions = torch.arange(max_seq_len, device=device)
        # angles [i, k] = i / theta^(2k/ d_k)
        angles = einsum(positions, inv_freq, "i, j -> i j")

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions.
        x shape: (..., seq_len, d_k)
        token_positions shape: (..., seq_len)
        output shape: (..., seq_len, d_k)
        """

        # (..., seq_len, d_k) -> # (..., seq_len, d_k / 2, 2)
        x_pair = rearrange(x, "... (pair two) -> ... pair two", two=2)
        x1 = x_pair[..., 0]
        x2 = x_pair[..., 1]

        # cos / sin shape: (..., seq_len, d_k / 2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # 2D rotation
        # [x1'] = [cos -sin][x1]
        # [x2'] = [sin  cos][x2]
        out1 = cos * x1 - sin * x2
        out2 = sin * x1 + cos * x2
        # (..., seq_len, d_k / 2, 2)
        out_pair = torch.stack([out1, out2], dim=-1)
        # (..., seq_len, d_k)
        out = rearrange(out_pair, "... pair two -> ... (pair two)")

        return out