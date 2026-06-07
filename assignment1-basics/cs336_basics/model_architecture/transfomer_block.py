import torch
from torch import nn, Tensor

from .rmsnorm import RMSNorm
from .multihead_self_attention import MultiheadSelfAttention
from .feedforward import build_SwiGLU, SwiGLU

class TransformerBlock(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention.
    d_ff: int Dimensionality of the position-wise feed-forward inner layer.
    """
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()

        self.attn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta,
            device=device, dtype=dtype,
        )

        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(
            d_model=d_model, d_ff=d_ff, device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y = x + self.attn(self.attn_norm(x), token_positions)
        z = y + self.ffn(self.ffn_norm(y))
        return z


def build_transformer_block(
        d_model: int, num_heads: int,  d_ff: int,
        max_seq_len: int, theta: float, weights: dict[str, Tensor]):

    block = TransformerBlock(
        d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, theta=theta
    )

    block.attn.q_proj.load_state_dict({"weight": weights["attn.q_proj.weight"]})
    block.attn.k_proj.load_state_dict({"weight": weights["attn.k_proj.weight"]})
    block.attn.v_proj.load_state_dict({"weight": weights["attn.v_proj.weight"]})
    block.attn.output_proj.load_state_dict({"weight": weights["attn.output_proj.weight"]})

    block.attn_norm.load_state_dict({"weight": weights["ln1.weight"]})
    block.ffn_norm.load_state_dict({"weight": weights["ln2.weight"]})

    block.ffn.w1.load_state_dict({"weight": weights["ffn.w1.weight"]})
    block.ffn.w2.load_state_dict({"weight": weights["ffn.w2.weight"]})
    block.ffn.w3.load_state_dict({"weight": weights["ffn.w3.weight"]})

    return block