import torch
from torch import nn
from einops import rearrange

from .linear import Linear
from .dot_product_attention import scaled_dot_product_attention
from .rope import RoPE

class MultiheadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # These are W_Q, W_K, W_V, W_O
        # Each projection outputs all heads at once:
        # (..., seq_len, d_model) -> (..., seq_len, num_heads * d_k)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Output projection W_O
        # concat heads: (..., seq_len, num_heads * d_v) -> (..., seq_len, d_model)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = (
                RoPE(theta=theta, d_k = self.d_k, max_seq_len=max_seq_len, device=device)
            )


    # x shape: (..., seq_len, d_model)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply causal multihead self-attention.
        """

        seq_len = x.shape[-2]

        # 1. Project x into Q, K, V.
        # q, k, v shape: (..., seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Split into heads.
        # (..., seq_len, d_model) -> (..., num_heads, seq_len, d_k)
        q = rearrange(q, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        k = rearrange(k, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        v = rearrange(v, "... seq (head d_v) -> ... head seq d_v", head=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)

            # token_positions: (..., seq)
            # q/k: (..., head, seq, d)
            # add a head-broadcast dimension
            token_positions = token_positions.unsqueeze(-2)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 3. Build causal mask.
        # mask[i, j] = True means token i can attend to token j.
        # Causal rule: token i can only attend to positions j <= i.
        # 保留主對角線以下（lower triangle），其他設 0。
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        # 4. Apply scaled dot-product attention.
        out = scaled_dot_product_attention(q, k, v, mask)

        # 5. Merge heads back.
        out = rearrange(out, "... head seq d_v -> ... seq (head d_v)")

        return self.output_proj(out)


def build_multihead_self_attention(
    d_model, num_heads,
    q_weight=None, k_weight=None, v_weight=None, o_weight=None,
    max_seq_len=None, theta=None, device=None, dtype=None,
):

    model = MultiheadSelfAttention(
        d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len,
        theta=theta, device=device, dtype=dtype,
    )

    if q_weight is not None:
        model.q_proj.load_state_dict({"weight": q_weight})

    if k_weight is not None:
        model.k_proj.load_state_dict({"weight": k_weight})

    if v_weight is not None:
        model.v_proj.load_state_dict({"weight": v_weight})

    if o_weight is not None:
        model.output_proj.load_state_dict({"weight": o_weight})

    return model