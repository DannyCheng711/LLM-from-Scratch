import math
import torch
from einops import einsum

def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Q, K shape: (batch_size, ..., seq_len, d_k)
    V shape:    (batch_size, ..., seq_len, d_v)
    mask shape: (seq_len, seq_len)
    """

    d_k = Q.shape[-1]

    # scores: (batch_size, ... , query_seq_len)
    scores = einsum(
        Q, K, "... q d_k, ... k d_k -> ... q k",
    ) / math.sqrt(d_k)

    # mask invalid attention positions (e.g., prevent looking ahead)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # softmax over keys
    attn = torch.softmax(scores, dim=-1)

    # output: (batch_size, ... , query_seq_len, d_v)
    # q, k are both token idx, so k d_v is token_idx d_v
    output = einsum(
        attn, V, "... q k, ... k d_v -> ... q d_v",
    )

    return output