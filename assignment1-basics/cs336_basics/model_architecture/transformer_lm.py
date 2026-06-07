import torch
from torch import nn

from tests.conftest import vocab_size
from .rmsnorm import RMSNorm
from .transfomer_block import TransformerBlock
from .embedding import Embedding
from .linear import Linear


class TransformerLM(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention.
    d_ff: int Dimensionality of the position-wise feed-forward inner layer.
    vocab_size: embedding matrix.
        int The size of the vocabulary, necessary for determining the dimensionality of the token
    context_length:
        int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
    num_layers: int The number of Transformer blocks to use
    """
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta,
                 vocab_size, context_length, num_layers, device=None, dtype=None):

        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        # embedding
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # multiple transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, max_seq_len, theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])


        # normalization (layer norm)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # Linear
        self.lm_head = Linear(
            in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids shape: (batch, seq_len)
        return logits shape: (batch, seq_len, vocab_size)
        """

        seq_len = token_ids.shape[-1]
        token_positions = torch.arange(seq_len).unsqueeze(0)

        x = self.embedding(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits


def build_transformer_lm(
    vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights):
    lm = TransformerLM(
        vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads,
        d_ff=d_ff, max_seq_len=context_length, theta=rope_theta
    )

    lm.embedding.load_state_dict({
        "weight": weights["token_embeddings.weight"]
    })

    for i in range(num_layers):
        layer = lm.layers[i]
        prefix = f"layers.{i}"

        layer.attn.q_proj.load_state_dict({
            "weight": weights[f"{prefix}.attn.q_proj.weight"]
        })
        layer.attn.k_proj.load_state_dict({
            "weight": weights[f"{prefix}.attn.k_proj.weight"]
        })
        layer.attn.v_proj.load_state_dict({
            "weight": weights[f"{prefix}.attn.v_proj.weight"]
        })
        layer.attn.output_proj.load_state_dict({
            "weight": weights[f"{prefix}.attn.output_proj.weight"]
        })

        layer.attn_norm.load_state_dict({
            "weight": weights[f"{prefix}.ln1.weight"]
        })
        layer.ffn_norm.load_state_dict({
            "weight": weights[f"{prefix}.ln2.weight"]
        })

        layer.ffn.w1.load_state_dict({
            "weight": weights[f"{prefix}.ffn.w1.weight"]
        })
        layer.ffn.w2.load_state_dict({
            "weight": weights[f"{prefix}.ffn.w2.weight"]
        })
        layer.ffn.w3.load_state_dict({
            "weight": weights[f"{prefix}.ffn.w3.weight"]
        })

    lm.ln_final.load_state_dict({
        "weight": weights["ln_final.weight"]
    })

    lm.lm_head.load_state_dict({
        "weight": weights["lm_head.weight"]
    })

    return lm
