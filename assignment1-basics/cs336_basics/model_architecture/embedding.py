import torch
from torch import nn
from einops import einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module.
        This function should accept the following parameters:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., d_model
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        #  1. Build empty embedding (token -> d_model)
        #  embedding.shape = (vocab, d_model)
        weight = torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype)

        # 2. Initialize
        # ~ N(0, 1) → truncated to [-3, 3]
        # σ² = 1
        # σ = 1
        nn.init.trunc_normal_(weight, mean=0, std=1, a=-3, b=3)

        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return self.weight[token_ids] # look up embedding

def build_embedding(num_embeddings, embedding_dim, weights=None, device=None, dtype=None):
    embedding = Embedding(num_embeddings, embedding_dim, device=device, dtype=dtype)

    if weights is not None:
        embedding.load_state_dict({
            "weight": weights.to(device=device, dtype=dtype)
        })

    return embedding