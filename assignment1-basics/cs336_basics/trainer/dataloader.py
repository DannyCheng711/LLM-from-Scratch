import numpy as np
import torch
from torch import dtype


def get_batch(x, batch_size, context_length, device):
    """
    Sample next-token prediction batches from a 1D token array.

    Args:
        x: Token IDs with shape (num_tokens,).
        batch_size: Number of sampled sequences.
        context_length: Number of tokens per sequence.
        device: Target PyTorch device, e.g. "cpu", "cuda:0", or "mps".

    Returns:
        inputs:  shape (batch_size, context_length)
        targets: shape (batch_size, context_length)
    """

    # high is excluded
    starts = np.random.randint(low = 0, high = len(x) - context_length, size = batch_size)

    # get inputs from starts
    inputs_np = np.stack([
        x[i : i + context_length] for i in starts
    ])
    targets_np = np.stack([
        x[i + 1: i + context_length + 1] for i in starts
    ])

    inputs = torch.as_tensor(inputs_np, dtype=torch.long, device=device)
    targets = torch.as_tensor(targets_np, dtype=torch.long, device=device)

    return inputs, targets
