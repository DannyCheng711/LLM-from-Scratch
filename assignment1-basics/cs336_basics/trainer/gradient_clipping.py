from collections.abc import Iterable

import torch
from torch import nn

def get_gradient_clip(parameters, max_l2_norm, eps=1e-6):

    # model.parameters() is an iterator, so materialize it
    # because we need to iterate over it twice.
    parameters = list(parameters)

    grads = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]

    if not grads:
        return

    # Combined L2 norm:
    # sqrt(sum over all gradient elements of g_i^2)
    total_norm = torch.sqrt(
        sum(torch.sum(grad.detach() ** 2) for grad in grads)
    )

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)

        # Modify gradients
        for grad in grads:
            grad.mul_(scale)
