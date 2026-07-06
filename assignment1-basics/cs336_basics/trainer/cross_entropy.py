import torch

def cross_entropy(logits, targets):
    """
    logits shape: (..., vocab_size)
    targets shape: (...)
    return: scalar average loss (cross batches)
    """

    # 1. subtract max for numerical stability
    max_logits = logits.max(dim=-1, keepdim=True).values # shape (batch, seq, 1)
    shifted_logits = logits - max_logits # shape (batch, seq, vocab_size)

    # 2. compute stable log_sum_exp
    # shape: (batch, seq)
    log_sum_exp = torch.log(
        torch.exp(shifted_logits).sum(dim = -1 ) # sum along vocab -> shape (batch, seq)
    ) + max_logits.squeeze(-1)

    # 3. get the logit corresponding to the true target token
    target_logits = logits.gather(
        dim=-1, index=targets.unsqueeze(-1),
    ).squeeze(-1) # shape: (batch, seq)

    # 4. CrossEntropy = log_sum_exp(logits) - target_logits
    loss = log_sum_exp - target_logits

    # 5. average over all batch / sequence positions
    return loss.mean()