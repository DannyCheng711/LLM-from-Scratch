import torch

def top_p_filter(probs: torch.Tensor, top_p: float):
    """
    Apply top-p (nucleus) filtering.

    Args:
        probs: Probability distribution with shape (batch, vocab_size).
        top_p: Cumulative probability threshold in the interval (0, 1].
    Returns:
        Filtered and re-normalized probabilities with the same shape.
    """

    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1]")

    if top_p == 1.0:
        return probs

    # Sort tokens from highest to lowest probability
    sorted_probs, sorted_indices = torch.sort (
        probs, dim=-1, descending=True
    )

    # Compute cumulative probability in sorted order.
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1) # (batch, vocab_size)

    # Remove tokens after the cumulative probability exceeds top_p
    # the first token exceeding top_p is still kept.
    remove_mask = cumulative_probs > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone() # right shift by one position
    remove_mask[..., 0] = False # always keep the first element

    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)

    # Re-normalize the retained probabilities so that they sum to 1
    sorted_probs = sorted_probs / sorted_probs.sum(dim= -1, keepdim=True)

    # Restore the original vocabulary order.
    filtered_probs = torch.zeros_like(probs)
    filtered_probs = filtered_probs.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_probs,
    )

    return filtered_probs




@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
):

    """
    Args:
        input_ids: Token IDs with shape (1, prompt_length).
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_id: Token ID corresponding to <|endoftext|>.
        temperature: Temperature used before softmax.
        top_p: Nucleus sampling threshold.

    Returns:
        Tensor containing prompt and generated tokens,
        with shape (1, prompt_length + generated_length).
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            "input_ids must have shape (1, sequence_length)"
        )

    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one token")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1]")

    was_training = model.training
    model.eval()
    generated = input_ids

    try:

        for _ in range(max_new_tokens):

            # Keep only tokens that fit within the model context window.
            context_length = getattr(
                model,
                "context_length",
                generated.shape[1],
            )

            model_input = generated[:, -context_length:]

            # logits shape: (1, current_seq_len, vocab_size)
            logits = model(model_input)

            # Use only the final position to predict the next token.
            # The final position has attended to the entire prompt and is used to predict the next token.
            next_token_logits = logits[:, -1, :]

            # Temperature scaling is applied before softmax.
            next_token_logits = next_token_logits / temperature

            # Convert logits into a probability distribution.
            probs = torch.softmax(next_token_logits, dim=-1)

            # Keep only the smallest set of likely tokens whose
            # cumulative probability reaches top_p.
            probs = top_p_filter(probs, top_p)

            # Sample one token from the filtered distribution.
            next_token = torch.multinomial(
                probs,  num_samples=1,
            )

            # Append the sampled token to the generated sequence.
            generated = torch.cat(
                [generated, next_token],
                dim=-1,
            )

            if next_token.item() == eos_token_id:
                break

    finally:
        if was_training:
            model.train()

    return generated