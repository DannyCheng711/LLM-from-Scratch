import math

def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    # Warm-up
    if t < T_w:
        return (t / T_w) * alpha_max

    # Cosine annealing
    if t <= T_c:
        cosine_term = math.cos(
            ((t - T_w) / (T_c - T_w)) * math.pi
        )

        return alpha_min + 0.5 * (1 + cosine_term) * (alpha_max - alpha_min)

    # Post-annealing
    return alpha_min