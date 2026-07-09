import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.01):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        # for closure template
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]

                state["step"] += 1

                t = state["step"]

                # Adam moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Adam update
                p.data.addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-lr)

                # Decoupled weight decay
                p.data.add_(p.data, alpha=-lr * weight_decay)

            return loss