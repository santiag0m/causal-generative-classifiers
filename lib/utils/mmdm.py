from typing import Callable, Sequence

import torch
import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.))


class MMDMOptim:
    """
    Modified Method of Differential Multipliers based on:
    https://www.engraved.blog/how-we-can-make-machine-learning-algorithms-tunable/
    """
    def __init__(
            self,
            params: Sequence[nn.Parameter],
            lr: float,
            epsilon: float = 0,
            damping: float = 10.0,
            lambda_lr: float = 1.0,
            model_optim: Callable = torch.optim.SGD,
            **kwargs
        ):

        device = next(model.parameters()).device

        self.epsilon = epsilon
        self.damping = damping
        self.lambda_ = Lambda().to(device)
        self.lambda_optim = torch.optim.SGD(self.lambda_.parameters(), lr=lambda_lr)
        self.model_optim = model_optim(params, lr=lr, **kwargs)

    def lagrangian(self, main_loss: torch.Tensor, constrained_loss: torch.Tensor) -> torch.Tensor:
        damp = self.damping * (self.epsilon - constrained_loss.detach())
        return main_loss - (self.lambda_ - damp) * (self.epsilon - constrained_loss)

    def step(self):
        self.lambda_.weight.grad *= -1  # Gradient Ascent
        self.lambda_optim.step()
        self.model_optim.step()

        if self.lambda_.weight < 0:
            self.lambda_.weight = 0

    

    
