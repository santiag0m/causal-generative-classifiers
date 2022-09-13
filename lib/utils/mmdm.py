from typing import Callable, Iterable

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
            params: Iterable[nn.Parameter],
            lr: float,
            damping: float = 10.0,
            lambda_lr: float = 1.0,
            model_optim: Callable = torch.optim.SGD,
            **kwargs
        ):

        device = next(params).device

        self.damping = damping
        self.lambda_ = Lambda().to(device)
        self.lambda_optim = torch.optim.SGD(self.lambda_.parameters(), lr=lambda_lr)
        self.model_optim = model_optim(params, lr=lr, **kwargs)

    def lagrangian(self, main_loss: torch.Tensor, constrained_loss: torch.Tensor, target_value: float) -> torch.Tensor:
        damp = self.damping * (target_value - constrained_loss.detach())
        return main_loss - (self.lambda_.weight - damp) * (target_value - constrained_loss)

    def zero_grad(self):
        self.lambda_optim.zero_grad()
        self.model_optim.zero_grad()

    def step(self):
        self.lambda_.weight.grad *= -1  # Gradient Ascent
        self.lambda_optim.step()
        self.model_optim.step()

        if self.lambda_.weight < 0:
            self.lambda_.weight = nn.Parameter(torch.tensor(0.))

    

    
