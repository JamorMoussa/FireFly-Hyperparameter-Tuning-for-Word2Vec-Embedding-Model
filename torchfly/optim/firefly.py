from torch.optim import Optimizer
import torch
import torch.nn as nn
from typing import Iterator, Callable, overload


__all__ = ["FireFlyOptimizer", ]

class FireFlyOptimizer(Optimizer):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.modules.loss._Loss,
        *,
        n_fireflies: int = 20,
        alpha: float = 0.5,
        beta0: float = 1.0,
        gamma: float = 1.0,

    ) -> None:
        
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        
        self.model: nn.Module = model
        self.loss: nn.modules.loss._Loss = loss

        self.fireflies: torch.Tensor
        self.intensities: torch.Tensor

        self.best_fitness = float('inf')
        
        defaults = dict(n_fireflies=n_fireflies, alpha=alpha, beta0=beta0, gamma=gamma)
        
        super(FireFlyOptimizer, self).__init__(model.parameters(), defaults)

    def fitness(self, weight: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.model[0].weight.data = torch.tensor(weight.clone().detach().reshape(1, -1))
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        return loss

    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        for group in self.param_groups:
            for i, p in enumerate(tuple(group['params'])):

                if i>= 1: continue
                
                self.fireflies = torch.randn((group['n_fireflies'], p.t().shape[0]))
                self.intensities = torch.zeros(group['n_fireflies'])

                for i in range(group["n_fireflies"]):
                    self.intensities[i] = self.fitness(self.fireflies[i], inputs, targets)

                min_idx = torch.argmin(self.intensities)
                if self.intensities[min_idx] < self.best_fitness:
                    self.best_solution = self.fireflies[min_idx].clone()
                    self.best_fitness = self.intensities[min_idx]

                for i in range(group['n_fireflies']):
                                for j in range(group['n_fireflies']):
                                    if self.intensities[j] < self.intensities[i]:
                                        r = torch.norm(self.fireflies[i] - self.fireflies[j])
                                        beta = group['beta0'] * torch.exp(-group['gamma'] * r**2)
                                        self.fireflies[i] += beta * (self.fireflies[j] - self.fireflies[i]) + group['alpha'] * (torch.rand(p.t().shape[0])) - 0.5



        self.model[0].weight.data = self.best_solution



