from dataclasses import dataclass

__all__ = ["FireFlyConfig", ]

@dataclass
class FireFlyConfig:

    pop_size: int
    alpha: float
    beta0: float
    gamma: float
    max_iters: int
    seed: int = None
    
    @staticmethod
    def get_defaults():
        return FireFlyConfig(pop_size=20, alpha=1.0, beta0=1.0, gamma=0.01, max_iters=100, seed=None)

    def to_dict(self,):
        return dict(pop_size=self.pop_size, alpha=self.alpha, beta0=self.beta0, gamma=self.gamma, max_iters=self.max_iters)