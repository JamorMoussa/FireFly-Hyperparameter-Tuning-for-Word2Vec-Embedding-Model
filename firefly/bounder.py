from typing import Union
from dataclasses import dataclass
import numpy as np

__all__ = ["FireFlyParameterBounder", ]

@dataclass
class FireFlyParameterBounder:
    bounds: list[tuple[Union[float, int]]]

    @staticmethod
    def get_defaults(dim=3):
        return FireFlyParameterBounder(bounds=[(-5, 5) for _ in range(dim)])

    def clip(self, value: Union[float, int] , lb, ub):
        if lb > value: return lb
        elif value > ub:
            return ub
        return value

    def apply(self, input: np.ndarray):
        lb = np.array(tuple(map(lambda item: item[0], self.bounds)))
        ub = np.array(tuple(map(lambda item: item[1], self.bounds)))

        return np.vectorize(self.clip)(input, lb, ub)