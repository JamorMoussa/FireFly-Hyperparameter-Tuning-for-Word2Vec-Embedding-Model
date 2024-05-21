from .base import FireFlyBase
from firefly import FireFlyConfig, FireFlyParameterBounder
import numpy as np
from typing import Callable, Any

__all__ = ["FireFlyOptimizer", ]

class FireFlyOptimizer(FireFlyBase):
    
    def __init__(
        self,
        config: FireFlyConfig  = FireFlyConfig.get_defaults(),
        bounder: FireFlyParameterBounder = FireFlyParameterBounder.get_defaults()
    ) -> None:
        super(FireFlyOptimizer, self).__init__(config=config, bounder=bounder)
    
    def run(
            self,
            func: Callable[[Any], Any],
            dim: int
    ) -> None:
        
        fireflies = self.gen_fireflies(dim=dim)
        intensity = self.get_intensity(func=func, fireflies=fireflies)
        
        self.best_intensity = np.min(intensity)
        self.best_pos = self.bounder.apply(fireflies[np.argmin(intensity)])
        
        iter = self.config.pop_size
        new_alpha = self.config.alpha

        diff = np.apply_along_axis(lambda item: item[1] - item[0],1, np.array([item for item in self.bounder.bounds]))
        
        for iter in range(self.config.max_iters):
            new_alpha *= 0.97
            
            for i in range(self.config.pop_size):
                
                for j in range(self.config.pop_size):
                
                    if intensity[i] > intensity[j] and not np.isnan(intensity[j]): 
                    
                        r = self.get_distance(fireflies[i], fireflies[j])
                        beta = self.compute_beta(r=r)
                        
                        steps = new_alpha * (np.random.rand(dim) - 0.5) * diff
                        
                        fireflies[i] += self.update_ffi(fireflies[j], fireflies[i], beta=beta, steps=steps)
                        fireflies[i] = self.bounder.apply(fireflies[i])
                        intensity[i] = func(fireflies[i])
                        
                        if not np.isnan(intensity[i]) and intensity[i] < self.best_intensity: 
                            self.best_pos = self.bounder.apply(fireflies[i].copy())
                            self.best_intensity = func(self.best_pos)