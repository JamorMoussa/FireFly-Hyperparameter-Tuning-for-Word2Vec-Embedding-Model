import numpy as np
from firefly import FireFlyConfig, FireFlyParameterBounder



class FireFlyBase:

    config: FireFlyConfig
    bounder: FireFlyParameterBounder
    
    def __init__(
        self,
        config: FireFlyConfig  = FireFlyConfig.get_defaults(),
        bounder: FireFlyParameterBounder = FireFlyParameterBounder.get_defaults()
    ) -> None:
        
        assert isinstance(config, FireFlyConfig), "the 'config' param must be an instance of 'FireFlyConfig'."
        assert isinstance(bounder, FireFlyParameterBounder), "the 'config' param must be an instance of 'FireFlyParameterBounder'."
        self.config = config
        self.bounder = bounder

    def gen_fireflies(self, dim: int = 3):
        return np.random.rand(self.config.pop_size, dim)

    def get_intensity(self, func, fireflies: np.ndarray):
        return np.apply_along_axis(func, 1, fireflies)

    def get_distance(self, fi, fj):
        return np.sum(np.square(fi - fj), axis=-1)

    def compute_beta(self, r: float):
        return self.config.beta0 * np.exp(-self.config.gamma * r)

    def update_ffi(self, fi, fj, beta, steps):
        return beta * (fj - fi) + steps