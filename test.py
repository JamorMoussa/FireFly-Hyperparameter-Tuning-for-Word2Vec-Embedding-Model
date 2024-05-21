from firefly import FireFlyOptimizer ,FireFlyConfig, FireFlyParameterBounder

bounder = FireFlyParameterBounder(bounds=[(-5, 5), (-5, 5)])
config = FireFlyConfig.get_defaults()

config.max_iters = 100
config.gamma = 0.001
config.alpha = 2

def fn(x): 
    return x[0]**2 + x[1]**2


FA = FireFlyOptimizer(bounder= bounder)


FA.run(func= fn, dim= 2)

print(FA.best_intensity)
print(FA.best_pos)