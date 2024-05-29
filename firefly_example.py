from firefly import FireFlyOptimizer, FireFlyConfig, FireFlyParameterBounder


def fitness(x):
    return x[0]**2 + x[1]**2


configs = FireFlyConfig.get_defaults()
configs.pop_size = 100
configs.max_iters = 10
print(configs)

bouder = FireFlyParameterBounder(bounds= [(-5, 5), (-5, 5)])
print(bouder)


FA = FireFlyOptimizer(config= configs, bounder= bouder)

FA.run(func= fitness, dim= 2)


print("Best Fitness:", FA.best_intensity)
print("Best position:", FA.best_pos)


