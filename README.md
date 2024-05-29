# FireFly Hyperparameter Tuning for Word2Vec Embedding Model

**Swarm intelligence** algorithms, inspired by the collective behavior of social creatures like birds, bees, and fireflies, have shown great promise in solving complex optimization problems. The **Firefly Algorithm**, in particular, mimics the flashing behavior of fireflies to find optimal solutions through iterative search and adjustment.

### Motivation

**Hyperparameter tuning** is a critical step in the development of deep learning models. This project aims to use the **FireFly Algorithm** to perform hyperparameter tuning for a Word2Vec embedding model. The goal is to leverage meta-heuristic optimization methods to find the best and most optimal hyperparameters.

## FireFly Algorithm

In this project, we provide an implementation of the FireFly Algorithm with some additional features, such as:
- Ensuring that parameters remain within specified ranges.
- Different parameters are given different scales, respecting their nature as described by the problem.

### Example

Below is an example demonstrating how to use the FireFly Algorithm for optimization:

```python
from firefly import FireFlyOptimizer, FireFlyConfig, FireFlyParameterBounder

def fitness(x):
    return x[0]**2 + x[1]**2

configs = FireFlyConfig.get_defaults()
configs.pop_size = 100
configs.max_iters = 10
print(configs)

bounder = FireFlyParameterBounder(bounds=[(-5, 5), (-5, 5)])
print(bounder)

FA = FireFlyOptimizer(config=configs, bounder=bounder)

FA.run(func=fitness, dim=2)

print("Best Fitness:", FA.best_intensity)
print("Best position:", FA.best_pos)
```

```bash
➜  FireFly-Optimizer-Deep-Learning git:(main) ✗ python3 firefly_example.py

FireFlyConfig(pop_size=100, alpha=1.0, beta0=1.0, gamma=0.01, max_iters=10, seed=None)
FireFlyParameterBounder(bounds=[(-5, 5), (-5, 5)])
Best Fitness: 0.009908765479110442
Best position: [0.06986411 0.07090678]
```

## Hyperparameter Tuning With FireFly

To use FireFly as an optimizer for hyperparameter tuning, run the following command and specify some arguments:

```bash
➜ python3 hyperparam.py --popsize=5 --alpha=1.0 --beta0=1.0 --gamma=0.01 --maxiters=5 
```