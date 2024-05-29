# FireFly Hyperparameter Tuning for Word2Vec Embedding Model

**Swarm intelligence** algorithms, inspired by the collective behavior of social creatures like birds, bees, and fireflies, have shown great promise in solving complex optimization problems. The **Firefly Algorithm**, in particular, mimics the flashing behavior of fireflies to find optimal solutions through
iterative search and adjustment.

### Motivation 

**Hyperparameter tuning** is a critical step in the development of deep learning models. How can we use the meta-heuristic optimization methods to find the best and optimal hyperparameters. This project aims to use the **FireFly Algorithm** to hyperparameter tuning for a word2vec embedding model.


## FireFly Algorithm

In this project we provide an implimentaion of FireFly algorithm with some additionally features, shuch as: 

- make sure that the paramerts don't be out the range.
- different paramters are given differents scall, resprect their nature described by the problem.

Let's have an example: 

```python
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

```

```bath
➜  FireFly-Optimizer-Deep-Learning git:(main) ✗ python3 firefly_example.py

FireFlyConfig(pop_size=100, alpha=1.0, beta0=1.0, gamma=0.01, max_iters=10, seed=None)
FireFlyParameterBounder(bounds=[(-5, 5), (-5, 5)])
Best Fitness: 0.009908765479110442
Best position: [0.06986411 0.07090678]
```