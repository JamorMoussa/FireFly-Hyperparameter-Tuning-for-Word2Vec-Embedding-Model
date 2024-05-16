from fireflyalgorithm import FireflyAlgorithm
from fireflyalgorithm.problems import sphere


def func(x):
    x, y = x 
    # return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    return x**2 + y**2 + x*y

FA = FireflyAlgorithm(gamma=0.01, alpha=0.2)
best, best_pos = FA.run(function=func, dim=2, lb=-5, ub=5, max_evals=1000)

print(best, best_pos)