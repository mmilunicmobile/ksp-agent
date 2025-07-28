import math
from typing import Callable
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from random import random


def de_casteljau(t: float, coefs: list[float]) -> float:
    """De Casteljau's algorithm."""
    beta = coefs.copy()  # values in this list are overridden
    n = len(beta)
    for j in range(1, n):
        for k in range(n - j):
            beta[k] = beta[k] * (1 - t) + beta[k + 1] * t
    return beta[0]


def triangularize(spline_points: list[float]):
    triangle = [[]]
    for i in spline_points:
        if len(triangle[-1]) >= 2 ** (len(triangle) - 1):
            triangle.append([])
        triangle[-1].append(i)
    return triangle


def bernstein_trajectory_calculator(t: float, spline_points: list[float]):
    total = 0
    for coefs in triangularize(spline_points):
        total += de_casteljau(t, coefs)
    return total


def bernstein_trajectory(spline_points: list[float]):
    return lambda t: bernstein_trajectory_calculator(t, spline_points)


def trajectory_length(float_supplier: Callable[[float], float]):
    total_length = 0
    for i in range(0, 100):
        position = i / 100
        next_position = (i + 1) / 100
        total_length += math.hypot(
            next_position - position,
            float_supplier(next_position) - float_supplier(position),
        )

    return total_length


def graph_trajectory(float_supplier: Callable[[float], float]):
    x = np.linspace(0, 1, 100)
    y = [float_supplier(t) for t in x]
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    x0 = np.array([random() for i in range(7)])
    # print(triangularize(x0))
    result = minimize(
        lambda x: trajectory_length(bernstein_trajectory(list(x))),
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-8},
    )
    print(result)
    graph_trajectory(bernstein_trajectory(list(result.x)))
