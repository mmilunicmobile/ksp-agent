from poliastro.bodies import Body
from poliastro.twobody import Orbit
from typing import Callable, Literal
import numpy as np
from abc import ABC, abstractmethod

class OptimizableSet(ABC):
    @abstractmethod
    def get_constraints(self) -> list[tuple[float,float]]:
        pass

    @abstractmethod
    def get_x(self) -> np.ndarray:
        pass

class OrbitSet(OptimizableSet):
    def __init__(self, orbit_generators: Callable[[np.ndarray], Orbit], orbit_x: np.ndarray, constraints: list[tuple[float,float]], orbit_cost: Callable[[np.ndarray], float]):
        self.get_orbit = orbit_generators
        self._initial_orbit_x = orbit_x
        self._constraints = constraints
        self.get_delta_v_cost = orbit_cost

    def get_constraints(self):
        return self._constraints
    
    def get_x(self):
        return self._initial_orbit_x

def solve_generalized_orbit(orbit_sets: list[OrbitSet]):
    """
    Minimizes the delta-v required to get through a series of orbit sets. It does so by connecting each orbit by solving the Lambert Problem.

    Args:
        orbit_generators: Callables that take a set of parameters to be varied upon and return Orbits.
        orbit_x: Initial variables for each orbit.
        constraints: A set of constraints on the orbit generator parameters.
    """

    NUMBER_OF_ORBITS = len(orbit_sets)
    NUMBER_OF_LAMBERTS = NUMBER_OF_ORBITS - 1

    

    def f(x):
