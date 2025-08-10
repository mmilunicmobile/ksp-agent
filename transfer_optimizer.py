from poliastro.bodies import Body
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from typing import Callable, Literal
import numpy as np
from abc import ABC, abstractmethod
from astropy.time import Time, TimeDelta

class OptimizableSet(ABC):
    @abstractmethod
    def get_constraints(self) -> list[tuple[float,float]]:
        pass

    @abstractmethod
    def get_x(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_cost(self, x: np.ndarray) -> float:
        pass

    def get_N(self) -> int:
        return len(self.get_x())

class OrbitSet(OptimizableSet):
    def __init__(self, orbit_generators: Callable[[np.ndarray], Orbit], orbit_x: np.ndarray, constraints: list[tuple[float,float]], orbit_cost: Callable[[np.ndarray], float]):
        self.get_orbit = orbit_generators
        self._initial_orbit_x = orbit_x
        self._constraints = constraints
        self._get_delta_v_cost = orbit_cost

    def get_constraints(self):
        return self._constraints
    
    def get_x(self):
        return self._initial_orbit_x
    
    def get_cost(self, x):
        return self._get_delta_v_cost(x)

class TwoOrbitSetTransfer(OptimizableSet):
    """
    Is an orbit set that allows minimization of the delta-v required for a Lambert transfer from orbital_set_1 to orbital_set_2 and then from orbital_set_2 to orbital_set_3
    """
    def __init__(self, orbit_set_1: OrbitSet, orbit_set_2: OrbitSet, maximum_time: float):
        self._orbit_set_1 = orbit_set_1
        self._orbit_set_2 = orbit_set_2
        self._maximum_time = maximum_time

        self._initial_x = np.concatenate([orbit_set_1.get_x(), orbit_set_2.get_x(), 0.33, 0.66])
        self._constraints = orbit_set_1.get_constraints() + orbit_set_2.get_constraints() + [(0,1), (0,1)]
    
        self._x1_len = self._orbit_set_1.get_N()
        self._x2_len = self._orbit_set_2.get_N()

    def get_x(self):
        return self._initial_x
    
    def get_constraints(self):
        return self._constraints
    
    def get_cost(self, x):
        x1 = x[0:self._x1_len]
        x2 = x[self._x1_len:self._x2_len]
        
        cost =  self._orbit_set_1.get_cost(x1) + self._orbit_set_2.get_cost(x2) + self.get_maneuver(x).get_total_cost()
        return cost

    def get_maneuver(self, x: np.ndarray) -> Maneuver:
        x1 = x[0:self._x1_len]
        x2 = x[self._x1_len:self._x2_len]

        # sort times so that they are always in order
        times = x[self._x2_len:] * self._maximum_time
        times = sorted(times)
        
        # technically this doesnt work
        transfer_start = self._orbit_set_1.get_orbit(x1).propagate(times[0])
        transfer_end = self._orbit_set_2.get_orbit(x2).propagate(times[1])

        theoretical_manuver = Maneuver.lambert(transfer_start, transfer_end)

        return theoretical_manuver

def optimize_set(set: OptimizableSet) -> np.ndarray:
    