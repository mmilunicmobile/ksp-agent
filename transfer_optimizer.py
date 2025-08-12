from poliastro.bodies import Body, Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from typing import Callable, Literal
import numpy as np
from abc import ABC, abstractmethod
from astropy.time import Time, TimeDelta
from scipy.optimize import differential_evolution, minimize
from astropy import units as u
from astropy.units import Quantity

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
    
    @staticmethod
    def from_orbit(orbit: Orbit):
        return OrbitSet(lambda x: orbit, np.array([]), [], lambda x: 0)
    
    def free_nu(self):
        def orbit_generator(x):
            initial_orbit = self.get_orbit(x[0:-1])
            attractor = initial_orbit.attractor
            a, ecc, inc, raan, argp, nu = initial_orbit.classical()
            return Orbit.from_classical(attractor, a, ecc, inc, raan, argp, x[-1] * u.deg)
            
        return OrbitSet(orbit_generator, np.concatenate([self.get_x(), np.array([0])]), self.get_constraints() + [(-360, 360)], lambda x: self.get_cost(x[0:-1]) + 0)

class TwoOrbitSetLambertTransfer(OptimizableSet):
    """
    Is an orbit set that allows minimization of the delta-v required for a Lambert transfer from orbital_set_1 to orbital_set_2 and then from orbital_set_2 to orbital_set_3
    """
    def __init__(self, orbit_set_1: OrbitSet, orbit_set_2: OrbitSet, maximum_time: Quantity):
        self._orbit_set_1 = orbit_set_1
        self._orbit_set_2 = orbit_set_2
        self._maximum_time = maximum_time

        self._initial_x = np.concatenate([orbit_set_1.get_x(), orbit_set_2.get_x(), [0.33, 0.66]])
        self._constraints = orbit_set_1.get_constraints() + orbit_set_2.get_constraints() + [(0,1), (0,1)]
    
        self._x1_end = self._orbit_set_1.get_N()
        self._x2_end = self._orbit_set_2.get_N() + self._x1_end

    def get_x(self):
        return self._initial_x
    
    def get_constraints(self):
        return self._constraints
    
    def get_cost(self, x):
        x1 = x[0:self._x1_end]
        x2 = x[self._x1_end:self._x2_end]
        
        cost =  self._orbit_set_1.get_cost(x1) + self._orbit_set_2.get_cost(x2) + self.get_maneuver(x).get_total_cost().to_value(u.m / u.s)
        return cost

    def get_maneuver(self, x: np.ndarray) -> Maneuver:
        x1 = x[0:self._x1_end]
        x2 = x[self._x1_end:self._x2_end]

        # sort times so that they are always in order
        times = x[self._x2_end:] * self._maximum_time
        times = sorted(times)
        
        # technically this doesnt work
        transfer_start = self._orbit_set_1.get_orbit(x1).propagate(times[0])
        transfer_end = self._orbit_set_2.get_orbit(x2).propagate(times[1] + 1 * u.s)

        theoretical_manuver = Maneuver.lambert(transfer_start, transfer_end)

        return theoretical_manuver
    
    def get_orbit_1(self, x: np.ndarray):
        x1 = x[0:self._x1_end]
        return self._orbit_set_1.get_orbit(x1)

    def get_orbit_1_cost(self, x: np.ndarray):
        x1 = x[0:self._x1_end]
        return self._orbit_set_1.get_cost(x1)

    def get_orbit_2(self, x: np.ndarray):
        x2 = x[self._x1_end:self._x2_end]
        return self._orbit_set_2.get_orbit(x2)

    def get_orbit_2_cost(self, x: np.ndarray):
        x2 = x[self._x1_end:self._x2_end]
        return self._orbit_set_2.get_cost(x2)
    
class TwoOrbitSetProgradeLambertTransfer(OptimizableSet):
    def __init__(self, orbit_set_1: OrbitSet, orbit_set_2: OrbitSet, maximum_time: Quantity):
        self._orbit_set_1 = orbit_set_1
        self._orbit_set_2 = orbit_set_2
        self._maximum_time = maximum_time

        self._initial_x = np.concatenate([orbit_set_1.get_x(), orbit_set_2.get_x(), [0, 0.25, 0.50, 0.75]])
        self._constraints = orbit_set_1.get_constraints() + orbit_set_2.get_constraints() + [(0,10000), (0,1), (0,1), (0,1)]
    
        self._x1_end = self._orbit_set_1.get_N()
        self._x2_end = self._orbit_set_2.get_N() + self._x1_end

    def get_x(self):
        return self._initial_x
    
    def get_constraints(self):
        return self._constraints
    
    def get_cost(self, x):
        x1 = x[0:self._x1_end]
        x2 = x[self._x1_end:self._x2_end]
        
        cost =  self._orbit_set_1.get_cost(x1) + self._orbit_set_2.get_cost(x2) + self.get_maneuver(x).get_total_cost().to_value(u.m / u.s)
        return cost
    
    def get_maneuver(self, x: np.ndarray) -> Maneuver:
        x1 = x[0:self._x1_end]
        x2 = x[self._x1_end:self._x2_end]

        # sort times so that they are always in order
        impulse = x[self._x2_end] * u.m / u.s
        times = x[self._x2_end+1:] * self._maximum_time
        times = sorted(times)
        
        # technically this doesnt work

        transfer_start = self._orbit_set_1.get_orbit(x1).propagate(times[0])
        prograde = transfer_start.v / np.linalg.norm(transfer_start.v)
        maneuver_a = Maneuver.impulse(prograde * impulse)
        transfer_middle = transfer_start.apply_maneuver(maneuver_a).propagate(times[1] - times[0] + 1 * u.s) # type: ignore
        transfer_end = self._orbit_set_2.get_orbit(x2).propagate(times[2] + 2 * u.s)

        manuver_b_c = Maneuver.lambert(transfer_middle, transfer_end)

        a_dt, a_dv = maneuver_a[0]
        b_dt, b_dv = manuver_b_c[0]
        c_dt, c_dv = manuver_b_c[1]

        return Maneuver((times[0], a_dv), (times[1], b_dv), (times[2], c_dv))

def optimize_set(set: OptimizableSet) -> np.ndarray:
    result = differential_evolution(set.get_cost, set.get_constraints(), x0=set.get_x(), tol=0.1)
    if result.success:
        return result.x
    else:
        raise Exception(f"Optimization failed: {result.message}")

def main():
    first_orbit = OrbitSet.from_orbit(Orbit.circular(Earth, 100 * u.km))
    second_orbit = OrbitSet.from_orbit(Orbit.circular(Earth, 100 * u.km, inc=90 * u.deg))
    transfer_set = TwoOrbitSetLambertTransfer(first_orbit, second_orbit, 1 * u.day)
    x = optimize_set(transfer_set)
    print(transfer_set.get_maneuver(x))

if __name__ == "__main__":
    main()