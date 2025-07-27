import math
from abc import ABC, abstractmethod
import poliastro
from astropy import units
from astropy.units import Quantity
from astropy.units import cds
import krpc
from typing import Callable
from poliastro.bodies import Body
from poliastro.twobody.events import Event
from poliastro.twobody.propagation import CowellPropagator
from poliastro.core.propagation import func_twobody
import numpy as np
from poliastro.twobody import Orbit
from poliastro.twobody.sampling import EpochsArray
from astropy.time import TimeDelta
from matplotlib import pyplot as plt
import krpc
from krpc.services.spacecenter import Vessel

class VelocityDownCrossEvent(Event):
    """Detect if a satellite crosses a specific threshold altitude while ascending."""

    def __init__(self, terminal=True):
        super().__init__(terminal=terminal, direction=-1) # Direction = 1 for ascending

    def __call__(self, t, u_, k):
        # Calculate current altitude
        self._last_t = t
        velocity = np.dot(u_[3:6], u_[0:3])
        return velocity

class RocketInfo(ABC):
    @abstractmethod
    def get_initial_body(self) -> Body:
        pass

    @abstractmethod
    def get_initial_position(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_initial_velocity(self) -> np.ndarray:
        pass

    @abstractmethod
    def thrust_acceleration_at_time_t(self, time, u, direction) -> Quantity:
        pass

    @abstractmethod
    def aerodynamic_acceleration_at_time_t(self, time, u, direction) -> Quantity:
        pass

    @abstractmethod
    def direction_controller(self, time, u) -> np.ndarray:
        pass

    def f(self, time, u, k):
        # print(u, time)
        du_kep = func_twobody(time, u, k)
        ship_direction = self.direction_controller(time*units.s, u)
        a_thrust_x, a_thrust_y, a_thrust_z = self.thrust_acceleration_at_time_t(time*units.s, u, ship_direction).to_value(units.km/units.s**2)
        du_thrust = np.array([0,0,0, a_thrust_x, a_thrust_y, a_thrust_z])
        a_drag_x, a_drag_y, a_drag_z = self.aerodynamic_acceleration_at_time_t(time*units.s, u, ship_direction).to_value(units.km/units.s**2)
        du_drag = np.array([0,0,0, a_drag_x, a_drag_y, a_drag_z])
        return du_kep + du_thrust + du_drag
    
    def simulate_launch(self):
        # tofs = TimeDelta(np.linspace(0 * units.s, 200 * units.s, num=500))
        initial_orbit = Orbit.from_vectors(self.get_initial_body(), self.get_initial_position(), self.get_initial_velocity())
        # final_ephem = initial_orbit.to_ephem(EpochsArray(initial_orbit.epoch + tofs, method=CowellPropagator(f=self.f)))
        events = [VelocityDownCrossEvent()]
        final_orbit = initial_orbit.propagate(1000 * units.s, method=CowellPropagator(f=self.f, events=events, rtol=0.00001))
        return final_orbit

class SimulatedRocket(RocketInfo):
    def thrust_acceleration_at_time_t(self, time, u, direction):
        self.thrust = 1380.66 * units.kN
        self.isp = 285 * units.s
        self.mass = 15.3 * units.Mg
        self.burn_time = 8 * units.s

        if (time < self.burn_time):
            g = 9.81 * (units.m / (units.s ** 2))
            thrust_portion = 1
            mass_flow_rate = self.thrust / (self.isp * g) * thrust_portion
            current_mass = self.mass - (mass_flow_rate * time)
            acceleration = self.thrust / current_mass
            return acceleration * direction
        else:
            return 0 * (units.m / (units.s ** 2)) * direction
    
    def get_initial_body(self):
        return Body.from_parameters(
            parent=None,
            k=3.5316000E12 * (units.m ** 3 / units.s ** 2),
            R=600 * units.km,
            name="Kerbin",
            symbol="K"
        )
    
    def get_initial_position(self):
        return np.array([600, 0, 0]) * units.km

    def get_initial_velocity(self):
        return np.array([0, 0.1, 0]) * (units.m / units.s)

    def aerodynamic_acceleration_at_time_t(self, time, u, direction):
        return np.array([0,0,0]) * (units.m / units.s ** 2)

    def direction_controller(self, time, u):
        return self.trajectory(np.linalg.norm(u[0:3]))

    def trajectory(self, altitude):
        return np.array([1,0,0])
    
class KRPCSimulatedRocket(RocketInfo):
    def __init__(self, vessel: Vessel):
        self.vessel = vessel
        self.body = vessel.orbit.body
        self.reference_frame = self.body.non_rotating_reference_frame
        self.flight = vessel.flight(self.reference_frame)

    def thrust_acceleration_at_time_t(self, time, u, direction):
        g = 9.81 * (units.m / (units.s ** 2))
        thrust_portion = 1
        altitude = self.body.altitude_at_position(tuple(u[0:3] * 1000), self.reference_frame)
        altitude = 0
        pressure = (self.body.pressure_at(altitude) * units.Pa).to_value(cds.atm)
        self.thrust = self.vessel.available_thrust_at(pressure)
        self.thrust *= units.N
        self.isp = self.vessel.specific_impulse_at(pressure)
        self.isp *= units.s
        self.mass = self.vessel.mass * units.kg
        resources = self.vessel.resources
        self.fuel_mass = resources.amount("LiquidFuel") * 5 + resources.amount("Oxidizer") * 5
        self.fuel_mass *= units.kg
        mass_flow_rate = self.thrust / (self.isp * g) * thrust_portion
        self.burn_time = (self.fuel_mass / mass_flow_rate).to(units.s)

        if (time < self.burn_time):
            g = 9.81 * (units.m / (units.s ** 2))
            thrust_portion = 1
            mass_flow_rate = self.thrust / (self.isp * g) * thrust_portion
            current_mass = self.mass - (mass_flow_rate * time)
            acceleration = self.thrust / current_mass
            return acceleration * direction
        else:
            return 0 * (units.m / (units.s ** 2)) * direction

    def get_initial_position(self):
        return np.array(self.vessel.position(self.reference_frame)) * units.m

    def get_initial_velocity(self):
        return np.array(self.vessel.velocity(self.reference_frame)) * (units.m / units.s)

    def aerodynamic_acceleration_at_time_t(self, time, u, direction):
        position = tuple(u[0:3] * 1000)
        velocity = tuple(u[3:6] * 1000)
        altitude = self.body.altitude_at_position(position, self.reference_frame)
        force = self.flight.simulate_aerodynamic_force_at(self.body, position, velocity)

        force = (0,0,0)
        return np.array(force) * (units.N) / (self.vessel.mass * units.kg)

    def direction_controller(self, time, u):
        return u[0:3] / np.linalg.norm(u[0:3])

    def trajectory(self, altitude):
        return np.array([1,0,0])
    
    def get_initial_body(self) -> Body:
        return Body.from_parameters(
            parent=None,
            k=(self.body.mass*  6.6743E-11) * (units.m ** 3 / units.s ** 2),
            R=self.body.equatorial_radius * units.m,
            name=self.body.name,
            symbol=self.body.name[0]
        )
    
def main():
    conn = krpc.connect()
    rocket = KRPCSimulatedRocket(conn.space_center.active_vessel)
    resultant = rocket.simulate_launch()
    
    print(resultant)
    print(resultant.r_a)
    print(resultant.r)


if __name__ == "__main__":
    main()