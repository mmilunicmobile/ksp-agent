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
from poliastro._math.ivp import DOP853, solve_ivp
from numba import njit

GRAVITY = 9.81 * (units.m / (units.s ** 2))
GAS_CONSTANT = 287.053 * (units.J / units.kg / units.K)

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
        du_kep = np.array([*func_twobody(time, u[0:6], k), 0])
        ship_direction = self.direction_controller(time*units.s, u)
        du_thrust = self.thrust_acceleration_at_time_t(time*units.s, u, ship_direction)
        du_drag = self.aerodynamic_acceleration_at_time_t(time*units.s, u, ship_direction)
        return du_kep + du_thrust + du_drag
    
    def simulate_launch(self):
        # tofs = TimeDelta(np.linspace(0 * units.s, 200 * units.s, num=500))
        initial_orbit = Orbit.from_vectors(self.get_initial_body(), self.get_initial_position(), self.get_initial_velocity())
        # final_ephem = initial_orbit.to_ephem(EpochsArray(initial_orbit.epoch + tofs, method=CowellPropagator(f=self.f)))
        events = [VelocityDownCrossEvent()]
        time_of_flight = (10000 * units.s).to_value(units.s)
        mass = self.get_initial_mass()
        u0 = [*initial_orbit.r.to_value(units.km), *initial_orbit.v.to_value(units.km/units.s), mass.to_value(units.kg)]
        result = solve_ivp(
            self.f,
            (0, time_of_flight),
            u0,
            args=(initial_orbit.attractor.k.to_value(units.km ** 3 / units.s**2),),
            rtol=1e-4,
            atol=1e-12,
            method=DOP853,
            dense_output=True,
            events=events,
        )
        if not result.success:
            raise RuntimeError("Integration failed")
        
        last_t = time_of_flight

        if events is not None:
            # Collect only the terminal events
            terminal_events = [event for event in events if event.terminal]

            # If there are no terminal events, then the last time of integration is the
            # greatest one from the original array of propagation times
            if not terminal_events:
                last_t = time_of_flight
            else:
                # Filter the event which triggered first
                last_t = min(event._last_t for event in terminal_events)
        
        final_orbit_vector = result.sol(last_t)
        # final_orbit = initial_orbit.propagate(1000 * units.s, method=CowellPropagator(f=self.f, events=events, rtol=0.00001))
        final_orbit = Orbit.from_vectors(initial_orbit.attractor, final_orbit_vector[0:3] * units.km, final_orbit_vector[3:6] * (units.km / units.s), initial_orbit.epoch)
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
        self.vessel_reference_frame = self.vessel.reference_frame
        self.flight = vessel.flight(self.reference_frame)
        self.body_radius = self.body.equatorial_radius * units.m
        self.inital_position = np.array(self.vessel.position(self.reference_frame)) * units.m
        self.initial_velocity = np.array(self.vessel.velocity(self.reference_frame)) * (units.m / units.s)
        self.body_rotational_angle = np.array((0, self.body.rotational_speed, 0)) * (units.rad / units.s)

        self.altitude_curve = np.linspace(0, self.body.atmosphere_depth*1.1, 100) * units.m
        self.pressure_curve = np.array(
            [self.body.pressure_at(a.to_value(units.m)) for a in self.altitude_curve]
        )  * units.Pa
        self.tempertature_curve = np.array(
            [self.body.temperature_at((0, a.to_value(units.m), 0), self.vessel_reference_frame) for a in self.altitude_curve]
        ) * (units.K)
        self.thrust_curve = np.array(
            [self.vessel.available_thrust_at(a.to_value(cds.atm)) for a in self.pressure_curve]
        )  * units.N
        self.isp_curve = np.array(
            [self.vessel.specific_impulse_at(a.to_value(cds.atm)) for a in self.pressure_curve]
        )  * units.s

        self.mass = self.vessel.mass * units.kg
        resources = self.vessel.resources
        self.fuel_mass = resources.amount("LiquidFuel") * 5 + resources.amount("Oxidizer") * 5
        self.fuel_mass *= units.kg

        self.initial_body = Body.from_parameters(
            parent=None,
            k=(self.body.gravitational_parameter) * (units.m ** 3 / units.s ** 2),
            R=self.body.equatorial_radius * units.m,
            name=self.body.name,
            symbol=self.body.name[0]
        )

        flight = vessel.flight(self.vessel_reference_frame)
        pos = vessel.position(self.vessel_reference_frame)
        def mach_number_to_Acof(velocity):
            velocity = np.array(velocity) * flight.speed_of_sound
            return np.linalg.norm(flight.simulate_aerodynamic_force_at(vessel.orbit.body, pos, tuple(velocity))) * 2 / (vessel.orbit.body.atmospheric_density_at_position(pos, self.vessel_reference_frame) * np.linalg.norm(velocity) ** 2)
        
        self.mach_curve = np.linspace(0.01, 30, 500)
        self.Acof_curve = np.array(
            [mach_number_to_Acof((0,a,0)) for a in self.mach_curve]
        ) * (units.m ** 2)
    
    def refresh(self):
        self.inital_position = np.array(self.vessel.position(self.reference_frame)) * units.m
        self.initial_velocity = np.array(self.vessel.velocity(self.reference_frame)) * (units.m / units.s)
        self.mass = self.vessel.mass * units.kg
        resources = self.vessel.resources
        self.fuel_mass = resources.amount("LiquidFuel") * 5 + resources.amount("Oxidizer") * 5
        self.fuel_mass *= units.kg

    def thrust_acceleration_at_time_t(self, time, u, direction):
        thrust_portion = 1
        altitude = np.linalg.norm(u[0:3] * units.km) - self.body_radius
        self.thrust = np.interp(altitude, self.altitude_curve, self.thrust_curve)
        self.isp = np.interp(altitude, self.altitude_curve, self.isp_curve)
        mass_flow_rate = self.thrust / (self.isp * GRAVITY) * thrust_portion
        current_mass = u[6] * units.kg

        if (current_mass > (self.mass - self.fuel_mass)):
            acceleration = self.thrust / current_mass
            return np.array([0,0,0, *(acceleration * direction).to_value(units.km / units.s**2), -mass_flow_rate.to_value(units.kg/units.s)])
        else:
            return np.array([0,0,0, 0,0,0, 0])

    def get_initial_position(self):
        return self.inital_position

    def get_initial_velocity(self):
        return self.initial_velocity

    def aerodynamic_acceleration_at_time_t(self, time, u, spaceship_direction):
        position = np.array(u[0:3]) * units.km
        true_velocity = np.array(u[3:6]) * units.km / units.s
        surface_velocity = np.cross(self.body_rotational_angle, position) / units.rad
        velocity = true_velocity - surface_velocity
        altitude = np.linalg.norm(u[0:3] * units.km) - self.body_radius
        pressure = np.interp(altitude, self.altitude_curve, self.pressure_curve)
        temperature = np.interp(altitude, self.altitude_curve, self.tempertature_curve)
        atmospheric_density = pressure / (GAS_CONSTANT * temperature)
        speed_of_sound = ((1.4 * GAS_CONSTANT * temperature) ** 0.5).to(units.m / units.s)
        mach_number = np.linalg.norm(velocity) / speed_of_sound
        Acof =  np.interp(mach_number, self.mach_curve, self.Acof_curve)
        force_drag = 0.5 * atmospheric_density * np.linalg.norm(velocity)**2 * Acof
        force = velocity / np.linalg.norm(velocity) * force_drag * -1
        
        current_mass = u[6] * units.kg

        acceleration = force / current_mass
        # position = tuple(u[0:3] * 1000)
        # velocity = tuple(u[3:6] * 1000)
        # altitude = self.body.altitude_at_position(position, self.reference_frame)
        # force = self.flight.simulate_aerodynamic_force_at(self.body, position, velocity)
        return np.array([0,0,0, *acceleration.to_value(units.km/units.s**2), 0])

    def direction_controller(self, time, u):
        return u[0:3] / np.linalg.norm(u[0:3])

    def trajectory(self, altitude):
        return np.array([1,0,0])
    
    def get_initial_body(self) -> Body:
        return self.initial_body
    
    def get_initial_mass(self) -> Quantity:
        return self.mass
    
def main():
    conn = krpc.connect()
    tot = 0
    rocket = KRPCSimulatedRocket(conn.space_center.active_vessel)
    while (True):
        resultant = rocket.simulate_launch()
        print(resultant.r_a)
        rocket.refresh()

def aerodynamic_test():
    conn = krpc.connect()
    vessel = conn.space_center.active_vessel
    reference_frame = vessel.reference_frame
    flight = vessel.flight(reference_frame)
    pos = vessel.position(reference_frame)
    def f(velocity):
        velocity = np.array(velocity) * flight.speed_of_sound
        return np.linalg.norm(flight.simulate_aerodynamic_force_at(vessel.orbit.body, pos, tuple(velocity))) * 2 / (vessel.orbit.body.atmospheric_density_at_position(pos, reference_frame) * np.linalg.norm(velocity) ** 2)
    for i in range(1,3000,5):
        print(f((0,i/100,0)))

if __name__ == "__main__":
    main()