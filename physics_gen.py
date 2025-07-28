from astropy import units
from astropy.units import cds
from poliastro.bodies import Body
from poliastro.twobody.events import Event
from poliastro.core.propagation import func_twobody
import numpy as np
from poliastro.twobody import Orbit
from matplotlib import pyplot as plt
from krpc.services.spacecenter import Vessel
from poliastro._math.ivp import DOP853, solve_ivp
from numba import njit

GRAVITY = 9.81 * (units.m / (units.s**2))
GRAVITY_NUMBA = 9.81
GAS_CONSTANT = 287.053 * (units.J / units.kg / units.K)
GAS_CONSTANT_NUMBA = 287.053


class VelocityDownCrossEvent(Event):
    """Detect if a satellite crosses a specific threshold altitude while ascending."""

    def __init__(self, terminal=True):
        super().__init__(terminal=terminal, direction=-1)  # Direction = 1 for ascending

    def __call__(self, t, u_, k, *others):
        # Calculate current altitude
        self._last_t = t
        velocity = np.dot(u_[3:6], u_[0:3])
        return velocity


# @jitclass
# class PhysicsSimulatorKSP:
#     def __init__(self, body_radius, altitude_curve, thrust_curve, isp_curve, mass, fuel_mass, body_rotational_angle, temperature_curve, pressure_curve, mach_curve, Acof_curve, initial_body, initial_position, initial_velocity):
#         self.body_radius = body_radius
#         self.altitude_curve = altitude_curve
#         self.thrust_curve = thrust_curve
#         self.isp_curve = isp_curve
#         self.mass = mass
#         self.fuel_mass = fuel_mass
#         self.body_rotational_angle = body_rotational_angle
#         self.temperature_curve = temperature_curve
#         self.pressure_curve = pressure_curve
#         self.mach_curve = mach_curve
#         self.Acof_curve = Acof_curve
#         self.initial_body = initial_body
#         self.initial_position = initial_position
#         self.initial_velocity = initial_velocity


@njit(cache=True)
def thrust_acceleration_at_time_t(
    time,
    u,
    body_radius,
    altitude_curve,
    thrust_curve,
    isp_curve,
    mass,
    fuel_mass,
    direction_controller_args,
):
    direction, thrust_portion = direction_controller(time, u, direction_controller_args)
    altitude = np.linalg.norm(u[0:3]) - body_radius
    thrust = np.interp(altitude, altitude_curve, thrust_curve) * thrust_portion
    isp = np.interp(altitude, altitude_curve, isp_curve)
    mass_flow_rate = thrust / (isp * GRAVITY_NUMBA)
    current_mass = u[6]

    if current_mass > (mass - fuel_mass):
        acceleration = thrust / current_mass
        return np.array([0, 0, 0, *(acceleration * direction), -mass_flow_rate])
    else:
        return np.array([0, 0, 0, 0, 0, 0, -mass_flow_rate * 0])


@njit(cache=True)
def f(
    time,
    u,
    k,
    body_radius,
    altitude_curve,
    thrust_curve,
    isp_curve,
    mass,
    fuel_mass,
    body_rotational_angle,
    pressure_curve,
    temperature_curve,
    mach_curve,
    Acof_curve,
    direction_controller_args,
):
    # print(u, time)
    du_kep = np.hstack(
        (func_twobody(time, u[0:6] / 1000, k / 1e9) * 1000, np.array([0]))
    )
    du_thrust = thrust_acceleration_at_time_t(
        time,
        u,
        body_radius,
        altitude_curve,
        thrust_curve,
        isp_curve,
        mass,
        fuel_mass,
        direction_controller_args,
    )
    du_drag = aerodynamic_acceleration_at_time_t(
        time,
        u,
        body_rotational_angle,
        body_radius,
        altitude_curve,
        pressure_curve,
        temperature_curve,
        mach_curve,
        Acof_curve,
        direction_controller_args,
    )
    return du_kep + du_thrust + du_drag


def simulate_launch(
    initial_body,
    initial_position,
    initial_velocity,
    body_radius,
    altitude_curve,
    thrust_curve,
    isp_curve,
    mass,
    fuel_mass,
    body_rotational_angle,
    tempertature_curve,
    pressure_curve,
    mach_curve,
    Acof_curve,
    direction_controller_args,
):
    # tofs = TimeDelta(np.linspace(0 * units.s, 200 * units.s, num=500))
    initial_orbit = Orbit.from_vectors(initial_body, initial_position, initial_velocity)
    # final_ephem = initial_orbit.to_ephem(EpochsArray(initial_orbit.epoch + tofs, method=CowellPropagator(f=self.f)))
    events = [VelocityDownCrossEvent()]
    time_of_flight = (10000 * units.s).to_value(units.s)
    mass = mass
    u0 = [
        *initial_orbit.r.to_value(units.m),
        *initial_orbit.v.to_value(units.m / units.s),
        mass.to_value(units.kg),
    ]
    result = solve_ivp(
        f,
        (0, time_of_flight),
        u0,
        args=(
            initial_orbit.attractor.k.to_value(units.m**3 / units.s**2),
            body_radius.to_value(units.m),
            altitude_curve.to_value(units.m),
            thrust_curve.to_value(units.N),
            isp_curve.to_value(units.s),
            mass.to_value(units.kg),
            fuel_mass.to_value(units.kg),
            body_rotational_angle.to_value(units.rad / units.s),
            pressure_curve.to_value(units.Pa),
            tempertature_curve.to_value(units.K),
            mach_curve,
            Acof_curve.to_value(units.m**2),
            direction_controller_args,
        ),
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
    final_orbit = Orbit.from_vectors(
        initial_orbit.attractor,
        final_orbit_vector[0:3] * units.m,
        final_orbit_vector[3:6] * (units.m / units.s),
        initial_orbit.epoch,
    )
    return final_orbit


@njit(cache=True)
def aerodynamic_acceleration_at_time_t(
    time,
    u,
    body_rotational_angle,
    body_radius,
    altitude_curve,
    pressure_curve,
    temperature_curve,
    mach_curve,
    Acof_curve,
    direction_controller_args,
):
    position = u[0:3]
    true_velocity = u[3:6]
    surface_velocity = np.cross(body_rotational_angle, position)
    velocity = true_velocity - surface_velocity
    altitude = np.linalg.norm(u[0:3]) - body_radius
    pressure = np.interp(altitude, altitude_curve, pressure_curve)
    temperature = np.interp(altitude, altitude_curve, temperature_curve)
    atmospheric_density = pressure / (GAS_CONSTANT_NUMBA * temperature)
    speed_of_sound = (1.4 * GAS_CONSTANT_NUMBA * temperature) ** 0.5
    mach_number = np.linalg.norm(velocity) / speed_of_sound
    Acof = np.interp(mach_number, mach_curve, Acof_curve)
    force_drag = 0.5 * atmospheric_density * np.linalg.norm(velocity) ** 2 * Acof
    force = velocity / np.linalg.norm(velocity) * force_drag * -1

    current_mass = u[6]

    acceleration = force / current_mass
    # position = tuple(u[0:3] * 1000)
    # velocity = tuple(u[3:6] * 1000)
    # altitude = self.body.altitude_at_position(position, self.reference_frame)
    # force = self.flight.simulate_aerodynamic_force_at(self.body, position, velocity)
    return np.hstack((np.array([0, 0, 0]), acceleration, np.array([0])))


@njit(cache=True)
def direction_controller(time, u, direction_controller_args):
    max_height = direction_controller_args[0]
    return (u[0:3] / np.linalg.norm(u[0:3])), (
        max_height
        # 1 if np.linalg.norm(u[0:3]) < max_height else 0
    )


class KRPCSimulatedRocket:
    def __init__(self, vessel: Vessel):
        self.vessel = vessel
        self.body = vessel.orbit.body
        self.reference_frame = self.body.non_rotating_reference_frame
        self.vessel_reference_frame = self.vessel.reference_frame
        self.flight = vessel.flight(self.reference_frame)
        self.body_radius = self.body.equatorial_radius * units.m
        self.initial_position = (
            np.array(self.vessel.position(self.reference_frame)) * units.m
        )
        self.initial_velocity = np.array(self.vessel.velocity(self.reference_frame)) * (
            units.m / units.s
        )
        self.body_rotational_angle = np.array((0, self.body.rotational_speed, 0)) * (
            units.rad / units.s
        )

        self.altitude_curve = (
            np.linspace(0, self.body.atmosphere_depth * 1.1, 100) * units.m
        )
        self.pressure_curve = (
            np.array(
                [
                    self.body.pressure_at(a.to_value(units.m))
                    for a in self.altitude_curve
                ]
            )
            * units.Pa
        )
        self.tempertature_curve = np.array(
            [
                self.body.temperature_at(
                    (0, a.to_value(units.m), 0), self.vessel_reference_frame
                )
                for a in self.altitude_curve
            ]
        ) * (units.K)
        self.thrust_curve = (
            np.array(
                [
                    self.vessel.available_thrust_at(a.to_value(cds.atm))
                    for a in self.pressure_curve
                ]
            )
            * units.N
        )
        self.isp_curve = (
            np.array(
                [
                    self.vessel.specific_impulse_at(a.to_value(cds.atm))
                    for a in self.pressure_curve
                ]
            )
            * units.s
        )

        self.mass = self.vessel.mass * units.kg
        resources = self.vessel.resources
        self.fuel_mass = (
            resources.amount("LiquidFuel") * 5 + resources.amount("Oxidizer") * 5
        )
        self.fuel_mass *= units.kg

        self.initial_body = Body.from_parameters(
            parent=None,
            k=(self.body.gravitational_parameter) * (units.m**3 / units.s**2),
            R=self.body.equatorial_radius * units.m,
            name=self.body.name,
            symbol=self.body.name[0],
        )

        flight = vessel.flight(self.vessel_reference_frame)
        pos = vessel.position(self.vessel_reference_frame)

        def mach_number_to_Acof(velocity):
            velocity = np.array(velocity) * flight.speed_of_sound
            return (
                np.linalg.norm(
                    flight.simulate_aerodynamic_force_at(
                        vessel.orbit.body, pos, tuple(velocity)
                    )
                )
                * 2
                / (
                    vessel.orbit.body.atmospheric_density_at_position(
                        pos, self.vessel_reference_frame
                    )
                    * np.linalg.norm(velocity) ** 2
                )
            )

        self.mach_curve = np.linspace(0.01, 30, 500)
        self.Acof_curve = np.array(
            [mach_number_to_Acof((0, a, 0)) for a in self.mach_curve]
        ) * (units.m**2)
        self.direction_controller_args = []

    def refresh(self):
        self.initial_position = (
            np.array(self.vessel.position(self.reference_frame)) * units.m
        )
        self.initial_velocity = np.array(self.vessel.velocity(self.reference_frame)) * (
            units.m / units.s
        )
        self.mass = self.vessel.mass * units.kg
        resources = self.vessel.resources
        self.fuel_mass = (
            resources.amount("LiquidFuel") * 5 + resources.amount("Oxidizer") * 5
        )
        self.fuel_mass *= units.kg

    def simulate_launch(self):
        return simulate_launch(
            self.initial_body,
            self.initial_position,
            self.initial_velocity,
            self.body_radius,
            self.altitude_curve,
            self.thrust_curve,
            self.isp_curve,
            self.mass,
            self.fuel_mass,
            self.body_rotational_angle,
            self.tempertature_curve,
            self.pressure_curve,
            self.mach_curve,
            self.Acof_curve,
            self.direction_controller_args,
        )

    def set_direction_controller_args(self, controller_args):
        self.direction_controller_args = controller_args
