from physics_gen import KRPCSimulatedRocket
import krpc
from astropy import units
import time
import numpy as np
from krpc.services.spacecenter import Vessel
from threading import Thread

def main():
    global controls_array
    conn = krpc.connect()
    # tot = 0
    space_center = conn.space_center
    if space_center is None:
        raise ValueError("No space center found.")

    rocket = KRPCSimulatedRocket(space_center.active_vessel)
    # def periapsis_calc(height):
    #     height = height[0]
    #     height *= 0.001
    #     rocket.direction_controller = lambda time,u: (u[0:3] / np.linalg.norm(u[0:3]), min(1, max(height,0)))
    #     resultant = rocket.simulate_launch()
    #     return ((resultant.r_a.to_value(units.m) - 600000) - 10000) ** 2
    height_shutoff = Thread(
        target=control_monitor, args=(space_center.active_vessel,)
    )
    height_shutoff.daemon = True
    height_shutoff.start()

    while True:
        controls_array = calculate_apoapsis(rocket)
        rocket.refresh()
        print(controls_array)


def calculate_apoapsis(rocket):
    start_turn = 610000

    max_height = 700000 * units.m
    min_height = 600000 * units.m
    target_height = 610000 * units.m

    start_time = time.time()
    while max_height > (min_height + 3 * units.m):
        mean_height = (max_height + min_height) / 2
        # def f(time, u):
        #     return (u[0:3] / np.linalg.norm(u[0:3]), 1 if (np.linalg.norm(u[0:3]) * units.km) < mean_height else 0)
        rocket.set_direction_controller_args(np.array([start_turn, mean_height.to_value(units.m), 0, np.pi / 2]))
        resultant_orbit = rocket.simulate_launch()
        height_at_max_height, velocity_at_max_height = resultant_orbit.rv()
        if np.linalg.norm(height_at_max_height) < target_height:
            min_height = mean_height
        else:
            max_height = mean_height
    end_time = time.time()
    # print(max_height, min_height)
    # print(end_time - start_time)
    return [start_turn, max_height.to_value(units.m), 0, np.pi / 2]

    # threading.Thread(target=shutoff_height_monitor, args=[conn.space_center.active_vessel]).start()
    # while (True):
    #     x0 = [0.5]
    #     result = minimize(periapsis_calc, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-3}, bounds=[(0,1)])
    #     print(result.x[0])
    #     shutoff_altitude = result.x[0]
    #     rocket.refresh()

controls_array = None

def control_monitor(vessel: Vessel):
    reference_frame = vessel.orbit.body.non_rotating_reference_frame
    while True:
        if controls_array is None:
            continue
        else:
            start_height, end_height, heading, final_pitch = controls_array
            initial_pitch = np.pi / 2
            height = np.linalg.norm(vessel.position(reference_frame))
            vertical_velocity = np.dot(vessel.velocity(reference_frame), vessel.position(reference_frame)) / height
            vessel.control.throttle = (
                1
                if height < end_height
                else 0
            )
            pitch = np.interp(height, np.array([start_height, end_height]), np.array([initial_pitch, final_pitch]))
            vessel.auto_pilot.engage()
            vessel.auto_pilot.target_heading = np.rad2deg(heading)
            vessel.auto_pilot.target_roll = 0
            vessel.auto_pilot.target_pitch = np.rad2deg(pitch)

def aerodynamic_test():
    conn = krpc.connect()
    vessel = conn.space_center.active_vessel
    reference_frame = vessel.reference_frame
    flight = vessel.flight(reference_frame)
    pos = vessel.position(reference_frame)

    def f(velocity):
        velocity = np.array(velocity) * flight.speed_of_sound
        return (
            np.linalg.norm(
                flight.simulate_aerodynamic_force_at(
                    vessel.orbit.body, pos, tuple(velocity)
                )
            )
            * 2
            / (
                vessel.orbit.body.atmospheric_density_at_position(pos, reference_frame)
                * np.linalg.norm(velocity) ** 2
            )
        )

    for i in range(1, 3000, 5):
        print(f((0, i / 100, 0)))


if __name__ == "__main__":
    main()
