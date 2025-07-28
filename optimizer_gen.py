from physics_gen import KRPCSimulatedRocket
import krpc
from astropy import units
import time
import numpy as np
from krpc.services.spacecenter import Vessel
from threading import Thread


shutoff_altitude = 0


def main():
    global shutoff_altitude
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
        target=height_shutoff_thread, args=(space_center.active_vessel,)
    )
    height_shutoff.daemon = True
    height_shutoff.start()

    while True:
        height = calculate_shutoff(rocket)
        shutoff_altitude = height
        rocket.refresh()
        print(height)


def height_shutoff_thread(vessel: Vessel):
    reference = vessel.orbit.body.reference_frame
    while True:
        vessel.control.throttle = shutoff_altitude
        # vessel.control.throttle = 1 if np.linalg.norm(vessel.position(reference)) < shutoff_altitude else 0


def calculate_shutoff(rocket):
    max_height = 1
    min_height = 0
    target_height = 610000 * units.m

    start_time = time.time()
    while max_height > (min_height + 0.01):
        mean_height = (max_height + min_height) / 2
        # def f(time, u):
        #     return (u[0:3] / np.linalg.norm(u[0:3]), 1 if (np.linalg.norm(u[0:3]) * units.km) < mean_height else 0)
        rocket.set_direction_controller_args([mean_height])
        resultant_orbit = rocket.simulate_launch()
        height_at_max_height, velocity_at_max_height = resultant_orbit.rv()
        if np.linalg.norm(height_at_max_height) < target_height:
            min_height = mean_height
        else:
            max_height = mean_height
    end_time = time.time()
    # print(max_height, min_height)
    # print(end_time - start_time)
    return min_height

    # threading.Thread(target=shutoff_height_monitor, args=[conn.space_center.active_vessel]).start()
    # while (True):
    #     x0 = [0.5]
    #     result = minimize(periapsis_calc, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-3}, bounds=[(0,1)])
    #     print(result.x[0])
    #     shutoff_altitude = result.x[0]
    #     rocket.refresh()


def shutoff_height_monitor():
    conn = krpc.connect()
    # tot = 0
    vessel = conn.space_center.active_vessel
    while True:
        vessel.control.throttle = (
            1
            if 1354.98046875 > vessel.flight(vessel.reference_frame).mean_altitude
            else 0
        )


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
