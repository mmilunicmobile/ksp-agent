import krpc
import time
import math
from scipy.optimize import minimize

def main():
    print("Hello from krpc-module!")
    conn = krpc.connect()

    time.sleep(5)

    target_altitude = 500
    

    current_vessel = conn.space_center.active_vessel

    current_vessel.control.sas = False
    current_vessel.control.rcs = False
    current_vessel.control.throttle = 1.0

    current_vessel.control.activate_next_stage()
    current_vessel.auto_pilot.engage()
    current_vessel.auto_pilot.target

    obt_frame = current_vessel.orbit.body.non_rotating_reference_frame
    srf_frame = current_vessel.orbit.body.reference_frame

    while True:
        apoapsis = current_vessel.orbit.apoapsis_altitude
        altitude = current_vessel.flight(srf_frame).mean_altitude
        tta = current_vessel.flight(srf_frame).vertical_speed
        theoretical_elevation = apoapsis if tta > 0 else altitude
        throttle = min(1, max(0,target_altitude - theoretical_elevation) / 10)
        print(throttle, theoretical_elevation, tta)
        current_vessel.control.throttle = throttle

if __name__ == "__main__":
    main()
