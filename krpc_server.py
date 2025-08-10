import datetime
from mcp.server.fastmcp import Context, FastMCP
import subprocess
import krpc
import asyncio
import enum
from pydantic import BaseModel
import math

class TimeReference(enum.Enum):
    ALTITUDE = "altitude"
    APOAPSIS = "apoapsis"
    CLOSEST_APPROACH = "closest_approach"
    COMPUTED = "computed"
    EQ_ASCENDING = "eq_ascending"
    EQ_DESCENDING = "eq_descending"
    EQ_HIGHEST_AD = "eq_highest_ad"
    EQ_NEAREST_AD = "eq_nearest_ad"
    PERIAPSIS = "periapsis"
    REL_ASCENDING = "rel_ascending"
    REL_DESCENDING = "rel_descending"
    REL_HIGHEST_AD = "rel_highest_ad"
    REL_NEAREST_AD = "rel_nearest_ad"
    X_FROM_NOW = "x_from_now"

def time_reference_converter(enum_reference: TimeReference):
    return getattr(mech_jeb.TimeReference, enum_reference.value)

class TimeSelector(BaseModel):
    time_reference: TimeReference
    lead_time: float | None = None
    circularize_altitude: float | None = None

mcp = FastMCP("Kerbal Space Program")

async def _execute_maneuver(op_name: str, op, ctx: Context):
    mech_jeb.staging_controller.enabled = True
    op.make_nodes()
    if hasattr(mech_jeb.maneuver_planner, "error_message"):
        warning = mech_jeb.maneuver_planner.error_message
        if warning:
            return f"MechJeb Warning: {warning}"

    executor = mech_jeb.node_executor
    executor.execute_all_nodes()

    await ctx.info(f"Executing {op_name} maneuver.")

    while executor.enabled:
        await asyncio.sleep(1)
    mech_jeb.staging_controller.enabled = False

    return f"{op_name} maneuver complete."

@mcp.tool()
async def operation_apoapsis(ctx: Context, new_apoapsis: float, time_selector: TimeSelector) -> str:
    """Creates a maneuver to set a new apoapsis.
    
    Args:
        new_apoapsis: The new apoapsis in meters.
        time_selector: The time selector for the maneuver.
    """
    
    op = mech_jeb.maneuver_planner.operation_apoapsis
    op.new_apoapsis = new_apoapsis
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("Apoapsis", op, ctx)

@mcp.tool()
async def operation_circularize(ctx: Context, time_selector: TimeSelector) -> str:
    """Creates a maneuver to circularize the orbit.
    
    Args:
        time_selector: The time selector for the maneuver.
    """
    
    op = mech_jeb.maneuver_planner.operation_circularize
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("Circularize", op, ctx)

class OrbitalInformation(BaseModel):
    body_name: str
    apoapsis_altitude: float
    periapsis_altitude: float
    semi_major_axis: float
    semi_minor_axis: float
    eccentricity: float
    inclination: float
    period: float
    time_to_apoapsis: float
    time_to_periapsis: float
    time_to_soi_change: float | None

@mcp.tool()
def get_orbital_information() -> OrbitalInformation:
    """Returns orbital information for the active vessel."""
    vessel = space_center.active_vessel
    orbit = vessel.orbit
    soi_time = orbit.time_to_soi_change
    return OrbitalInformation(
        body_name=orbit.body.name,
        apoapsis_altitude=orbit.apoapsis_altitude,
        periapsis_altitude=orbit.periapsis_altitude,
        semi_major_axis=orbit.semi_major_axis,
        semi_minor_axis=orbit.semi_minor_axis,
        eccentricity=orbit.eccentricity,
        inclination=orbit.inclination,
        period=orbit.period,
        time_to_apoapsis=orbit.time_to_apoapsis,
        time_to_periapsis=orbit.time_to_periapsis,
        time_to_soi_change= soi_time if not math.isnan(soi_time) else None,
    )

@mcp.tool()
async def operation_course_correction(ctx: Context, course_correct_final_pe_a: float, intercept_distance: float) -> str:
    """Creates a maneuver to fine-tune the closest approach to a target. Note, if you are aproaching a body, this does not account for gravity, make sure to add at least the body radius so that you do not crash into the surface.
    
    Args:
        course_correct_final_pe_a: This is the distance to the center of the body in meters.
        intercept_distance: The intercept distance. This matters if you are aiming for a vessel.
    """
    op = mech_jeb.maneuver_planner.operation_course_correction
    op.course_correct_final_pe_a = course_correct_final_pe_a
    op.intercept_distance = intercept_distance
    return await _execute_maneuver("Course Correction", op, ctx)

@mcp.tool()
async def operation_ellipticize(ctx: Context, new_apoapsis: float, new_periapsis: float, time_selector: TimeSelector) -> str:
    """Creates a maneuver to change both periapsis and apoapsis.
    
    Args:
        new_apoapsis: The new apoapsis in meters.
        new_periapsis: The new periapsis in meters.
        time_selector: The time selector for the maneuver.
    """
    
    op = mech_jeb.maneuver_planner.operation_ellipticize
    op.new_apoapsis = new_apoapsis
    op.new_periapsis = new_periapsis
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("Ellipticize", op, ctx)

@mcp.tool()
async def operation_inclination(ctx: Context, new_inclination: float, time_selector: TimeSelector) -> str:
    """Creates a maneuver to change inclination.
    
    Args:
        new_inclination: The new inclination in degrees.
        time_selector: The time selector for the maneuver.
    """
    
    op = mech_jeb.maneuver_planner.operation_inclination
    op.new_inclination = new_inclination
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("Inclination", op, ctx)

@mcp.tool()
async def operation_interplanetary_transfer(ctx: Context, wait_for_phase_angle: bool) -> str:
    """Creates a maneuver to transfer to another planet.
    
    Args:
        wait_for_phase_angle: Whether to wait for the correct phase angle.
    """
    
    op = mech_jeb.maneuver_planner.operation_interplanetary_transfer
    op.wait_for_phase_angle = wait_for_phase_angle
    return await _execute_maneuver("Interplanetary Transfer", op, ctx)

@mcp.tool()
async def operation_kill_rel_vel(ctx: Context, time_selector: TimeSelector) -> str:
    """Matches velocities with the target.
    
    Args:
        time_selector: The time selector for the maneuver.
    """
    
    op = mech_jeb.maneuver_planner.operation_kill_rel_vel
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("Kill Relative Velocity", op, ctx)

@mcp.tool()
async def operation_lambert(ctx: Context, intercept_interval: float, time_selector: TimeSelector) -> str:
    """Creates a maneuver to intercept a target at a chosen time.
    
    Args:
        intercept_interval: The intercept interval in seconds.
        time_selector: The time selector for the maneuver.
    """
    
    op = mech_jeb.maneuver_planner.operation_lambert
    op.intercept_interval = intercept_interval
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("Lambert", op, ctx)

@mcp.tool()
async def operation_lan(ctx: Context, new_lan: float, time_selector: TimeSelector) -> str:
    """Changes the longitude of the ascending node (LAN).
    
    Args:
        new_lan: The new LAN in degrees.
        time_selector: The time selector for the maneuver.
    """
    
    op = mech_jeb.maneuver_planner.operation_lan
    op.new_lan = new_lan
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("LAN", op, ctx)

@mcp.tool()
async def operation_longitude(ctx: Context, new_surface_longitude: float, time_selector: TimeSelector) -> str:
    """Changes the surface longitude of an apsis.
    
    Args:
        new_surface_longitude: The new surface longitude in degrees.
        time_selector: The time selector for the maneuver.
    """
    
    op = mech_jeb.maneuver_planner.operation_longitude
    op.new_surface_longitude = new_surface_longitude
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("Longitude", op, ctx)

@mcp.tool()
async def operation_moon_return(ctx: Context, moon_return_altitude: float) -> str:
    """Creates a maneuver to return from a moon to its parent body.
    
    Args:
        moon_return_altitude: The approximate return altitude in meters.
    """
    
    op = mech_jeb.maneuver_planner.operation_moon_return
    op.moon_return_altitude = moon_return_altitude
    return await _execute_maneuver("Moon Return", op, ctx)

@mcp.tool()
async def operation_periapsis(ctx: Context, new_periapsis: float, time_selector: TimeSelector) -> str:
    """Creates a maneuver to set a new periapsis.
    
    Args:
        new_periapsis: The new periapsis in meters.
        time_selector: The time selector for the maneuver.
    """
    
    op = mech_jeb.maneuver_planner.operation_periapsis
    op.new_periapsis = new_periapsis
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("Periapsis", op, ctx)

@mcp.tool()
async def operation_plane(ctx: Context, time_selector: TimeSelector) -> str:
    """Creates a maneuver to match orbital planes with a target.
    
    Args:
        time_selector: The time selector for the maneuver.
    """
    
    op = mech_jeb.maneuver_planner.operation_plane
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("Plane", op, ctx)

@mcp.tool()
async def operation_resonant_orbit(ctx: Context, resonance_denominator: int, resonance_numerator: int, time_selector: TimeSelector) -> str:
    """Creates a maneuver to establish a resonant orbit.
    
    Args:
        resonance_denominator: The denominator of the resonance ratio.
        resonance_numerator: The numerator of the resonance ratio.
        time_selector: The time selector for the maneuver.
    """
    
    op = mech_jeb.maneuver_planner.operation_resonant_orbit
    op.resonance_denominator = resonance_denominator
    op.resonance_numerator = resonance_numerator
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("Resonant Orbit", op, ctx)

@mcp.tool()
async def operation_semi_major(ctx: Context, new_semi_major_axis: float, time_selector: TimeSelector) -> str:
    """Creates a maneuver to set a new semi-major axis.
    
    Args:
        new_semi_major_axis: The new semi-major axis in meters.
        time_selector: The time selector for the maneuver.
    """
    op = mech_jeb.maneuver_planner.operation_semi_major
    op.new_semi_major_axis = new_semi_major_axis
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("Semi-Major Axis", op, ctx)

@mcp.tool()
async def operation_transfer(ctx: Context, intercept_only: bool, period_offset: float, simple_transfer: bool, time_selector: TimeSelector) -> str:
    """Plans a bi-impulsive (Hohmann) transfer to a target.
    
    Args:
        intercept_only: If true, plans for impact/flyby without a capture burn.
        period_offset: Fractional target period offset.
        simple_transfer: If true, uses a simple coplanar transfer, ignoring time_selector.
        time_selector: The time selector for the maneuver.
    """
    op = mech_jeb.maneuver_planner.operation_transfer
    op.intercept_only = intercept_only
    op.period_offset = period_offset
    op.simple_transfer = simple_transfer
    op.time_selector.time_reference = time_reference_converter(time_selector.time_reference)
    if time_selector.lead_time:
        op.time_selector.lead_time = time_selector.lead_time
    if time_selector.circularize_altitude:
        op.time_selector.circularize_altitude = time_selector.circularize_altitude
    return await _execute_maneuver("Transfer", op, ctx)

@mcp.tool()
async def launch_rocket(ctx: Context, orbit_altitude: float, orbit_inclination: float = 0) -> str:
    """Launches the active vessel on a default trajectory to the supplied orbital altitude. It's inclination is by default also zero. 
    This is a long running task which will provide progress updates.
    
    Args:
        orbit_altitude: The orbit_altitude of which the rocket will launch to in orbit in meters. This is in meters above sea level.
        orbit_inclination: The orbit_inclination of which the rocket will launch to in orbit in degrees.
    """
    vessel = space_center.active_vessel

    ascent = mech_jeb.ascent_autopilot
    ascent.desired_orbit_altitude = orbit_altitude

    ascent.desired_inclination = orbit_inclination

    ascent.force_roll = True
    ascent.vertical_roll = 90
    ascent.turn_roll = 90

    ascent.autostage = True

    ascent.enabled = True
    vessel.control.activate_next_stage() #launch the vessel

    flight = vessel.flight(vessel.orbit.body.reference_frame)

    await ctx.info("Launching rocket right now (liftoff).")

    while ascent.enabled:
        await asyncio.sleep(1)
        current_altitude = flight.mean_altitude
        await ctx.report_progress(progress=min(1,max(0,(current_altitude / orbit_altitude))), total=1.0, message=f"Rocket is now at {int(current_altitude)} meters")
    
    return f"Finished launching rocket to {orbit_altitude} meters."

@mcp.tool()
async def launch_to_rendezvous(ctx: Context) -> str:
    """Launches the active vessel to rendezvous with the current target.
    This is a long running task which will provide progress updates.
    """
    vessel = space_center.active_vessel
    ascent = mech_jeb.ascent_autopilot

    if space_center.target_vessel is None:
        return "No target vessel selected. Please select a target vessel in-game."

    ascent.launch_to_rendezvous_with_target()

    ascent.autostage = True
    ascent.enabled = True
    vessel.control.activate_next_stage() #launch the vessel

    flight = vessel.flight(vessel.orbit.body.reference_frame)
    target_vessel = space_center.target_vessel

    await ctx.info("Launching to rendezvous with target.")

    while ascent.enabled:
        await asyncio.sleep(1)
        # distance = flight.distance(target_vessel.position(vessel.orbit.body.reference_frame))
        # await ctx.report_progress(progress=0.5, total=1.0, message=f"Distance to target: {int(distance)} meters")

    return "Rendezvous ascent complete."

@mcp.tool()
async def say_message(msg: str) -> str:
    """Says a message to the terminal outloud so the user can hear.
    
    Args:
        msg: The message to be said out loud in the terminal.
    """

    subprocess.call(["say", msg])

    return f"Said: {msg}"

@mcp.tool()
async def warp_to_next_sphere_of_influence(ctx: Context) -> str:
    """Warps the active vessel to the next sphere of influence."""
    vessel = space_center.active_vessel
    orbit = vessel.orbit
    
    time_to_soi_change = orbit.time_to_soi_change

    if time_to_soi_change is None or time_to_soi_change <= 0:
        return "No upcoming sphere of influence change."

    # add a bit of a buffer just in case
    warp_time = space_center.ut + time_to_soi_change + 10

    space_center.warp_to(warp_time)

    await ctx.info(f"Warping to next sphere of influence.")

    while space_center.warp_factor > 0:
        await asyncio.sleep(1)

    return "Warp complete."

@mcp.tool()
def list_celestial_bodies() -> list[str]:
    """Returns a list of all celestial bodies by their names."""
    celestial_bodies = space_center.bodies
    return [body.name for body in celestial_bodies.values()]

@mcp.tool()
def set_target_celestial_body(name: str) -> str:
    """Sets the target to a celestial body by its name.

    Args:
        name: The name of the celestial body to target.
    """
    try:
        celestial_bodies = space_center.bodies
        for body in celestial_bodies.values():
            if body.name.lower() == name.lower():
                space_center.target_body = body
                return f"Target set to {body.name}."
        return f"Error: Celestial body '{name}' not found."
    except Exception as e:
        return f"An error occurred: {e}"

@mcp.tool()
def get_current_time() -> str:
    """Returns the current time."""
    return str(datetime.datetime.now())

def main():
    global conn
    global space_center
    global mech_jeb

    conn = krpc.connect("KSP Agent")
    space_center_temp = conn.space_center
    if space_center_temp is None:
        raise IOError("Failed to setup space center.")
    else:
        space_center = space_center_temp
    mech_jeb = conn.mech_jeb  # type: ignore

    mcp.run(transport='stdio')
    
if __name__ == "__main__":
    main()