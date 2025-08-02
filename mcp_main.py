import datetime
from mcp.server.fastmcp import Context, FastMCP
import subprocess
import krpc
import asyncio

mcp = FastMCP("Kerbal Space Program MechJeb")

@mcp.tool()
async def launch_rocket(height: float, ctx: Context) -> str:
    """Launches the active vessel on a default trajectory to the supplied height. It's inclination is by default also zero. 
    This is a long running task which will provide progress updates.
    
    Args:
        height: The height of which the rocket will launch to in orbit in meters.
    """

    vessel = conn.space_center.active_vessel
    mech_jeb = conn.mech_jeb

    ascent = mech_jeb.ascent_autopilot
    ascent.desired_orbit_altitude = height

    ascent.desired_inclination = 0

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
        await ctx.report_progress(progress=min(1,max(0,(current_altitude / height))), total=1.0, message=f"Rocket is now at {int(current_altitude)} meters")
    
    return f"Finished launching rocket to {height} meters."

@mcp.tool()
async def say_message(msg: str) -> str:
    """Says a message to the terminal outloud so the user can hear.
    
    Args:
        msg: The message to be said out loud in the terminal.
    """

    subprocess.call(["say", msg])

    return f"Said: {msg}"

@mcp.tool()
def get_current_time() -> str:
    """Returns the current time."""
    return str(datetime.datetime.now())

def setup():
    global conn
    conn = krpc.connect("KSP Agent")

if __name__ == "__main__":
    # Initialize and run the server
    setup()
    try:
        mcp.run(transport='stdio')
    finally:
        conn.close()