# KRPC Smart Autopilot

> [!WARNING]
> Currently this is in very early stages of development. Likely everything is going to change.

## Project Goals
- [ ] MCP server bindings for MechJeb actions
  - [x] *Launch to orbit*
  - [ ] *Land from orbit*
  - [ ] *Change orbit*
  - [ ] *Set target*
  - [ ] *Rendezvous with target*
  - [ ] *Return from moon*
  - [ ] Docking to target
  - [ ] Interplanetary transfer
- [ ] MCP server bindngs for information
  - [ ] *Vessel information*
  - [ ] *Planetary information*
  - [ ] Delta-v information
- [ ] [Model-and-acceleration pursuit for generic path](https://arxiv.org/abs/2209.04346)

Items in *italics* are necessary for what I would consider the first major release which would be capable of performing a Mun or Minmus landing.

The MechJeb2 KRPC library is somewhat buggy and I would perfer to go strictly with KRPC for all control if possible, as that would remove the need for MechJeb2 or MechJeb2 KRPC and would allow multiple craft to be piloted simultaneously with an extended physics range. Ideally a Model-and-acceleration pursuit model plus a generic orbital transfer calculator and path planner would be able to replace MechJeb and function as a native and hopefully more accurate model. (Aerodynamics really makes things difficult LOL.)

KRPC Smart Autopilot is a bot that using Poliastro and KRPC and simulation can plan very precise maneuvers in KSP using simulation optimization and then execute them in KSP.

At the moment the physics simulation can simulate simple launch trajectories through an atmosphere to apoapsis. This simulation includes a high quality approximation of drag, thrust, and mass changes.
