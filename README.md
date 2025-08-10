# KRPC Smart Autopilot

This is an MCP server for KSP. To run make sure you have KSP running, the KRPC, MechJeb2, and KRPC.MechJeb mods installed. Make sure you have uv installed as well.

You can then setup any MCP capable agent you want to locally access this tool, though gemini-cli is what I have been testing with

To use gemini-cli just install gemini-cli then run `gemini` while in the root directory of this project and it should automatically be able to use the tool due to [`.gemini/settings.json`](.gemini/settings.json) as well as [`AGENTS.md`](AGENTS.md) being preconfigured.

> [!WARNING]
> Currently this is in very early stages of development. Likely everything is going to change.

## Project goals

- [ ] MCP server bindings for MechJeb actions
  - [x] *Launch to orbit*
  - [ ] *Land from orbit*
  - [x] *Change orbit*
  - [x] *Set target to planet*
  - [x] *Rendezvous with target*
  - [x] *Return from moon*
  - [ ] Set target to vessel
  - [ ] Docking to target
  - [ ] Interplanetary transfer
- [ ] MCP server bindngs for information
  - [x] *Vessel orbit information*
  - [ ] *Planetary information*
  - [ ] Vessel information
  - [ ] Delta-v information
- [ ] [Model-and-acceleration pursuit for generic path](https://arxiv.org/abs/2209.04346)

> [!NOTE]
> Items in *italics* are necessary for what I would consider the first major release which would be capable of performing a Mun or Minmus landing and return.

## How it works

The current entrypoint is [`krpc_server.py`](krpc_server.py) which runs an MCP server using the `mcp[cli]` library AKA FastMCP. From here the MCP server provides individual tools to

- launch into orbit
- set target body
- get possible bodies
- get orbital info

and a bunch of tools that each perform some sort of orbital manuever including but not limited to

- Hohmann transfers
- circularization
- return from moon transfers

Additionally there are some tips in [`AGENTS.md`](AGENTS.md) for any agents using this.

## Current issues

The MechJeb2 KRPC library is somewhat buggy. Additionally, the agents can be pretty stupid and not understand the best way to make a manuever. This leads to some pretty rediculous delta-v charges if you need to change your inclination a lot for example.

## Future plans

### MechJeb2 independance

Ideally the project would not rely on MechJeb2 at all and only on KRPC. This requires a custom launch generator (not too hard to create) and a way to transfer orbits.

### Unified orbital transfer

Ideally, there would be one unified orbital transfer tool which would calculate the optimal series of impulse manuevers to transfer to any other given orbit subject to some tolerances. This would make tool calling much easier on the agent and reduce its ability to make mistakes.
