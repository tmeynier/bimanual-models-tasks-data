# Simulation

`simulation.base.SimulationBackend` is the task-neutral boundary used by the
LeRobot adapter. `simulation.backends.mujoco.MujocoBackend` implements the
current robot-only environment: 12 actuators, the floor, collision geometry,
and four fixed robot cameras.

Task objects and behaviors do not belong in the base backend. The seven paper
tasks are organized under `simulation/tasks/`, with one description and one IK
entry point per task. Run `bimanual-tasks` to inspect their current status.

External backends can register through the `bimanual.simulation_backends`
entry-point group and must implement `SimulationBackend`.
