# Paper-to-code architecture

The accompanying paper is the experimental specification. This document maps
its concepts to code and records which boundaries must remain stable as the
implementation grows.

## Embodiment and observations

The paper's robot is a fixed-base 12-DoF bimanual platform. Joint names and
ordering are shared by `hardware/axes.py` and
`simulation/backends/mujoco/backend.py`. The robot MJCF is
`simulation/backends/mujoco/assets/robot.xml`.

The four paper views map to `head_camera`, `floor_camera`, `wrist_l_camera`, and
`wrist_r_camera`. MuJoCo's free camera is for interactive inspection only and is
not a learned-policy observation. The cameras exist in the MJCF, but the current
simulation and physical LeRobot adapters expose one selected stream per process.
Synchronized four-view capture is therefore an explicit prerequisite for final
paper dataset collection, not a capability that the current code claims.

## Task boundary

Every task lives in exactly one `simulation/tasks/task_NN_*` package:

```text
task_NN_name/
  README.md     task instruction, owned state, and success definition
  __init__.py   public TASK metadata
  scene.xml     task-local MuJoCo objects and fixtures
  ik.py         task-specific Cartesian plan and controller factory
```

Shared numerical IK and waypoint execution live in `simulation/tasks/ik.py`, but
waypoint sequences, object assets, initialization, subgoal evaluation, and task
timing remain inside the owning task. All seven tasks, T1–T7, are implemented.
This prevents assumptions from one benchmark task leaking into the
task-neutral robot backend.

The current demonstration generator solves arm motion numerically. Objects are
never attached to or teleported with a gripper: grasping, carrying, release, and
settling are governed by MuJoCo contact physics.

## Models and data

`bimanual_robot/` is the LeRobot-facing layer:

- `bimanual_simulation.py` adapts any registered backend to LeRobot's Robot API;
- `datasets/` constructs native LeRobot datasets;
- `training/` launches official policy training;
- `evaluation/` loads and evaluates official ACT and $\pi_{0.5}$ policies;
- `recording/` contains physical demonstration capture.

The paper's main matrix must be represented in artifact names and run metadata:

| Axis | Required levels |
| --- | --- |
| Model | ACT, $\pi_{0.5}$ |
| Robot data | real only, real + simulation |
| Task training | seven specialists, one joint seven-task model |
| Human data | absent/present for $\pi_{0.5}$ only |
| Evaluation | nominal and controlled perturbations |

Generated datasets and model weights are not versioned in Git. Their hub IDs,
commits, seeds, configurations, and result tables should be recorded in an
experiment manifest when the final runs are produced.

## Metrics

Task packages should expose ordered subgoals so evaluation can report complete
success and paper-defined partial success consistently. Evaluation additionally
records inference latency and controlled-perturbation recovery. Aggregation must
retain trial counts and 95% confidence intervals rather than only mean scores.

## Implementation order

For each task: finalize its scene and initial-state sampler; implement its local
IK demonstration controller; add deterministic success/subgoal tests; record
simulation demonstrations; validate against physical demonstrations; then add
the task to specialist and multi-task experiment manifests. A task remains
`implemented=False` until all first three steps pass tests.
