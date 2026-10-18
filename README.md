# Models, tasks, and data for low-cost bimanual manipulation

This repository is the reference implementation for the paper **“What Makes a
Generalist Robot Policy? A Study of Models, Tasks, and Data for Low-Cost
Bimanual Manipulation.”** It contains the CAD, hardware interface, MuJoCo model,
LeRobot integration, task specifications, and experiment entry points for an
unnamed custom-built fixed-base bimanual robot.

The robot has 12 controlled joints: five arm joints and one gripper joint on
each side. The paper's observation setup uses four RGB views—head, lower
body/floor, left wrist, and right wrist. The study compares official LeRobot ACT and
$\pi_{0.5}$ policies using real-robot, simulated, multi-task, and (for
$\pi_{0.5}$) egocentric-human data.

## Current status

- The robot-only MuJoCo environment and all four fixed cameras are implemented;
  the current LeRobot adapter selects one camera per process, while final paper
  datasets require synchronized capture from all four.
- Physical Feetech calibration, teleoperation, recording, and policy rollout are
  available.
- ACT/$\pi_{0.5}$ training and evaluation launchers use the official LeRobot
  implementations.
- All seven paper tasks have task-local MuJoCo scenes, numerical IK
  demonstrations, and deterministic success evaluators.
- Cubic containers replace bowls in T1, T2, T3, and T6. Every cube is 30 mm per
  side. Every container is 70 × 70 × 35 mm externally, with a 60 × 60 mm clear
  opening, 30 mm usable depth, and 5 mm walls. Arm trajectories are solved with
  numerical IK. Cubes use MuJoCo contact physics plus the reversible grasp lock
  while a gripper is fully closed, as documented by each task.

Run `bimanual-tasks` to see the implementation status from the source itself.

## Benchmark tasks

| ID | Task | Nominal duration |
| --- | --- | ---: |
| T1 | Basic object placement: cube into target container | 10 s |
| T2 | Object retrieval: cube from container to workspace | 10 s |
| T3 | Color-conditioned sorting into matching containers | 10 s |
| T4 | Number-conditioned drawer opening | 10 s |
| T5 | Green cube stacked stably on blue cube | 10 s |
| T6 | Closest-arm mid-air handover between containers | 10 s |
| T7 | Warehouse stocking into a requested drawer | 20 s |

Each task owns one directory under [`simulation/tasks`](simulation/tasks), with
its specification in `README.md`, task scene in `scene.xml`, and task-specific IK
entry point in `ik.py`. Assets, initialization, success checks, and demonstrations
remain task-local.

## Install and inspect the robot

Python 3.12 is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
bimanual-view --backend mujoco
```

Run an implemented task demonstration with:

```bash
bimanual-task-demo 1
```

On macOS, `bimanual-view` automatically relaunches with `mjpython` when it is
available. In the viewer:

- use Left/Right Arrow to cycle the four fixed cameras;
- press Escape to return to MuJoCo's free camera;
- pass `--debug` to show collision geometry and camera axes;
- use MuJoCo's native Control panel to move the 12 actuators.

The viewer therefore exposes five viewpoints in total: MuJoCo's free camera plus
`head_camera`, `wrist_l_camera`, `wrist_r_camera`, and `floor_camera`.

## Repository map

```text
paper/                 LaTeX source of the accompanying paper
cad/                   printable robot and task-fixture CAD
hardware/              calibration, Feetech control, and camera tools
simulation/
  backends/mujoco/     task-neutral robot MJCF and meshes
  tasks/               one specification and IK module per paper task
bimanual_robot/        LeRobot adapter, data, training, and evaluation code
scripts/               convenience wrappers for common commands
tests/                 robot, camera, hardware, data, and task-contract tests
```

See [`docs/repository-architecture.md`](docs/repository-architecture.md) for the
paper-to-code mapping and [`docs/quickstart.md`](docs/quickstart.md) for commands.

## Reproducibility contract

The paper defines the experimental scope and is stored at
[`paper/main.tex`](paper/main.tex). Code and datasets should preserve:

- the same 12-dimensional joint action space across simulation and hardware;
- the four robot RGB views described in the paper;
- separate single-task and joint seven-task training runs;
- matched real-only versus real-plus-simulation comparisons for ACT and
  $\pi_{0.5}$;
- egocentric-human-data ablations only for $\pi_{0.5}$;
- complete success, ordered partial success, inference latency, recovery under
  controlled perturbations, and 95% confidence intervals.

Generated datasets, logs, checkpoints, and experiment outputs are deliberately
ignored by Git. Publish immutable dataset/model artifacts separately and record
their identifiers and exact configurations with each reported result.
