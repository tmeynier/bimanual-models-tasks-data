# Quickstart

## Environment

```bash
cd /path/to/bimanual-models-tasks-data
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Add `pi05` when running $\pi_{0.5}$ experiments, and `remote` when using the
asynchronous policy server:

```bash
python -m pip install -e ".[dev,pi05,remote]"
```

## Robot-only MuJoCo environment

```bash
bimanual-view --backend mujoco
```

On macOS the command finds the environment's `mjpython` automatically. Use the
Left/Right Arrow keys for the four fixed views and Escape for the free camera.
To inspect collision shapes and camera axes:

```bash
bimanual-view --backend mujoco --debug
```

The base MJCF intentionally contains only the bimanual robot and floor. Task
objects belong to task-local scenes.

## Inspect benchmark task status

```bash
bimanual-tasks
```

The seven packages under `simulation/tasks/` are the source of truth for task
instructions, nominal durations, subgoals, scenes, and IK controllers. All
seven tasks are implemented.

Run an implemented demonstration in the native MuJoCo viewer:

```bash
bimanual-task-demo 1
```

Choose a task ID from 1 through 7. Add `--debug` to expose collision geometry,
`--seed N` to select the request and randomized layout, or `--headless --no-real-time`
for automated evaluation. Arm motion between Cartesian waypoints uses numerical
IK, while grasping and object motion rely on MuJoCo contact physics.

## Record real demonstrations

Calibrate the leader and follower before applying torque. Then record a task by
using its exact paper instruction:

```bash
bimanual-physical-record \
  --repo-id YOUR_USER/bimanual-t01-real \
  --root datasets/bimanual-t01-real \
  --task "Pick up the cube and place it inside the target cubic container."
```

The current recorder supports one selected camera stream. The paper protocol
requires four synchronized robot views, so a paper reproduction must extend or
compose capture accordingly before collecting the final benchmark datasets.

## Train official LeRobot policies

ACT:

```bash
bimanual-train \
  --repo-id YOUR_USER/bimanual-t01-real \
  --root datasets/bimanual-t01-real \
  --policy-type act \
  --output-dir outputs/act-t01-real
```

$\pi_{0.5}$ uses the policy type registered by the installed LeRobot release.
List available names before launching:

```bash
bimanual-policies
```

Pass additional official `lerobot-train` options after the repository wrapper's
arguments. The wrapper does not maintain custom ACT derivatives.

## Evaluate

Print a simulation rollout command without running it:

```bash
bimanual-evaluate outputs/act-t01-real/checkpoints/last/pretrained_model \
  --task "Pick up the cube and place it inside the target cubic container." \
  --dry-run
```

T1–T7 provide task-local scenes and success evaluators. Physical rollout has
explicit safety gates:

```bash
bimanual-physical-rollout \
  --policy-path outputs/act-t01-real/checkpoints/last/pretrained_model \
  --task "Pick up the cube and place it inside the target cubic container." \
  --dry-run
```

Run tests and formatting checks with:

```bash
pytest
ruff check .
```
