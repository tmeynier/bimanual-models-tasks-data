# LeRobot integration

This package connects the custom bimanual platform to official LeRobot data,
ACT, and $\pi_{0.5}$ workflows.

- `config_bimanual_simulation.py` and `bimanual_simulation.py` provide the
  `bimanual_simulation` Robot plugin.
- `datasets/` maps backend observations and actions to LeRobot features.
- `recording/` records physical leader/follower demonstrations.
- `training/` launches official `lerobot-train` policy implementations.
- `evaluation/` provides simulation, offline-loss, physical, and remote rollout
  utilities.

Benchmark-specific scenes, IK, and success criteria live in `simulation/tasks/`,
not in this package.
