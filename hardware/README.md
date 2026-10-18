# Hardware tools

This directory contains the physical interface for the unnamed 12-DoF bimanual
robot: Feetech discovery, calibration, leader/follower control, USB camera
capture, and safety-limited teleoperation.

Typical setup:

```bash
source .venv/bin/activate
bimanual-leader-calibrate
bimanual-follower-calibrate
bimanual-follow
```

Inspect the available command options before applying power:

```bash
bimanual-follow --help
bimanual-physical-record --help
bimanual-physical-rollout --help
```

Physical policy rollout first loads the policy, keeps follower torque disabled
for a prediction check, and requires an explicit operator confirmation before
enabling torque. Keep the workspace clear and power removal within reach.

The paper uses four synchronized RGB views. The current physical recorder accepts
one selected USB camera, so final paper-data collection requires a synchronized
four-camera capture path using the canonical names `head_camera`, `floor_camera`,
`wrist_l_camera`, and `wrist_r_camera`.
