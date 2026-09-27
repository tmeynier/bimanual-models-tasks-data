# Feetech STS3215 calibration and teleoperation

## Install Feetech support

The repository already uses `.venv`. Install the optional Feetech SDK once:

```bash
.venv/bin/python -m pip install "lerobot[feetech]==0.6.1"
```

The STS3215 bus must have the correct external supply connected. USB powers the
bus adapter, not the servos. Use the voltage specified on your servo model.

## Detect one bus, discover its IDs, and monitor positions

Connect and power one arm's bus, then run:

```bash
.venv/bin/python hardware/read_feetech_positions.py
```

The command discovers the serial adapter and responding servo IDs, then displays
each joint's raw and calibrated position. Press `Ctrl+C` to stop. Useful variants:

```bash
.venv/bin/python hardware/read_feetech_positions.py --format csv
.venv/bin/python hardware/read_feetech_positions.py --port /dev/cu.usbmodem101
.venv/bin/python hardware/scan_feetech_ids.py
```

Every servo on a bus must have a unique ID. Duplicate IDs cannot be reliably
distinguished because their replies collide on the shared wire.

## Read a USB camera

```bash
.venv/bin/python hardware/read_usb_camera.py --list
.venv/bin/python hardware/read_usb_camera.py
```

Select a camera with `--index 0`. Use `--no-display` for a terminal-only test
and `--frames 100` to stop after 100 frames. On macOS, camera access must be
enabled for Terminal or Codex under System Settings, Privacy & Security, Camera.

## Record physical leader-to-follower teleoperation

The physical recorder commands the calibrated follower from the leader while
recording measured joint positions, absolute joint commands, and RGB frames. Its
LeRobot schema contains the 12 arm and finger joints plus the camera image.

```bash
.venv/bin/hf auth login
.venv/bin/python -m hepha_lerobot.recording.physical_teleop \
  --repo-id YOUR_HF_USER/hepha_physical_teleop \
  --root datasets/hepha_physical_teleop \
  --leader-port /dev/cu.usbmodem58FA1019951 \
  --follower-port /dev/cu.usbmodem58FA1020401 \
  --camera-index 0 \
  --episodes 20 \
  --episode-seconds 60 \
  --push-to-hub
```

The camera preview is enabled by default. At the end of each attempt, choose
`SAVE`, `DISCARD`, `RETRY`, or `QUIT`. Only saved episodes are finalized and
uploaded. Use a new `--root` for another dataset; `--overwrite` intentionally
replaces an existing local recording.

## Run a trained ACT or PI0 policy on the physical follower

The physical rollout uses the same 12-joint order, calibrated radians, camera
preprocessing, and saved LeRobot pre/postprocessors as physical recording. It
does not require the leader. Always start torque-disabled:

```bash
.venv/bin/python -m hepha_lerobot.evaluation.physical_rollout \
  --policy-path tmeynier/hepha_act_physical \
  --follower-port /dev/cu.usbmodem58FA1020401 \
  --camera-index 0 \
  --device mps \
  --dry-run
```

The dry run displays predictions without writing goal positions. Support the
unpowered arms and confirm that the checkpoint reports 12 state/action
dimensions and produces finite targets before enabling motion.

For PI0, install the optional dependencies and test torque-disabled first:

```bash
uv pip install --python .venv/bin/python -e ".[pi0]"
.venv/bin/python -m hepha_lerobot.evaluation.physical_rollout \
  --policy-path tmeynier/hepha_pi0_physical_v2 \
  --follower-port /dev/cu.usbmodem58FA1020401 \
  --camera-index 0 \
  --device mps \
  --task "Put the white cube in the bowl" \
  --n-action-steps 10 \
  --fps 5 \
  --watchdog-seconds 120 \
  --dry-run
```

Do not pass `--temporal-ensemble-coeff` to PI0. PI0 uses its own flow-matching
inference and action queue; temporal ensembling is ACT-only.

### Run policy inference on a remote GPU

LeRobot 0.6.1 includes an asynchronous gRPC policy server. Hepha can use that
server while keeping the camera, servo bus, safety limiter, UI, and torque
shutdown on the Mac. The remote server returns postprocessed chunks of 12
absolute joint targets; Hepha overlaps the next request with execution of the
current chunk.

The transport is unauthenticated and serializes Python objects. Never publish
its port to the internet. Bind it to loopback on RunPod and carry it through an
SSH tunnel. On RunPod:

```bash
cd /workspace/hepha
source .venv/bin/activate
uv pip install --python .venv/bin/python -e ".[pi0,remote]"
python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=5 \
  --inference_latency=0 \
  --obs_queue_timeout=10
```

In a separate Mac terminal, use the SSH hostname and port shown by RunPod:

```bash
ssh -i /Users/YOUR_USER/.ssh/runpod_ed25519 \
  -p RUNPOD_SSH_PORT \
  -N \
  -L 8080:127.0.0.1:8080 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  root@RUNPOD_SSH_HOST
```

Install the lightweight client dependencies on the Mac and run a dry test:

```bash
.venv/bin/python -m pip install -e ".[remote]"
.venv/bin/python -m hepha_lerobot.evaluation.physical_rollout \
  --policy-path tmeynier/hepha_pi0_physical_v2 \
  --remote-server 127.0.0.1:8080 \
  --remote-policy-type pi0 \
  --remote-policy-device cuda \
  --follower-port /dev/cu.usbmodem58FA1020401 \
  --camera-index 0 \
  --task "Put the white cube in the bowl" \
  --n-action-steps 10 \
  --fps 5 \
  --remote-load-timeout 300 \
  --remote-request-timeout 10 \
  --watchdog-seconds 15 \
  --duration-seconds 60 \
  --dry-run
```

A lost tunnel, failed request, invalid action, or watchdog expiry exits the
follower context and disables torque. Use the same LeRobot version on both
machines because the protocol sends version-specific Python data classes.

For a powered test, repeat without `--dry-run`. SPACE in the camera window is
the final torque-enable confirmation. Q, Escape, Ctrl+C, a camera or bus error,
or a non-finite action stops the rollout and disables torque. Commands are
clamped to calibrated ranges and rate-limited.

## Calibrate a complete leader or follower arm

Calibration covers all 12 servos on the selected bus with a simultaneous range
sweep. Support the complete arm before starting because torque is disabled on
all joints together.

```bash
.venv/bin/python hardware/calibrate_feetech_positions.py \
  --port /dev/cu.usbmodem58FA1019951
.venv/bin/python hardware/calibrate_feetech_follower.py \
  --port /dev/cu.usbmodem58FA1020401
```

Move every joint through its complete physical range, then press Enter. The
command validates all ranges and atomically updates the selected calibration
file. Ctrl+C cancels without changing the existing JSON.

## Teleoperate a physical follower

Begin with one joint and `--dry-run`:

```bash
.venv/bin/hepha-follow \
  --leader-port /dev/cu.LEADER_PORT \
  --follower-port /dev/cu.FOLLOWER_PORT \
  --joints shoulder_l \
  --dry-run
```

When readings and directions are correct, remove `--dry-run`. The program seeds
each follower goal from its present position, approaches the leader pose at a
limited speed, clamps joint limits during operation, and disables torque on
exit or communication failure.

## Test calibrated joints in MuJoCo

```bash
.venv/bin/python hardware/teleoperate_mujoco_joint.py --id 2
.venv/bin/python hardware/teleoperate_mujoco_joint.py --ids 2 4
```

The utility reads the selected torque-disabled servos and drives only the
corresponding simulated arm joints. Close the viewer or press `Ctrl+C` to stop.

## Record MuJoCo episodes from the physical leader

```bash
.venv/bin/python -m hepha_lerobot.recording.teleop \
  --ids 1 2 3 4 5 6 7 8 9 10 11 12 \
  --repo-id hepha/mujoco_feetech_teleop \
  --root datasets/hepha_mujoco_feetech_teleop \
  --episodes 20 \
  --episode-seconds 60 \
  --fps 30
```

The interactive recorder stores camera observations, simulated joint state, and
the exact actions applied to MuJoCo. It asks whether to save, discard, retry, or
quit after every attempt.
