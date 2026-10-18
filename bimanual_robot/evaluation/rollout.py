"""Run an official LeRobot ACT or PI0.5 policy on a simulation backend."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from bimanual_robot.cli import find_environment_executable
from bimanual_robot.evaluation.policy import load_policy
from bimanual_robot.training.train import resolve_device
from simulation import SimulationConfig, create_backend
from simulation.base import parse_backend_options
from simulation.view import _parse_bool

DEFAULT_TASK = "Move the bimanual robot safely"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_path")
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--backend-option", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument("--direct", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def build_rollout_command(args: argparse.Namespace) -> list[str]:
    """Build the direct command, using mjpython for a visible macOS viewer."""

    mjpython = find_environment_executable("mjpython")
    executable = (
        mjpython
        if sys.platform == "darwin"
        and not args.headless
        and args.backend == "mujoco"
        and Path(mjpython).is_file()
        else sys.executable
    )
    command = [
        executable,
        "-m",
        "bimanual_robot.evaluation.rollout",
        args.policy_path,
        "--direct",
        f"--backend={args.backend}",
        f"--duration={args.duration}",
        f"--fps={args.fps}",
        f"--width={args.width}",
        f"--height={args.height}",
        f"--camera={args.camera}",
        f"--task={args.task}",
        f"--seed={args.seed}",
        f"--device={resolve_device(args.device)}",
        f"--debug={str(args.debug).lower()}",
    ]
    if args.headless:
        command.append("--headless")
    if args.n_action_steps is not None:
        command.append(f"--n-action-steps={args.n_action_steps}")
    if args.temporal_ensemble_coeff is not None:
        command.append(f"--temporal-ensemble-coeff={args.temporal_ensemble_coeff}")
    for option in args.backend_option:
        command.append(f"--backend-option={option}")
    return command


def _policy_observation(
    raw_observation: dict,
    *,
    state_names: tuple[str, ...],
    camera: str,
) -> dict[str, np.ndarray]:
    return {
        OBS_STATE: np.asarray(
            [raw_observation[name] for name in state_names], dtype=np.float32
        ),
        f"{OBS_IMAGES}.{camera}": np.asarray(raw_observation[camera]),
    }


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    policy, _, preprocessor, postprocessor = load_policy(
        args.policy_path,
        device,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    config = SimulationConfig(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        render=True,
        viewer=not args.headless,
        debug=args.debug,
        options=parse_backend_options(args.backend_option),
    )
    with create_backend(args.backend, config) as backend:
        backend.reset(seed=args.seed)
        state_names = tuple(
            name
            for name, feature_type in backend.observation_features.items()
            if feature_type is float
        )
        action_names = tuple(backend.action_features)
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        deadline = time.perf_counter() + args.duration
        interval = 1.0 / args.fps
        while time.perf_counter() < deadline and backend.is_open:
            started = time.perf_counter()
            raw = backend.get_observation()
            observation = _policy_observation(
                raw, state_names=state_names, camera=args.camera
            )
            batch = prepare_observation_for_inference(
                observation,
                torch.device(device),
                task=args.task,
                robot_type=f"bimanual_{backend.name}",
            )
            with torch.inference_mode():
                action = postprocessor(policy.select_action(preprocessor(batch)))
            values = action.squeeze(0).detach().cpu().numpy()
            if values.shape != (len(action_names),):
                raise ValueError(
                    f"Policy returned {values.shape}; backend expects "
                    f"({len(action_names)},)"
                )
            backend.send_action(dict(zip(action_names, values, strict=True)))
            time.sleep(max(0.0, interval - (time.perf_counter() - started)))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.duration <= 0 or args.fps <= 0:
        raise ValueError("--duration and --fps must be positive")
    if args.dry_run:
        print(" ".join(build_rollout_command(args)))
        return
    if args.direct:
        run(args)
        return
    subprocess.run(build_rollout_command(args), check=True)


if __name__ == "__main__":
    main()
