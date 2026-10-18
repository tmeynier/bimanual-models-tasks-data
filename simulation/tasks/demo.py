"""Run one task-specific IK demonstration in MuJoCo."""

from __future__ import annotations

import argparse
import time

from simulation import SimulationConfig
from simulation.tasks import create_task_backend, create_task_controller, get_task
from simulation.view import _ensure_mjpython_on_macos

FINAL_REST_SETTLE_SECONDS = 2.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", type=int, choices=range(1, 8))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--real-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace the visible demonstration in real time (default: true).",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace):
    task = get_task(args.task_id)
    config = SimulationConfig(
        camera=args.camera,
        fps=args.fps,
        render=False,
        viewer=False,
        debug=args.debug,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(args.task_id, config) as backend:
        controller = create_task_controller(args.task_id, backend)
        controller.reset(seed=args.seed)
        if not args.headless:
            backend.open_viewer(debug=args.debug)
        if args.debug:
            if args.task_id in (1, 2):
                release_target = (
                    "container release target" if args.task_id == 1 else "workspace release target"
                )
                workspace_boundary = (
                    "30 x 10.5 cm Task 1 sampling workspace"
                    if args.task_id == 1
                    else "30 x 10.5 cm Task 2 sampling workspace"
                )
                print(
                    "Frames: yellow origin = cube object frame; "
                    f"white origin = {release_target}; "
                    "magenta origin = fingertip TCP midway between fingers; "
                    f"cyan rectangle = {workspace_boundary}; "
                    "XYZ = red/green/blue."
                )
            elif args.task_id == 4:
                print(
                    "Frames: white origin = selected drawer-handle IK target; "
                    "magenta origin = fingertip TCP midway between fingers; "
                    "drawer pull = closed gripper moving along rack-local +Y; "
                    "cyan rectangle = 50 x 30 cm workspace boundary; "
                    "XYZ = red/green/blue."
                )
            elif args.task_id == 3:
                print("Cyan rectangle = 30 x 10.5 cm Task 3 sampling workspace.")
            elif args.task_id == 5:
                print("Cyan rectangle = 30 x 10.5 cm Task 5 sampling workspace.")
            elif args.task_id == 6:
                print("Cyan rectangle = 30 x 10.5 cm Task 6 sampling workspace.")
            elif args.task_id == 7:
                print(
                    "Frames: white drawer frame = requested handle IK; "
                    "white stocking frame = cube release target inside the open drawer; "
                    "yellow frames = randomized cubes; magenta frame = fingertip TCP; "
                    "cyan rectangle = 30 x 10.5 cm cube sampling workspace."
                )
            else:
                print("Cyan rectangle = 50 x 30 cm workspace boundary.")
        frames = task.nominal_duration_seconds * args.fps
        for frame in range(frames + 1):
            started = time.perf_counter()
            progress = frame / frames
            backend.send_action(controller.action(progress))
            backend.step()
            if args.real_time and not args.headless:
                time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - started)))
        final_action = controller.action(1.0)
        settle_frames = round(FINAL_REST_SETTLE_SECONDS * args.fps)
        for _ in range(settle_frames):
            started = time.perf_counter()
            backend.send_action(final_action)
            backend.step()
            if args.real_time and not args.headless:
                time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - started)))
        result = controller.evaluate()
        status = "SUCCESS" if result.success else "FAILURE"
        print(f"T{task.task_id} {task.title}: {status}")
        print(f"Subgoals: {result.subgoals}")
        print(f"Details: {result.details}")
        return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not args.headless:
        _ensure_mjpython_on_macos("simulation.tasks.demo")
    result = run(args)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
