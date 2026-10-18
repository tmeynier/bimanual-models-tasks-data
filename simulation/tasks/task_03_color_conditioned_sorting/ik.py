"""IK demonstration for T3 color-conditioned sorting."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from simulation.backends.mujoco import MujocoBackend
from simulation.tasks.base import TaskDefinition
from simulation.tasks.ik import (
    CUBE_CLOSED_GRIPPER,
    CUBE_GRASP_QUATERNIONS,
    FINGER_ACTION_INDICES,
    IKConvergenceError,
    OPEN_GRIPPER,
    CubicTaskController,
    TaskEvaluation,
    Waypoint,
    body_position,
    hand_tcp_position,
    inside_container,
)

TASK = TaskDefinition(
    task_id=3,
    slug="color_conditioned_sorting",
    title="Color-conditioned sorting",
    instruction="Sort the blue and green cubes into their matching cubic containers.",
    nominal_duration_seconds=10,
    subgoals=("place the blue cube", "place the green cube"),
    implemented=True,
)

WORKSPACE_CENTER = (0.0, 0.1175)
WORKSPACE_LENGTH = 0.30
WORKSPACE_WIDTH = 0.105
CUBE_HALF_WIDTH = 0.015
CONTAINER_OUTER_HALF_WIDTH = 0.035
MINIMUM_EDGE_CLEARANCE = 0.030
CUBE_X_RANGE = (
    WORKSPACE_CENTER[0] - WORKSPACE_LENGTH / 2.0 + CUBE_HALF_WIDTH,
    WORKSPACE_CENTER[0] + WORKSPACE_LENGTH / 2.0 - CUBE_HALF_WIDTH,
)
CUBE_Y_RANGE = (
    WORKSPACE_CENTER[1] - WORKSPACE_WIDTH / 2.0 + CUBE_HALF_WIDTH,
    WORKSPACE_CENTER[1] + WORKSPACE_WIDTH / 2.0 - CUBE_HALF_WIDTH,
)
CONTAINER_X_RANGE = (
    WORKSPACE_CENTER[0] - WORKSPACE_LENGTH / 2.0 + CONTAINER_OUTER_HALF_WIDTH,
    WORKSPACE_CENTER[0] + WORKSPACE_LENGTH / 2.0 - CONTAINER_OUTER_HALF_WIDTH,
)
CONTAINER_Y_RANGE = (
    WORKSPACE_CENTER[1] - WORKSPACE_WIDTH / 2.0 + CONTAINER_OUTER_HALF_WIDTH,
    WORKSPACE_CENTER[1] + WORKSPACE_WIDTH / 2.0 - CONTAINER_OUTER_HALF_WIDTH,
)
MAX_LAYOUT_ATTEMPTS = 64
CUBE_CENTER_Z = 0.035
CONTAINER_DELIVERY_TCP_Z = 0.135
CONTAINER_APPROACH_TCP_Z = 0.140
GRASP_TARGET_Z_OFFSET = 0.010
PREGRASP_TARGET_Z_OFFSET = 0.035
COMPACT_ENTRY_GRIPPER = 0.20
CUBE_GRASP_OPEN_GRIPPER = 0.40
GRIPPER_STATE_TOLERANCE = 1e-6
COLORS = ("blue", "green")


@dataclass(frozen=True)
class TaskLayout:
    blue_cube: tuple[float, float, float]
    green_cube: tuple[float, float, float]
    blue_container: tuple[float, float, float]
    green_container: tuple[float, float, float]
    first_color: str
    blue_arm: str
    green_arm: str

    @property
    def color_order(self) -> tuple[str, str]:
        other_color = "green" if self.first_color == "blue" else "blue"
        return (self.first_color, other_color)

    def cube(self, color: str) -> tuple[float, float, float]:
        return self.blue_cube if color == "blue" else self.green_cube

    def container(self, color: str) -> tuple[float, float, float]:
        return self.blue_container if color == "blue" else self.green_container

    def arm(self, color: str) -> str:
        return self.blue_arm if color == "blue" else self.green_arm


def _closest_resting_arm(
    backend: MujocoBackend,
    cube: tuple[float, float, float],
) -> str:
    cube_xy = np.asarray(cube[:2])
    distances = {
        side: float(np.linalg.norm(hand_tcp_position(backend, side)[:2] - cube_xy))
        for side in ("l", "r")
    }
    return min(distances, key=distances.__getitem__)


def _footprints_are_separate(
    first: tuple[float, float, float],
    first_half_width: float,
    second: tuple[float, float, float],
    second_half_width: float,
    edge_clearance: float = MINIMUM_EDGE_CLEARANCE,
) -> bool:
    required_separation = (
        first_half_width + second_half_width + edge_clearance
    )
    return bool(
        abs(first[0] - second[0]) >= required_separation
        or abs(first[1] - second[1]) >= required_separation
    )


def sample_layout(backend: MujocoBackend, seed: int) -> TaskLayout:
    """Sample all four objects across the full Task 3 workspace."""

    rng = np.random.default_rng(seed)
    for _ in range(512):
        containers: list[tuple[float, float, float]] = []
        for _ in COLORS:
            for _ in range(512):
                position = (
                    float(rng.uniform(*CONTAINER_X_RANGE)),
                    float(rng.uniform(*CONTAINER_Y_RANGE)),
                    0.0,
                )
                if all(
                    _footprints_are_separate(
                        position,
                        CONTAINER_OUTER_HALF_WIDTH,
                        other,
                        CONTAINER_OUTER_HALF_WIDTH,
                        edge_clearance=0.0,
                    )
                    for other in containers
                ):
                    containers.append(position)
                    break
            else:
                break
        if len(containers) != len(COLORS):
            continue

        cubes: list[tuple[float, float, float]] = []
        for _ in COLORS:
            for _ in range(512):
                position = (
                    float(rng.uniform(*CUBE_X_RANGE)),
                    float(rng.uniform(*CUBE_Y_RANGE)),
                    CUBE_CENTER_Z,
                )
                clear_of_containers = all(
                    _footprints_are_separate(
                        position,
                        CUBE_HALF_WIDTH,
                        container,
                        CONTAINER_OUTER_HALF_WIDTH,
                    )
                    for container in containers
                )
                clear_of_cubes = all(
                    _footprints_are_separate(
                        position,
                        CUBE_HALF_WIDTH,
                        other,
                        CUBE_HALF_WIDTH,
                        edge_clearance=0.0,
                    )
                    for other in cubes
                )
                if clear_of_containers and clear_of_cubes:
                    cubes.append(position)
                    break
            else:
                break
        if len(cubes) != len(COLORS):
            continue

        container_order = rng.permutation(len(COLORS))
        cube_order = rng.permutation(len(COLORS))
        blue_container = containers[int(container_order[0])]
        green_container = containers[int(container_order[1])]
        blue_cube = cubes[int(cube_order[0])]
        green_cube = cubes[int(cube_order[1])]
        first_color = COLORS[int(rng.integers(0, len(COLORS)))]
        return TaskLayout(
            blue_cube=blue_cube,
            green_cube=green_cube,
            blue_container=blue_container,
            green_container=green_container,
            first_color=first_color,
            blue_arm=_closest_resting_arm(backend, blue_cube),
            green_arm=_closest_resting_arm(backend, green_cube),
        )
    raise RuntimeError("Could not sample four separated Task 3 objects")


def _layout_attempt_seed(seed: int, attempt: int) -> int:
    """Derive a reproducible fresh layout seed while preserving attempt zero."""

    if attempt == 0:
        return seed
    sequence = np.random.SeedSequence((seed, attempt))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _configure_workspace_debug_rectangle(backend: MujocoBackend) -> None:
    """Resize the shared debug marker to Task 3's sampling workspace."""

    body_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "workspace_debug_rectangle",
    )
    backend.model.body_pos[body_id] = (*WORKSPACE_CENTER, 0.0225)
    half_length = WORKSPACE_LENGTH / 2.0
    half_width = WORKSPACE_WIDTH / 2.0
    marker_geometry = {
        "task_ik_marker_workspace_rectangle_back": ((0.0, -half_width, 0.0), half_length),
        "task_ik_marker_workspace_rectangle_right": ((half_length, 0.0, 0.0), half_width),
        "task_ik_marker_workspace_rectangle_front": ((0.0, half_width, 0.0), half_length),
        "task_ik_marker_workspace_rectangle_left": ((-half_length, 0.0, 0.0), half_width),
    }
    for name, (position, half_segment_length) in marker_geometry.items():
        geom_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        backend.model.geom_pos[geom_id] = position
        backend.model.geom_size[geom_id, 1] = half_segment_length


def _set_free_body_position(
    backend: MujocoBackend,
    body_name: str,
    position: tuple[float, float, float],
) -> None:
    body_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = int(backend.model.body_jntadr[body_id])
    qpos_id = int(backend.model.jnt_qposadr[joint_id])
    dof_id = int(backend.model.jnt_dofadr[joint_id])
    backend.data.qpos[qpos_id : qpos_id + 3] = position
    backend.data.qpos[qpos_id + 3 : qpos_id + 7] = (1.0, 0.0, 0.0, 0.0)
    backend.data.qvel[dof_id : dof_id + 6] = 0.0


def _set_mocap_body_position(
    backend: MujocoBackend,
    body_name: str,
    position: tuple[float, float, float],
) -> None:
    body_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = int(backend.model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"Body {body_name!r} is not a mocap body")
    backend.data.mocap_pos[mocap_id] = position
    backend.data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)


def _pick_and_place_waypoints(
    layout: TaskLayout,
    color: str,
    start_time: float,
    *,
    final_sequence: bool,
) -> tuple[Waypoint, ...]:
    cube_x, cube_y, cube_z = layout.cube(color)
    container_x, container_y, _ = layout.container(color)
    side = layout.arm(color)
    arm_key = "left" if side == "l" else "right"
    gripper_key = f"{arm_key}_gripper"
    preferred_quaternion_key = f"{arm_key}_preferred_quaternion"
    grasp_quaternion = CUBE_GRASP_QUATERNIONS[side]
    lift_x = 0.040 if side == "l" else -0.040

    def point(
        offset: float,
        xyz: tuple[float, float, float],
        **kwargs,
    ) -> Waypoint:
        return Waypoint(start_time + offset, **{arm_key: xyz, **kwargs})

    return (
        Waypoint(
            start_time + 0.5,
            **{gripper_key: COMPACT_ENTRY_GRIPPER},
            label=f"compact {side} gripper for {color} cube",
        ),
        point(
            1.7,
            (cube_x, cube_y, cube_z + PREGRASP_TARGET_Z_OFFSET),
            **{
                gripper_key: COMPACT_ENTRY_GRIPPER,
                preferred_quaternion_key: grasp_quaternion,
            },
            label=f"position compact gripper above {color} cube",
        ),
        Waypoint(
            start_time + 2.1,
            **{gripper_key: CUBE_GRASP_OPEN_GRIPPER},
            label=f"open gripper before reaching {color} cube",
        ),
        point(
            2.7,
            (cube_x, cube_y, cube_z + GRASP_TARGET_Z_OFFSET),
            **{
                gripper_key: CUBE_GRASP_OPEN_GRIPPER,
                preferred_quaternion_key: grasp_quaternion,
            },
            label=f"descend open gripper to {color} cube",
        ),
        point(
            3.2,
            (cube_x, cube_y, cube_z + GRASP_TARGET_Z_OFFSET),
            **{
                gripper_key: CUBE_CLOSED_GRIPPER,
                preferred_quaternion_key: grasp_quaternion,
            },
            label=f"close gripper around {color} cube",
        ),
        point(
            4.2,
            (lift_x, cube_y, 0.160),
            **{
                gripper_key: CUBE_CLOSED_GRIPPER,
                preferred_quaternion_key: grasp_quaternion,
            },
            label=f"lift {color} cube",
        ),
        point(
            5.5,
            (container_x, container_y, CONTAINER_APPROACH_TCP_Z),
            **{gripper_key: CUBE_CLOSED_GRIPPER},
            label=f"move {color} cube above matching container",
        ),
        point(
            6.5,
            (container_x, container_y, CONTAINER_DELIVERY_TCP_Z),
            **{gripper_key: CUBE_CLOSED_GRIPPER},
            label=f"lower {color} cube into matching container",
        ),
        Waypoint(
            start_time + 7.5,
            **{gripper_key: OPEN_GRIPPER},
            label=f"release {color} cube",
        ),
        point(
            8.4,
            (container_x, container_y, CONTAINER_APPROACH_TCP_Z),
            **{gripper_key: OPEN_GRIPPER},
            label=f"retreat after placing {color} cube",
        ),
        Waypoint(
            start_time + 9.5,
            return_to_rest=True,
            label=(
                "both arms return to rest"
                if final_sequence
                else f"return to rest after {color} cube"
            ),
        ),
    )


def create_controller(backend: MujocoBackend) -> CubicTaskController:
    state: dict[str, object] = {}
    cached_layout_key: tuple[int, int] | None = None
    cached_layout: TaskLayout | None = None
    layout_attempt = 0

    def initialize_scene(backend: MujocoBackend, seed: int) -> None:
        nonlocal cached_layout, cached_layout_key
        _configure_workspace_debug_rectangle(backend)
        layout_key = (seed, layout_attempt)
        if cached_layout is None or cached_layout_key != layout_key:
            cached_layout = sample_layout(
                backend,
                _layout_attempt_seed(seed, layout_attempt),
            )
            cached_layout_key = layout_key
        layout = cached_layout
        state["layout"] = layout
        state["layout_attempts"] = layout_attempt + 1
        _set_free_body_position(backend, "blue_cube", layout.blue_cube)
        _set_free_body_position(backend, "green_cube", layout.green_cube)
        _set_mocap_body_position(backend, "blue_container", layout.blue_container)
        _set_mocap_body_position(backend, "green_container", layout.green_container)
        mujoco.mj_forward(backend.model, backend.data)
        backend.update_hand_tcp_markers()
        backend.sync_viewer()

    def waypoints(seed: int) -> tuple[Waypoint, ...]:
        del seed
        layout = state["layout"]
        if not isinstance(layout, TaskLayout):
            raise TypeError("Task 3 layout was not initialized")
        first_color, second_color = layout.color_order
        first_start = 1.0
        second_start = 10.5
        state["grasp_windows"] = {
            first_color: (first_start + 3.2, first_start + 7.5),
            second_color: (second_start + 3.2, second_start + 7.5),
        }
        return (
            Waypoint(
                0.0,
                left_gripper=OPEN_GRIPPER,
                right_gripper=OPEN_GRIPPER,
                label="both arms at rest",
            ),
            Waypoint(
                1.0,
                left_gripper=OPEN_GRIPPER,
                right_gripper=OPEN_GRIPPER,
                label="hold rest pose",
            ),
            *_pick_and_place_waypoints(
                layout,
                first_color,
                first_start,
                final_sequence=False,
            ),
            *_pick_and_place_waypoints(
                layout,
                second_color,
                second_start,
                final_sequence=True,
            ),
        )

    def evaluate(backend: MujocoBackend) -> TaskEvaluation:
        layout = state["layout"]
        if not isinstance(layout, TaskLayout):
            raise TypeError("Task 3 layout was not initialized")
        blue = body_position(backend, "blue_cube")
        green = body_position(backend, "green_cube")
        blue_sorted = inside_container(
            blue,
            layout.blue_container[:2],
            half_width=0.040,
        )
        green_sorted = inside_container(
            green,
            layout.green_container[:2],
            half_width=0.040,
        )
        return TaskEvaluation(
            success=blue_sorted and green_sorted,
            subgoals={"blue_sorted": blue_sorted, "green_sorted": green_sorted},
            details={
                "blue_position": blue.tolist(),
                "green_position": green.tolist(),
                "sampled_blue_cube_position": layout.blue_cube,
                "sampled_green_cube_position": layout.green_cube,
                "sampled_blue_container_position": layout.blue_container,
                "sampled_green_container_position": layout.green_container,
                "color_order": layout.color_order,
                "blue_arm": layout.blue_arm,
                "green_arm": layout.green_arm,
                "layout_attempts": state["layout_attempts"],
            },
        )

    class TaskThreeController(CubicTaskController):
        def reset(self, *, seed: int) -> None:
            nonlocal layout_attempt
            last_error: IKConvergenceError | None = None
            for layout_attempt in range(MAX_LAYOUT_ATTEMPTS):
                for color in COLORS:
                    self.backend.detach_free_body(f"{color}_cube")
                try:
                    super().reset(seed=seed)
                except IKConvergenceError as error:
                    last_error = error
                    continue
                return
            assert last_error is not None
            raise IKConvergenceError(
                f"Task 3 seed {seed} had no reachable sampled layout after "
                f"{MAX_LAYOUT_ATTEMPTS} actual-trajectory attempts; last failure: "
                f"{last_error}"
            ) from last_error

        def action(self, progress: float) -> np.ndarray:
            action = super().action(progress)
            layout = state["layout"]
            grasp_windows = state["grasp_windows"]
            if not isinstance(layout, TaskLayout) or not isinstance(
                grasp_windows, dict
            ):
                raise TypeError("Task 3 grasp state was not initialized")
            elapsed = float(np.clip(progress, 0.0, 1.0)) * self.waypoints[-1].time
            for color in COLORS:
                side = layout.arm(color)
                finger_index = FINGER_ACTION_INDICES[side]
                close_time, release_time = grasp_windows[color]
                fingers_closed = bool(
                    action[finger_index]
                    <= CUBE_CLOSED_GRIPPER + GRIPPER_STATE_TOLERANCE
                )
                body_name = f"{color}_cube"
                cube_attached = self.backend.body_is_attached(body_name)
                should_attach = bool(
                    close_time <= elapsed < release_time and fingers_closed
                )
                if should_attach and not cube_attached:
                    self.backend.attach_free_body_to_hand_tcp(body_name, side)
                elif not should_attach and cube_attached:
                    self.backend.detach_free_body(body_name)
            return action

    return TaskThreeController(
        backend,
        waypoints,
        evaluate,
        scene_initializer=initialize_scene,
        ik_seed_factory=lambda seed: seed,
    )
