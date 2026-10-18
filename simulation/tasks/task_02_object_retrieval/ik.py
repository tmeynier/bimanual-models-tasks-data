"""IK demonstration for T2 object retrieval."""

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
    task_id=2,
    slug="object_retrieval",
    title="Object retrieval",
    instruction=(
        "Pick up the cube from the cubic container and place it on the workspace "
        "outside the container."
    ),
    nominal_duration_seconds=10,
    subgoals=("grasp the cube inside the container", "place it on the workspace"),
    implemented=True,
)

WORKSPACE_CENTER = (0.0, 0.1175)
WORKSPACE_LENGTH = 0.30
WORKSPACE_WIDTH = 0.105
CUBE_HALF_WIDTH = 0.015
CONTAINER_OUTER_HALF_WIDTH = 0.035
CONTAINER_INTERIOR_HALF_WIDTH = 0.030
MINIMUM_EDGE_CLEARANCE = 0.030
INSIDE_CONTAINER_MARGIN = 0.005
MAXIMUM_CUBE_OFFSET = (
    CONTAINER_INTERIOR_HALF_WIDTH
    - CUBE_HALF_WIDTH
    - INSIDE_CONTAINER_MARGIN
)
CONTAINER_X_RANGE = (
    WORKSPACE_CENTER[0] - WORKSPACE_LENGTH / 2.0 + CONTAINER_OUTER_HALF_WIDTH,
    WORKSPACE_CENTER[0] + WORKSPACE_LENGTH / 2.0 - CONTAINER_OUTER_HALF_WIDTH,
)
CONTAINER_Y_RANGE = (
    WORKSPACE_CENTER[1] - WORKSPACE_WIDTH / 2.0 + CONTAINER_OUTER_HALF_WIDTH,
    WORKSPACE_CENTER[1] + WORKSPACE_WIDTH / 2.0 - CONTAINER_OUTER_HALF_WIDTH,
)
CUBE_OFFSET_RANGE = (-MAXIMUM_CUBE_OFFSET, MAXIMUM_CUBE_OFFSET)
DESTINATION_X_RANGE = (
    WORKSPACE_CENTER[0] - WORKSPACE_LENGTH / 2.0 + CUBE_HALF_WIDTH,
    WORKSPACE_CENTER[0] + WORKSPACE_LENGTH / 2.0 - CUBE_HALF_WIDTH,
)
DESTINATION_Y_RANGE = (
    WORKSPACE_CENTER[1] - WORKSPACE_WIDTH / 2.0 + CUBE_HALF_WIDTH,
    WORKSPACE_CENTER[1] + WORKSPACE_WIDTH / 2.0 - CUBE_HALF_WIDTH,
)
MAX_LAYOUT_ATTEMPTS = 32
CONTAINER_CUBE_CENTER_Z = 0.045
WORKSPACE_CUBE_CENTER_Z = 0.035
COMPACT_ENTRY_GRIPPER = 0.20
CUBE_GRASP_OPEN_GRIPPER = 0.40
GRASP_TARGET_Z_OFFSET = 0.005
CUBE_LOCK_TIME = 4.8
PREGRASP_TARGET_Z_OFFSET = 0.035
CONTAINER_ENTRY_TARGET_Z_OFFSET = 0.015
CONTAINER_CLEAR_TCP_Z = 0.125
WORKSPACE_RELEASE_CLEARANCE = 0.030
WORKSPACE_DELIVERY_TCP_Z = (
    WORKSPACE_CUBE_CENTER_Z
    + GRASP_TARGET_Z_OFFSET
    + WORKSPACE_RELEASE_CLEARANCE
)
GRIPPER_STATE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class TaskLayout:
    cube: tuple[float, float, float]
    container: tuple[float, float, float]
    destination: tuple[float, float, float]
    working_arm: str


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
    cube: tuple[float, float, float],
    container: tuple[float, float, float],
) -> bool:
    required_separation = (
        CUBE_HALF_WIDTH + CONTAINER_OUTER_HALF_WIDTH + MINIMUM_EDGE_CLEARANCE
    )
    return bool(
        abs(cube[0] - container[0]) >= required_separation
        or abs(cube[1] - container[1]) >= required_separation
    )


def sample_layout(backend: MujocoBackend, seed: int) -> TaskLayout:
    """Sample the source and destination across the full Task 2 workspace."""

    rng = np.random.default_rng(seed)
    for _ in range(512):
        container = (
            float(rng.uniform(*CONTAINER_X_RANGE)),
            float(rng.uniform(*CONTAINER_Y_RANGE)),
            0.0,
        )
        cube = (
            container[0] + float(rng.uniform(*CUBE_OFFSET_RANGE)),
            container[1] + float(rng.uniform(*CUBE_OFFSET_RANGE)),
            CONTAINER_CUBE_CENTER_Z,
        )
        destination = (
            float(rng.uniform(*DESTINATION_X_RANGE)),
            float(rng.uniform(*DESTINATION_Y_RANGE)),
            WORKSPACE_CUBE_CENTER_Z,
        )
        if not _footprints_are_separate(
            destination,
            container,
        ):
            continue
        return TaskLayout(
            cube,
            container,
            destination,
            _closest_resting_arm(backend, cube),
        )
    raise RuntimeError(
        "Could not sample a Task 2 retrieval layout with an outside destination"
    )


def _layout_attempt_seed(seed: int, attempt: int) -> int:
    """Derive a reproducible fresh layout seed while preserving attempt zero."""

    if attempt == 0:
        return seed
    sequence = np.random.SeedSequence((seed, attempt))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _configure_workspace_debug_rectangle(backend: MujocoBackend) -> None:
    """Resize the shared debug marker to Task 2's sampling workspace."""

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


def _cube_is_on_workspace(cube: np.ndarray) -> bool:
    planar_offset = np.abs(cube[:2] - np.asarray(WORKSPACE_CENTER))
    maximum_center_offset = np.asarray(
        (
            WORKSPACE_LENGTH / 2.0 - CUBE_HALF_WIDTH,
            WORKSPACE_WIDTH / 2.0 - CUBE_HALF_WIDTH,
        )
    )
    return bool(
        np.all(planar_offset <= maximum_center_offset + 0.005)
        and WORKSPACE_CUBE_CENTER_Z - 0.010
        <= float(cube[2])
        <= WORKSPACE_CUBE_CENTER_Z + 0.025
    )


def create_controller(backend: MujocoBackend) -> CubicTaskController:
    state: dict[str, TaskLayout] = {}
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
        _set_free_body_position(backend, "cube", layout.cube)
        _set_mocap_body_position(backend, "source_container", layout.container)
        target_ik_position = (
            layout.destination[0],
            layout.destination[1],
            WORKSPACE_DELIVERY_TCP_Z,
        )
        _set_mocap_body_position(backend, "target_ik_frame", target_ik_position)
        mujoco.mj_forward(backend.model, backend.data)
        backend.update_hand_tcp_markers()
        backend.sync_viewer()

    def waypoints(seed: int) -> tuple[Waypoint, ...]:
        del seed
        layout = state["layout"]
        cube_x, cube_y, cube_z = layout.cube
        destination_x, destination_y, _ = layout.destination
        arm_key = "left" if layout.working_arm == "l" else "right"
        gripper_key = f"{arm_key}_gripper"
        preferred_quaternion_key = f"{arm_key}_preferred_quaternion"
        grasp_quaternion = CUBE_GRASP_QUATERNIONS[layout.working_arm]

        def point(
            time: float,
            xyz: tuple[float, float, float],
            **kwargs,
        ) -> Waypoint:
            return Waypoint(time, **{arm_key: xyz, **kwargs})

        return (
            Waypoint(
                0.0,
                left_gripper=OPEN_GRIPPER,
                right_gripper=OPEN_GRIPPER,
                label="both arms at rest",
            ),
            Waypoint(
                1.0,
                **{
                    "left_gripper": (
                        COMPACT_ENTRY_GRIPPER
                        if layout.working_arm == "l"
                        else OPEN_GRIPPER
                    ),
                    "right_gripper": (
                        COMPACT_ENTRY_GRIPPER
                        if layout.working_arm == "r"
                        else OPEN_GRIPPER
                    ),
                },
                label="compact working gripper at rest",
            ),
            point(
                2.5,
                (cube_x, cube_y, cube_z + PREGRASP_TARGET_Z_OFFSET),
                **{
                    gripper_key: COMPACT_ENTRY_GRIPPER,
                    preferred_quaternion_key: grasp_quaternion,
                },
                label="move directly above cube with compact gripper",
            ),
            point(
                3.3,
                (cube_x, cube_y, cube_z + CONTAINER_ENTRY_TARGET_Z_OFFSET),
                **{
                    gripper_key: COMPACT_ENTRY_GRIPPER,
                    preferred_quaternion_key: grasp_quaternion,
                },
                label="lower compact gripper inside container",
            ),
            Waypoint(
                3.7,
                **{gripper_key: CUBE_GRASP_OPEN_GRIPPER},
                label="open fingers inside container before cube",
            ),
            point(
                4.3,
                (cube_x, cube_y, cube_z + GRASP_TARGET_Z_OFFSET),
                **{
                    gripper_key: CUBE_GRASP_OPEN_GRIPPER,
                    preferred_quaternion_key: grasp_quaternion,
                },
                label="descend open gripper to cube",
            ),
            point(
                4.8,
                (cube_x, cube_y, cube_z + GRASP_TARGET_Z_OFFSET),
                **{
                    gripper_key: CUBE_CLOSED_GRIPPER,
                    preferred_quaternion_key: grasp_quaternion,
                },
                label="close fingers around cube",
            ),
            point(
                6.0,
                (cube_x, cube_y, CONTAINER_CLEAR_TCP_Z),
                **{
                    gripper_key: CUBE_CLOSED_GRIPPER,
                    preferred_quaternion_key: grasp_quaternion,
                },
                label="lift cube vertically clear of container",
            ),
            point(
                7.5,
                (destination_x, destination_y, WORKSPACE_DELIVERY_TCP_Z),
                **{gripper_key: CUBE_CLOSED_GRIPPER},
                label="move cube to requested placement target",
            ),
            point(
                8.5,
                (destination_x, destination_y, WORKSPACE_DELIVERY_TCP_Z),
                **{gripper_key: OPEN_GRIPPER},
                label="release cube on workspace",
            ),
            Waypoint(
                11.0,
                return_to_rest=True,
                label="both arms return to rest",
            ),
        )

    def evaluate(backend: MujocoBackend) -> TaskEvaluation:
        layout = state["layout"]
        cube = body_position(backend, "cube")
        retrieved = not inside_container(cube, layout.container[:2])
        placed = retrieved and _cube_is_on_workspace(cube)
        destination_error = float(
            np.linalg.norm(cube[:2] - np.asarray(layout.destination[:2]))
        )
        return TaskEvaluation(
            success=bool(placed),
            subgoals={
                "retrieved": retrieved,
                "placed_on_workspace": bool(placed),
            },
            details={
                "cube_position": cube.tolist(),
                "sampled_cube_position": layout.cube,
                "sampled_container_position": layout.container,
                "sampled_destination_position": layout.destination,
                "destination_planar_error": destination_error,
                "working_arm": layout.working_arm,
                "layout_attempts": state["layout_attempts"],
            },
        )

    class TaskTwoController(CubicTaskController):
        def reset(self, *, seed: int) -> None:
            nonlocal layout_attempt
            last_error: IKConvergenceError | None = None
            for layout_attempt in range(MAX_LAYOUT_ATTEMPTS):
                self.backend.detach_free_body("cube")
                try:
                    super().reset(seed=seed)
                except IKConvergenceError as error:
                    last_error = error
                    continue
                return
            assert last_error is not None
            raise IKConvergenceError(
                f"Task 2 seed {seed} had no reachable sampled layout after "
                f"{MAX_LAYOUT_ATTEMPTS} actual-trajectory attempts; last failure: "
                f"{last_error}"
            ) from last_error

        def action(self, progress: float) -> np.ndarray:
            action = super().action(progress)
            elapsed = (
                float(np.clip(progress, 0.0, 1.0)) * self.waypoints[-1].time
            )
            side = state["layout"].working_arm
            finger_index = FINGER_ACTION_INDICES[side]
            fingers_closed = bool(
                action[finger_index]
                <= CUBE_CLOSED_GRIPPER + GRIPPER_STATE_TOLERANCE
            )
            cube_attached = self.backend.body_is_attached("cube")
            if elapsed >= CUBE_LOCK_TIME and fingers_closed and not cube_attached:
                self.backend.attach_free_body_to_hand_tcp("cube", side)
            elif not fingers_closed and cube_attached:
                self.backend.detach_free_body("cube")
            return action

    return TaskTwoController(
        backend,
        waypoints,
        evaluate,
        scene_initializer=initialize_scene,
        ik_seed_factory=lambda seed: seed,
    )
