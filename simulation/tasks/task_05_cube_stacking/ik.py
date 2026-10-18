"""IK demonstration for T5 precision cube stacking."""

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
    WIDE_OPEN_GRIPPER,
    CubicTaskController,
    TaskEvaluation,
    Waypoint,
    body_position,
    hand_tcp_position,
)

TASK = TaskDefinition(
    task_id=5,
    slug="cube_stacking",
    title="Precision cube stacking",
    instruction="Place the green cube stably on top of the blue cube.",
    nominal_duration_seconds=10,
    subgoals=("grasp the green cube", "align above blue", "release stably"),
    implemented=True,
)

WORKSPACE_CENTER = (0.0, 0.1175)
WORKSPACE_LENGTH = 0.30
WORKSPACE_WIDTH = 0.105
CUBE_WIDTH = 0.030
CUBE_HALF_WIDTH = CUBE_WIDTH / 2.0
CUBE_CENTER_Z = 0.035
MINIMUM_EDGE_CLEARANCE = 0.030
CUBE_X_RANGE = (
    WORKSPACE_CENTER[0] - WORKSPACE_LENGTH / 2.0 + CUBE_HALF_WIDTH,
    WORKSPACE_CENTER[0] + WORKSPACE_LENGTH / 2.0 - CUBE_HALF_WIDTH,
)
CUBE_Y_RANGE = (
    WORKSPACE_CENTER[1] - WORKSPACE_WIDTH / 2.0 + CUBE_HALF_WIDTH,
    WORKSPACE_CENTER[1] + WORKSPACE_WIDTH / 2.0 - CUBE_HALF_WIDTH,
)
MAX_LAYOUT_ATTEMPTS = 32
GRASP_TARGET_Z_OFFSET = 0.010
POST_GRASP_LIFT_DISTANCE = 0.040
CUBE_RELEASE_GRIPPER = OPEN_GRIPPER
STACK_RELEASE_CLEARANCE = 0.020
STACK_DELIVERY_TCP_Z = (
    CUBE_CENTER_Z
    + CUBE_WIDTH
    + GRASP_TARGET_Z_OFFSET
    + STACK_RELEASE_CLEARANCE
)
STACK_APPROACH_CLEARANCE = 0.040
STACK_APPROACH_TCP_Z = STACK_DELIVERY_TCP_Z + STACK_APPROACH_CLEARANCE
STACK_RETREAT_TCP_Z = 0.16
GRIPPER_STATE_TOLERANCE = 1e-6
CUBE_LOCK_TIME = 3.0


@dataclass(frozen=True)
class TaskLayout:
    green_cube: tuple[float, float, float]
    blue_cube: tuple[float, float, float]
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
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> bool:
    required_separation = 2.0 * CUBE_HALF_WIDTH + MINIMUM_EDGE_CLEARANCE
    return bool(
        abs(first[0] - second[0]) >= required_separation
        or abs(first[1] - second[1]) >= required_separation
    )


def sample_layout(backend: MujocoBackend, seed: int) -> TaskLayout:
    """Sample both cube centers across the full Task 5 workspace."""

    rng = np.random.default_rng(seed)
    for _ in range(512):
        blue = (
            float(rng.uniform(*CUBE_X_RANGE)),
            float(rng.uniform(*CUBE_Y_RANGE)),
            CUBE_CENTER_Z,
        )
        green = (
            float(rng.uniform(*CUBE_X_RANGE)),
            float(rng.uniform(*CUBE_Y_RANGE)),
            CUBE_CENTER_Z,
        )
        if not _footprints_are_separate(green, blue):
            continue
        return TaskLayout(
            green_cube=green,
            blue_cube=blue,
            working_arm=_closest_resting_arm(backend, green),
        )
    raise RuntimeError("Could not sample two separated Task 5 cubes")


def _layout_attempt_seed(seed: int, attempt: int) -> int:
    """Derive a reproducible fresh layout seed while preserving attempt zero."""

    if attempt == 0:
        return seed
    sequence = np.random.SeedSequence((seed, attempt))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _configure_workspace_debug_rectangle(backend: MujocoBackend) -> None:
    """Resize the shared debug marker to Task 5's sampling workspace."""

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


def create_controller(backend: MujocoBackend) -> CubicTaskController:
    state: dict[str, TaskLayout] = {}
    grasp_state = {"grasped": False}
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
        _set_free_body_position(backend, "green_cube", layout.green_cube)
        _set_free_body_position(backend, "blue_cube", layout.blue_cube)
        mujoco.mj_forward(backend.model, backend.data)
        backend.update_hand_tcp_markers()
        backend.sync_viewer()

    def waypoints(seed: int) -> tuple[Waypoint, ...]:
        del seed
        layout = state["layout"]
        green_x, green_y, green_z = layout.green_cube
        blue_x, blue_y, _ = layout.blue_cube
        side = layout.working_arm
        arm_key = "left" if side == "l" else "right"
        gripper_key = f"{arm_key}_gripper"
        quaternion_key = f"{arm_key}_preferred_quaternion"
        grasp_quaternion = CUBE_GRASP_QUATERNIONS[side]

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
            point(
                2.0,
                (green_x, green_y, green_z + GRASP_TARGET_Z_OFFSET),
                **{
                    gripper_key: WIDE_OPEN_GRIPPER,
                    quaternion_key: grasp_quaternion,
                },
                label="grasp IK above green cube",
            ),
            point(
                2.6,
                (green_x, green_y, green_z + GRASP_TARGET_Z_OFFSET),
                **{
                    gripper_key: CUBE_CLOSED_GRIPPER,
                    quaternion_key: grasp_quaternion,
                },
                label="close gripper around green cube",
            ),
            Waypoint(
                CUBE_LOCK_TIME,
                **{gripper_key: CUBE_CLOSED_GRIPPER},
                label="lock grasped green cube",
            ),
            point(
                3.8,
                (
                    green_x,
                    green_y,
                    green_z
                    + GRASP_TARGET_Z_OFFSET
                    + POST_GRASP_LIFT_DISTANCE,
                ),
                **{
                    gripper_key: CUBE_CLOSED_GRIPPER,
                    quaternion_key: grasp_quaternion,
                },
                label="lift green cube vertically by 40 mm",
            ),
            point(
                5.0,
                (blue_x, blue_y, STACK_APPROACH_TCP_Z),
                **{
                    gripper_key: CUBE_CLOSED_GRIPPER,
                    quaternion_key: grasp_quaternion,
                },
                label="move high above blue cube",
            ),
            point(
                6.0,
                (blue_x, blue_y, STACK_DELIVERY_TCP_Z),
                **{
                    gripper_key: CUBE_CLOSED_GRIPPER,
                    quaternion_key: grasp_quaternion,
                },
                label="descend vertically to stack target",
            ),
            point(
                6.7,
                (blue_x, blue_y, STACK_DELIVERY_TCP_Z),
                **{
                    gripper_key: CUBE_RELEASE_GRIPPER,
                    quaternion_key: grasp_quaternion,
                },
                label="release green cube",
            ),
            point(
                8.0,
                (blue_x, blue_y, STACK_RETREAT_TCP_Z),
                **{
                    gripper_key: CUBE_RELEASE_GRIPPER,
                    quaternion_key: grasp_quaternion,
                },
                label="retreat IK above stacked cubes",
            ),
            Waypoint(
                9.8,
                return_to_rest=True,
                label="both arms return to rest",
            ),
        )

    def evaluate(backend: MujocoBackend) -> TaskEvaluation:
        layout = state["layout"]
        blue = body_position(backend, "blue_cube")
        green = body_position(backend, "green_cube")
        horizontal_error = float(np.linalg.norm(green[:2] - blue[:2]))
        vertical_offset = float(green[2] - blue[2])
        aligned = horizontal_error <= 0.018
        supported = 0.025 <= vertical_offset <= 0.040
        return TaskEvaluation(
            success=grasp_state["grasped"] and aligned and supported,
            subgoals={
                "grasped": grasp_state["grasped"],
                "aligned": aligned,
                "supported_by_blue_cube": supported,
            },
            details={
                "blue_position": blue.tolist(),
                "green_position": green.tolist(),
                "sampled_blue_position": layout.blue_cube,
                "sampled_green_position": layout.green_cube,
                "working_arm": layout.working_arm,
                "horizontal_error": horizontal_error,
                "vertical_offset": vertical_offset,
                "layout_attempts": layout_attempt + 1,
            },
        )

    class TaskFiveController(CubicTaskController):
        def reset(self, *, seed: int) -> None:
            nonlocal layout_attempt
            last_error: IKConvergenceError | None = None
            for layout_attempt in range(MAX_LAYOUT_ATTEMPTS):
                grasp_state["grasped"] = False
                self.backend.detach_free_body("green_cube")
                try:
                    super().reset(seed=seed)
                except IKConvergenceError as error:
                    last_error = error
                    continue
                return
            assert last_error is not None
            raise IKConvergenceError(
                f"Task 5 seed {seed} had no reachable sampled layout after "
                f"{MAX_LAYOUT_ATTEMPTS} actual-trajectory attempts; last failure: "
                f"{last_error}"
            ) from last_error

        def action(self, progress: float) -> np.ndarray:
            action = super().action(progress)
            elapsed = float(np.clip(progress, 0.0, 1.0)) * self.waypoints[-1].time
            side = state["layout"].working_arm
            finger_index = FINGER_ACTION_INDICES[side]
            fingers_closed = bool(
                action[finger_index]
                <= CUBE_CLOSED_GRIPPER + GRIPPER_STATE_TOLERANCE
            )
            fingers_open = bool(
                action[finger_index] >= CUBE_RELEASE_GRIPPER - GRIPPER_STATE_TOLERANCE
            )
            cube_attached = self.backend.body_is_attached("green_cube")
            if elapsed >= CUBE_LOCK_TIME and fingers_closed and not cube_attached:
                grasp_tcp = hand_tcp_position(self.backend, side)
                _set_free_body_position(
                    self.backend,
                    "green_cube",
                    tuple(grasp_tcp - np.array((0.0, 0.0, GRASP_TARGET_Z_OFFSET))),
                )
                mujoco.mj_forward(self.backend.model, self.backend.data)
                self.backend.attach_free_body_to_hand_tcp("green_cube", side)
                grasp_state["grasped"] = True
            elif fingers_open and cube_attached:
                blue_position = body_position(self.backend, "blue_cube")
                release_position = (
                    float(blue_position[0]),
                    float(blue_position[1]),
                    float(
                        blue_position[2]
                        + CUBE_WIDTH
                        + STACK_RELEASE_CLEARANCE
                    ),
                )
                self.backend.detach_free_body("green_cube")
                _set_free_body_position(
                    self.backend,
                    "green_cube",
                    release_position,
                )
                mujoco.mj_forward(self.backend.model, self.backend.data)
            return action

    return TaskFiveController(
        backend,
        waypoints,
        evaluate,
        scene_initializer=initialize_scene,
        ik_seed_factory=lambda seed: seed,
    )
