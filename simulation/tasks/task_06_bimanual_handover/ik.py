"""IK demonstration for T6 closest-arm mid-air handover."""

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
    OPEN_GRIPPER,
    CubicTaskController,
    IKConvergenceError,
    TaskEvaluation,
    Waypoint,
    body_position,
    hand_tcp_position,
    inside_container,
)

TASK = TaskDefinition(
    task_id=6,
    slug="bimanual_handover",
    title="Bimanual mid-air handover",
    instruction=(
        "Pick up the cube from the first cubic container with the closest arm, "
        "transfer it to the opposite arm in mid-air, and place it inside the "
        "second cubic container."
    ),
    nominal_duration_seconds=10,
    subgoals=(
        "closest arm grasps the cube",
        "opposite arm receives it in mid-air",
        "receiving arm places it in the second container",
    ),
    implemented=True,
)

COMPACT_ENTRY_GRIPPER = 0.20
CUBE_GRASP_OPEN_GRIPPER = 0.40
GRASP_TARGET_Z_OFFSET = 0.005
PREGRASP_TARGET_Z_OFFSET = 0.035
CONTAINER_ENTRY_TARGET_Z_OFFSET = 0.015
CONTAINER_CLEAR_TCP_Z = 0.125
SECOND_CONTAINER_APPROACH_TCP_Z = 0.17
SECOND_CONTAINER_DELIVERY_TCP_Z = 0.11
SECOND_CONTAINER_RETREAT_TCP_Z = 0.15
SECOND_CONTAINER_OUTWARD_RETREAT = 0.04
GRIPPER_STATE_TOLERANCE = 1e-6
CUBE_CENTER_Z = 0.045
WORKSPACE_CENTER = (0.0, 0.1175)
WORKSPACE_LENGTH = 0.30
WORKSPACE_WIDTH = 0.105
CUBE_HALF_WIDTH = 0.015
CONTAINER_OUTER_HALF_WIDTH = 0.035
CONTAINER_INTERIOR_HALF_WIDTH = 0.030
INSIDE_CONTAINER_MARGIN = 0.005
MINIMUM_EDGE_CLEARANCE = 0.030
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
MAX_LAYOUT_ATTEMPTS = 32
HANDOVER_CUBE = (0.0, 0.10, 0.16)
HANDOVER_TCP = (
    HANDOVER_CUBE[0],
    HANDOVER_CUBE[1],
    HANDOVER_CUBE[2] + GRASP_TARGET_Z_OFFSET,
)
RECEIVER_HANDOVER_LOWERING = 0.010
RECEIVER_HANDOVER_TCP = (
    HANDOVER_TCP[0],
    HANDOVER_TCP[1],
    HANDOVER_TCP[2] - RECEIVER_HANDOVER_LOWERING,
)
HANDOVER_PREGRASP_OFFSET = 0.05
PICKUP_GRASP_TIME = 4.8
RECEIVER_GRASP_TIME = 11.8


@dataclass(frozen=True)
class TaskLayout:
    cube: tuple[float, float, float]
    first_container: tuple[float, float, float]
    second_container: tuple[float, float, float]
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


def sample_layout(backend: MujocoBackend, seed: int) -> TaskLayout:
    """Sample both containers across the full Task 6 workspace."""

    rng = np.random.default_rng(seed)
    required_separation = (
        2.0 * CONTAINER_OUTER_HALF_WIDTH + MINIMUM_EDGE_CLEARANCE
    )
    for _ in range(512):
        first_container = (
            float(rng.uniform(*CONTAINER_X_RANGE)),
            float(rng.uniform(*CONTAINER_Y_RANGE)),
            0.0,
        )
        cube = (
            first_container[0] + float(rng.uniform(*CUBE_OFFSET_RANGE)),
            first_container[1] + float(rng.uniform(*CUBE_OFFSET_RANGE)),
            CUBE_CENTER_Z,
        )
        second_container = (
            float(rng.uniform(*CONTAINER_X_RANGE)),
            float(rng.uniform(*CONTAINER_Y_RANGE)),
            0.0,
        )
        separated = bool(
            abs(first_container[0] - second_container[0]) >= required_separation
            or abs(first_container[1] - second_container[1]) >= required_separation
        )
        if separated:
            return TaskLayout(
                cube,
                first_container,
                second_container,
                _closest_resting_arm(backend, cube),
            )
    raise RuntimeError("Could not sample two separated Task 6 containers")


def _layout_attempt_seed(seed: int, attempt: int) -> int:
    """Derive a reproducible fresh layout seed while preserving attempt zero."""

    if attempt == 0:
        return seed
    sequence = np.random.SeedSequence((seed, attempt))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _configure_workspace_debug_rectangle(backend: MujocoBackend) -> None:
    """Resize the shared debug marker to Task 6's sampling workspace."""

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


def create_controller(backend: MujocoBackend) -> CubicTaskController:
    state: dict[str, TaskLayout] = {}
    transfer_state: dict[str, str | bool | None] = {
        "owner": None,
        "pickup_grasped": False,
        "transferred": False,
    }
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
        _set_free_body_position(backend, "cube", layout.cube)
        _set_mocap_body_position(
            backend,
            "first_container",
            layout.first_container,
        )
        _set_mocap_body_position(
            backend,
            "second_container",
            layout.second_container,
        )
        mujoco.mj_forward(backend.model, backend.data)
        backend.update_hand_tcp_markers()
        backend.sync_viewer()

    def waypoints(seed: int) -> tuple[Waypoint, ...]:
        del seed
        layout = state["layout"]
        cube_x, cube_y, cube_z = layout.cube
        second_x, second_y, _ = layout.second_container
        pickup_side = layout.working_arm
        receiver_side = "r" if pickup_side == "l" else "l"
        pickup_arm_key = "left" if pickup_side == "l" else "right"
        receiver_arm_key = "left" if receiver_side == "l" else "right"
        pickup_gripper_key = f"{pickup_arm_key}_gripper"
        receiver_gripper_key = f"{receiver_arm_key}_gripper"
        pickup_orientation_key = f"{pickup_arm_key}_preferred_quaternion"
        receiver_orientation_key = f"{receiver_arm_key}_preferred_quaternion"
        pickup_quaternion = CUBE_GRASP_QUATERNIONS[pickup_side]
        receiver_quaternion = CUBE_GRASP_QUATERNIONS[receiver_side]
        pickup_side_sign = 1.0 if pickup_side == "l" else -1.0
        receiver_side_sign = 1.0 if receiver_side == "l" else -1.0
        receiver_pregrasp_tcp = (
            RECEIVER_HANDOVER_TCP[0]
            + receiver_side_sign * HANDOVER_PREGRASP_OFFSET,
            RECEIVER_HANDOVER_TCP[1],
            RECEIVER_HANDOVER_TCP[2],
        )

        def point(
            side: str,
            time: float,
            xyz: tuple[float, float, float],
            **kwargs,
        ) -> Waypoint:
            arm_key = "left" if side == "l" else "right"
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
                    pickup_gripper_key: COMPACT_ENTRY_GRIPPER,
                    receiver_gripper_key: OPEN_GRIPPER,
                },
                label="compact pickup gripper at rest",
            ),
            point(
                pickup_side,
                2.5,
                (cube_x, cube_y, cube_z + PREGRASP_TARGET_Z_OFFSET),
                **{
                    pickup_gripper_key: COMPACT_ENTRY_GRIPPER,
                    pickup_orientation_key: pickup_quaternion,
                },
                label="move pickup arm directly above cube with compact gripper",
            ),
            point(
                pickup_side,
                3.3,
                (cube_x, cube_y, cube_z + CONTAINER_ENTRY_TARGET_Z_OFFSET),
                **{
                    pickup_gripper_key: COMPACT_ENTRY_GRIPPER,
                    pickup_orientation_key: pickup_quaternion,
                },
                label="lower compact pickup gripper inside first container",
            ),
            Waypoint(
                3.7,
                **{pickup_gripper_key: CUBE_GRASP_OPEN_GRIPPER},
                label="open pickup fingers inside first container before cube",
            ),
            point(
                pickup_side,
                4.3,
                (cube_x, cube_y, cube_z + GRASP_TARGET_Z_OFFSET),
                **{
                    pickup_gripper_key: CUBE_GRASP_OPEN_GRIPPER,
                    pickup_orientation_key: pickup_quaternion,
                },
                label="descend open pickup gripper to cube",
            ),
            point(
                pickup_side,
                PICKUP_GRASP_TIME,
                (cube_x, cube_y, cube_z + GRASP_TARGET_Z_OFFSET),
                **{
                    pickup_gripper_key: CUBE_CLOSED_GRIPPER,
                    pickup_orientation_key: pickup_quaternion,
                },
                label="close pickup gripper around cube",
            ),
            point(
                pickup_side,
                6.0,
                (cube_x, cube_y, CONTAINER_CLEAR_TCP_Z),
                **{
                    pickup_gripper_key: CUBE_CLOSED_GRIPPER,
                    pickup_orientation_key: pickup_quaternion,
                },
                label="lift cube vertically clear of first container",
            ),
            point(
                pickup_side,
                7.5,
                HANDOVER_TCP,
                **{
                    pickup_gripper_key: CUBE_CLOSED_GRIPPER,
                    pickup_orientation_key: pickup_quaternion,
                },
                label="pickup arm alone carries cube to handover",
            ),
            Waypoint(
                8.0,
                **{
                    pickup_gripper_key: CUBE_CLOSED_GRIPPER,
                    receiver_gripper_key: COMPACT_ENTRY_GRIPPER,
                },
                label="compact receiving gripper at rest",
            ),
            point(
                receiver_side,
                9.5,
                receiver_pregrasp_tcp,
                **{
                    pickup_gripper_key: CUBE_CLOSED_GRIPPER,
                    receiver_gripper_key: COMPACT_ENTRY_GRIPPER,
                    receiver_orientation_key: receiver_quaternion,
                },
                label="move receiving arm to 50 mm handover pregrasp",
            ),
            point(
                receiver_side,
                10.2,
                receiver_pregrasp_tcp,
                **{
                    pickup_gripper_key: CUBE_CLOSED_GRIPPER,
                    receiver_gripper_key: CUBE_GRASP_OPEN_GRIPPER,
                    receiver_orientation_key: receiver_quaternion,
                },
                label="open receiving fingers at handover pregrasp",
            ),
            point(
                receiver_side,
                11.2,
                RECEIVER_HANDOVER_TCP,
                **{
                    pickup_gripper_key: CUBE_CLOSED_GRIPPER,
                    receiver_gripper_key: CUBE_GRASP_OPEN_GRIPPER,
                    receiver_orientation_key: receiver_quaternion,
                },
                label="move receiving hand horizontally onto held cube",
            ),
            point(
                receiver_side,
                RECEIVER_GRASP_TIME,
                RECEIVER_HANDOVER_TCP,
                **{
                    pickup_gripper_key: CUBE_CLOSED_GRIPPER,
                    receiver_gripper_key: CUBE_CLOSED_GRIPPER,
                    receiver_orientation_key: receiver_quaternion,
                },
                label="close receiving gripper and receive cube",
            ),
            Waypoint(
                12.4,
                **{
                    pickup_gripper_key: OPEN_GRIPPER,
                    receiver_gripper_key: CUBE_CLOSED_GRIPPER,
                },
                label="open pickup gripper after transfer",
            ),
            point(
                pickup_side,
                13.3,
                (pickup_side_sign * 0.04, 0.08, 0.17),
                **{
                    pickup_gripper_key: OPEN_GRIPPER,
                    receiver_gripper_key: CUBE_CLOSED_GRIPPER,
                },
                label="retreat empty pickup hand from handover",
            ),
            point(
                receiver_side,
                14.6,
                (
                    -receiver_side_sign * 0.04,
                    second_y,
                    SECOND_CONTAINER_APPROACH_TCP_Z,
                ),
                **{
                    receiver_gripper_key: CUBE_CLOSED_GRIPPER,
                    receiver_orientation_key: receiver_quaternion,
                },
                label="move above second container",
            ),
            point(
                receiver_side,
                15.8,
                (second_x, second_y, SECOND_CONTAINER_DELIVERY_TCP_Z),
                **{
                    receiver_gripper_key: CUBE_CLOSED_GRIPPER,
                    receiver_orientation_key: receiver_quaternion,
                },
                label="lower cube into second container",
            ),
            point(
                receiver_side,
                16.5,
                (second_x, second_y, SECOND_CONTAINER_DELIVERY_TCP_Z),
                **{
                    receiver_gripper_key: OPEN_GRIPPER,
                    receiver_orientation_key: receiver_quaternion,
                },
                label="release cube into second container",
            ),
            point(
                receiver_side,
                17.6,
                (
                    second_x + receiver_side_sign * SECOND_CONTAINER_OUTWARD_RETREAT,
                    second_y,
                    SECOND_CONTAINER_RETREAT_TCP_Z,
                ),
                **{
                    receiver_gripper_key: OPEN_GRIPPER,
                    receiver_orientation_key: receiver_quaternion,
                },
                label="retreat",
            ),
            Waypoint(
                20.0,
                return_to_rest=True,
                label="both arms return to rest",
            ),
        )

    def evaluate(backend: MujocoBackend) -> TaskEvaluation:
        layout = state["layout"]
        cube = body_position(backend, "cube")
        placed = inside_container(
            cube,
            layout.second_container[:2],
            half_width=0.04,
        )
        pickup_grasped = bool(transfer_state["pickup_grasped"])
        transferred = bool(transfer_state["transferred"])
        receiver_side = "r" if layout.working_arm == "l" else "l"
        return TaskEvaluation(
            success=pickup_grasped and transferred and placed,
            subgoals={
                "initial_arm_pick": pickup_grasped,
                "mid_air_transfer": transferred,
                "placed_in_second_container": placed,
            },
            details={
                "cube_position": cube.tolist(),
                "sampled_cube_position": layout.cube,
                "sampled_first_container_position": layout.first_container,
                "sampled_second_container_position": layout.second_container,
                "working_arm": layout.working_arm,
                "receiving_arm": receiver_side,
                "layout_attempts": layout_attempt + 1,
            },
        )

    class TaskSixController(CubicTaskController):
        def reset(self, *, seed: int) -> None:
            nonlocal layout_attempt
            last_error: IKConvergenceError | None = None
            for attempt in range(MAX_LAYOUT_ATTEMPTS):
                layout_attempt = attempt
                self.backend.detach_free_body("cube")
                transfer_state.update(
                    owner=None,
                    pickup_grasped=False,
                    transferred=False,
                )
                try:
                    super().reset(seed=seed)
                except IKConvergenceError as error:
                    last_error = error
                    continue
                return
            assert last_error is not None
            raise IKConvergenceError(
                f"Task 6 seed {seed} had no reachable sampled layout after "
                f"{MAX_LAYOUT_ATTEMPTS} actual-trajectory attempts; last failure: "
                f"{last_error}"
            ) from last_error

        def action(self, progress: float) -> np.ndarray:
            action = super().action(progress)
            elapsed = float(np.clip(progress, 0.0, 1.0)) * self.waypoints[-1].time
            layout = state["layout"]
            pickup_side = layout.working_arm
            receiver_side = "r" if pickup_side == "l" else "l"
            pickup_closed = bool(
                action[FINGER_ACTION_INDICES[pickup_side]]
                <= CUBE_CLOSED_GRIPPER + GRIPPER_STATE_TOLERANCE
            )
            receiver_closed = bool(
                action[FINGER_ACTION_INDICES[receiver_side]]
                <= CUBE_CLOSED_GRIPPER + GRIPPER_STATE_TOLERANCE
            )
            pickup_open = bool(
                action[FINGER_ACTION_INDICES[pickup_side]]
                >= OPEN_GRIPPER - GRIPPER_STATE_TOLERANCE
            )
            receiver_open = bool(
                action[FINGER_ACTION_INDICES[receiver_side]]
                >= OPEN_GRIPPER - GRIPPER_STATE_TOLERANCE
            )
            owner = transfer_state["owner"]

            if owner is None and elapsed >= PICKUP_GRASP_TIME and pickup_closed:
                self.backend.attach_free_body_to_hand_tcp("cube", pickup_side)
                transfer_state["owner"] = pickup_side
                transfer_state["pickup_grasped"] = True
                owner = pickup_side

            if (
                owner == pickup_side
                and elapsed >= RECEIVER_GRASP_TIME
                and receiver_closed
            ):
                self.backend.detach_free_body("cube")
                # Keep the cube at the giver's live pose.  The receiver is
                # intentionally lower, so snapping the cube to its TCP would
                # introduce an artificial downward jump during the transfer.
                self.backend.attach_free_body_to_hand_tcp("cube", receiver_side)
                transfer_state["owner"] = receiver_side
                transfer_state["transferred"] = True
                owner = receiver_side
            if (owner == pickup_side and pickup_open) or (
                owner == receiver_side and receiver_open
            ):
                self.backend.detach_free_body("cube")
                transfer_state["owner"] = None
            return action

    return TaskSixController(
        backend,
        waypoints,
        evaluate,
        scene_initializer=initialize_scene,
        ik_seed_factory=lambda seed: seed,
    )
