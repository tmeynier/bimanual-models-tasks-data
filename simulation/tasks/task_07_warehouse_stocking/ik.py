"""IK demonstration for T7 warehouse stocking."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from simulation.backends.mujoco import ACTUATOR_NAMES, MujocoBackend
from simulation.tasks.base import TaskDefinition
from simulation.tasks.ik import (
    CLOSED_GRIPPER,
    CUBE_CLOSED_GRIPPER,
    CUBE_GRASP_QUATERNIONS,
    FINGER_ACTION_INDICES,
    OPEN_GRIPPER,
    WIDE_OPEN_GRIPPER,
    CubicTaskController,
    IKConvergenceError,
    TaskEvaluation,
    Waypoint,
    body_position,
    hand_tcp_position,
    joint_position,
)

TASK = TaskDefinition(
    task_id=7,
    slug="warehouse_stocking",
    title="Warehouse stocking",
    instruction=(
        "Open the requested drawer, retrieve the cube of the requested color, "
        "place the cube inside, and close the drawer."
    ),
    nominal_duration_seconds=20,
    subgoals=(
        "open the requested drawer",
        "retrieve the requested-color cube",
        "place the cube inside",
        "close the drawer",
    ),
    implemented=True,
)

DRAWER_COUNT = 9
DRAWER_OPEN_DISTANCE = 0.05
TCP_PULL_DISTANCES = {
    1: 0.06,
    2: 0.06,
    3: 0.06,
    4: 0.06,
    5: 0.06,
    6: 0.06,
    7: 0.08,
    8: 0.08,
    9: 0.08,
}
RACK_TRANSLATION_RANGE = (-0.02, 0.02)
RACK_YAW_RANGE = (-np.deg2rad(5.0), np.deg2rad(5.0))
DRAWER_INITIAL_OPENING_RANGE = (-0.01, 0.01)
RACK_ROTATION_PIVOT_XY = np.asarray((-0.0078594, -0.1044), dtype=float)
RACK_BODY_NAMES = (
    "drawer_stack",
    *(f"drawer_{number}" for number in range(1, DRAWER_COUNT + 1)),
)

WORKSPACE_CENTER = (0.0, 0.1175)
WORKSPACE_LENGTH = 0.30
WORKSPACE_WIDTH = 0.105
CUBE_WIDTH = 0.030
CUBE_HALF_WIDTH = CUBE_WIDTH / 2.0
CUBE_CENTER_Z = 0.035
CUBE_MINIMUM_EDGE_CLEARANCE = 0.030
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
CUBE_APPROACH_HEIGHT = 0.135
CUBE_LIFT_TCP_Z = 0.16
DRAWER_DROP_CENTER_HEIGHT_ABOVE_RIM = 0.020
DRAWER_DROP_SIDE_OFFSET = 0.025
DRAWER_DROP_FRONT_OFFSET = 0.0
DRAWER_APPROACH_FORWARD_CLEARANCE = 0.080
POST_RELEASE_RETREAT_DISTANCE = 0.045
HANDLE_APPROACH_GRIPPER = 0.20
HANDLE_RELEASE_GRIPPER = 0.40
DRAWER_OPEN_HOLD_FRICTION = 5.0
GRIPPER_STATE_TOLERANCE = 1e-6
OPEN_HANDLE_ATTACH_TIME = 4.0
OPEN_HANDLE_RELEASE_TIME = 7.3
CUBE_ATTACH_TIME = 13.0
CUBE_RELEASE_TIME = 18.0
CLOSE_HANDLE_ATTACH_TIME = 24.0
CLOSE_HANDLE_RELEASE_TIME = 27.3
HANDLE_TARGET_FRAME_BODY = "drawer_handle_target_frame"
STOCKING_TARGET_FRAME_BODY = "stocking_target_frame"
SQRT_HALF = 2**-0.5
VERTICAL_SCISSOR_QUATERNIONS = {
    1: (SQRT_HALF, 0.0, -SQRT_HALF, 0.0),
    2: (SQRT_HALF, 0.0, SQRT_HALF, 0.0),
    3: (SQRT_HALF, 0.0, SQRT_HALF, 0.0),
    4: (SQRT_HALF, 0.0, -SQRT_HALF, 0.0),
    5: (0.0, -SQRT_HALF, 0.0, SQRT_HALF),
    6: (SQRT_HALF, 0.0, SQRT_HALF, 0.0),
    7: (0.0, SQRT_HALF, 0.0, SQRT_HALF),
    8: (SQRT_HALF, 0.0, SQRT_HALF, 0.0),
    9: (0.0, -SQRT_HALF, 0.0, SQRT_HALF),
}


@dataclass(frozen=True)
class TaskLayout:
    rack_translation: tuple[float, float]
    rack_yaw: float
    drawer_openings: tuple[float, ...]
    blue_cube: tuple[float, float, float]
    green_cube: tuple[float, float, float]


def _cube_footprints_are_separate(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> bool:
    required_separation = 2.0 * CUBE_HALF_WIDTH + CUBE_MINIMUM_EDGE_CLEARANCE
    return bool(
        abs(first[0] - second[0]) >= required_separation
        or abs(first[1] - second[1]) >= required_separation
    )


def sample_layout(seed: int) -> TaskLayout:
    """Sample the rack, drawer offsets, and two non-mirrored cube positions."""

    rng = np.random.default_rng(np.random.SeedSequence((seed, 0x5437)))
    rack_translation = (
        float(rng.uniform(*RACK_TRANSLATION_RANGE)),
        float(rng.uniform(*RACK_TRANSLATION_RANGE)),
    )
    rack_yaw = float(rng.uniform(*RACK_YAW_RANGE))
    drawer_openings = tuple(
        float(rng.uniform(*DRAWER_INITIAL_OPENING_RANGE))
        for _ in range(DRAWER_COUNT)
    )
    for _ in range(512):
        blue_cube = (
            float(rng.uniform(*CUBE_X_RANGE)),
            float(rng.uniform(*CUBE_Y_RANGE)),
            CUBE_CENTER_Z,
        )
        green_cube = (
            float(rng.uniform(*CUBE_X_RANGE)),
            float(rng.uniform(*CUBE_Y_RANGE)),
            CUBE_CENTER_Z,
        )
        if _cube_footprints_are_separate(blue_cube, green_cube):
            return TaskLayout(
                rack_translation,
                rack_yaw,
                drawer_openings,
                blue_cube,
                green_cube,
            )
    raise RuntimeError("Could not sample two separated Task 7 cubes")


def _layout_attempt_seed(seed: int, attempt: int) -> int:
    if attempt == 0:
        return seed
    sequence = np.random.SeedSequence((seed, attempt))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray(
        (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)),
        dtype=float,
    )


def _rotate_quaternion_about_z(
    quaternion: tuple[float, float, float, float] | np.ndarray,
    yaw: float,
) -> np.ndarray:
    rotated = np.empty(4)
    mujoco.mju_mulQuat(rotated, _yaw_quaternion(yaw), np.asarray(quaternion))
    return rotated


def _handle_position(backend: MujocoBackend, number: int) -> np.ndarray:
    geom_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"drawer_{number}_handle",
    )
    if geom_id < 0:
        raise KeyError(f"Drawer {number} handle geometry is missing")
    return backend.data.geom_xpos[geom_id].copy()


def _drawer_dof_id(backend: MujocoBackend, number: int) -> int:
    joint_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"drawer_{number}_slide",
    )
    return int(backend.model.jnt_dofadr[joint_id])


def _drawer_bottom_pose(
    backend: MujocoBackend,
    number: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"drawer_{number}_bottom",
    )
    if geom_id < 0:
        raise KeyError(f"Drawer {number} bottom geometry is missing")
    return (
        backend.data.geom_xpos[geom_id].copy(),
        backend.data.geom_xmat[geom_id].reshape(3, 3).copy(),
        backend.model.geom_size[geom_id].copy(),
    )


def _drawer_rim_height(backend: MujocoBackend, number: int) -> float:
    front_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"drawer_{number}_front",
    )
    if front_id < 0:
        raise KeyError(f"Drawer {number} front geometry is missing")
    return float(
        backend.data.geom_xpos[front_id, 2] + backend.model.geom_size[front_id, 2]
    )


def _closest_arm(backend: MujocoBackend, position: np.ndarray) -> str:
    return min(
        ("l", "r"),
        key=lambda side: float(
            np.linalg.norm(hand_tcp_position(backend, side) - position)
        ),
    )


def _closest_resting_arm(
    backend: MujocoBackend,
    cube: tuple[float, float, float],
) -> str:
    cube_xy = np.asarray(cube[:2])
    return min(
        ("l", "r"),
        key=lambda side: float(
            np.linalg.norm(hand_tcp_position(backend, side)[:2] - cube_xy)
        ),
    )


def _configure_handle_contacts(backend: MujocoBackend) -> None:
    for number in range(1, DRAWER_COUNT + 1):
        geom_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"drawer_{number}_handle",
        )
        backend.model.geom_contype[geom_id] = 0
        backend.model.geom_conaffinity[geom_id] = 0


def _set_cube_drawer_contacts(
    backend: MujocoBackend,
    cube_name: str,
    enabled: bool,
) -> None:
    geom_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"{cube_name}_collision",
    )
    backend.model.geom_conaffinity[geom_id] = 3 if enabled else 1


def _attachment_equality_id(backend: MujocoBackend, side: str) -> int:
    equality_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        f"drawer_handle_attach_{side}",
    )
    if equality_id < 0:
        raise KeyError(f"Task 7 {side!r}-hand attachment constraint is missing")
    return equality_id


def _deactivate_handle_attachments(backend: MujocoBackend) -> None:
    for side in ("l", "r"):
        backend.data.eq_active[_attachment_equality_id(backend, side)] = 0


def _activate_handle_attachment(
    backend: MujocoBackend,
    number: int,
    side: str,
) -> None:
    """Connect the live drawer-handle center to the live fingertip TCP."""

    mujoco.mj_forward(backend.model, backend.data)
    equality_id = _attachment_equality_id(backend, side)
    drawer_body_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        f"drawer_{number}",
    )
    hand_body_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        f"hand_{side}_link",
    )
    backend.model.eq_obj1id[equality_id] = drawer_body_id
    backend.model.eq_obj2id[equality_id] = hand_body_id
    drawer_rotation = backend.data.xmat[drawer_body_id].reshape(3, 3)
    hand_rotation = backend.data.xmat[hand_body_id].reshape(3, 3)
    handle_position = _handle_position(backend, number)
    tcp_position = hand_tcp_position(backend, side)
    backend.model.eq_data[equality_id, :3] = drawer_rotation.T @ (
        handle_position - backend.data.xpos[drawer_body_id]
    )
    backend.model.eq_data[equality_id, 3:6] = hand_rotation.T @ (
        tcp_position - backend.data.xpos[hand_body_id]
    )
    backend.data.eq_active[equality_id] = 1
    mujoco.mj_forward(backend.model, backend.data)


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


def _set_mocap_frame(
    backend: MujocoBackend,
    body_name: str,
    position: np.ndarray,
    quaternion: tuple[float, float, float, float] | np.ndarray,
) -> None:
    body_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = int(backend.model.body_mocapid[body_id])
    backend.data.mocap_pos[mocap_id] = position
    backend.data.mocap_quat[mocap_id] = quaternion
    mujoco.mj_forward(backend.model, backend.data)
    backend.update_hand_tcp_markers()
    backend.sync_viewer()


def _configure_workspace_debug_rectangle(backend: MujocoBackend) -> None:
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


def _task_seven_rest_action(backend: MujocoBackend) -> np.ndarray:
    rest = backend.rest_action().astype(float)
    rest[ACTUATOR_NAMES.index("wrist_l")] = np.pi / 8.0
    rest[ACTUATOR_NAMES.index("wrist_r")] = -np.pi / 8.0
    return rest


def _cube_inside_drawer(
    backend: MujocoBackend,
    cube_name: str,
    drawer_number: int,
) -> bool:
    cube = body_position(backend, cube_name)
    bottom_position, bottom_rotation, bottom_size = _drawer_bottom_pose(
        backend,
        drawer_number,
    )
    local = bottom_rotation.T @ (cube - bottom_position)
    planar_limits = bottom_size[:2] - CUBE_HALF_WIDTH - 0.002
    return bool(
        abs(local[0]) <= planar_limits[0]
        and abs(local[1]) <= planar_limits[1]
        and bottom_size[2] + CUBE_HALF_WIDTH - 0.004
        <= local[2]
        <= bottom_size[2] + CUBE_HALF_WIDTH + 0.015
    )


def create_controller(backend: MujocoBackend) -> CubicTaskController:
    assembly_body_ids = {
        name: mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )
        for name in RACK_BODY_NAMES
    }
    if any(body_id < 0 for body_id in assembly_body_ids.values()):
        raise KeyError("Task 7 drawer assembly body is missing")
    base_body_positions = {
        name: backend.model.body_pos[body_id].copy()
        for name, body_id in assembly_body_ids.items()
    }
    base_body_quaternions = {
        name: backend.model.body_quat[body_id].copy()
        for name, body_id in assembly_body_ids.items()
    }
    base_drawer_frictionloss = {
        number: float(backend.model.dof_frictionloss[_drawer_dof_id(backend, number)])
        for number in range(1, DRAWER_COUNT + 1)
    }
    scene_state: dict[str, TaskLayout] = {}
    requested: dict[str, object] = {}
    plan: dict[str, object] = {}
    cached_layout_key: tuple[int, int] | None = None
    cached_layout: TaskLayout | None = None
    layout_attempt = 0
    execution_state: dict[str, object] = {
        "handle_phase": "idle",
        "open_handle_grasped": False,
        "close_handle_grasped": False,
        "drawer_ever_open": False,
        "cube_grasped": False,
        "cube_released": False,
    }

    def initialize_scene(backend: MujocoBackend, seed: int) -> None:
        nonlocal cached_layout, cached_layout_key
        _configure_handle_contacts(backend)
        _configure_workspace_debug_rectangle(backend)
        requested_drawer_number = seed % DRAWER_COUNT + 1
        layout_key = (seed, layout_attempt)
        if cached_layout is None or cached_layout_key != layout_key:
            cached_layout = sample_layout(_layout_attempt_seed(seed, layout_attempt))
            cached_layout_key = layout_key
        layout = cached_layout
        scene_state["layout"] = layout

        cosine = float(np.cos(layout.rack_yaw))
        sine = float(np.sin(layout.rack_yaw))
        rotation_xy = np.asarray(((cosine, -sine), (sine, cosine)))
        translation_xy = np.asarray(layout.rack_translation)
        for name, body_id in assembly_body_ids.items():
            base_position = base_body_positions[name]
            rotated_xy = RACK_ROTATION_PIVOT_XY + rotation_xy @ (
                base_position[:2] - RACK_ROTATION_PIVOT_XY
            )
            backend.model.body_pos[body_id] = (
                *(rotated_xy + translation_xy),
                base_position[2],
            )
            backend.model.body_quat[body_id] = _rotate_quaternion_about_z(
                base_body_quaternions[name],
                layout.rack_yaw,
            )

        for number, opening in enumerate(layout.drawer_openings, start=1):
            joint_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"drawer_{number}_slide",
            )
            qpos_id = int(backend.model.jnt_qposadr[joint_id])
            dof_id = int(backend.model.jnt_dofadr[joint_id])
            backend.data.qpos[qpos_id] = opening
            backend.data.qvel[dof_id] = 0.0
            backend.model.dof_frictionloss[dof_id] = (
                base_drawer_frictionloss[number]
                if number == requested_drawer_number
                else DRAWER_OPEN_HOLD_FRICTION
            )

        _set_free_body_position(backend, "blue_cube", layout.blue_cube)
        _set_free_body_position(backend, "green_cube", layout.green_cube)
        mujoco.mj_forward(backend.model, backend.data)
        backend.update_hand_tcp_markers()
        backend.sync_viewer()

    def waypoints(seed: int) -> tuple[Waypoint, ...]:
        layout = scene_state["layout"]
        drawer_number = seed % DRAWER_COUNT + 1
        requested_color = "blue" if seed % 2 == 0 else "green"
        requested_cube_name = f"{requested_color}_cube"
        cube_position = getattr(layout, requested_cube_name)
        cube_x, cube_y, cube_z = cube_position
        cube_side = _closest_resting_arm(backend, cube_position)

        handle_target = _handle_position(backend, drawer_number)
        opening_side = _closest_arm(backend, handle_target)
        pull_distance = TCP_PULL_DISTANCES.get(drawer_number, DRAWER_OPEN_DISTANCE)
        pull_direction = np.asarray(
            (-np.sin(layout.rack_yaw), np.cos(layout.rack_yaw), 0.0),
            dtype=float,
        )
        open_handle_target = handle_target + pull_distance * pull_direction
        opening_retreat_target = (
            open_handle_target + POST_RELEASE_RETREAT_DISTANCE * pull_direction
        )

        bottom_position, bottom_rotation, _ = _drawer_bottom_pose(
            backend,
            drawer_number,
        )
        open_bottom_position = bottom_position + pull_distance * pull_direction
        cube_side_sign = 1.0 if cube_side == "l" else -1.0
        drawer_drop_cube_target = open_bottom_position + bottom_rotation @ np.asarray(
            (
                cube_side_sign * DRAWER_DROP_SIDE_OFFSET,
                DRAWER_DROP_FRONT_OFFSET,
                0.0,
            )
        )
        drawer_drop_cube_target[2] = (
            _drawer_rim_height(backend, drawer_number)
            + DRAWER_DROP_CENTER_HEIGHT_ABOVE_RIM
        )
        drawer_release_tcp = drawer_drop_cube_target + np.asarray(
            (0.0, 0.0, GRASP_TARGET_Z_OFFSET)
        )
        drawer_approach_tcp = (
            drawer_release_tcp
            + DRAWER_APPROACH_FORWARD_CLEARANCE * pull_direction
        )

        closing_side = _closest_arm(backend, open_handle_target)
        vertical_quaternion = tuple(
            float(value)
            for value in _rotate_quaternion_about_z(
                VERTICAL_SCISSOR_QUATERNIONS[drawer_number],
                layout.rack_yaw,
            )
        )
        cube_quaternion = CUBE_GRASP_QUATERNIONS[cube_side]
        cube_lift_x = 0.040 if cube_side == "l" else -0.040
        closed_handle_retreat_target = (
            handle_target + POST_RELEASE_RETREAT_DISTANCE * pull_direction
        )

        requested.update(
            drawer_number=drawer_number,
            color=requested_color,
            cube_name=requested_cube_name,
        )
        plan.update(
            opening_side=opening_side,
            cube_side=cube_side,
            closing_side=closing_side,
            handle_target=handle_target,
            open_handle_target=open_handle_target,
            drawer_drop_cube_target=drawer_drop_cube_target,
            drawer_release_tcp=drawer_release_tcp,
            vertical_quaternion=vertical_quaternion,
        )
        _set_mocap_frame(
            backend,
            HANDLE_TARGET_FRAME_BODY,
            handle_target,
            vertical_quaternion,
        )
        _set_mocap_frame(
            backend,
            STOCKING_TARGET_FRAME_BODY,
            drawer_release_tcp,
            cube_quaternion,
        )

        def point(
            side: str,
            time: float,
            xyz: tuple[float, float, float] | np.ndarray,
            **kwargs,
        ) -> Waypoint:
            arm_key = "left" if side == "l" else "right"
            return Waypoint(
                time,
                **{arm_key: tuple(float(value) for value in xyz), **kwargs},
            )

        def gripper_key(side: str) -> str:
            return "left_gripper" if side == "l" else "right_gripper"

        def orientation_key(side: str) -> str:
            arm_key = "left" if side == "l" else "right"
            return f"{arm_key}_preferred_quaternion"

        opening_gripper = gripper_key(opening_side)
        opening_orientation = orientation_key(opening_side)
        cube_gripper = gripper_key(cube_side)
        cube_orientation = orientation_key(cube_side)
        closing_gripper = gripper_key(closing_side)
        closing_orientation = orientation_key(closing_side)

        return (
            Waypoint(
                0.0,
                left_gripper=OPEN_GRIPPER,
                right_gripper=OPEN_GRIPPER,
                label=f"request {requested_color} cube and drawer {drawer_number}",
            ),
            Waypoint(
                0.6,
                **{opening_gripper: HANDLE_APPROACH_GRIPPER},
                label="compact drawer-opening gripper",
            ),
            point(
                opening_side,
                2.5,
                handle_target,
                **{
                    opening_gripper: HANDLE_APPROACH_GRIPPER,
                    opening_orientation: vertical_quaternion,
                },
                label="match vertical hand IK to requested drawer handle",
            ),
            point(
                opening_side,
                3.0,
                handle_target,
                **{
                    opening_gripper: HANDLE_APPROACH_GRIPPER,
                    opening_orientation: vertical_quaternion,
                },
                label="hold drawer-opening handle IK",
            ),
            point(
                opening_side,
                3.6,
                handle_target,
                **{
                    opening_gripper: CLOSED_GRIPPER,
                    opening_orientation: vertical_quaternion,
                },
                label="close scissor grasp around opening handle",
            ),
            point(
                opening_side,
                OPEN_HANDLE_ATTACH_TIME,
                handle_target,
                **{
                    opening_gripper: CLOSED_GRIPPER,
                    opening_orientation: vertical_quaternion,
                    "locked_actuators": (f"hand_{opening_side}",),
                },
                label="attach gripper to requested drawer handle",
            ),
            point(
                opening_side,
                6.5,
                open_handle_target,
                **{
                    opening_gripper: CLOSED_GRIPPER,
                    opening_orientation: vertical_quaternion,
                    "locked_actuators": (f"hand_{opening_side}",),
                },
                label="pull requested drawer open",
            ),
            Waypoint(
                OPEN_HANDLE_RELEASE_TIME,
                **{opening_gripper: HANDLE_RELEASE_GRIPPER},
                label="release opened drawer handle",
            ),
            point(
                opening_side,
                8.8,
                opening_retreat_target,
                **{
                    opening_gripper: HANDLE_RELEASE_GRIPPER,
                    opening_orientation: vertical_quaternion,
                },
                label="retreat from opened drawer",
            ),
            Waypoint(
                10.5,
                return_to_rest=True,
                label="reset both arms before cube pickup",
            ),
            point(
                cube_side,
                11.5,
                (cube_x, cube_y, cube_z + CUBE_APPROACH_HEIGHT),
                **{
                    cube_gripper: WIDE_OPEN_GRIPPER,
                    cube_orientation: cube_quaternion,
                },
                label="approach requested cube",
            ),
            point(
                cube_side,
                12.3,
                (cube_x, cube_y, cube_z + GRASP_TARGET_Z_OFFSET),
                **{
                    cube_gripper: WIDE_OPEN_GRIPPER,
                    cube_orientation: cube_quaternion,
                },
                label="align open gripper above requested cube",
            ),
            Waypoint(
                12.6,
                **{cube_gripper: WIDE_OPEN_GRIPPER},
                label="hold completed requested-cube IK",
            ),
            point(
                cube_side,
                CUBE_ATTACH_TIME,
                (cube_x, cube_y, cube_z + GRASP_TARGET_Z_OFFSET),
                **{
                    cube_gripper: CUBE_CLOSED_GRIPPER,
                    cube_orientation: cube_quaternion,
                },
                label="close fingers around requested cube",
            ),
            point(
                cube_side,
                14.0,
                (cube_lift_x, cube_y, CUBE_LIFT_TCP_Z),
                **{
                    cube_gripper: CUBE_CLOSED_GRIPPER,
                    cube_orientation: cube_quaternion,
                },
                label="lift requested cube clear of workspace",
            ),
            point(
                cube_side,
                16.0,
                drawer_approach_tcp,
                **{
                    cube_gripper: CUBE_CLOSED_GRIPPER,
                },
                label="raise cube in front of open requested drawer",
            ),
            point(
                cube_side,
                17.2,
                drawer_release_tcp,
                **{
                    cube_gripper: CUBE_CLOSED_GRIPPER,
                },
                label="position requested cube above open drawer",
            ),
            point(
                cube_side,
                CUBE_RELEASE_TIME,
                drawer_release_tcp,
                **{
                    cube_gripper: OPEN_GRIPPER,
                },
                label="drop requested cube into open drawer",
            ),
            point(
                cube_side,
                19.3,
                drawer_approach_tcp,
                **{
                    cube_gripper: OPEN_GRIPPER,
                },
                label="retreat above stocked drawer",
            ),
            Waypoint(
                20.8,
                return_to_rest=True,
                label="reset both arms before drawer closing",
            ),
            Waypoint(
                21.4,
                **{closing_gripper: HANDLE_APPROACH_GRIPPER},
                label="compact drawer-closing gripper",
            ),
            point(
                closing_side,
                22.8,
                open_handle_target,
                **{
                    closing_gripper: HANDLE_APPROACH_GRIPPER,
                    closing_orientation: vertical_quaternion,
                },
                label="match vertical hand IK to open drawer handle",
            ),
            point(
                closing_side,
                23.4,
                open_handle_target,
                **{
                    closing_gripper: CLOSED_GRIPPER,
                    closing_orientation: vertical_quaternion,
                },
                label="close scissor grasp around closing handle",
            ),
            point(
                closing_side,
                CLOSE_HANDLE_ATTACH_TIME,
                open_handle_target,
                **{
                    closing_gripper: CLOSED_GRIPPER,
                    closing_orientation: vertical_quaternion,
                    "locked_actuators": (f"hand_{closing_side}",),
                },
                label="attach gripper to open drawer handle",
            ),
            point(
                closing_side,
                26.5,
                handle_target,
                **{
                    closing_gripper: CLOSED_GRIPPER,
                    closing_orientation: vertical_quaternion,
                    "locked_actuators": (f"hand_{closing_side}",),
                },
                label="push stocked drawer closed",
            ),
            Waypoint(
                CLOSE_HANDLE_RELEASE_TIME,
                **{closing_gripper: HANDLE_RELEASE_GRIPPER},
                label="release closed drawer handle",
            ),
            point(
                closing_side,
                28.8,
                closed_handle_retreat_target,
                **{
                    closing_gripper: HANDLE_RELEASE_GRIPPER,
                    closing_orientation: vertical_quaternion,
                },
                label="retreat from closed stocked drawer",
            ),
            Waypoint(
                30.5,
                return_to_rest=True,
                label="both arms return to rest",
            ),
        )

    def evaluate(backend: MujocoBackend) -> TaskEvaluation:
        layout = scene_state["layout"]
        drawer_number = int(requested["drawer_number"])
        cube_name = str(requested["cube_name"])
        openings = {
            number: joint_position(backend, f"drawer_{number}_slide")
            for number in range(1, DRAWER_COUNT + 1)
        }
        initial_openings = {
            number: layout.drawer_openings[number - 1]
            for number in range(1, DRAWER_COUNT + 1)
        }
        drawer_closed = bool(
            abs(openings[drawer_number] - initial_openings[drawer_number]) <= 0.015
        )
        other_drawers_unchanged = all(
            abs(opening - initial_openings[number]) <= 0.015
            for number, opening in openings.items()
            if number != drawer_number
        )
        cube_inside = _cube_inside_drawer(backend, cube_name, drawer_number)
        drawer_opened = bool(execution_state["drawer_ever_open"])
        cube_grasped = bool(execution_state["cube_grasped"])
        cube_released = bool(execution_state["cube_released"])
        drawer_reclosed = bool(
            execution_state["close_handle_grasped"] and drawer_closed
        )
        return TaskEvaluation(
            success=(
                bool(execution_state["open_handle_grasped"])
                and drawer_opened
                and cube_grasped
                and cube_released
                and cube_inside
                and drawer_reclosed
                and other_drawers_unchanged
            ),
            subgoals={
                "requested_drawer_opened": drawer_opened,
                "requested_cube_retrieved": cube_grasped,
                "cube_placed_inside": cube_inside,
                "requested_drawer_closed": drawer_reclosed,
            },
            details={
                "requested_drawer": drawer_number,
                "requested_color": requested["color"],
                "opening_arm": plan["opening_side"],
                "cube_arm": plan["cube_side"],
                "closing_arm": plan["closing_side"],
                "cube_position": body_position(backend, cube_name).tolist(),
                "sampled_blue_position": layout.blue_cube,
                "sampled_green_position": layout.green_cube,
                "drawer_openings": openings,
                "initial_drawer_openings": initial_openings,
                "rack_translation": layout.rack_translation,
                "rack_yaw_degrees": float(np.rad2deg(layout.rack_yaw)),
                "layout_attempts": layout_attempt + 1,
            },
        )

    def initialize_pose(controller: CubicTaskController) -> None:
        labels = [waypoint.label for waypoint in controller.waypoints]
        opening_index = labels.index(
            "match vertical hand IK to requested drawer handle"
        )
        delivery_index = labels.index("position requested cube above open drawer")
        opening_side = str(plan["opening_side"])
        cube_side = str(plan["cube_side"])
        handle_position, handle_orientation = controller.pose_target(
            opening_index,
            opening_side,
        )
        delivery_position, delivery_orientation = controller.pose_target(
            delivery_index,
            cube_side,
        )
        initial_action = controller._actions[0]
        backend.data.qpos[backend.qpos_ids] = initial_action
        backend.data.ctrl[backend.actuator_ids] = initial_action
        mujoco.mj_forward(backend.model, backend.data)
        _set_mocap_frame(
            backend,
            HANDLE_TARGET_FRAME_BODY,
            handle_position,
            handle_orientation,
        )
        _set_mocap_frame(
            backend,
            STOCKING_TARGET_FRAME_BODY,
            delivery_position,
            delivery_orientation,
        )

    class WarehouseStockingController(CubicTaskController):
        def reset(self, *, seed: int) -> None:
            nonlocal layout_attempt
            last_error: IKConvergenceError | None = None
            for attempt in range(MAX_LAYOUT_ATTEMPTS):
                layout_attempt = attempt
                _deactivate_handle_attachments(self.backend)
                self.backend.detach_free_body("blue_cube")
                self.backend.detach_free_body("green_cube")
                _set_cube_drawer_contacts(self.backend, "blue_cube", True)
                _set_cube_drawer_contacts(self.backend, "green_cube", True)
                execution_state.update(
                    handle_phase="idle",
                    open_handle_grasped=False,
                    close_handle_grasped=False,
                    drawer_ever_open=False,
                    cube_grasped=False,
                    cube_released=False,
                )
                try:
                    super().reset(seed=seed)
                except IKConvergenceError as error:
                    last_error = error
                    continue
                return
            assert last_error is not None
            raise IKConvergenceError(
                f"Task 7 seed {seed} had no reachable randomized trajectory after "
                f"{MAX_LAYOUT_ATTEMPTS} attempts; last failure: {last_error}"
            ) from last_error

        def action(self, progress: float) -> np.ndarray:
            action = super().action(progress)
            elapsed = float(np.clip(progress, 0.0, 1.0)) * self.waypoints[-1].time
            drawer_number = int(requested["drawer_number"])
            layout = scene_state["layout"]
            opening_side = str(plan["opening_side"])
            cube_side = str(plan["cube_side"])
            closing_side = str(plan["closing_side"])
            cube_name = str(requested["cube_name"])
            opening = joint_position(
                self.backend,
                f"drawer_{drawer_number}_slide",
            )
            if opening - layout.drawer_openings[drawer_number - 1] >= 0.035:
                execution_state["drawer_ever_open"] = True

            opening_finger = action[FINGER_ACTION_INDICES[opening_side]]
            closing_finger = action[FINGER_ACTION_INDICES[closing_side]]
            cube_finger = action[FINGER_ACTION_INDICES[cube_side]]
            handle_phase = str(execution_state["handle_phase"])

            if (
                elapsed >= OPEN_HANDLE_ATTACH_TIME
                and elapsed < OPEN_HANDLE_RELEASE_TIME
                and opening_finger <= CLOSED_GRIPPER + GRIPPER_STATE_TOLERANCE
                and handle_phase == "idle"
            ):
                _activate_handle_attachment(
                    self.backend,
                    drawer_number,
                    opening_side,
                )
                execution_state["handle_phase"] = "opening"
                execution_state["open_handle_grasped"] = True
                handle_phase = "opening"
            if (
                handle_phase == "opening"
                and elapsed >= OPEN_HANDLE_RELEASE_TIME
                and opening_finger
                >= HANDLE_RELEASE_GRIPPER - GRIPPER_STATE_TOLERANCE
            ):
                _deactivate_handle_attachments(self.backend)
                self.backend.model.dof_frictionloss[
                    _drawer_dof_id(self.backend, drawer_number)
                ] = DRAWER_OPEN_HOLD_FRICTION
                execution_state["handle_phase"] = "drawer_open"
                handle_phase = "drawer_open"

            cube_attached = self.backend.body_is_attached(cube_name)
            if (
                elapsed >= CUBE_ATTACH_TIME
                and elapsed < CUBE_RELEASE_TIME
                and cube_finger
                <= CUBE_CLOSED_GRIPPER + GRIPPER_STATE_TOLERANCE
                and not cube_attached
            ):
                self.backend.attach_free_body_to_hand_tcp(cube_name, cube_side)
                execution_state["cube_grasped"] = True
                cube_attached = True
            if (
                elapsed >= CUBE_RELEASE_TIME
                and cube_finger >= OPEN_GRIPPER - GRIPPER_STATE_TOLERANCE
                and cube_attached
            ):
                self.backend.detach_free_body(cube_name)
                execution_state["cube_released"] = True

            if (
                elapsed >= CLOSE_HANDLE_ATTACH_TIME
                and elapsed < CLOSE_HANDLE_RELEASE_TIME
                and closing_finger <= CLOSED_GRIPPER + GRIPPER_STATE_TOLERANCE
                and handle_phase == "drawer_open"
            ):
                self.backend.model.dof_frictionloss[
                    _drawer_dof_id(self.backend, drawer_number)
                ] = base_drawer_frictionloss[drawer_number]
                _activate_handle_attachment(
                    self.backend,
                    drawer_number,
                    closing_side,
                )
                execution_state["handle_phase"] = "closing"
                execution_state["close_handle_grasped"] = True
                handle_phase = "closing"
            if (
                handle_phase == "closing"
                and elapsed >= CLOSE_HANDLE_RELEASE_TIME
                and closing_finger
                >= HANDLE_RELEASE_GRIPPER - GRIPPER_STATE_TOLERANCE
            ):
                _deactivate_handle_attachments(self.backend)
                execution_state["handle_phase"] = "closed"
            return action

    return WarehouseStockingController(
        backend,
        waypoints,
        evaluate,
        scene_initializer=initialize_scene,
        pose_initializer=initialize_pose,
        ik_seed_factory=lambda seed: seed + 7,
        rest_action_factory=_task_seven_rest_action,
    )
