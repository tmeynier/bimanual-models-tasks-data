"""IK demonstration for T4 number-conditioned drawer opening."""

from dataclasses import dataclass

import mujoco
import numpy as np

from simulation.backends.mujoco import ACTUATOR_NAMES, MujocoBackend
from simulation.tasks.base import TaskDefinition
from simulation.tasks.ik import (
    CLOSED_GRIPPER,
    FINGER_ACTION_INDICES,
    IKConvergenceError,
    OPEN_GRIPPER,
    CubicTaskController,
    TaskEvaluation,
    Waypoint,
    hand_tcp_position,
    joint_position,
)

TASK = TaskDefinition(
    task_id=4,
    slug="drawer_opening",
    title="Number-conditioned drawer opening",
    instruction="Open the drawer with the requested number.",
    nominal_duration_seconds=10,
    subgoals=("locate the requested drawer", "grasp its handle", "open it"),
    implemented=True,
)

DRAWER_COUNT = 9
DRAWER_OPEN_DISTANCE = 0.05
RACK_TRANSLATION_RANGE = (-0.02, 0.02)
RACK_YAW_RANGE = (-np.deg2rad(5.0), np.deg2rad(5.0))
DRAWER_INITIAL_OPENING_RANGE = (-0.01, 0.01)
MAX_LAYOUT_ATTEMPTS = 32
RACK_ROTATION_PIVOT_XY = np.asarray((-0.0078594, -0.1044), dtype=float)
RACK_BODY_NAMES = ("drawer_stack",) + tuple(
    f"drawer_{number}" for number in range(1, DRAWER_COUNT + 1)
)
TCP_PULL_DISTANCES = {2: 0.06, 4: 0.06, 5: 0.06, 8: 0.05, 9: 0.05}
POST_RELEASE_RETREAT_DISTANCE = 0.045
HANDLE_APPROACH_GRIPPER = 0.20
HANDLE_RELEASE_GRIPPER = 0.40
GRIPPER_STATE_TOLERANCE = 1e-6
HANDLE_ATTACH_TIME = 4.0
TARGET_FRAME_BODY = "drawer_handle_target_frame"
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


def sample_layout(seed: int) -> TaskLayout:
    """Sample a reproducible rack transform and nine initial drawer offsets."""

    rng = np.random.default_rng(np.random.SeedSequence((seed, 0x5434)))
    return TaskLayout(
        rack_translation=(
            float(rng.uniform(*RACK_TRANSLATION_RANGE)),
            float(rng.uniform(*RACK_TRANSLATION_RANGE)),
        ),
        rack_yaw=float(rng.uniform(*RACK_YAW_RANGE)),
        drawer_openings=tuple(
            float(rng.uniform(*DRAWER_INITIAL_OPENING_RANGE))
            for _ in range(DRAWER_COUNT)
        ),
    )


def _layout_attempt_seed(seed: int, attempt: int) -> int:
    """Derive a reproducible fresh layout seed while preserving attempt zero."""

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
    mujoco.mju_mulQuat(
        rotated,
        _yaw_quaternion(yaw),
        np.asarray(quaternion, dtype=float),
    )
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


def _closest_arm(backend: MujocoBackend, position: np.ndarray) -> str:
    return min(
        ("l", "r"),
        key=lambda side: float(np.linalg.norm(hand_tcp_position(backend, side) - position)),
    )


def _handle_ik_target(
    backend: MujocoBackend,
    number: int,
) -> tuple[np.ndarray, str]:
    """Return the exact live center of a drawer handle as its IK target."""
    handle_center = _handle_position(backend, number)
    side = _closest_arm(backend, handle_center)
    return handle_center, side


def _configure_handle_contacts(backend: MujocoBackend, seed: int) -> None:
    """Use the temporary constraint, not collision, for handle coupling."""
    del seed
    for number in range(1, DRAWER_COUNT + 1):
        geom_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"drawer_{number}_handle",
        )
        backend.model.geom_contype[geom_id] = 0
        backend.model.geom_conaffinity[geom_id] = 0


def _set_handle_contact_enabled(
    backend: MujocoBackend,
    number: int,
    enabled: bool,
) -> None:
    geom_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"drawer_{number}_handle",
    )
    backend.model.geom_contype[geom_id] = int(enabled)
    backend.model.geom_conaffinity[geom_id] = int(enabled)


def _task_four_rest_action(backend: MujocoBackend) -> np.ndarray:
    rest = backend.rest_action().astype(float)
    rest[ACTUATOR_NAMES.index("wrist_l")] = np.pi / 8.0
    rest[ACTUATOR_NAMES.index("wrist_r")] = -np.pi / 8.0
    return rest


def _set_target_frame(
    backend: MujocoBackend,
    position: np.ndarray,
    quaternion: tuple[float, float, float, float] | np.ndarray,
) -> None:
    body_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_BODY,
        TARGET_FRAME_BODY,
    )
    mocap_id = int(backend.model.body_mocapid[body_id])
    backend.data.mocap_pos[mocap_id] = position
    backend.data.mocap_quat[mocap_id] = quaternion
    mujoco.mj_forward(backend.model, backend.data)
    backend.update_hand_tcp_markers()
    backend.sync_viewer()


def _attachment_equality_id(backend: MujocoBackend, side: str) -> int:
    equality_id = mujoco.mj_name2id(
        backend.model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        f"drawer_handle_attach_{side}",
    )
    if equality_id < 0:
        raise KeyError(f"Task 4 {side!r}-hand attachment constraint is missing")
    return equality_id


def _deactivate_handle_attachments(backend: MujocoBackend) -> None:
    for side in ("l", "r"):
        backend.data.eq_active[_attachment_equality_id(backend, side)] = 0


def _activate_handle_attachment(
    backend: MujocoBackend,
    number: int,
    side: str,
    handle_target: np.ndarray,
) -> None:
    """Connect the drawer handle point to the live fingertip TCP."""

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
    tcp_position = hand_tcp_position(backend, side)
    backend.model.eq_data[equality_id, :3] = drawer_rotation.T @ (
        handle_target - backend.data.xpos[drawer_body_id]
    )
    backend.model.eq_data[equality_id, 3:6] = hand_rotation.T @ (
        tcp_position - backend.data.xpos[hand_body_id]
    )
    backend.data.eq_active[equality_id] = 1
    mujoco.mj_forward(backend.model, backend.data)


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
        raise KeyError("Task 4 drawer assembly body is missing")
    base_body_positions = {
        name: backend.model.body_pos[body_id].copy()
        for name, body_id in assembly_body_ids.items()
    }
    base_body_quaternions = {
        name: backend.model.body_quat[body_id].copy()
        for name, body_id in assembly_body_ids.items()
    }
    requested: dict[str, int | str | np.ndarray] = {
        "number": 1,
        "side": "l",
        "handle_target": np.zeros(3),
    }
    scene_state: dict[str, TaskLayout] = {}
    cached_layout_key: tuple[int, int] | None = None
    cached_layout: TaskLayout | None = None
    layout_attempt = 0
    attachment_state = {
        "attached": False,
        "ever_attached": False,
        "detached_after_pull": False,
    }

    def initialize_scene(backend: MujocoBackend, seed: int) -> None:
        nonlocal cached_layout, cached_layout_key
        _configure_handle_contacts(backend, seed)
        layout_key = (seed, layout_attempt)
        if cached_layout is None or cached_layout_key != layout_key:
            cached_layout = sample_layout(
                _layout_attempt_seed(seed, layout_attempt)
            )
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

        mujoco.mj_forward(backend.model, backend.data)
        backend.update_hand_tcp_markers()
        backend.sync_viewer()

    def waypoints(seed: int) -> tuple[Waypoint, ...]:
        number = seed % DRAWER_COUNT + 1
        requested["number"] = number
        handle, side = _handle_ik_target(backend, number)
        requested["side"] = side
        requested["handle_target"] = handle.copy()
        x, y, z = (float(value) for value in handle)
        pull_distance = TCP_PULL_DISTANCES.get(number, DRAWER_OPEN_DISTANCE)
        arm_key = "right" if side == "r" else "left"
        gripper_key = "right_gripper" if side == "r" else "left_gripper"
        hand_actuator = f"hand_{side}"
        preferred_quaternion_key = f"{arm_key}_preferred_quaternion"
        layout = scene_state["layout"]
        vertical_quaternion = tuple(
            float(value)
            for value in _rotate_quaternion_about_z(
                VERTICAL_SCISSOR_QUATERNIONS[number],
                layout.rack_yaw,
            )
        )
        pull_direction = np.asarray(
            (-np.sin(layout.rack_yaw), np.cos(layout.rack_yaw), 0.0),
            dtype=float,
        )
        pull_target = handle + pull_distance * pull_direction
        retreat_target = pull_target + POST_RELEASE_RETREAT_DISTANCE * pull_direction
        _set_target_frame(backend, handle, vertical_quaternion)

        def point(time: float, xyz: tuple[float, float, float], **kwargs) -> Waypoint:
            return Waypoint(time, **{arm_key: xyz, **kwargs})

        return (
            Waypoint(
                0.0,
                left_gripper=OPEN_GRIPPER,
                right_gripper=OPEN_GRIPPER,
                label=f"request drawer {number}",
            ),
            Waypoint(
                0.6,
                **{gripper_key: HANDLE_APPROACH_GRIPPER},
                label="compact working gripper before approaching handle",
            ),
            point(
                2.5,
                (x, y, z),
                **{
                    gripper_key: HANDLE_APPROACH_GRIPPER,
                    preferred_quaternion_key: vertical_quaternion,
                },
                label="match vertical hand IK to handle target",
            ),
            point(
                3.0,
                (x, y, z),
                **{
                    gripper_key: HANDLE_APPROACH_GRIPPER,
                    preferred_quaternion_key: vertical_quaternion,
                },
                label="hold vertical hand IK on handle",
            ),
            point(
                3.6,
                (x, y, z),
                **{
                    gripper_key: CLOSED_GRIPPER,
                    preferred_quaternion_key: vertical_quaternion,
                },
                label="close scissor grasp around handle",
            ),
            point(
                HANDLE_ATTACH_TIME,
                (x, y, z),
                **{
                    gripper_key: CLOSED_GRIPPER,
                    preferred_quaternion_key: vertical_quaternion,
                    "locked_actuators": (hand_actuator,),
                },
                label="attach closed gripper to handle",
            ),
            point(
                6.5,
                tuple(float(value) for value in pull_target),
                **{
                    gripper_key: CLOSED_GRIPPER,
                    preferred_quaternion_key: vertical_quaternion,
                    "locked_actuators": (hand_actuator,),
                },
                label="pull attached drawer open",
            ),
            Waypoint(
                7.3,
                **{gripper_key: HANDLE_RELEASE_GRIPPER},
                label="open fingers and detach handle",
            ),
            point(
                8.8,
                tuple(float(value) for value in retreat_target),
                **{
                    gripper_key: HANDLE_RELEASE_GRIPPER,
                    preferred_quaternion_key: vertical_quaternion,
                },
                label="retreat released hand backward from drawer",
            ),
            point(
                9.6,
                tuple(float(value) for value in retreat_target),
                **{
                    gripper_key: HANDLE_RELEASE_GRIPPER,
                    preferred_quaternion_key: vertical_quaternion,
                },
                label="hold backward retreat clear of drawer",
            ),
            Waypoint(
                11.5,
                return_to_rest=True,
                label="both arms return to rest",
            ),
        )

    def evaluate(backend: MujocoBackend) -> TaskEvaluation:
        number = int(requested["number"])
        layout = scene_state["layout"]
        openings = {
            index: joint_position(backend, f"drawer_{index}_slide")
            for index in range(1, DRAWER_COUNT + 1)
        }
        initial_openings = {
            index: layout.drawer_openings[index - 1]
            for index in range(1, DRAWER_COUNT + 1)
        }
        requested_open = openings[number] - initial_openings[number] >= 0.035
        others_closed = all(
            abs(opening - initial_openings[index]) <= 0.015
            for index, opening in openings.items()
            if index != number
        )
        return TaskEvaluation(
            success=(
                attachment_state["ever_attached"]
                and attachment_state["detached_after_pull"]
                and requested_open
                and others_closed
            ),
            subgoals={
                "requested_handle_grasped": attachment_state["ever_attached"],
                "handle_detached_after_pull": attachment_state["detached_after_pull"],
                "requested_drawer_open": requested_open,
                "other_drawers_closed": others_closed,
            },
            details={
                "requested_drawer": number,
                "openings": openings,
                "initial_openings": initial_openings,
                "rack_translation": layout.rack_translation,
                "rack_yaw_degrees": float(np.rad2deg(layout.rack_yaw)),
                "layout_attempts": layout_attempt + 1,
            },
        )

    def initialize_pose(controller: CubicTaskController) -> None:
        align_index = next(
            index
            for index, waypoint in enumerate(controller.waypoints)
            if waypoint.label == "match vertical hand IK to handle target"
        )
        side = str(requested["side"])
        position, resolved_orientation = controller.pose_target(align_index, side)
        initial_action = controller._actions[0]
        backend.data.qpos[backend.qpos_ids] = initial_action
        backend.data.ctrl[backend.actuator_ids] = initial_action
        mujoco.mj_forward(backend.model, backend.data)
        _set_target_frame(backend, position, resolved_orientation)

    class HandleAttachmentController(CubicTaskController):
        def reset(self, *, seed: int) -> None:
            nonlocal layout_attempt
            last_error: IKConvergenceError | None = None
            for layout_attempt in range(MAX_LAYOUT_ATTEMPTS):
                _deactivate_handle_attachments(self.backend)
                attachment_state.update(
                    attached=False,
                    ever_attached=False,
                    detached_after_pull=False,
                )
                try:
                    super().reset(seed=seed)
                except IKConvergenceError as error:
                    last_error = error
                    continue
                return
            assert last_error is not None
            raise IKConvergenceError(
                f"Task 4 seed {seed} had no reachable randomized rack pose after "
                f"{MAX_LAYOUT_ATTEMPTS} actual-trajectory attempts; last failure: "
                f"{last_error}"
            ) from last_error

        def action(self, progress: float) -> np.ndarray:
            action = super().action(progress)
            elapsed = float(np.clip(progress, 0.0, 1.0)) * self.waypoints[-1].time
            side = str(requested["side"])
            if self.current_label == "pull attached drawer open":
                pull_index = next(
                    index
                    for index, waypoint in enumerate(self.waypoints)
                    if waypoint.label == "pull attached drawer open"
                )
                pull_target, pull_orientation = self.pose_target(pull_index, side)
                _set_target_frame(self.backend, pull_target, pull_orientation)
            finger_position = action[FINGER_ACTION_INDICES[side]]
            fingers_closed = bool(finger_position <= CLOSED_GRIPPER + GRIPPER_STATE_TOLERANCE)
            fingers_open = bool(
                finger_position >= HANDLE_RELEASE_GRIPPER - GRIPPER_STATE_TOLERANCE
            )
            if (
                elapsed >= HANDLE_ATTACH_TIME
                and fingers_closed
                and not attachment_state["attached"]
                and not attachment_state["ever_attached"]
            ):
                _activate_handle_attachment(
                    self.backend,
                    int(requested["number"]),
                    side,
                    np.asarray(requested["handle_target"]),
                )
                attachment_state["attached"] = True
                attachment_state["ever_attached"] = True
            elif attachment_state["attached"] and fingers_open:
                _deactivate_handle_attachments(self.backend)
                _set_handle_contact_enabled(
                    self.backend,
                    int(requested["number"]),
                    False,
                )
                attachment_state["attached"] = False
                attachment_state["detached_after_pull"] = True
            return action

    return HandleAttachmentController(
        backend,
        waypoints,
        evaluate,
        scene_initializer=initialize_scene,
        pose_initializer=initialize_pose,
        ik_seed_factory=lambda seed: seed + 1,
        rest_action_factory=_task_four_rest_action,
    )
