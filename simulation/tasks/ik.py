"""Shared numerical IK and waypoint execution for task-local demonstrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import mujoco
import numpy as np

from simulation.backends.mujoco import ACTUATOR_NAMES, MujocoBackend

ARM_ACTION_INDICES = {
    "l": np.arange(0, 5, dtype=int),
    "r": np.arange(6, 11, dtype=int),
}
FINGER_ACTION_INDICES = {
    "l": ACTUATOR_NAMES.index("finger_l"),
    "r": ACTUATOR_NAMES.index("finger_r"),
}
GRIPPER_TIP_SITES = {
    "l": ("hand_l_fixed_tip_site", "finger_l_moving_tip_site"),
    "r": ("hand_r_fixed_tip_site", "finger_r_moving_tip_site"),
}
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)
CUBE_GRASP_QUATERNION = (2**-0.5, 0.0, 0.0, -(2**-0.5))
MIRRORED_CUBE_GRASP_QUATERNION = (2**-0.5, 0.0, 0.0, 2**-0.5)
CUBE_GRASP_QUATERNIONS = {
    "l": CUBE_GRASP_QUATERNION,
    "r": MIRRORED_CUBE_GRASP_QUATERNION,
}
OPEN_GRIPPER = 1.40
WIDE_OPEN_GRIPPER = float(np.pi / 2.0)
CLOSED_GRIPPER = 0.15
# Use a gentler cube hold than the drawer-handle grasp.  The scripted cube lock
# maintains the relative hand/cube pose once this aperture has been reached.
CUBE_CLOSED_GRIPPER = 0.32
ORIENTATION_PREFERENCE_WEIGHT = 0.05


class IKConvergenceError(RuntimeError):
    """Raised when a Cartesian waypoint is outside the arm's reachable workspace."""


@dataclass(frozen=True)
class Waypoint:
    """One timed Cartesian waypoint in a task-specific demonstration."""

    time: float
    left: tuple[float, float, float] | None = None
    right: tuple[float, float, float] | None = None
    left_quaternion: tuple[float, float, float, float] | None = None
    right_quaternion: tuple[float, float, float, float] | None = None
    left_preferred_quaternion: tuple[float, float, float, float] | None = None
    right_preferred_quaternion: tuple[float, float, float, float] | None = None
    left_gripper: float | None = None
    right_gripper: float | None = None
    locked_actuators: tuple[str, ...] = ()
    joint_positions: tuple[tuple[str, float], ...] = ()
    return_to_rest: bool = False
    label: str = ""


@dataclass(frozen=True)
class TaskEvaluation:
    success: bool
    subgoals: dict[str, bool]
    details: dict[str, Any] = field(default_factory=dict)


WaypointFactory = Callable[[int], tuple[Waypoint, ...]]
Evaluator = Callable[[MujocoBackend], TaskEvaluation]
SceneInitializer = Callable[[MujocoBackend, int], None]
PoseInitializer = Callable[["CubicTaskController"], None]
IKSeedFactory = Callable[[int], int]
RestActionFactory = Callable[[MujocoBackend], np.ndarray]


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _sample_mirrored_arm_positions(
    backend: MujocoBackend,
    side: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample corresponding arm coordinates from one canonical left-arm space."""

    left_indices = ARM_ACTION_INDICES["l"]
    canonical = rng.uniform(
        backend.control_low[left_indices],
        backend.control_high[left_indices],
    )
    return canonical if side == "l" else -canonical


def body_position(backend: MujocoBackend, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise KeyError(f"MuJoCo body not found: {name}")
    return backend.data.xpos[body_id].copy()


def joint_position(backend: MujocoBackend, name: str) -> float:
    joint_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(f"MuJoCo joint not found: {name}")
    return float(backend.data.qpos[int(backend.model.jnt_qposadr[joint_id])])


def inside_container(
    position: np.ndarray,
    center: tuple[float, float],
    *,
    half_width: float = 0.045,
    maximum_height: float = 0.16,
) -> bool:
    return bool(
        abs(float(position[0]) - center[0]) <= half_width
        and abs(float(position[1]) - center[1]) <= half_width
        and 0.025 <= float(position[2]) <= maximum_height
    )


def _tip_site_ids(backend: MujocoBackend, side: str) -> tuple[int, int]:
    ids = tuple(
        mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in GRIPPER_TIP_SITES[side]
    )
    if any(site_id < 0 for site_id in ids):
        raise KeyError(f"MuJoCo fingertip sites not found for {side!r} arm")
    return ids  # type: ignore[return-value]


def hand_tcp_position(backend: MujocoBackend, side: str) -> np.ndarray:
    """Return the live midpoint of the fixed and moving distal finger tips."""

    fixed_site_id, moving_site_id = _tip_site_ids(backend, side)
    return 0.5 * (backend.data.site_xpos[fixed_site_id] + backend.data.site_xpos[moving_site_id])


def hand_tcp_quaternion(backend: MujocoBackend, side: str) -> np.ndarray:
    fixed_site_id, _ = _tip_site_ids(backend, side)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, backend.data.site_xmat[fixed_site_id])
    return quaternion


def _hand_tcp_jacobians(
    backend: MujocoBackend,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    fixed_site_id, moving_site_id = _tip_site_ids(backend, side)
    fixed_position = np.zeros((3, backend.model.nv))
    moving_position = np.zeros((3, backend.model.nv))
    rotation = np.zeros((3, backend.model.nv))
    mujoco.mj_jacSite(
        backend.model,
        backend.data,
        fixed_position,
        rotation,
        fixed_site_id,
    )
    mujoco.mj_jacSite(
        backend.model,
        backend.data,
        moving_position,
        None,
        moving_site_id,
    )
    return 0.5 * (fixed_position + moving_position), rotation


def solve_arm_position(
    backend: MujocoBackend,
    side: str,
    target_position: np.ndarray,
    initial_action: np.ndarray,
    locked_action_indices: np.ndarray | None = None,
    locked_action_values: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Solve one position target, without imposing an orientation."""

    action = initial_action.copy()
    indices = ARM_ACTION_INDICES[side]
    locked_action_indices = np.asarray(
        () if locked_action_indices is None else locked_action_indices,
        dtype=int,
    )
    locked_action_values = np.asarray(
        () if locked_action_values is None else locked_action_values,
        dtype=float,
    )
    movable_indices = indices[~np.isin(indices, locked_action_indices)]
    action[locked_action_indices] = locked_action_values
    joint_ids = backend.joint_ids[movable_indices]
    dof_ids = backend.model.jnt_dofadr[joint_ids]
    for _ in range(300):
        backend.data.qpos[backend.qpos_ids] = action
        mujoco.mj_forward(backend.model, backend.data)
        position_error = target_position - hand_tcp_position(backend, side)
        error_norm = float(np.linalg.norm(position_error))
        if error_norm < 0.002:
            return action, error_norm
        position_jacobian, _ = _hand_tcp_jacobians(backend, side)
        arm_jacobian = position_jacobian[:, dof_ids]
        delta = arm_jacobian.T @ np.linalg.solve(
            arm_jacobian @ arm_jacobian.T + 5e-4 * np.eye(3),
            position_error,
        )
        action[movable_indices] += np.clip(delta, -0.08, 0.08)
        action[movable_indices] = np.clip(
            action[movable_indices],
            backend.control_low[movable_indices] + 0.01,
            backend.control_high[movable_indices] - 0.01,
        )

    backend.data.qpos[backend.qpos_ids] = action
    mujoco.mj_forward(backend.model, backend.data)
    error = target_position - hand_tcp_position(backend, side)
    return action, float(np.linalg.norm(error))


class CubicTaskController:
    """Solve Cartesian waypoints and produce smooth joint-space demonstrations."""

    def __init__(
        self,
        backend: MujocoBackend,
        waypoints: tuple[Waypoint, ...] | WaypointFactory,
        evaluator: Evaluator,
        *,
        scene_initializer: SceneInitializer | None = None,
        pose_initializer: PoseInitializer | None = None,
        ik_seed_factory: IKSeedFactory | None = None,
        rest_action_factory: RestActionFactory | None = None,
    ) -> None:
        self.backend = backend
        self._waypoints_or_factory = waypoints
        self._evaluator = evaluator
        self._scene_initializer = scene_initializer
        self._pose_initializer = pose_initializer
        self._ik_seed_factory = ik_seed_factory or (lambda seed: seed)
        self._rest_action_factory = rest_action_factory or (lambda backend: backend.rest_action())
        self._solving_waypoint_label = ""
        self.waypoints: tuple[Waypoint, ...] = ()
        self._actions: tuple[np.ndarray, ...] = ()
        self._joint_states: tuple[dict[str, float], ...] = ()
        self._pose_targets: dict[
            tuple[int, str],
            tuple[np.ndarray, np.ndarray],
        ] = {}
        self.current_label = ""

    def reset(self, *, seed: int) -> None:
        self.backend.reset(seed=seed)
        if self._scene_initializer is not None:
            self._scene_initializer(self.backend, seed)
        self.waypoints = (
            self._waypoints_or_factory(seed)
            if callable(self._waypoints_or_factory)
            else self._waypoints_or_factory
        )
        if len(self.waypoints) < 2:
            raise ValueError("A task demonstration needs at least two waypoints")
        times = tuple(waypoint.time for waypoint in self.waypoints)
        if times[0] != 0.0 or any(right <= left for left, right in pairwise(times)):
            raise ValueError("Waypoint times must start at zero and increase strictly")
        self._actions = self._solve_actions(self._ik_seed_factory(seed))
        self._joint_states = self._build_joint_states()
        self.current_label = self.waypoints[0].label
        self.backend.reset(seed=seed)
        if self._scene_initializer is not None:
            self._scene_initializer(self.backend, seed)
        if self._pose_initializer is not None:
            self._pose_initializer(self)

    def _solve_actions(self, seed: int) -> tuple[np.ndarray, ...]:
        rng = np.random.default_rng(seed)
        action = self._rest_action()
        solved: list[np.ndarray] = []
        self._pose_targets = {}
        for waypoint_index, waypoint in enumerate(self.waypoints):
            self._solving_waypoint_label = waypoint.label
            if waypoint.return_to_rest:
                action = self._rest_action()
            if waypoint.left_gripper is not None:
                action[FINGER_ACTION_INDICES["l"]] = waypoint.left_gripper
            if waypoint.right_gripper is not None:
                action[FINGER_ACTION_INDICES["r"]] = waypoint.right_gripper
            if waypoint.left is not None:
                action, orientation = self._solve_arm(
                    "l",
                    waypoint.left,
                    waypoint.left_quaternion,
                    waypoint.left_preferred_quaternion,
                    action,
                    rng,
                    waypoint.locked_actuators,
                )
                self._pose_targets[(waypoint_index, "l")] = (
                    np.asarray(waypoint.left),
                    orientation,
                )
            if waypoint.right is not None:
                action, orientation = self._solve_arm(
                    "r",
                    waypoint.right,
                    waypoint.right_quaternion,
                    waypoint.right_preferred_quaternion,
                    action,
                    rng,
                    waypoint.locked_actuators,
                )
                self._pose_targets[(waypoint_index, "r")] = (
                    np.asarray(waypoint.right),
                    orientation,
                )
            solved.append(action.copy())
        return tuple(solved)

    def _solve_arm(
        self,
        side: str,
        target_position: tuple[float, float, float],
        target_quaternion: tuple[float, float, float, float] | None,
        preferred_quaternion: tuple[float, float, float, float] | None,
        seed_action: np.ndarray,
        rng: np.random.Generator,
        locked_actuators: tuple[str, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = ARM_ACTION_INDICES[side]
        try:
            locked_indices = np.asarray(
                [ACTUATOR_NAMES.index(name) for name in locked_actuators],
                dtype=int,
            )
        except ValueError as error:
            raise KeyError(f"Unknown locked actuator in {locked_actuators!r}") from error
        if not set(locked_indices).issubset(set(indices)):
            raise ValueError(
                f"Locked actuators {locked_actuators!r} must belong to the {side}-arm"
            )
        locked_values = seed_action[locked_indices].copy()
        rest_arm_candidate = seed_action.copy()
        rest_arm_candidate[indices] = self._rest_action()[indices]
        rest_arm_candidate[locked_indices] = locked_values
        candidates = [seed_action.copy(), rest_arm_candidate]
        for _ in range(12):
            candidate = seed_action.copy()
            candidate[indices] = _sample_mirrored_arm_positions(
                self.backend,
                side,
                rng,
            )
            candidate[locked_indices] = locked_values
            candidates.append(candidate)
        if target_quaternion is None:
            position_solutions = []
            for candidate in candidates:
                solved, position_error = self._iterate_arm_position(
                    side,
                    np.asarray(target_position),
                    candidate,
                    locked_indices,
                    locked_values,
                )
                position_solutions.append((solved, position_error))
            reachable = [result for result in position_solutions if result[1] <= 0.012]
            if not reachable:
                _, best_error = min(
                    position_solutions,
                    key=lambda result: result[1],
                )
                raise IKConvergenceError(
                    f"{side}-arm position {target_position} remained "
                    f"{best_error:.4f} m away"
                )
            orientation_preference = np.asarray(
                preferred_quaternion or IDENTITY_QUATERNION,
                dtype=float,
            )
            orientation_preference /= np.linalg.norm(orientation_preference)
            ranked: list[tuple[float, np.ndarray, np.ndarray]] = []
            for solved, position_error in reachable:
                achieved_quaternion = self._site_quaternion(side, solved)
                rotation_error = np.empty(3)
                mujoco.mju_subQuat(
                    rotation_error,
                    orientation_preference,
                    achieved_quaternion,
                )
                score = position_error + ORIENTATION_PREFERENCE_WEIGHT * float(
                    np.linalg.norm(rotation_error)
                )
                ranked.append((score, solved, achieved_quaternion))
            _, reachable_action, target_quaternion_array = min(
                ranked,
                key=lambda result: result[0],
            )
            candidates.insert(0, reachable_action)
        else:
            target_quaternion_array = np.asarray(target_quaternion, dtype=float)
            quaternion_norm = float(np.linalg.norm(target_quaternion_array))
            if quaternion_norm == 0.0:
                raise ValueError("Target quaternion cannot be zero")
            target_quaternion_array = target_quaternion_array / quaternion_norm

        best_score = float("inf")
        best_position_error = float("inf")
        best_orientation_error = float("inf")
        best_action = candidates[0]
        for candidate in candidates:
            solved, position_error, orientation_error = self._iterate_arm(
                side,
                np.asarray(target_position),
                target_quaternion_array,
                candidate,
                locked_indices,
                locked_values,
            )
            score = position_error + 0.05 * orientation_error
            if score < best_score:
                best_score = score
                best_position_error = position_error
                best_orientation_error = orientation_error
                best_action = solved
            if position_error < 0.002 and orientation_error < np.deg2rad(2.0):
                break
        if best_position_error > 0.012 or best_orientation_error > np.deg2rad(8.0):
            self.backend.data.qpos[self.backend.qpos_ids] = best_action
            mujoco.mj_forward(self.backend.model, self.backend.data)
            achieved_position = tuple(
                float(value) for value in hand_tcp_position(self.backend, side)
            )
            diagnostic = (
                f"{side}-arm {self._solving_waypoint_label!r}: target "
                f"{tuple(float(value) for value in target_position)}, residual "
                f"{best_position_error:.4f} m / "
                f"{np.rad2deg(best_orientation_error):.1f} deg, "
                f"achieved {achieved_position}"
            )
            raise IKConvergenceError(diagnostic)
        return best_action, target_quaternion_array.copy()

    def _site_quaternion(self, side: str, action: np.ndarray) -> np.ndarray:
        self.backend.data.qpos[self.backend.qpos_ids] = action
        mujoco.mj_forward(self.backend.model, self.backend.data)
        return hand_tcp_quaternion(self.backend, side)

    def _iterate_arm_position(
        self,
        side: str,
        target_position: np.ndarray,
        initial_action: np.ndarray,
        locked_action_indices: np.ndarray,
        locked_action_values: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        return solve_arm_position(
            self.backend,
            side,
            target_position,
            initial_action,
            locked_action_indices,
            locked_action_values,
        )

    def pose_target(
        self,
        waypoint_index: int,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the resolved position and orientation for one arm waypoint."""

        try:
            position, quaternion = self._pose_targets[(waypoint_index, side)]
        except KeyError as error:
            raise KeyError(f"No {side!r}-arm pose target at waypoint {waypoint_index}") from error
        return position.copy(), quaternion.copy()

    def _iterate_arm(
        self,
        side: str,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        initial_action: np.ndarray,
        locked_action_indices: np.ndarray,
        locked_action_values: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        action = initial_action.copy()
        indices = ARM_ACTION_INDICES[side]
        movable_indices = indices[~np.isin(indices, locked_action_indices)]
        action[locked_action_indices] = locked_action_values
        joint_ids = self.backend.joint_ids[movable_indices]
        dof_ids = self.backend.model.jnt_dofadr[joint_ids]
        current_quaternion = np.empty(4)
        orientation_error = np.empty(3)
        orientation_weight = 0.05
        best_action = action.copy()
        best_position_error = float("inf")
        best_orientation_error = float("inf")
        best_score = float("inf")

        for _ in range(500):
            self.backend.data.qpos[self.backend.qpos_ids] = action
            mujoco.mj_forward(self.backend.model, self.backend.data)
            point = hand_tcp_position(self.backend, side)
            position_error = target_position - point
            current_quaternion[:] = hand_tcp_quaternion(self.backend, side)
            mujoco.mju_subQuat(
                orientation_error,
                target_quaternion,
                current_quaternion,
            )
            position_error_norm = float(np.linalg.norm(position_error))
            orientation_error_norm = float(np.linalg.norm(orientation_error))
            score = position_error_norm + orientation_weight * orientation_error_norm
            if score < best_score:
                best_score = score
                best_action = action.copy()
                best_position_error = position_error_norm
                best_orientation_error = orientation_error_norm
            if position_error_norm < 0.002 and orientation_error_norm < np.deg2rad(2.0):
                return action, position_error_norm, orientation_error_norm
            position_jacobian, rotation_jacobian = _hand_tcp_jacobians(
                self.backend,
                side,
            )
            arm_jacobian = np.vstack(
                (
                    position_jacobian[:, dof_ids],
                    orientation_weight * rotation_jacobian[:, dof_ids],
                )
            )
            pose_error = np.concatenate((position_error, orientation_weight * orientation_error))
            delta = np.linalg.solve(
                arm_jacobian.T @ arm_jacobian + 5e-4 * np.eye(len(movable_indices)),
                arm_jacobian.T @ pose_error,
            )
            action[movable_indices] += np.clip(delta, -0.08, 0.08)
            action[movable_indices] = np.clip(
                action[movable_indices],
                self.backend.control_low[movable_indices] + 0.01,
                self.backend.control_high[movable_indices] - 0.01,
            )

        return best_action, best_position_error, best_orientation_error

    def _rest_action(self) -> np.ndarray:
        action = np.asarray(self._rest_action_factory(self.backend), dtype=float)
        if action.shape != self.backend.control_low.shape:
            raise ValueError(
                f"Task rest action must match the backend action shape, got {action.shape}"
            )
        return np.clip(
            action,
            self.backend.control_low,
            self.backend.control_high,
        ).copy()

    def _build_joint_states(self) -> tuple[dict[str, float], ...]:
        state: dict[str, float] = {}
        states: list[dict[str, float]] = []
        for waypoint in self.waypoints:
            state = {**state, **dict(waypoint.joint_positions)}
            states.append(state.copy())
        return tuple(states)

    def _segment(self, elapsed: float) -> tuple[int, int, float]:
        if elapsed <= 0.0:
            return 0, 1, 0.0
        for right in range(1, len(self.waypoints)):
            if elapsed <= self.waypoints[right].time:
                left = right - 1
                span = self.waypoints[right].time - self.waypoints[left].time
                return left, right, smoothstep((elapsed - self.waypoints[left].time) / span)
        return len(self.waypoints) - 2, len(self.waypoints) - 1, 1.0

    def action(self, progress: float) -> np.ndarray:
        elapsed = float(np.clip(progress, 0.0, 1.0)) * self.waypoints[-1].time
        left, right, blend = self._segment(elapsed)
        self.current_label = self.waypoints[right].label
        action = (1.0 - blend) * self._actions[left] + blend * self._actions[right]
        self._update_task_joints(left, right, blend)
        return action

    def _update_task_joints(self, left: int, right: int, blend: float) -> None:
        names = set(self._joint_states[left]) | set(self._joint_states[right])
        for name in names:
            start = self._joint_states[left].get(name, 0.0)
            end = self._joint_states[right].get(name, start)
            self._set_joint_position(name, (1.0 - blend) * start + blend * end)

    def _set_joint_position(self, name: str, value: float) -> None:
        joint_id = mujoco.mj_name2id(self.backend.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"MuJoCo joint not found: {name}")
        qpos_id = int(self.backend.model.jnt_qposadr[joint_id])
        dof_id = int(self.backend.model.jnt_dofadr[joint_id])
        self.backend.data.qpos[qpos_id] = value
        self.backend.data.qvel[dof_id] = 0.0
        mujoco.mj_forward(self.backend.model, self.backend.data)

    def evaluate(self) -> TaskEvaluation:
        mujoco.mj_forward(self.backend.model, self.backend.data)
        return self._evaluator(self.backend)
