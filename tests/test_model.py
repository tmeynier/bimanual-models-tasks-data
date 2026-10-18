from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from simulation import SimulationConfig, available_backends, create_backend
from simulation.backends.mujoco import ACTUATOR_NAMES, MujocoBackend
from simulation.backends.mujoco.backend import default_model_path
from simulation.tasks.ik import (
    ARM_ACTION_INDICES,
    hand_tcp_position,
    hand_tcp_quaternion,
)

ROBOT_MESH_NAMES = {
    "base_link",
    "head_link",
    "shoulder_l_link",
    "forearm_l_link",
    "arm_l_link",
    "wrist_l_link",
    "hand_l_link",
    "finger_l_link",
    "shoulder_r_link",
    "forearm_r_link",
    "arm_r_link",
    "wrist_r_link",
    "hand_r_link",
    "finger_r_link",
}
REMOVED_SCENE_TERMS = ("drawer", "rack", "storage", "cube")
ROBOT_CAMERA_NAMES = {
    "head_camera",
    "wrist_l_camera",
    "wrist_r_camera",
    "floor_camera",
}


def _names(model: mujoco.MjModel, kind: mujoco.mjtObj, count: int) -> set[str]:
    return {
        name
        for object_id in range(count)
        if (name := mujoco.mj_id2name(model, kind, object_id)) is not None
    }


def test_mujoco_model_matches_control_contract() -> None:
    assert "mujoco" in available_backends()
    with create_backend("mujoco", SimulationConfig(render=False)) as backend:
        assert isinstance(backend, MujocoBackend)
        assert len(backend.actuator_ids) == len(ACTUATOR_NAMES) == 12
        assert backend.joint_positions().shape == (12,)
        assert tuple(backend.action_features) == tuple(
            f"{name}.pos" for name in ACTUATOR_NAMES
        )
        assert backend.observation_features["head_camera"] == (256, 256, 3)


def test_default_scene_contains_only_the_bimanual_robot() -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        model = backend.model
        assert (model.nq, model.nv, model.nu) == (12, 12, 12)
        assert model.neq == 0
        assert model.npair == 0
        assert _names(model, mujoco.mjtObj.mjOBJ_MESH, model.nmesh) == ROBOT_MESH_NAMES
        assert (
            _names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
            == ROBOT_CAMERA_NAMES
        )
        camera_parents = {
            name: mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(
                    model.cam_bodyid[
                        mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_CAMERA, name
                        )
                    ]
                ),
            )
            for name in ("wrist_l_camera", "wrist_r_camera", "floor_camera")
        }
        assert camera_parents == {
            "wrist_l_camera": "wrist_l_link",
            "wrist_r_camera": "wrist_r_link",
            "floor_camera": "base_link",
        }

        scene_names = set().union(
            _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
            _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
            _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
            _names(model, mujoco.mjtObj.mjOBJ_MESH, model.nmesh),
        )
        assert not {
            name
            for name in scene_names
            if any(term in name.lower() for term in REMOVED_SCENE_TERMS)
        }


def test_asset_directory_contains_only_referenced_robot_meshes() -> None:
    mesh_directory = Path(default_model_path()).parent / "meshes"
    assert {path.stem for path in mesh_directory.glob("*.stl")} == ROBOT_MESH_NAMES


def test_actuators_control_exactly_the_bimanual_joints() -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        names = tuple(
            mujoco.mj_id2name(
                backend.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                int(actuator_id),
            )
            for actuator_id in backend.actuator_ids
        )
        assert names == ACTUATOR_NAMES
        assert len(set(backend.joint_ids.tolist())) == 12


def test_grasp_sites_are_between_the_distal_finger_tips() -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        centers: dict[tuple[float, str], np.ndarray] = {}
        for finger_angle in (0.15, 1.40):
            for side in ("l", "r"):
                finger_joint = mujoco.mj_name2id(
                    backend.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"hand_{side}_link_hand_{side}_finger_{side}_joint",
                )
                backend.data.qpos[backend.model.jnt_qposadr[finger_joint]] = finger_angle
            mujoco.mj_forward(backend.model, backend.data)
            backend.update_hand_tcp_markers()

            for side in ("l", "r"):
                fixed_site = mujoco.mj_name2id(
                    backend.model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    f"hand_{side}_fixed_tip_site",
                )
                moving_site = mujoco.mj_name2id(
                    backend.model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    f"finger_{side}_moving_tip_site",
                )
                expected_center = 0.5 * (
                    backend.data.site_xpos[fixed_site]
                    + backend.data.site_xpos[moving_site]
                )
                marker_body = mujoco.mj_name2id(
                    backend.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"hand_{side}_grasp_frame",
                )
                assert np.allclose(
                    backend.data.xpos[marker_body],
                    expected_center,
                )
                centers[(finger_angle, side)] = expected_center.copy()

        for side in ("l", "r"):
            assert not np.allclose(centers[(0.15, side)], centers[(1.40, side)])


def test_robot_resets_and_steps_with_finite_joint_state() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        backend.reset(seed=42)
        backend.send_action(backend.home_action())
        backend.step()
        positions = backend.joint_positions()
        assert positions.shape == (12,)
        assert np.all(np.isfinite(positions))
    # MuJoCo's constraint solver can permit tiny hard-stop penetration during
    # one integration step; 1e-3 rad is about 0.057 degrees.
    assert np.all(positions >= backend.control_low - 1e-3)
    assert np.all(positions <= backend.control_high + 1e-3)


def test_robot_uses_mirrored_arm_joint_limits() -> None:
    expected = {
        "forearm_l": (0.0, 3.0 * np.pi / 4.0),
        "arm_l": (-np.pi / 2.0, 0.0),
        "wrist_l": (0.0, np.pi / 2.0),
        "hand_l": (-np.pi / 4.0, np.pi / 4.0),
        "forearm_r": (-3.0 * np.pi / 4.0, 0.0),
        "arm_r": (0.0, np.pi / 2.0),
        "wrist_r": (-np.pi / 2.0, 0.0),
        "hand_r": (-np.pi / 4.0, np.pi / 4.0),
    }
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        for actuator_name, requested_range in expected.items():
            actuator_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )
            joint_id = int(backend.model.actuator_trnid[actuator_id, 0])
            assert np.allclose(
                backend.model.jnt_range[joint_id],
                requested_range,
            )
            assert np.allclose(
                backend.model.actuator_ctrlrange[actuator_id],
                requested_range,
            )


def test_robot_tcp_kinematics_are_exactly_mirrored() -> None:
    reflection = np.diag((-1.0, 1.0, 1.0))
    samples = (
        np.zeros(5),
        np.array((0.3, 0.8, -0.4, 0.6, 0.2)),
        np.array((-0.7, 1.4, 0.5, 1.1, -0.9)),
    )
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        for left_arm in samples:
            action = backend.rest_action().astype(float)
            action[ARM_ACTION_INDICES["l"]] = left_arm
            action[ARM_ACTION_INDICES["r"]] = -left_arm
            backend.data.qpos[backend.qpos_ids] = action
            mujoco.mj_forward(backend.model, backend.data)

            left_rotation = np.empty(9)
            right_rotation = np.empty(9)
            mujoco.mju_quat2Mat(
                left_rotation,
                hand_tcp_quaternion(backend, "l"),
            )
            mujoco.mju_quat2Mat(
                right_rotation,
                hand_tcp_quaternion(backend, "r"),
            )
            assert np.allclose(
                hand_tcp_position(backend, "r"),
                reflection @ hand_tcp_position(backend, "l"),
                atol=1e-12,
            )
            assert np.allclose(
                right_rotation.reshape(3, 3),
                reflection @ left_rotation.reshape(3, 3) @ reflection,
                atol=1e-12,
            )
