from __future__ import annotations

from importlib import import_module
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np
import pytest

from simulation import SimulationConfig
from simulation.tasks import (
    TASKS,
    create_task_backend,
    create_task_controller,
    get_task,
    task_scene_path,
)
from simulation.tasks.ik import (
    ARM_ACTION_INDICES,
    CLOSED_GRIPPER,
    CUBE_CLOSED_GRIPPER,
    CUBE_GRASP_QUATERNION,
    FINGER_ACTION_INDICES,
    MIRRORED_CUBE_GRASP_QUATERNION,
    OPEN_GRIPPER,
    WIDE_OPEN_GRIPPER,
    IKConvergenceError,
    hand_tcp_position,
    hand_tcp_quaternion,
)

EXPECTED_TITLES = (
    "Basic object placement",
    "Object retrieval",
    "Color-conditioned sorting",
    "Number-conditioned drawer opening",
    "Precision cube stacking",
    "Bimanual mid-air handover",
    "Warehouse stocking",
)


def _reachable_task_one_layout(task_module):
    return task_module.TaskLayout(
        cube=(-0.117, 0.123, 0.035),
        container=(0.011, 0.114, 0.0),
        working_arm="r",
    )


def test_paper_defines_seven_ordered_tasks() -> None:
    assert tuple(task.task_id for task in TASKS) == tuple(range(1, 8))
    assert tuple(task.title for task in TASKS) == EXPECTED_TITLES
    assert tuple(task.nominal_duration_seconds for task in TASKS) == (
        10,
        10,
        10,
        10,
        10,
        10,
        20,
    )
    assert all(task.instruction and task.subgoals for task in TASKS)


def test_each_task_owns_documentation_and_ik_entry_point() -> None:
    tasks_root = Path(__file__).parents[1] / "simulation" / "tasks"
    task_directories = sorted(tasks_root.glob("task_[0-9][0-9]_*"))
    assert len(task_directories) == len(TASKS)

    for task, directory in zip(TASKS, task_directories, strict=True):
        assert (directory / "README.md").is_file()
        assert (directory / "ik.py").is_file()
        module = import_module(f"simulation.tasks.{directory.name}.ik")
        assert module.TASK is task


def test_all_seven_tasks_own_implemented_scenes() -> None:
    assert all(task.implemented for task in TASKS)
    for task in TASKS:
        assert task_scene_path(task.task_id).is_file()


def test_task_one_ik_matches_resolved_position_and_orientation_targets(
    monkeypatch,
) -> None:
    task_module = import_module("simulation.tasks.task_01_basic_object_placement.ik")
    monkeypatch.setattr(
        task_module,
        "sample_layout",
        lambda backend, seed: _reachable_task_one_layout(task_module),
    )
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(1, config) as backend:
        controller = create_task_controller(1, backend)
        controller.reset(seed=0)

        cube_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "cube",
        )
        source_origin_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "task_ik_marker_source_origin",
        )
        assert backend.model.geom_bodyid[source_origin_id] == cube_id
        assert np.allclose(
            backend.data.xmat[cube_id].reshape(3, 3),
            np.eye(3),
        )
        assert backend.data.xpos[cube_id, 2] == pytest.approx(0.035)

        for waypoint_index, waypoint in enumerate(controller.waypoints):
            for side, position in (("l", waypoint.left), ("r", waypoint.right)):
                if position is None:
                    continue
                target_position, target_quaternion = controller.pose_target(
                    waypoint_index,
                    side,
                )
                backend.data.qpos[backend.qpos_ids] = controller._actions[waypoint_index]
                mujoco.mj_forward(backend.model, backend.data)
                achieved_quaternion = hand_tcp_quaternion(backend, side)
                orientation_error = np.empty(3)
                mujoco.mju_subQuat(
                    orientation_error,
                    target_quaternion,
                    achieved_quaternion,
                )
                assert np.linalg.norm(target_position - hand_tcp_position(backend, side)) <= 0.012
                assert np.linalg.norm(orientation_error) <= np.deg2rad(8.0)


def test_cube_grasp_orientation_uses_requested_axis_mapping() -> None:
    left_rotation = np.empty(9)
    right_rotation = np.empty(9)
    mujoco.mju_quat2Mat(left_rotation, np.asarray(CUBE_GRASP_QUATERNION))
    mujoco.mju_quat2Mat(
        right_rotation,
        np.asarray(MIRRORED_CUBE_GRASP_QUATERNION),
    )
    left_rotation = left_rotation.reshape(3, 3)
    right_rotation = right_rotation.reshape(3, 3)
    reflection = np.diag((-1.0, 1.0, 1.0))

    assert np.allclose(left_rotation[:, 2], (0.0, 0.0, 1.0))
    assert np.allclose(left_rotation[:, 1], (1.0, 0.0, 0.0))
    assert np.allclose(left_rotation[:, 0], (0.0, -1.0, 0.0))
    assert np.allclose(right_rotation, reflection @ left_rotation @ reflection)


def test_task_one_aligns_open_tcp_10_mm_above_cube_before_closing(monkeypatch) -> None:
    task_module = import_module("simulation.tasks.task_01_basic_object_placement.ik")
    monkeypatch.setattr(
        task_module,
        "sample_layout",
        lambda backend, seed: _reachable_task_one_layout(task_module),
    )
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(1, config) as backend:
        controller = create_task_controller(1, backend)
        controller.reset(seed=0)
        labels = [waypoint.label for waypoint in controller.waypoints]
        align_index = labels.index("align open gripper 10 mm above cube center")
        hold_index = labels.index("hold completed cube IK")
        close_index = labels.index("close fingers around cube")
        lift_index = labels.index("lift cube")
        assert align_index < hold_index < close_index < lift_index
        lower_index = labels.index("lower cube")
        release_index = labels.index("release cube")

        align = controller.waypoints[align_index]
        side = "l" if align.left is not None else "r"
        cube_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "cube",
        )
        cube_center = backend.data.xpos[cube_id].copy()
        target_position, _ = controller.pose_target(align_index, side)
        expected_target = cube_center + np.array((0.0, 0.0, 0.010))
        assert np.allclose(target_position, expected_target)

        finger_index = FINGER_ACTION_INDICES[side]
        arm_indices = ARM_ACTION_INDICES[side]
        align_action = controller._actions[align_index]
        hold_action = controller._actions[hold_index]
        close_action = controller._actions[close_index]
        assert align_action[finger_index] == pytest.approx(np.pi / 2.0)
        assert hold_action[finger_index] == pytest.approx(np.pi / 2.0)
        assert close_action[finger_index] == pytest.approx(CUBE_CLOSED_GRIPPER)
        assert np.allclose(align_action[arm_indices], hold_action[arm_indices])
        assert not np.allclose(hold_action[arm_indices], close_action[arm_indices])

        backend.data.qpos[backend.qpos_ids] = align_action
        mujoco.mj_forward(backend.model, backend.data)
        assert np.linalg.norm(hand_tcp_position(backend, side) - expected_target) <= 0.002

        close_target, _ = controller.pose_target(close_index, side)
        assert np.allclose(close_target, expected_target)
        backend.data.qpos[backend.qpos_ids] = close_action
        mujoco.mj_forward(backend.model, backend.data)
        assert np.linalg.norm(hand_tcp_position(backend, side) - expected_target) <= 0.002

        lower_target, _ = controller.pose_target(lower_index, side)
        release_target, _ = controller.pose_target(release_index, side)
        assert lower_target[2] == pytest.approx(0.135)
        assert release_target[2] == pytest.approx(0.135)
        container_target_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target_ik_frame",
        )
        assert backend.data.xpos[container_target_id, 2] == pytest.approx(0.135)


def test_task_one_samples_layout_once_per_controller_reset(monkeypatch) -> None:
    task_module = import_module("simulation.tasks.task_01_basic_object_placement.ik")
    sampled_seeds: list[int] = []

    def counted_sample_layout(backend, seed):
        sampled_seeds.append(seed)
        return _reachable_task_one_layout(task_module)

    monkeypatch.setattr(task_module, "sample_layout", counted_sample_layout)
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(1, config) as backend:
        controller = create_task_controller(1, backend)
        controller.reset(seed=0)

    assert sampled_seeds == [0]


def test_task_one_resamples_only_after_actual_trajectory_failure(monkeypatch) -> None:
    task_module = import_module("simulation.tasks.task_01_basic_object_placement.ik")
    sampled_seeds: list[int] = []
    reachable = _reachable_task_one_layout(task_module)

    def sampled_layout(backend, seed):
        sampled_seeds.append(seed)
        if len(sampled_seeds) == 1:
            return task_module.TaskLayout(
                cube=(10.0, 10.0, task_module.CUBE_CENTER_Z),
                container=(0.0, 0.115, 0.0),
                working_arm="l",
            )
        return reachable

    monkeypatch.setattr(task_module, "sample_layout", sampled_layout)
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(1, config) as backend:
        controller = create_task_controller(1, backend)
        controller.reset(seed=4)
        details = controller.evaluate().details

    assert sampled_seeds == [4, task_module._layout_attempt_seed(4, 1)]
    assert details["sampled_cube_position"] == reachable.cube
    assert details["layout_attempts"] == 2


def test_task_one_samples_independent_reachable_layouts() -> None:
    task_module = import_module("simulation.tasks.task_01_basic_object_placement.ik")
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(1, config) as backend:
        layouts = [task_module.sample_layout(backend, seed) for seed in range(40)]
        assert task_module.sample_layout(backend, 7) == task_module.sample_layout(
            backend,
            7,
        )
        for layout in layouts:
            assert layout.working_arm == task_module._closest_resting_arm(
                backend,
                layout.cube,
            )
            assert task_module.CONTAINER_X_RANGE[0] <= layout.container[0] <= (
                task_module.CONTAINER_X_RANGE[1]
            )
            assert task_module.CONTAINER_Y_RANGE[0] <= layout.container[1] <= (
                task_module.CONTAINER_Y_RANGE[1]
            )
            assert task_module.CUBE_X_RANGE[0] <= layout.cube[0] <= (
                task_module.CUBE_X_RANGE[1]
            )
            assert task_module.CUBE_Y_RANGE[0] <= layout.cube[1] <= (
                task_module.CUBE_Y_RANGE[1]
            )
            assert task_module._footprints_are_separate(
                layout.cube,
                layout.container,
            )

    assert {layout.working_arm for layout in layouts} == {"l", "r"}
    assert not np.allclose(
        layouts[0].cube,
        (-layouts[1].cube[0], layouts[1].cube[1], layouts[1].cube[2]),
    )
    assert np.ptp([layout.cube[0] for layout in layouts]) > 0.20
    assert np.ptp([layout.cube[1] for layout in layouts]) > 0.05
    assert np.ptp([layout.container[0] for layout in layouts]) > 0.18
    assert np.ptp([layout.container[1] for layout in layouts]) > 0.025


def test_task_one_reachable_sampler_plans_across_seed_sweep() -> None:
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    used_sides = set()
    with create_task_backend(1, config) as backend:
        controller = create_task_controller(1, backend)
        for seed in range(20):
            controller.reset(seed=seed)
            used_sides.add(
                "l"
                if any(waypoint.left is not None for waypoint in controller.waypoints)
                else "r"
            )
    assert used_sides == {"l", "r"}


def test_task_one_locks_cube_after_close_and_releases_before_open(monkeypatch) -> None:
    task_module = import_module("simulation.tasks.task_01_basic_object_placement.ik")
    monkeypatch.setattr(
        task_module,
        "sample_layout",
        lambda backend, seed: _reachable_task_one_layout(task_module),
    )
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(1, config) as backend:
        controller = create_task_controller(1, backend)
        controller.reset(seed=0)
        labels = [waypoint.label for waypoint in controller.waypoints]
        close_index = labels.index("close fingers around cube")
        lift_index = labels.index("lift cube")
        transport_index = labels.index("move above target")
        lower_index = labels.index("lower cube")
        release_index = labels.index("release cube")
        retreat_index = labels.index("retreat")
        side = "l" if controller.waypoints[lift_index].left is not None else "r"
        cube_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "cube",
        )

        def cube_pose_in_hand() -> tuple[np.ndarray, np.ndarray]:
            hand_quaternion = hand_tcp_quaternion(backend, side)
            hand_rotation = np.empty(9)
            mujoco.mju_quat2Mat(hand_rotation, hand_quaternion)
            relative_position = hand_rotation.reshape(3, 3).T @ (
                backend.data.xpos[cube_id] - hand_tcp_position(backend, side)
            )
            inverse_hand = hand_quaternion.copy()
            inverse_hand[1:] *= -1.0
            relative_quaternion = np.empty(4)
            mujoco.mju_mulQuat(
                relative_quaternion,
                inverse_hand,
                backend.data.xquat[cube_id],
            )
            return relative_position, relative_quaternion

        close_action = controller._actions[close_index]
        finger_index = FINGER_ACTION_INDICES[side]
        for waypoint_index in (
            close_index,
            lift_index,
            transport_index,
            lower_index,
        ):
            assert controller._actions[waypoint_index][finger_index] == pytest.approx(
                CUBE_CLOSED_GRIPPER
            )
        for waypoint_index in (release_index, retreat_index):
            assert controller._actions[waypoint_index][finger_index] == pytest.approx(OPEN_GRIPPER)
        backend.data.qpos[backend.qpos_ids] = close_action
        backend.data.ctrl[backend.actuator_ids] = close_action
        mujoco.mj_forward(backend.model, backend.data)
        controller.action(0.401)
        assert backend.body_is_attached("cube")
        initial_position, initial_quaternion = cube_pose_in_hand()

        lift_action = controller._actions[lift_index]
        backend.data.qpos[backend.qpos_ids] = lift_action
        backend.data.ctrl[backend.actuator_ids] = lift_action
        mujoco.mj_forward(backend.model, backend.data)
        backend.step()
        moved_position, moved_quaternion = cube_pose_in_hand()
        assert np.allclose(moved_position, initial_position, atol=1e-4)
        assert abs(float(np.dot(moved_quaternion, initial_quaternion))) == pytest.approx(
            1.0,
            abs=1e-6,
        )

        controller.action(0.751)
        assert not backend.body_is_attached("cube")


def test_task_one_requires_30_mm_cube_container_clearance() -> None:
    task_module = import_module("simulation.tasks.task_01_basic_object_placement.ik")
    required_center_separation = (
        task_module.CUBE_HALF_WIDTH
        + task_module.CONTAINER_OUTER_HALF_WIDTH
        + task_module.MINIMUM_EDGE_CLEARANCE
    )
    container = (0.0, 0.0, 0.0)
    almost_clear = (required_center_separation - 1e-6, 0.0, 0.015)
    clear = (required_center_separation + 1e-6, 0.0, 0.015)

    assert pytest.approx(0.030) == task_module.MINIMUM_EDGE_CLEARANCE
    assert not task_module._footprints_are_separate(almost_clear, container)
    assert task_module._footprints_are_separate(clear, container)


def test_task_two_samples_full_workspace_layouts_and_uses_closest_arm() -> None:
    task_module = import_module("simulation.tasks.task_02_object_retrieval.ik")
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(2, config) as backend:
        layouts = [task_module.sample_layout(backend, seed) for seed in range(40)]
        assert task_module.sample_layout(backend, 7) == task_module.sample_layout(
            backend,
            7,
        )
        for layout in layouts:
            assert layout.working_arm == task_module._closest_resting_arm(
                backend,
                layout.cube,
            )
            assert task_module.CONTAINER_X_RANGE[0] <= layout.container[0] <= (
                task_module.CONTAINER_X_RANGE[1]
            )
            assert task_module.CONTAINER_Y_RANGE[0] <= layout.container[1] <= (
                task_module.CONTAINER_Y_RANGE[1]
            )
            assert task_module.DESTINATION_X_RANGE[0] <= layout.destination[0] <= (
                task_module.DESTINATION_X_RANGE[1]
            )
            assert task_module.DESTINATION_Y_RANGE[0] <= layout.destination[1] <= (
                task_module.DESTINATION_Y_RANGE[1]
            )

    maximum_inside_offset = (
        task_module.CONTAINER_INTERIOR_HALF_WIDTH
        - task_module.CUBE_HALF_WIDTH
        - task_module.INSIDE_CONTAINER_MARGIN
    )
    for layout in layouts:
        assert abs(layout.cube[0] - layout.container[0]) <= maximum_inside_offset
        assert abs(layout.cube[1] - layout.container[1]) <= maximum_inside_offset
        assert layout.cube[2] == pytest.approx(0.045)
        assert task_module._footprints_are_separate(
            layout.destination,
            layout.container,
        )

    assert {layout.working_arm for layout in layouts} == {"l", "r"}
    assert np.ptp([layout.container[0] for layout in layouts]) > 0.18
    assert np.ptp([layout.container[1] for layout in layouts]) > 0.025
    assert np.ptp([layout.destination[0] for layout in layouts]) > 0.20
    assert np.ptp([layout.destination[1] for layout in layouts]) > 0.05


def test_task_two_resamples_only_after_actual_trajectory_failure(monkeypatch) -> None:
    task_module = import_module("simulation.tasks.task_02_object_retrieval.ik")
    sampled_seeds: list[int] = []
    reachable = task_module.TaskLayout(
        cube=(-0.10363055972693495, 0.11969561145788811, 0.045),
        container=(-0.09893407243132131, 0.11403199868443305, 0.0),
        destination=(0.02510597077140663, 0.1496835504712892, 0.035),
        working_arm="r",
    )

    def sampled_layout(backend, seed):
        sampled_seeds.append(seed)
        if len(sampled_seeds) == 1:
            return task_module.TaskLayout(
                cube=(10.0, 10.0, task_module.CONTAINER_CUBE_CENTER_Z),
                container=(10.0, 10.0, 0.0),
                destination=(0.0, 0.115, task_module.WORKSPACE_CUBE_CENTER_Z),
                working_arm="l",
            )
        return reachable

    monkeypatch.setattr(task_module, "sample_layout", sampled_layout)
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(2, config) as backend:
        controller = create_task_controller(2, backend)
        controller.reset(seed=4)
        details = controller.evaluate().details

    assert sampled_seeds == [4, task_module._layout_attempt_seed(4, 1)]
    assert details["sampled_cube_position"] == reachable.cube
    assert details["layout_attempts"] == 2


def test_task_workspace_boundary_dimensions() -> None:
    task_one = import_module(
        "simulation.tasks.task_01_basic_object_placement.ik"
    )
    assert task_one.WORKSPACE_CENTER == (0.0, 0.1175)
    assert pytest.approx(0.30) == task_one.WORKSPACE_LENGTH
    assert pytest.approx(0.105) == task_one.WORKSPACE_WIDTH

    task_two = import_module("simulation.tasks.task_02_object_retrieval.ik")
    assert task_two.WORKSPACE_CENTER == (0.0, 0.1175)
    assert pytest.approx(0.30) == task_two.WORKSPACE_LENGTH
    assert pytest.approx(0.105) == task_two.WORKSPACE_WIDTH

    task_three = import_module("simulation.tasks.task_03_color_conditioned_sorting.ik")
    assert task_three.WORKSPACE_CENTER == (0.0, 0.1175)
    assert pytest.approx(0.30) == task_three.WORKSPACE_LENGTH
    assert pytest.approx(0.105) == task_three.WORKSPACE_WIDTH

    task_five = import_module("simulation.tasks.task_05_cube_stacking.ik")
    assert task_five.WORKSPACE_CENTER == (0.0, 0.1175)
    assert pytest.approx(0.30) == task_five.WORKSPACE_LENGTH
    assert pytest.approx(0.105) == task_five.WORKSPACE_WIDTH

    task_six = import_module("simulation.tasks.task_06_bimanual_handover.ik")
    assert task_six.WORKSPACE_CENTER == (0.0, 0.1175)
    assert pytest.approx(0.30) == task_six.WORKSPACE_LENGTH
    assert pytest.approx(0.105) == task_six.WORKSPACE_WIDTH

    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    for task_id in (1, 2, 3, 5, 6):
        with create_task_backend(task_id, config) as backend:
            controller = create_task_controller(task_id, backend)
            controller.reset(seed=4)
            body_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "workspace_debug_rectangle",
            )
            assert backend.model.body_pos[body_id] == pytest.approx(
                (0.0, 0.1175, 0.0225)
            )
            expected_half_lengths = {
                "task_ik_marker_workspace_rectangle_back": 0.15,
                "task_ik_marker_workspace_rectangle_right": 0.0525,
                "task_ik_marker_workspace_rectangle_front": 0.15,
                "task_ik_marker_workspace_rectangle_left": 0.0525,
            }
            for name, expected in expected_half_lengths.items():
                geom_id = mujoco.mj_name2id(
                    backend.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    name,
                )
                assert backend.model.geom_size[geom_id, 1] == pytest.approx(expected)

    robot_scene = (
        Path(__file__).parents[1] / "simulation" / "backends" / "mujoco" / "assets" / "robot.xml"
    )
    root = ElementTree.parse(robot_scene).getroot()
    rectangle_body = next(
        body for body in root.iter("body") if body.attrib.get("name") == "workspace_debug_rectangle"
    )
    assert tuple(float(value) for value in rectangle_body.attrib["pos"].split()) == (
        0.0,
        0.175,
        0.0225,
    )
    segments = list(rectangle_body.findall("geom"))
    assert len(segments) == 4
    boundary_points = []
    for segment in segments:
        assert segment.attrib["name"].startswith("task_ik_marker_workspace_rectangle_")
        assert segment.attrib["group"] == "3"
        assert segment.attrib["contype"] == "0"
        assert segment.attrib["conaffinity"] == "0"
        endpoints = np.asarray(
            [float(value) for value in segment.attrib["fromto"].split()]
        ).reshape(2, 3)
        assert np.allclose(endpoints[:, 2], 0.0)
        boundary_points.extend(endpoints[:, :2])
    assert {tuple(point) for point in np.unique(np.asarray(boundary_points), axis=0)} == {
        (-0.25, -0.15),
        (-0.25, 0.15),
        (0.25, -0.15),
        (0.25, 0.15),
    }

    config = SimulationConfig(render=False)
    for task_id in range(1, 8):
        with create_task_backend(task_id, config) as backend:
            rectangle_body_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "workspace_debug_rectangle",
            )
            assert rectangle_body_id >= 0
            assert np.allclose(
                backend.data.xpos[rectangle_body_id],
                (0.0, 0.175, 0.0225),
            )
            model_segments = [
                geom_id
                for geom_id in range(backend.model.ngeom)
                if (
                    mujoco.mj_id2name(
                        backend.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        geom_id,
                    )
                    or ""
                ).startswith("task_ik_marker_workspace_rectangle_")
            ]
            assert len(model_segments) == 4
            assert all(backend.model.geom_group[geom_id] == 3 for geom_id in model_segments)
            assert all(backend.model.geom_contype[geom_id] == 0 for geom_id in model_segments)
            assert all(backend.model.geom_conaffinity[geom_id] == 0 for geom_id in model_segments)


def test_task_two_uses_symmetric_grasp_transport_and_release(monkeypatch) -> None:
    task_module = import_module("simulation.tasks.task_02_object_retrieval.ik")
    left_layout = task_module.TaskLayout(
        cube=(0.142, 0.120, 0.045),
        container=(0.155, 0.135, 0.0),
        destination=(0.038, 0.100, 0.035),
        working_arm="l",
    )
    right_layout = task_module.TaskLayout(
        cube=(-left_layout.cube[0], left_layout.cube[1], left_layout.cube[2]),
        container=(
            -left_layout.container[0],
            left_layout.container[1],
            left_layout.container[2],
        ),
        destination=(
            -left_layout.destination[0],
            left_layout.destination[1],
            left_layout.destination[2],
        ),
        working_arm="r",
    )
    layouts = {0: right_layout, 1: left_layout}
    monkeypatch.setattr(
        task_module,
        "sample_layout",
        lambda backend, seed: layouts[seed],
    )
    config = SimulationConfig(
        fps=60,
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    records = {}
    for seed, side in ((0, "r"), (1, "l")):
        with create_task_backend(2, config) as backend:
            controller = create_task_controller(2, backend)
            controller.reset(seed=seed)
            labels = [waypoint.label for waypoint in controller.waypoints]
            compact_index = labels.index("compact working gripper at rest")
            above_index = labels.index("move directly above cube with compact gripper")
            inside_index = labels.index("lower compact gripper inside container")
            open_index = labels.index("open fingers inside container before cube")
            align_index = labels.index("descend open gripper to cube")
            close_index = labels.index("close fingers around cube")
            lift_index = labels.index("lift cube vertically clear of container")
            placement_index = labels.index("move cube to requested placement target")
            release_index = labels.index("release cube on workspace")
            assert (
                compact_index
                < above_index
                < inside_index
                < open_index
                < align_index
                < close_index
                < lift_index
                < placement_index
                < release_index
            )

            above_target, _ = controller.pose_target(above_index, side)
            assert np.allclose(
                above_target,
                np.asarray(layouts[seed].cube) + np.array((0.0, 0.0, 0.035)),
            )
            inside_target, _ = controller.pose_target(inside_index, side)
            assert np.allclose(
                inside_target,
                np.asarray(layouts[seed].cube) + np.array((0.0, 0.0, 0.015)),
            )
            align_target, _ = controller.pose_target(align_index, side)
            assert np.allclose(
                align_target,
                np.asarray(layouts[seed].cube) + np.array((0.0, 0.0, 0.005)),
            )
            lift_target, _ = controller.pose_target(lift_index, side)
            placement_target, _ = controller.pose_target(placement_index, side)
            release_target, _ = controller.pose_target(release_index, side)
            assert np.allclose(lift_target[:2], np.asarray(layouts[seed].cube[:2]))
            assert lift_target[2] == pytest.approx(0.125)
            assert placement_target[2] == pytest.approx(
                task_module.WORKSPACE_DELIVERY_TCP_Z
            )
            assert release_target[2] == pytest.approx(
                task_module.WORKSPACE_DELIVERY_TCP_Z
            )

            finger_index = FINGER_ACTION_INDICES[side]
            other_side = "l" if side == "r" else "r"
            other_finger_index = FINGER_ACTION_INDICES[other_side]
            arm_indices = ARM_ACTION_INDICES[side]
            actions = np.asarray(controller._actions)
            assert actions[compact_index, finger_index] == pytest.approx(0.20)
            assert actions[compact_index, other_finger_index] == pytest.approx(OPEN_GRIPPER)
            assert actions[above_index, finger_index] == pytest.approx(0.20)
            assert actions[inside_index, finger_index] == pytest.approx(0.20)
            assert actions[open_index, finger_index] == pytest.approx(0.40)
            assert actions[align_index, finger_index] == pytest.approx(0.40)
            assert actions[close_index, finger_index] == pytest.approx(
                CUBE_CLOSED_GRIPPER
            )
            assert np.allclose(
                actions[inside_index, arm_indices],
                actions[open_index, arm_indices],
            )
            assert not np.allclose(
                actions[align_index, arm_indices],
                actions[close_index, arm_indices],
            )
            close_target, _ = controller.pose_target(close_index, side)
            assert np.allclose(close_target, align_target)
            for waypoint_index in (
                close_index,
                lift_index,
                placement_index,
            ):
                assert actions[waypoint_index, finger_index] == pytest.approx(
                    CUBE_CLOSED_GRIPPER
                )
            assert actions[release_index, finger_index] == pytest.approx(OPEN_GRIPPER)

            source_container_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "source_container",
            )
            target_frame_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "target_ik_frame",
            )
            assert backend.model.body_mocapid[source_container_id] >= 0
            assert backend.model.body_mocapid[target_frame_id] >= 0
            assert backend.data.xpos[target_frame_id, 2] == pytest.approx(
                task_module.WORKSPACE_DELIVERY_TCP_Z
            )

            controller.action(0.20)
            assert not backend.body_is_attached("cube")
            backend.data.qpos[backend.qpos_ids] = actions[close_index]
            backend.data.ctrl[backend.actuator_ids] = actions[close_index]
            mujoco.mj_forward(backend.model, backend.data)
            controller.action(0.46)
            assert backend.body_is_attached("cube")
            controller.action(0.80)
            assert not backend.body_is_attached("cube")

            records[side] = (controller.waypoints, actions)

    right_waypoints, right_actions = records["r"]
    left_waypoints, left_actions = records["l"]
    assert [waypoint.label for waypoint in right_waypoints] == [
        waypoint.label for waypoint in left_waypoints
    ]
    assert np.allclose(
        right_actions[-1, ARM_ACTION_INDICES["r"]],
        -left_actions[-1, ARM_ACTION_INDICES["l"]],
        atol=1e-12,
    )
    assert np.allclose(
        right_actions[:, FINGER_ACTION_INDICES["r"]],
        left_actions[:, FINGER_ACTION_INDICES["l"]],
        atol=1e-12,
    )


def test_task_three_randomizes_and_sorts_cubes_sequentially(monkeypatch) -> None:
    task_module = import_module("simulation.tasks.task_03_color_conditioned_sorting.ik")
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )

    with create_task_backend(3, config) as backend:
        layouts = [task_module.sample_layout(backend, seed) for seed in range(20)]
        assert {layout.first_color for layout in layouts} == {"blue", "green"}
        assert any(layout.blue_arm == layout.green_arm for layout in layouts)
        assert any(layout.blue_arm != layout.green_arm for layout in layouts)
        assert len({layout.blue_cube[:2] for layout in layouts}) > 2
        assert len({layout.green_cube[:2] for layout in layouts}) > 2
        assert len({layout.blue_container[:2] for layout in layouts}) > 2
        assert len({layout.green_container[:2] for layout in layouts}) > 2
        assert task_module.sample_layout(backend, 7) == task_module.sample_layout(
            backend,
            7,
        )
        for layout in layouts:
            for cube in (layout.blue_cube, layout.green_cube):
                assert task_module.CUBE_X_RANGE[0] <= cube[0] <= (
                    task_module.CUBE_X_RANGE[1]
                )
                assert task_module.CUBE_Y_RANGE[0] <= cube[1] <= (
                    task_module.CUBE_Y_RANGE[1]
                )
            for container in (layout.blue_container, layout.green_container):
                assert task_module.CONTAINER_X_RANGE[0] <= container[0] <= (
                    task_module.CONTAINER_X_RANGE[1]
                )
                assert task_module.CONTAINER_Y_RANGE[0] <= container[1] <= (
                    task_module.CONTAINER_Y_RANGE[1]
                )
            objects = (
                ("cube", layout.blue_cube, task_module.CUBE_HALF_WIDTH),
                ("cube", layout.green_cube, task_module.CUBE_HALF_WIDTH),
                (
                    "container",
                    layout.blue_container,
                    task_module.CONTAINER_OUTER_HALF_WIDTH,
                ),
                (
                    "container",
                    layout.green_container,
                    task_module.CONTAINER_OUTER_HALF_WIDTH,
                ),
            )
            for first_index, (first_kind, first, first_half_width) in enumerate(objects):
                for second_kind, second, second_half_width in objects[first_index + 1 :]:
                    clearance = (
                        task_module.MINIMUM_EDGE_CLEARANCE
                        if first_kind != second_kind
                        else 0.0
                    )
                    assert task_module._footprints_are_separate(
                        first,
                        first_half_width,
                        second,
                        second_half_width,
                        edge_clearance=clearance,
                    )

        assert not np.allclose(
            layouts[0].blue_cube,
            (-layouts[1].blue_cube[0], layouts[1].blue_cube[1], layouts[1].blue_cube[2]),
        )

    layout = task_module.TaskLayout(
        blue_cube=(0.115, 0.124, 0.035),
        green_cube=(0.155, 0.127, 0.035),
        blue_container=(-0.010, 0.090, 0.0),
        green_container=(-0.012, 0.215, 0.0),
        first_color="green",
        blue_arm="l",
        green_arm="l",
    )
    monkeypatch.setattr(task_module, "sample_layout", lambda backend, seed: layout)

    with create_task_backend(3, config) as backend:
        controller = create_task_controller(3, backend)
        controller.reset(seed=1)
        labels = [waypoint.label for waypoint in controller.waypoints]
        actions = np.asarray(controller._actions)
        assert labels.index("close gripper around green cube") < labels.index(
            "position compact gripper above blue cube"
        )
        assert labels.index("release green cube") < labels.index("close gripper around blue cube")

        for color, cube in (("green", layout.green_cube), ("blue", layout.blue_cube)):
            pregrasp_index = labels.index(f"position compact gripper above {color} cube")
            open_index = labels.index(f"open gripper before reaching {color} cube")
            grasp_index = labels.index(f"descend open gripper to {color} cube")
            close_index = labels.index(f"close gripper around {color} cube")
            release_index = labels.index(f"release {color} cube")
            assert pregrasp_index < open_index < grasp_index < close_index
            pregrasp_target, _ = controller.pose_target(pregrasp_index, "l")
            grasp_target, _ = controller.pose_target(grasp_index, "l")
            assert np.allclose(
                pregrasp_target,
                np.asarray(cube) + np.array((0.0, 0.0, 0.035)),
            )
            assert np.allclose(
                grasp_target,
                np.asarray(cube) + np.array((0.0, 0.0, 0.010)),
            )
            finger_index = FINGER_ACTION_INDICES["l"]
            assert actions[pregrasp_index, finger_index] == pytest.approx(0.20)
            assert actions[open_index, finger_index] == pytest.approx(
                task_module.CUBE_GRASP_OPEN_GRIPPER
            )
            assert actions[grasp_index, finger_index] == pytest.approx(
                task_module.CUBE_GRASP_OPEN_GRIPPER
            )
            assert actions[close_index, finger_index] == pytest.approx(
                CUBE_CLOSED_GRIPPER
            )
            close_target, _ = controller.pose_target(close_index, "l")
            assert np.allclose(close_target, grasp_target)

            backend.data.qpos[backend.qpos_ids] = actions[close_index]
            backend.data.ctrl[backend.actuator_ids] = actions[close_index]
            mujoco.mj_forward(backend.model, backend.data)
            close_progress = controller.waypoints[close_index].time / controller.waypoints[-1].time
            controller.action(close_progress)
            assert backend.body_is_attached(f"{color}_cube")
            release_progress = (
                controller.waypoints[release_index].time / controller.waypoints[-1].time
            )
            controller.action(release_progress)
            assert not backend.body_is_attached(f"{color}_cube")


def test_task_three_resamples_only_after_actual_trajectory_failure(monkeypatch) -> None:
    task_module = import_module("simulation.tasks.task_03_color_conditioned_sorting.ik")
    sampled_seeds: list[int] = []
    reachable = task_module.TaskLayout(
        blue_cube=(0.115, 0.124, 0.035),
        green_cube=(0.155, 0.127, 0.035),
        blue_container=(-0.010, 0.090, 0.0),
        green_container=(-0.012, 0.215, 0.0),
        first_color="green",
        blue_arm="l",
        green_arm="l",
    )

    def sampled_layout(backend, seed):
        sampled_seeds.append(seed)
        if len(sampled_seeds) == 1:
            return task_module.TaskLayout(
                blue_cube=(10.0, 10.0, task_module.CUBE_CENTER_Z),
                green_cube=(11.0, 10.0, task_module.CUBE_CENTER_Z),
                blue_container=(10.0, 11.0, 0.0),
                green_container=(11.0, 11.0, 0.0),
                first_color="blue",
                blue_arm="l",
                green_arm="l",
            )
        return reachable

    monkeypatch.setattr(task_module, "sample_layout", sampled_layout)
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(3, config) as backend:
        controller = create_task_controller(3, backend)
        controller.reset(seed=1)
        details = controller.evaluate().details

    assert sampled_seeds == [1, task_module._layout_attempt_seed(1, 1)]
    assert details["sampled_blue_cube_position"] == reachable.blue_cube
    assert details["layout_attempts"] == 2


def test_task_four_pulls_reachable_drawers_forward_with_closed_gripper() -> None:
    task_module = import_module("simulation.tasks.task_04_drawer_opening.ik")
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(4, config) as backend:
        assert (
            mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "drawer_stack",
            )
            >= 0
        )
        for number in range(1, 10):
            handle_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"drawer_{number}_handle",
            )
            assert backend.data.geom_xpos[handle_id, 1] == pytest.approx(-0.035)

        drawer_eight_handle = task_module._handle_position(backend, 8)
        drawer_eight_target, drawer_eight_side = task_module._handle_ik_target(backend, 8)
        assert drawer_eight_side == task_module._closest_arm(backend, drawer_eight_handle)
        assert drawer_eight_target == pytest.approx(drawer_eight_handle)

        controller = create_task_controller(4, backend)
        for number in range(1, 10):
            controller.reset(seed=number - 1)
            labels = [waypoint.label for waypoint in controller.waypoints]
            align_index = labels.index("match vertical hand IK to handle target")
            close_index = labels.index("close scissor grasp around handle")
            attach_index = labels.index("attach closed gripper to handle")
            pull_index = labels.index("pull attached drawer open")
            release_index = labels.index("open fingers and detach handle")
            retreat_index = labels.index("retreat released hand backward from drawer")
            retreat_hold_index = labels.index("hold backward retreat clear of drawer")
            assert (
                align_index
                < close_index
                < attach_index
                < pull_index
                < release_index
                < retreat_index
                < retreat_hold_index
            )
            assert all(not waypoint.joint_positions for waypoint in controller.waypoints)

            align_waypoint = controller.waypoints[align_index]
            side = "l" if align_waypoint.left is not None else "r"
            preferred_orientation = (
                align_waypoint.left_preferred_quaternion
                if side == "l"
                else align_waypoint.right_preferred_quaternion
            )
            initial_details = controller.evaluate().details
            rack_yaw = np.deg2rad(initial_details["rack_yaw_degrees"])
            assert preferred_orientation == pytest.approx(
                task_module._rotate_quaternion_about_z(
                    task_module.VERTICAL_SCISSOR_QUATERNIONS[number],
                    rack_yaw,
                )
            )
            handle_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"drawer_{number}_handle",
            )
            handle_target, handle_orientation = controller.pose_target(
                align_index,
                side,
            )
            close_target, _ = controller.pose_target(close_index, side)
            pull_target, _ = controller.pose_target(pull_index, side)
            retreat_target, _ = controller.pose_target(retreat_index, side)
            finger_index = FINGER_ACTION_INDICES[side]
            assert controller._actions[align_index][finger_index] == pytest.approx(
                task_module.HANDLE_APPROACH_GRIPPER
            )
            assert controller._actions[close_index][finger_index] == pytest.approx(
                CLOSED_GRIPPER
            )
            assert controller._actions[attach_index][finger_index] == pytest.approx(
                CLOSED_GRIPPER
            )
            assert controller._actions[pull_index][finger_index] == pytest.approx(
                CLOSED_GRIPPER
            )
            hand_index = int(ARM_ACTION_INDICES[side][-1])
            assert controller._actions[attach_index][hand_index] == pytest.approx(
                controller._actions[close_index][hand_index]
            )
            assert controller._actions[pull_index][hand_index] == pytest.approx(
                controller._actions[close_index][hand_index]
            )
            assert close_target == pytest.approx(handle_target)
            pull_direction = np.asarray(
                (-np.sin(rack_yaw), np.cos(rack_yaw), 0.0)
            )
            expected_pull_distance = task_module.TCP_PULL_DISTANCES.get(
                number,
                task_module.DRAWER_OPEN_DISTANCE,
            )
            assert pull_target - handle_target == pytest.approx(
                expected_pull_distance * pull_direction
            )
            assert retreat_target - pull_target == pytest.approx(
                task_module.POST_RELEASE_RETREAT_DISTANCE * pull_direction
            )
            assert controller._actions[retreat_index][finger_index] == pytest.approx(
                task_module.HANDLE_RELEASE_GRIPPER
            )
            assert controller._actions[release_index][finger_index] == pytest.approx(
                task_module.HANDLE_RELEASE_GRIPPER
            )
            hold_target, _ = controller.pose_target(retreat_hold_index, side)
            assert hold_target == pytest.approx(retreat_target)
            slide_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"drawer_{number}_slide",
            )
            assert backend.model.jnt_axis[slide_id] == pytest.approx((0.0, 1.0, 0.0))
            assert backend.model.jnt_range[slide_id] == pytest.approx((-0.01, 0.08))
            slide_qpos_id = int(backend.model.jnt_qposadr[slide_id])
            assert backend.data.qpos[slide_qpos_id] == pytest.approx(
                initial_details["initial_openings"][number]
            )
            backend.data.qpos[backend.qpos_ids] = controller._actions[close_index]
            mujoco.mj_forward(backend.model, backend.data)
            assert hand_tcp_position(backend, side) == pytest.approx(
                handle_target,
                abs=0.012,
            )
            assert np.allclose(handle_target, backend.data.geom_xpos[handle_id])
            assert np.linalg.norm(handle_orientation) == pytest.approx(1.0)
            target_body_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_BODY,
                task_module.TARGET_FRAME_BODY,
            )
            assert backend.data.xpos[target_body_id] == pytest.approx(handle_target)
            assert abs(
                float(backend.data.xquat[target_body_id] @ handle_orientation)
            ) == pytest.approx(1.0)
            assert backend.model.geom_contype[handle_id] == 0
            for other_number in set(range(1, 10)) - {number}:
                other_handle_id = mujoco.mj_name2id(
                    backend.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"drawer_{other_number}_handle",
                )
                assert backend.model.geom_contype[other_handle_id] == 0

            frames = round(controller.waypoints[-1].time * config.fps)
            for frame in range(frames + 1):
                backend.send_action(controller.action(frame / frames))
                backend.step()
            result = controller.evaluate()
            assert result.success
            assert result.details["requested_drawer"] == number
            assert set(result.details["openings"]) == set(range(1, 10))
            assert (
                result.details["openings"][number]
                - result.details["initial_openings"][number]
                >= 0.035
            )
            assert result.subgoals["handle_detached_after_pull"]
            assert not np.any(backend.data.eq_active)


def test_task_four_randomizes_rack_pose_and_drawer_openings() -> None:
    task_module = import_module("simulation.tasks.task_04_drawer_opening.ik")
    layouts = [task_module.sample_layout(seed) for seed in range(40)]

    assert task_module.sample_layout(7) == task_module.sample_layout(7)
    for layout in layouts:
        assert all(
            task_module.RACK_TRANSLATION_RANGE[0]
            <= value
            <= task_module.RACK_TRANSLATION_RANGE[1]
            for value in layout.rack_translation
        )
        assert task_module.RACK_YAW_RANGE[0] <= layout.rack_yaw <= (
            task_module.RACK_YAW_RANGE[1]
        )
        assert len(layout.drawer_openings) == task_module.DRAWER_COUNT
        assert all(
            task_module.DRAWER_INITIAL_OPENING_RANGE[0]
            <= opening
            <= task_module.DRAWER_INITIAL_OPENING_RANGE[1]
            for opening in layout.drawer_openings
        )

    assert np.ptp([layout.rack_translation[0] for layout in layouts]) > 0.03
    assert np.ptp([layout.rack_translation[1] for layout in layouts]) > 0.03
    assert np.rad2deg(np.ptp([layout.rack_yaw for layout in layouts])) > 8.0
    assert np.ptp([opening for layout in layouts for opening in layout.drawer_openings]) > 0.018


def test_task_seven_randomizes_rack_drawers_and_cubes_without_mirroring() -> None:
    task_module = import_module(
        "simulation.tasks.task_07_warehouse_stocking.ik"
    )
    layouts = [task_module.sample_layout(seed) for seed in range(40)]

    assert task_module.sample_layout(7) == task_module.sample_layout(7)
    for layout in layouts:
        assert all(
            task_module.RACK_TRANSLATION_RANGE[0]
            <= value
            <= task_module.RACK_TRANSLATION_RANGE[1]
            for value in layout.rack_translation
        )
        assert task_module.RACK_YAW_RANGE[0] <= layout.rack_yaw <= (
            task_module.RACK_YAW_RANGE[1]
        )
        assert all(
            task_module.DRAWER_INITIAL_OPENING_RANGE[0]
            <= opening
            <= task_module.DRAWER_INITIAL_OPENING_RANGE[1]
            for opening in layout.drawer_openings
        )
        for cube in (layout.blue_cube, layout.green_cube):
            assert task_module.CUBE_X_RANGE[0] <= cube[0] <= (
                task_module.CUBE_X_RANGE[1]
            )
            assert task_module.CUBE_Y_RANGE[0] <= cube[1] <= (
                task_module.CUBE_Y_RANGE[1]
            )
            assert cube[2] == pytest.approx(task_module.CUBE_CENTER_Z)
        assert task_module._cube_footprints_are_separate(
            layout.blue_cube,
            layout.green_cube,
        )

    # Independent seeded samples exercise both sides; there is no paired-seed
    # reflection rule tying one episode to the next.
    assert {np.sign(layout.blue_cube[0]) for layout in layouts} == {-1.0, 1.0}
    assert {np.sign(layout.green_cube[0]) for layout in layouts} == {-1.0, 1.0}
    assert np.ptp([layout.blue_cube[0] for layout in layouts]) > 0.20
    assert np.ptp([layout.green_cube[0] for layout in layouts]) > 0.20


def test_task_seven_drawer_collisions_match_exported_wall_thicknesses() -> None:
    scene = (
        Path(__file__).parents[1]
        / "simulation"
        / "tasks"
        / "task_07_warehouse_stocking"
        / "scene.xml"
    )
    geometries = {
        geom.attrib["name"]: geom
        for geom in ElementTree.parse(scene).getroot().iter("geom")
        if "name" in geom.attrib
    }

    for number in range(1, 10):
        bottom_size = tuple(
            float(value)
            for value in geometries[f"drawer_{number}_bottom"].attrib["size"].split()
        )
        back_size = tuple(
            float(value)
            for value in geometries[f"drawer_{number}_back"].attrib["size"].split()
        )
        side_positive_size = tuple(
            float(value)
            for value in geometries[f"drawer_{number}_side_xp"].attrib["size"].split()
        )
        side_negative_size = tuple(
            float(value)
            for value in geometries[f"drawer_{number}_side_xn"].attrib["size"].split()
        )
        handle_size = tuple(
            float(value)
            for value in geometries[f"drawer_{number}_handle"].attrib["size"].split()
        )

        assert 2.0 * bottom_size[2] == pytest.approx(0.004)
        assert 2.0 * back_size[1] == pytest.approx(0.004)
        assert 2.0 * side_positive_size[0] == pytest.approx(0.004)
        assert 2.0 * side_negative_size[0] == pytest.approx(0.004)
        assert 2.0 * handle_size[1] == pytest.approx(0.002)


def test_task_seven_executes_open_stock_close_sequence_for_bottom_center() -> None:
    task_module = import_module(
        "simulation.tasks.task_07_warehouse_stocking.ik"
    )
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(7, config) as backend:
        controller = create_task_controller(7, backend)
        controller.reset(seed=7)
        labels = [waypoint.label for waypoint in controller.waypoints]
        ordered_labels = (
            "match vertical hand IK to requested drawer handle",
            "pull requested drawer open",
            "release opened drawer handle",
            "reset both arms before cube pickup",
            "align open gripper above requested cube",
            "close fingers around requested cube",
            "raise cube in front of open requested drawer",
            "position requested cube above open drawer",
            "drop requested cube into open drawer",
            "reset both arms before drawer closing",
            "match vertical hand IK to open drawer handle",
            "push stocked drawer closed",
            "release closed drawer handle",
            "both arms return to rest",
        )
        indices = [labels.index(label) for label in ordered_labels]
        assert indices == sorted(indices)

        initial = controller.evaluate()
        assert initial.details["requested_drawer"] == 8
        assert initial.details["requested_color"] == "green"

        release_index = labels.index("drop requested cube into open drawer")
        cube_side = initial.details["cube_arm"]
        release_tcp, _ = controller.pose_target(release_index, cube_side)
        front_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "drawer_8_front",
        )
        rim_height = (
            backend.data.geom_xpos[front_id, 2]
            + backend.model.geom_size[front_id, 2]
        )
        released_cube_center_z = release_tcp[2] - task_module.GRASP_TARGET_Z_OFFSET
        assert released_cube_center_z - rim_height == pytest.approx(
            task_module.DRAWER_DROP_CENTER_HEIGHT_ABOVE_RIM
        )

        frames = round(controller.waypoints[-1].time * config.fps)
        for frame in range(frames + 1):
            backend.send_action(controller.action(frame / frames))
            backend.step()
        result = controller.evaluate()

        assert result.success
        assert all(result.subgoals.values())
        assert result.details["requested_drawer"] == 8
        assert result.details["requested_color"] == "green"
        assert not backend.body_is_attached("green_cube")
        assert not np.any(backend.data.eq_active)


def test_all_implemented_tasks_finish_at_the_exported_rest_pose(monkeypatch) -> None:
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    for task_id in range(1, 8):
        if task_id == 1:
            task_module = import_module(
                "simulation.tasks.task_01_basic_object_placement.ik"
            )
            monkeypatch.setattr(
                task_module,
                "sample_layout",
                lambda backend, seed, module=task_module: (
                    _reachable_task_one_layout(module)
                ),
            )
        with create_task_backend(task_id, config) as backend:
            controller = create_task_controller(task_id, backend)
            try:
                controller.reset(seed=0)
            except IKConvergenceError:
                assert task_id == 4
                continue
            assert controller.waypoints[-1].return_to_rest
            assert controller.waypoints[-1].label == "both arms return to rest"
            expected_rest = backend.rest_action()
            if task_id == 4:
                task_module = import_module("simulation.tasks.task_04_drawer_opening.ik")
                expected_rest = task_module._task_four_rest_action(backend)
            elif task_id == 7:
                task_module = import_module(
                    "simulation.tasks.task_07_warehouse_stocking.ik"
                )
                expected_rest = task_module._task_seven_rest_action(backend)
            assert np.allclose(controller._actions[-1], expected_rest)


def test_task_five_uses_vertical_lift_and_top_down_stack_approach() -> None:
    task_module = import_module("simulation.tasks.task_05_cube_stacking.ik")
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(5, config) as backend:
        controller = create_task_controller(5, backend)
        controller.reset(seed=0)
        details = controller.evaluate().details
        green_cube = tuple(details["sampled_green_position"])
        blue_cube = tuple(details["sampled_blue_position"])
        side = details["working_arm"]
        labels = [waypoint.label for waypoint in controller.waypoints]
        actions = np.asarray(controller._actions)
        grasp_index = labels.index("grasp IK above green cube")
        close_index = labels.index("close gripper around green cube")
        lift_index = labels.index("lift green cube vertically by 40 mm")
        approach_index = labels.index("move high above blue cube")
        place_index = labels.index("descend vertically to stack target")
        release_index = labels.index("release green cube")
        retreat_index = labels.index("retreat IK above stacked cubes")
        rest_index = labels.index("both arms return to rest")
        assert (
            grasp_index
            < close_index
            < lift_index
            < approach_index
            < place_index
            < release_index
            < retreat_index
            < rest_index
        )

        grasp_target, _ = controller.pose_target(grasp_index, side)
        close_target, _ = controller.pose_target(close_index, side)
        lift_target, _ = controller.pose_target(lift_index, side)
        approach_target, _ = controller.pose_target(approach_index, side)
        place_target, _ = controller.pose_target(place_index, side)
        release_target, _ = controller.pose_target(release_index, side)
        retreat_target, _ = controller.pose_target(retreat_index, side)
        cube_position = np.asarray(green_cube)
        assert np.allclose(
            grasp_target,
            cube_position + np.array((0.0, 0.0, 0.010)),
        )
        assert np.allclose(close_target, grasp_target)
        assert np.allclose(
            lift_target,
            grasp_target + np.array((0.0, 0.0, task_module.POST_GRASP_LIFT_DISTANCE)),
        )
        assert np.allclose(
            approach_target,
            np.array(
                (
                    blue_cube[0],
                    blue_cube[1],
                    task_module.STACK_APPROACH_TCP_Z,
                )
            ),
        )
        assert np.allclose(
            place_target,
            np.asarray(blue_cube)
            + np.array(
                (
                    0.0,
                    0.0,
                    task_module.CUBE_WIDTH
                    + task_module.GRASP_TARGET_Z_OFFSET
                    + task_module.STACK_RELEASE_CLEARANCE,
                )
            ),
        )
        assert np.allclose(release_target, place_target)
        assert np.allclose(
            retreat_target,
            np.array(
                (
                    blue_cube[0],
                    blue_cube[1],
                    task_module.STACK_RETREAT_TCP_Z,
                )
            ),
        )
        finger_index = FINGER_ACTION_INDICES[side]
        assert actions[grasp_index, finger_index] == pytest.approx(WIDE_OPEN_GRIPPER)
        assert actions[close_index, finger_index] == pytest.approx(CUBE_CLOSED_GRIPPER)
        assert actions[lift_index, finger_index] == pytest.approx(CUBE_CLOSED_GRIPPER)
        assert actions[approach_index, finger_index] == pytest.approx(CUBE_CLOSED_GRIPPER)
        assert actions[place_index, finger_index] == pytest.approx(CUBE_CLOSED_GRIPPER)
        assert actions[release_index, finger_index] == pytest.approx(OPEN_GRIPPER)
        assert actions[retreat_index, finger_index] == pytest.approx(OPEN_GRIPPER)
        cartesian_waypoints = [
            waypoint
            for waypoint in controller.waypoints
            if (waypoint.left if side == "l" else waypoint.right) is not None
        ]
        assert len(cartesian_waypoints) == 7
        unique_spatial_targets = {
            tuple(waypoint.left if side == "l" else waypoint.right)
            for waypoint in cartesian_waypoints
        }
        assert len(unique_spatial_targets) == 5

        backend.data.qpos[backend.qpos_ids] = actions[close_index]
        backend.data.ctrl[backend.actuator_ids] = actions[close_index]
        mujoco.mj_forward(backend.model, backend.data)
        controller.action(
            (task_module.CUBE_LOCK_TIME + 0.01) / controller.waypoints[-1].time
        )
        assert backend.body_is_attached("green_cube")
        live_blue_position = (0.025, 0.120, task_module.CUBE_CENTER_Z)
        task_module._set_free_body_position(
            backend,
            "blue_cube",
            live_blue_position,
        )
        mujoco.mj_forward(backend.model, backend.data)
        controller.action(
            controller.waypoints[release_index].time / controller.waypoints[-1].time
        )
        assert not backend.body_is_attached("green_cube")
        assert task_module.body_position(backend, "green_cube") == pytest.approx(
            (
                live_blue_position[0],
                live_blue_position[1],
                live_blue_position[2]
                + task_module.CUBE_WIDTH
                + task_module.STACK_RELEASE_CLEARANCE,
            )
        )


@pytest.mark.parametrize("seed", (0, 1))
def test_task_six_uses_task_two_container_grasp_sequence_for_closest_arm(
    seed: int,
) -> None:
    task_module = import_module("simulation.tasks.task_06_bimanual_handover.ik")
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(6, config) as backend:
        controller = create_task_controller(6, backend)
        controller.reset(seed=seed)
        details = controller.evaluate().details
        pickup_side = details["working_arm"]
        labels = [waypoint.label for waypoint in controller.waypoints]
        actions = np.asarray(controller._actions)
        pregrasp_index = labels.index(
            "move pickup arm directly above cube with compact gripper"
        )
        entry_index = labels.index(
            "lower compact pickup gripper inside first container"
        )
        open_index = labels.index(
            "open pickup fingers inside first container before cube"
        )
        grasp_index = labels.index("descend open pickup gripper to cube")
        close_index = labels.index("close pickup gripper around cube")
        assert pregrasp_index < entry_index < open_index < grasp_index < close_index

        cube_position = np.asarray(details["sampled_cube_position"])
        pregrasp_target, _ = controller.pose_target(pregrasp_index, pickup_side)
        entry_target, _ = controller.pose_target(entry_index, pickup_side)
        grasp_target, _ = controller.pose_target(grasp_index, pickup_side)
        assert np.allclose(
            pregrasp_target,
            cube_position + np.array((0.0, 0.0, 0.035)),
        )
        assert np.allclose(
            entry_target,
            cube_position + np.array((0.0, 0.0, 0.015)),
        )
        assert np.allclose(
            grasp_target,
            cube_position + np.array((0.0, 0.0, 0.005)),
        )
        finger_index = FINGER_ACTION_INDICES[pickup_side]
        assert actions[pregrasp_index, finger_index] == pytest.approx(0.20)
        assert actions[entry_index, finger_index] == pytest.approx(0.20)
        assert actions[open_index, finger_index] == pytest.approx(
            task_module.CUBE_GRASP_OPEN_GRIPPER
        )
        assert actions[grasp_index, finger_index] == pytest.approx(
            task_module.CUBE_GRASP_OPEN_GRIPPER
        )
        assert actions[close_index, finger_index] == pytest.approx(CUBE_CLOSED_GRIPPER)


def test_tasks_five_and_six_use_reproducible_seeded_object_layouts() -> None:
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )

    task_five = import_module("simulation.tasks.task_05_cube_stacking.ik")
    with create_task_backend(5, config) as backend:
        layouts = [task_five.sample_layout(backend, seed) for seed in range(40)]
        assert task_five.sample_layout(backend, 2) == task_five.sample_layout(
            backend,
            2,
        )
        assert len({layout.green_cube for layout in layouts}) == 40
        assert len({layout.blue_cube for layout in layouts}) == 40
        assert {layout.working_arm for layout in layouts} == {"l", "r"}
        for layout in layouts:
            for cube in (layout.green_cube, layout.blue_cube):
                assert task_five.CUBE_X_RANGE[0] <= cube[0] <= (
                    task_five.CUBE_X_RANGE[1]
                )
                assert task_five.CUBE_Y_RANGE[0] <= cube[1] <= (
                    task_five.CUBE_Y_RANGE[1]
                )
            assert task_five._footprints_are_separate(
                layout.green_cube,
                layout.blue_cube,
            )
        assert not np.allclose(
            layouts[0].green_cube,
            (
                -layouts[1].green_cube[0],
                layouts[1].green_cube[1],
                layouts[1].green_cube[2],
            ),
        )
        assert np.ptp([layout.green_cube[0] for layout in layouts]) > 0.20
        assert np.ptp([layout.blue_cube[0] for layout in layouts]) > 0.20
        controller = create_task_controller(5, backend)
        controller.reset(seed=2)
        accepted = controller.evaluate().details
        for body_name, expected_position in (
            ("green_cube", accepted["sampled_green_position"]),
            ("blue_cube", accepted["sampled_blue_position"]),
        ):
            body_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            assert backend.data.xpos[body_id] == pytest.approx(expected_position)

    task_six = import_module("simulation.tasks.task_06_bimanual_handover.ik")
    with create_task_backend(6, config) as backend:
        layouts = [task_six.sample_layout(backend, seed) for seed in range(40)]
        assert task_six.sample_layout(backend, 2) == task_six.sample_layout(
            backend,
            2,
        )
        assert len({layout.cube for layout in layouts}) == 40
        assert len({layout.first_container for layout in layouts}) == 40
        assert len({layout.second_container for layout in layouts}) == 40
        assert {layout.working_arm for layout in layouts} == {"l", "r"}
        for layout in layouts:
            assert layout.working_arm == task_six._closest_resting_arm(
                backend,
                layout.cube,
            )
            for container in (layout.first_container, layout.second_container):
                assert task_six.CONTAINER_X_RANGE[0] <= container[0] <= (
                    task_six.CONTAINER_X_RANGE[1]
                )
                assert task_six.CONTAINER_Y_RANGE[0] <= container[1] <= (
                    task_six.CONTAINER_Y_RANGE[1]
                )
            assert abs(layout.cube[0] - layout.first_container[0]) <= (
                task_six.MAXIMUM_CUBE_OFFSET
            )
            assert abs(layout.cube[1] - layout.first_container[1]) <= (
                task_six.MAXIMUM_CUBE_OFFSET
            )
            required_separation = (
                2.0 * task_six.CONTAINER_OUTER_HALF_WIDTH
                + task_six.MINIMUM_EDGE_CLEARANCE
            )
            assert (
                abs(layout.first_container[0] - layout.second_container[0])
                >= required_separation
                or abs(layout.first_container[1] - layout.second_container[1])
                >= required_separation
            )
        assert np.ptp([layout.first_container[0] for layout in layouts]) > 0.18
        assert np.ptp([layout.second_container[0] for layout in layouts]) > 0.18
        assert not np.allclose(
            layouts[0].first_container,
            (
                -layouts[1].first_container[0],
                layouts[1].first_container[1],
                layouts[1].first_container[2],
            ),
        )
        controller = create_task_controller(6, backend)
        controller.reset(seed=2)
        accepted = controller.evaluate().details
        assert accepted["working_arm"] in {"l", "r"}
        assert accepted["receiving_arm"] != accepted["working_arm"]
        for body_name, expected_position in (
            ("cube", accepted["sampled_cube_position"]),
            ("first_container", accepted["sampled_first_container_position"]),
            ("second_container", accepted["sampled_second_container_position"]),
        ):
            body_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            assert backend.data.xpos[body_id] == pytest.approx(expected_position)


@pytest.mark.parametrize("seed", (0, 1))
def test_task_six_receiver_opens_before_matching_handover_cube_target(
    seed: int,
) -> None:
    task_module = import_module("simulation.tasks.task_06_bimanual_handover.ik")
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with create_task_backend(6, config) as backend:
        controller = create_task_controller(6, backend)
        controller.reset(seed=seed)
        details = controller.evaluate().details
        pickup_side = details["working_arm"]
        receiver_side = details["receiving_arm"]
        labels = [waypoint.label for waypoint in controller.waypoints]
        pickup_handover_index = labels.index(
            "pickup arm alone carries cube to handover"
        )
        compact_index = labels.index("compact receiving gripper at rest")
        pregrasp_index = labels.index(
            "move receiving arm to 50 mm handover pregrasp"
        )
        aperture_index = labels.index(
            "open receiving fingers at handover pregrasp"
        )
        align_index = labels.index(
            "move receiving hand horizontally onto held cube"
        )
        close_index = labels.index("close receiving gripper and receive cube")
        transfer_index = labels.index("open pickup gripper after transfer")
        assert (
            pickup_handover_index
            < compact_index
            < pregrasp_index
            < aperture_index
            < align_index
            < close_index
            < transfer_index
        )
        pickup_target, _ = controller.pose_target(
            pickup_handover_index,
            pickup_side,
        )
        pregrasp_target, _ = controller.pose_target(pregrasp_index, receiver_side)
        aperture_target, _ = controller.pose_target(aperture_index, receiver_side)
        target, _ = controller.pose_target(align_index, receiver_side)
        assert np.allclose(pickup_target, task_module.HANDOVER_TCP)
        assert np.allclose(aperture_target, pregrasp_target)
        receiver_side_sign = 1.0 if receiver_side == "l" else -1.0
        assert pregrasp_target[0] == pytest.approx(
            target[0]
            + receiver_side_sign * task_module.HANDOVER_PREGRASP_OFFSET
        )
        assert pregrasp_target[1:] == pytest.approx(target[1:])
        assert np.allclose(target, task_module.RECEIVER_HANDOVER_TCP)
        assert pickup_target[2] - target[2] == pytest.approx(
            task_module.RECEIVER_HANDOVER_LOWERING
        )
        finger_index = FINGER_ACTION_INDICES[receiver_side]
        actions = np.asarray(controller._actions)
        assert actions[compact_index, finger_index] == pytest.approx(
            task_module.COMPACT_ENTRY_GRIPPER
        )
        assert actions[pregrasp_index, finger_index] == pytest.approx(
            task_module.COMPACT_ENTRY_GRIPPER
        )
        assert actions[aperture_index, finger_index] == pytest.approx(
            task_module.CUBE_GRASP_OPEN_GRIPPER
        )
        assert actions[align_index, finger_index] == pytest.approx(
            task_module.CUBE_GRASP_OPEN_GRIPPER
        )
        assert actions[close_index, finger_index] == pytest.approx(CUBE_CLOSED_GRIPPER)
        assert actions[transfer_index, finger_index] == pytest.approx(CUBE_CLOSED_GRIPPER)


def test_all_task_cubes_are_30_mm() -> None:
    tasks_root = Path(__file__).parents[1] / "simulation" / "tasks"
    cube_geometries = []
    for scene in tasks_root.glob("task_*/scene.xml"):
        root = ElementTree.parse(scene).getroot()
        cube_geometries.extend(
            geom for geom in root.iter("geom") if "cube" in geom.attrib.get("name", "")
        )
    assert cube_geometries
    assert all(geom.attrib["size"] == "0.015 0.015 0.015" for geom in cube_geometries)


def test_all_task_cubes_have_enabled_collision_geometries() -> None:
    tasks_root = Path(__file__).parents[1] / "simulation" / "tasks"
    collision_geometries = []
    for scene in tasks_root.glob("task_*/scene.xml"):
        root = ElementTree.parse(scene).getroot()
        collision_geometries.extend(
            geom
            for geom in root.iter("geom")
            if "cube" in geom.attrib.get("name", "")
            and "collision" in geom.attrib.get("name", "")
        )
    assert collision_geometries
    for geom in collision_geometries:
        assert geom.attrib["type"] == "box"
        assert geom.attrib["contype"] == "1"
        # Every cube must collide with the ordinary scene collision group.
        # Task 7 additionally accepts the drawer-only contact bit.
        assert int(geom.attrib["conaffinity"]) & 1
        assert geom.attrib["group"] == "0"


def test_task_cubes_start_on_their_support_surfaces() -> None:
    tasks_root = Path(__file__).parents[1] / "simulation" / "tasks"
    expected_center_heights = {
        "task_01_basic_object_placement": {"cube": 0.035},
        "task_02_object_retrieval": {"cube": 0.045},
        "task_03_color_conditioned_sorting": {
            "blue_cube": 0.035,
            "green_cube": 0.035,
        },
        "task_05_cube_stacking": {
            "blue_cube": 0.035,
            "green_cube": 0.035,
        },
        "task_06_bimanual_handover": {"cube": 0.045},
        "task_07_warehouse_stocking": {
            "blue_cube": 0.035,
            "green_cube": 0.035,
        },
    }
    for task_directory, cube_heights in expected_center_heights.items():
        root = ElementTree.parse(tasks_root / task_directory / "scene.xml").getroot()
        bodies = {body.attrib["name"]: body for body in root.iter("body")}
        for cube_name, expected_height in cube_heights.items():
            actual_height = float(bodies[cube_name].attrib["pos"].split()[2])
            assert actual_height == pytest.approx(expected_height)


def test_all_container_panels_are_5_mm_thick() -> None:
    tasks_root = Path(__file__).parents[1] / "simulation" / "tasks"
    panels = []
    container_scenes = (
        tasks_root / "task_01_basic_object_placement" / "scene.xml",
        tasks_root / "task_02_object_retrieval" / "scene.xml",
        tasks_root / "task_03_color_conditioned_sorting" / "scene.xml",
        tasks_root / "task_06_bimanual_handover" / "scene.xml",
    )
    for scene in container_scenes:
        root = ElementTree.parse(scene).getroot()
        panels.extend(
            geom
            for geom in root.iter("geom")
            if "_bottom" in geom.attrib.get("name", "") or "_wall_" in geom.attrib.get("name", "")
        )
    assert panels
    for panel in panels:
        name = panel.attrib["name"]
        half_extents = tuple(float(value) for value in panel.attrib["size"].split())
        thickness_axis = 2 if name.endswith("_bottom") else (0 if "_wall_x" in name else 1)
        assert half_extents[thickness_axis] == pytest.approx(0.0025)


def test_container_exterior_and_interior_dimensions_are_preserved() -> None:
    tasks_root = Path(__file__).parents[1] / "simulation" / "tasks"
    specifications = (
        ("task_01_basic_object_placement", "target", 0.060, 0.030, 0.070, 0.035),
        ("task_02_object_retrieval", "source", 0.060, 0.030, 0.070, 0.035),
        ("task_03_color_conditioned_sorting", "blue", 0.060, 0.030, 0.070, 0.035),
        ("task_03_color_conditioned_sorting", "green", 0.060, 0.030, 0.070, 0.035),
        ("task_06_bimanual_handover", "first", 0.060, 0.030, 0.070, 0.035),
        ("task_06_bimanual_handover", "second", 0.060, 0.030, 0.070, 0.035),
    )
    for (
        package,
        prefix,
        expected_opening,
        expected_depth,
        expected_exterior_width,
        expected_exterior_height,
    ) in specifications:
        root = ElementTree.parse(tasks_root / package / "scene.xml").getroot()
        geometries = {geom.attrib["name"]: geom for geom in root.iter("geom")}

        def bounds(
            name: str,
            axis: int,
            scene_geometries=geometries,
        ) -> tuple[float, float]:
            geom = scene_geometries[name]
            position = tuple(float(value) for value in geom.attrib["pos"].split())
            size = tuple(float(value) for value in geom.attrib["size"].split())
            return position[axis] - size[axis], position[axis] + size[axis]

        x_inner_negative = bounds(f"{prefix}_wall_xn", 0)[1]
        x_inner_positive = bounds(f"{prefix}_wall_xp", 0)[0]
        y_inner_negative = bounds(f"{prefix}_wall_yn", 1)[1]
        y_inner_positive = bounds(f"{prefix}_wall_yp", 1)[0]
        bottom_top = bounds(f"{prefix}_bottom", 2)[1]
        wall_top = bounds(f"{prefix}_wall_xp", 2)[1]

        x_outer_negative = bounds(f"{prefix}_wall_xn", 0)[0]
        x_outer_positive = bounds(f"{prefix}_wall_xp", 0)[1]
        y_outer_negative = bounds(f"{prefix}_wall_yn", 1)[0]
        y_outer_positive = bounds(f"{prefix}_wall_yp", 1)[1]
        bottom_lower = bounds(f"{prefix}_bottom", 2)[0]

        assert x_inner_positive - x_inner_negative == pytest.approx(expected_opening)
        assert y_inner_positive - y_inner_negative == pytest.approx(expected_opening)
        assert wall_top - bottom_top == pytest.approx(expected_depth)
        assert x_outer_positive - x_outer_negative == pytest.approx(expected_exterior_width)
        assert y_outer_positive - y_outer_negative == pytest.approx(expected_exterior_width)
        assert wall_top - bottom_lower == pytest.approx(expected_exterior_height)


def test_demo_handles_task_four_ik_without_raising() -> None:
    demo = import_module("simulation.tasks.demo")

    exit_code = demo.main(["4", "--headless", "--seed", "5"])
    debug_exit_code = demo.main(
        ["4", "--debug", "--headless", "--no-real-time", "--seed", "5"]
    )

    assert exit_code == 0
    assert debug_exit_code == 0


def test_task_lookup_rejects_unknown_ids() -> None:
    assert get_task(1) is TASKS[0]
    assert get_task(7) is TASKS[-1]
    with pytest.raises(ValueError, match="between 1 and 7"):
        get_task(0)
    with pytest.raises(ValueError, match="between 1 and 7"):
        get_task(8)
