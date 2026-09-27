from __future__ import annotations

import numpy as np
import pytest
from hepha_lerobot.conditioning import (
    DRAWER_CONDITION_NAMES,
    drawer_condition,
    drawer_condition_feature,
    drawer_task,
)
from hepha_lerobot.datasets.builder import create_dataset
from lerobot.utils.constants import OBS_ENV_STATE


def test_drawer_condition_is_one_hot() -> None:
    condition = drawer_condition(5)

    np.testing.assert_array_equal(
        condition,
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    )


def test_drawer_condition_uses_native_lerobot_environment_state() -> None:
    feature = drawer_condition_feature()[OBS_ENV_STATE]

    assert feature["shape"] == (9,)
    assert feature["names"] == list(DRAWER_CONDITION_NAMES)


def test_drawer_task_identifies_the_requested_drawer() -> None:
    assert drawer_task("Move the cube to drawer {drawer_index}.", 3) == (
        "Move the cube to drawer 3."
    )


def test_ik_dataset_environment_state_includes_drawer_and_phase(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(
        "hepha_lerobot.datasets.builder.LeRobotDataset.create",
        lambda **kwargs: captured.update(kwargs),
    )
    backend = type(
        "Backend",
        (),
        {
            "name": "test",
            "action_features": {"joint": float},
            "observation_features": {"joint": float},
        },
    )()

    create_dataset(
        backend=backend,
        repo_id="hepha/test",
        root=tmp_path,
        fps=30,
        use_videos=False,
    )

    feature = captured["features"][OBS_ENV_STATE]
    assert feature["shape"] == (14,)


@pytest.mark.parametrize("drawer_index", [0, 10])
def test_drawer_condition_rejects_invalid_indices(drawer_index: int) -> None:
    with pytest.raises(ValueError, match="Drawer index"):
        drawer_condition(drawer_index)
