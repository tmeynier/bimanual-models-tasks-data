from __future__ import annotations

import pytest

from bimanual_robot.policies import (
    SUPPORTED_POLICY_TYPES,
    available_policy_types,
    require_supported_policy_type,
)


def test_only_paper_policy_families_are_exposed() -> None:
    assert SUPPORTED_POLICY_TYPES == ("act", "pi05")
    assert available_policy_types() == SUPPORTED_POLICY_TYPES


def test_other_policy_families_are_rejected() -> None:
    with pytest.raises(ValueError, match="supports only act, pi05"):
        require_supported_policy_type("unsupported")
