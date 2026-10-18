"""Shared loading utilities for official LeRobot policies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins

from bimanual_robot.policies import require_supported_policy_type


def apply_inference_overrides(
    policy_config: Any,
    *,
    n_action_steps: int | None,
    temporal_ensemble_coeff: float | None,
) -> None:
    """Apply supported chunking overrides without changing a checkpoint on disk."""

    if n_action_steps is not None:
        if n_action_steps <= 0:
            raise ValueError("--n-action-steps must be positive")
        if not hasattr(policy_config, "n_action_steps"):
            raise ValueError(f"Policy {policy_config.type!r} has no action chunks")
        chunk_size = getattr(policy_config, "chunk_size", None)
        if chunk_size is not None and n_action_steps > chunk_size:
            raise ValueError(
                f"--n-action-steps ({n_action_steps}) cannot exceed chunk size "
                f"({chunk_size})"
            )
        policy_config.n_action_steps = n_action_steps

    if temporal_ensemble_coeff is None:
        return
    if not math.isfinite(temporal_ensemble_coeff) or temporal_ensemble_coeff < 0.0:
        raise ValueError("--temporal-ensemble-coeff must be finite and non-negative")
    if policy_config.type != "act" or not hasattr(
        policy_config, "temporal_ensemble_coeff"
    ):
        raise ValueError("--temporal-ensemble-coeff is supported only by ACT")
    if getattr(policy_config, "n_action_steps", None) != 1:
        raise ValueError("Temporal ensembling requires --n-action-steps 1")
    policy_config.temporal_ensemble_coeff = temporal_ensemble_coeff


def load_policy(
    policy_path: str | Path,
    device: str,
    *,
    n_action_steps: int | None = None,
    temporal_ensemble_coeff: float | None = None,
):
    """Load a policy and its saved preprocessing pipeline for inference."""

    register_third_party_plugins()
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    require_supported_policy_type(policy_config.type)
    policy_config.device = device
    policy_config.pretrained_path = policy_path
    apply_inference_overrides(
        policy_config,
        n_action_steps=n_action_steps,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
    )
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(policy_path, config=policy_config)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(policy_path),
        pretrained_revision=policy_config.pretrained_revision,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, policy_config, preprocessor, postprocessor
