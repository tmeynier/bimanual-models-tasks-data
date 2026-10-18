"""Expose only the two policy families evaluated by this repository."""

from __future__ import annotations

SUPPORTED_POLICY_TYPES = ("act", "pi05")


def available_policy_types() -> tuple[str, ...]:
    """Return the supported paper policies available in this environment."""

    import lerobot.policies  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    registered = PreTrainedConfig.get_known_choices()
    return tuple(name for name in SUPPORTED_POLICY_TYPES if name in registered)


def require_supported_policy_type(policy_type: str) -> None:
    if policy_type not in SUPPORTED_POLICY_TYPES:
        supported = ", ".join(SUPPORTED_POLICY_TYPES)
        raise ValueError(
            f"Unsupported policy type {policy_type!r}; this repository supports "
            f"only {supported}."
        )


def main() -> None:
    print("\n".join(available_policy_types()))


if __name__ == "__main__":
    main()
