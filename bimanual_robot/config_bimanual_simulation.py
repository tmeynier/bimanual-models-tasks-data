"""Backend-neutral LeRobot configuration for the simulated bimanual robot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("bimanual_simulation")
@dataclass
class BimanualSimulationConfig(RobotConfig):
    backend: str = "mujoco"
    backend_options: dict[str, Any] = field(default_factory=dict)
    camera: str = "head_camera"
    width: int = 256
    height: int = 256
    fps: int = 30
    viewer: bool = False
