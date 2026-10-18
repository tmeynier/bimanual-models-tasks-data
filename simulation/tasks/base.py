"""Shared contracts for task-specific simulated demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TaskDefinition:
    """Paper task metadata colocated with its task-specific IK implementation."""

    task_id: int
    slug: str
    title: str
    instruction: str
    nominal_duration_seconds: int
    subgoals: tuple[str, ...]
    implemented: bool = False


class TaskController(Protocol):
    """Interface implemented by task-specific demonstration controllers."""

    def reset(self, *, seed: int) -> None: ...

    def action(self, progress: float) -> Any: ...

    def evaluate(self) -> Any: ...


class TaskNotImplementedError(NotImplementedError):
    """Raised when a benchmark task has not received an IK implementation yet."""


def unavailable_controller(task: TaskDefinition) -> TaskController:
    raise TaskNotImplementedError(
        f"T{task.task_id} ({task.title}) has a specification but no IK implementation yet"
    )
