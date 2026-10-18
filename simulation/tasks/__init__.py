"""Seven task-scoped simulation packages defined by the accompanying paper."""

from dataclasses import replace
from importlib import import_module
from pathlib import Path

from simulation import SimulationConfig, create_backend
from simulation.backends.mujoco import MujocoBackend

from .base import TaskController, TaskDefinition, TaskNotImplementedError
from .task_01_basic_object_placement import TASK as TASK_01
from .task_02_object_retrieval import TASK as TASK_02
from .task_03_color_conditioned_sorting import TASK as TASK_03
from .task_04_drawer_opening import TASK as TASK_04
from .task_05_cube_stacking import TASK as TASK_05
from .task_06_bimanual_handover import TASK as TASK_06
from .task_07_warehouse_stocking import TASK as TASK_07

TASKS = (TASK_01, TASK_02, TASK_03, TASK_04, TASK_05, TASK_06, TASK_07)
TASK_PACKAGES = (
    "task_01_basic_object_placement",
    "task_02_object_retrieval",
    "task_03_color_conditioned_sorting",
    "task_04_drawer_opening",
    "task_05_cube_stacking",
    "task_06_bimanual_handover",
    "task_07_warehouse_stocking",
)


def get_task(task_id: int) -> TaskDefinition:
    if not 1 <= task_id <= len(TASKS):
        raise ValueError(f"Task ID must be between 1 and {len(TASKS)}, got {task_id}")
    return TASKS[task_id - 1]


def task_scene_path(task_id: int) -> Path:
    task = get_task(task_id)
    if not task.implemented:
        raise TaskNotImplementedError(f"T{task_id} has no implemented scene")
    return Path(__file__).parent / TASK_PACKAGES[task_id - 1] / "scene.xml"


def create_task_backend(task_id: int, config: SimulationConfig) -> MujocoBackend:
    """Create the MuJoCo scene owned by one implemented task package."""

    options = {**config.options, "model_path": str(task_scene_path(task_id))}
    backend = create_backend("mujoco", replace(config, options=options))
    if not isinstance(backend, MujocoBackend):
        raise TypeError("Task demonstrations currently require the MuJoCo backend")
    return backend


def create_task_controller(task_id: int, backend: MujocoBackend) -> TaskController:
    task = get_task(task_id)
    if not task.implemented:
        raise TaskNotImplementedError(f"T{task_id} has no implemented IK controller")
    module = import_module(f"simulation.tasks.{TASK_PACKAGES[task_id - 1]}.ik")
    return module.create_controller(backend)


def main() -> None:
    for task in TASKS:
        status = "implemented" if task.implemented else "planned"
        print(f"T{task.task_id}: {task.title} [{status}]")


__all__ = [
    "TASKS",
    "TaskController",
    "TaskDefinition",
    "TaskNotImplementedError",
    "create_task_backend",
    "create_task_controller",
    "get_task",
    "main",
    "task_scene_path",
]
