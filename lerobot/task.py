"""Shared constants for the cube-and-drawer task."""

from __future__ import annotations

from enum import IntEnum

DRAWER_COUNT = 9


class TaskPhase(IntEnum):
    PREPARE = 1
    OPEN_DRAWER = 2
    PICK_CUBE = 3
    PLACE_AND_CLOSE = 4
    COMPLETE = 5

