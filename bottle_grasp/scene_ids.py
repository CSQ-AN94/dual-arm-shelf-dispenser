"""Pure helpers for MoveIt world-object id ownership.

This module is imported both from the vision environment (tests) and from the
ROS helper scripts (which run `python3 bottle_grasp/<helper>.py`, putting this
directory on sys.path), so it must stay free of ROS and project imports.
"""

from __future__ import annotations

from typing import Iterable

# Every world collision object this demo ever creates starts with one of
# these prefixes. Helpers delete all matching objects from the live scene
# before adding the current request, so a crashed or interrupted previous
# helper can never leave stale geometry behind.
MANAGED_ID_PREFIXES: tuple[str, ...] = (
    "rgbd",
    "replan_",
    "fence_",
    "collision_selftest",
)

RGBD_VOXELS_ID = "rgbd_voxels"


def managed_scene_ids(existing_ids: Iterable[str]) -> list[str]:
    """Return the ids owned by this demo, in stable order, without duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for object_id in existing_ids:
        if object_id in seen:
            continue
        if any(object_id.startswith(prefix) for prefix in MANAGED_ID_PREFIXES):
            seen.add(object_id)
            result.append(object_id)
    return result
