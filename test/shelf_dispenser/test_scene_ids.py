"""Stateless scene-cleanup id ownership; no ROS required."""

from shelf_dispenser.ros.scene_ids import RGBD_VOXELS_ID, managed_scene_ids


def test_owned_prefixes_are_selected_for_removal():
    existing = [
        RGBD_VOXELS_ID,
        "rgbd_17",  # legacy per-voxel id from an interrupted old run
        "replan_03",
        "fence_table_top",
        "collision_selftest_box",
    ]
    assert managed_scene_ids(existing) == existing


def test_foreign_objects_are_never_touched():
    assert managed_scene_ids(["table", "operator_zone", "bottle"]) == []


def test_duplicates_collapse_and_order_is_stable():
    assert managed_scene_ids(
        ["replan_01", "rgbd_voxels", "replan_01"]
    ) == ["replan_01", "rgbd_voxels"]
