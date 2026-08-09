"""The diagnostic must catch a voxel the fingers really hit and clear one they miss."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_gripper_voxel_contact",
    ROOT / "scripts" / "diagnose_gripper_voxel_contact.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ARMS_CONFIG = {
    "arms": [
        {
            "arm_id": "right_arm",
            "tcp_transform_from_ik_link": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.1682],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    ]
}

# TCP +Z is the approach axis, so a quaternion that maps TCP +Z onto world -Y
# reproduces the shelf geometry: the arm drives in along -Y.  This is the same
# rotation the live scenarios carry.
APPROACH_QUAT = [0.5, 0.5, -0.5, 0.5]


def _scenario(voxels, voxel_size=0.025):
    return {
        "frame_id": "platform_base_link",
        "planning_arm_id": "right_arm",
        "source_approach_direction": [0.0, -1.0, 0.0],
        "source_pregrasp_offset_m": 0.085,
        "source_grasp_pose": {"xyz": [0.0, -0.8, -0.1], "quat_xyzw": APPROACH_QUAT},
        "source_support_surface_id": "fence_shelf_bottom",
        "bottle": {"radius_m": 0.033, "pose": {"xyz": [0.0, -0.8, -0.1]}},
        "shelf_boxes": [],
        "dynamic_obstacle_id": "head_rgbd_non_target",
        "obstacle_voxel_size_m": voxel_size,
        "obstacle_voxels": voxels,
    }


def _opening_axis_world():
    """World direction of the finger box's 151 mm axis at the test grasp pose."""
    rot = MODULE._quat_to_matrix(APPROACH_QUAT)
    yaw = MODULE.MOUNT_YAW_RAD
    mount = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return (rot @ mount)[:, 0]


def test_voxel_beside_the_fingers_is_reported_with_an_opening_axis_bite():
    # 60 mm out along the opening axis from the grasp point: inside the 75.5 mm
    # half width, so the fully-open envelope swallows it.
    point = np.array([0.0, -0.8, -0.1]) + _opening_axis_world() * 0.060
    report = MODULE.analyze(_scenario([point.tolist()]), ARMS_CONFIG)

    candidate = report["candidates"][0]
    assert candidate["contact_count"] == 1
    contact = candidate["contacts"][0]
    assert contact["obstacle"] == "head_rgbd_non_target"
    assert contact["center_xyz"] == pytest.approx(point.tolist(), abs=1e-9)
    # 75.5 mm half width + 12.5 mm voxel half - 60 mm offset = 28 mm.
    assert candidate["max_opening_axis_overlap_m"] == pytest.approx(0.028, abs=1e-6)


def test_voxel_clear_of_the_envelope_is_not_reported():
    # 110 mm out is beyond 75.5 mm + 12.5 mm, and the cross-product axes must
    # not turn that near miss into a false contact.
    point = np.array([0.0, -0.8, -0.1]) + _opening_axis_world() * 0.110
    report = MODULE.analyze(_scenario([point.tolist()]), ARMS_CONFIG)

    candidate = report["candidates"][0]
    assert candidate["contact_count"] == 0
    assert candidate["pregrasp_clear"] is True
    # 110 mm out - 75.5 mm half width - 12.5 mm voxel half = 22 mm of daylight,
    # and the report has to say so rather than only "no contact".
    assert candidate["closest_approach"]["gap_m"] == pytest.approx(0.022, abs=1e-6)
    assert candidate["closest_approach"]["gripper_box"] == "fingers"


def test_a_voxel_only_the_deep_insertion_reaches_leaves_the_pregrasp_clear():
    # On the approach axis at the grasp point: the fingertips are still 13.5 mm
    # short of it at the pregrasp, so first contact must come later.
    report = MODULE.analyze(_scenario([[0.0, -0.8, -0.1]]), ARMS_CONFIG)

    candidate = report["candidates"][0]
    assert candidate["contact_count"] == 1
    assert candidate["pregrasp_clear"] is True
    assert candidate["contacts"][0]["first_contact_insertion_m"] > 0.0
