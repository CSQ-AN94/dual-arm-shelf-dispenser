"""The left arm's measured model has to hold, in both directions."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.left_arm import (
    arrival_error_deg,
    left_joint_signs,
    left_view,
    open_left_arm,
)
from shelf_dispenser.safety import load_safety_profile

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "shelf_dispenser" / "safety_profiles.json"


def _profile():
    return load_safety_profile(PROFILES, "shelf_template", require_verified=False)


def test_the_measured_model_is_present_and_closed():
    """0.678 mm / 0.108 deg over 24 states, with the joints left alone.

    An earlier model negated joints 2, 4 and 6 and matched link7 to 0.5 mm --
    and put the elbow in a mirrored configuration that MoveIt reported as the
    arm colliding with its own head.  Endpoint agreement is not configuration
    agreement on a 7-DoF arm, so the signs staying at +1 is part of the result.
    """
    model = _profile().left_arm_model
    assert model is not None
    assert model["joint_signs"] == [1] * 7
    assert model["samples"] >= 12
    assert model["max_position_residual_m"] < 0.003
    assert model["max_orientation_residual_deg"] < 0.5

    # The whole 180 degrees lives on the tool side, where the URDF puts it.
    tool = np.asarray(model["T_left_link7_tool_offset"], dtype=float)
    yaw = np.degrees(np.arctan2(tool[1, 0], tool[0, 0]))
    assert abs(abs(yaw) - 180.0) < 1.0, yaw


def test_left_view_swaps_the_bridge_and_the_tool_record():
    profile = _profile()
    left = left_view(profile)

    assert left.name.endswith("__left")
    # Planning must go through the left arm's own bridge, not the right one's.
    assert not np.allclose(left.T_moveit_from_profile, profile.T_moveit_from_profile)
    assert np.allclose(
        left.T_moveit_from_profile,
        np.asarray(profile.left_arm_model["T_moveit_from_left_profile"]),
    )
    # And through the left tool record, which carries the tool-side constant.
    assert left.tool_mount_calibration is profile.left_tool_mount_calibration
    assert left.tool_mount_calibration.grasping_allowed is False
    # The right-arm view is untouched.
    assert _profile().tcp_frame_transform is None


def test_the_fence_conversion_is_consistent_with_both_bridges():
    """Derived from the two measured bridges, not from the suspect config one.

    config.yaml's dual-arm transform came from two head-camera eye-to-hand
    rounds and does not describe the controller base frames -- using it is what
    produced the 179.9 deg surprise.
    """
    profile = _profile()
    left = left_view(profile)
    right_bridge = np.asarray(profile.T_moveit_from_profile, dtype=float)
    left_bridge = np.asarray(
        profile.left_arm_model["T_moveit_from_left_profile"], dtype=float
    )
    # A point in the left base frame, taken to MoveIt two ways, must land twice.
    point = np.array([0.12, -0.30, 0.44, 1.0])
    via_conversion = right_bridge @ np.append(
        left.in_fence_frame(point[:3]), 1.0
    )
    via_left_bridge = left_bridge @ point
    assert via_conversion == pytest.approx(via_left_bridge, abs=1e-9)


def test_joint_signs_are_validated_not_trusted():
    profile = _profile()
    assert left_joint_signs(profile) == (1,) * 7

    broken = replace(profile, left_arm_model={"joint_signs": [1, 1, 1]})
    with pytest.raises(SafetyAbort, match="7 个"):
        left_joint_signs(broken)
    missing = replace(profile, left_arm_model=None)
    with pytest.raises(SafetyAbort, match="left_arm_model"):
        left_joint_signs(missing)
    with pytest.raises(SafetyAbort, match="left_arm_model"):
        left_view(missing)


def test_the_left_arm_needs_its_own_tool_record():
    profile = _profile()
    assert "do not transfer to the left arm" in (
        profile.tool_mount_calibration.evidence_id
    )
    stripped = replace(profile, left_tool_mount_calibration=None)
    with pytest.raises(SafetyAbort, match="左臂工具标定"):
        open_left_arm(
            SimpleNamespace(connections=SimpleNamespace()),
            SimpleNamespace(
                tcp_z_m=0.151, moveit_link7_to_controller_flange_m=0.0172
            ),
            stripped,
            take_control=False,
        )


def test_arrival_error_rejects_malformed_joints():
    assert arrival_error_deg([1.0] * 7, [1.5] * 7) == pytest.approx(0.5)
    with pytest.raises(SafetyAbort, match="7 个数"):
        arrival_error_deg([1.0] * 6, [1.0] * 7)
    with pytest.raises(SafetyAbort, match="非有限"):
        arrival_error_deg([float("nan")] * 7, [1.0] * 7)
