"""The left arm's fence must mean the same physical place as the right's."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.left_arm import LeftArmFence, arrival_error_deg, open_left_arm
from shelf_dispenser.safety import load_safety_profile

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "shelf_dispenser" / "safety_profiles.json"

# The measured dual-arm transform from config.yaml, 2026-07-14.
BASE_RIGHT_TO_BASE_LEFT = np.array(
    [
        [0.999691058769, 0.016811128493, 0.018307730006, -0.119987534677],
        [-0.016767487109, 0.999856203057, -0.002534676056, 0.007718909353],
        [-0.018347708175, 0.002226918364, 0.999829186631, 0.014391259738],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _profile():
    return load_safety_profile(PROFILES, "shelf_template", require_verified=False)


def test_the_verdict_is_identical_to_the_right_arm_fence_everywhere():
    """Regression: bounding a rotated box grew the allowed zones.

    The earlier version rewrote each box into the left base frame and took the
    axis-aligned hull, which is larger than the box it came from.  For an
    allowed zone that hands out space the fence never granted -- a point could
    pass the left-framed check while being outside every real zone.  Converting
    the point instead is exact, so the two verdicts must agree on every sample,
    including ones deliberately placed either side of a zone edge.
    """
    profile = _profile()
    fence = LeftArmFence(profile, BASE_RIGHT_TO_BASE_LEFT)
    rotation = BASE_RIGHT_TO_BASE_LEFT[:3, :3]
    translation = BASE_RIGHT_TO_BASE_LEFT[:3, 3]

    def right_verdict(point) -> bool:
        return bool(
            profile.tcp_workspace.contains(point, profile.clearance_m)
            and any(z.contains(point, 0.0) for z in profile.allowed_tcp_zones)
            and not any(b.contains(point, 0.0) for b in profile.keepout_boxes)
        )

    rng = np.random.default_rng(0)
    lower = np.asarray(profile.tcp_workspace.minimum) - 0.1
    upper = np.asarray(profile.tcp_workspace.maximum) + 0.1
    samples = [lower + rng.random(3) * (upper - lower) for _ in range(1000)]
    for zone in profile.allowed_tcp_zones:
        for corner in (zone.minimum, zone.maximum):
            samples.append(np.asarray(corner) + 1e-4)
            samples.append(np.asarray(corner) - 1e-4)

    for in_right in samples:
        in_left = rotation.T @ (np.asarray(in_right) - translation)
        assert fence.contains(in_left) == right_verdict(in_right), in_right


def test_the_point_conversion_round_trips():
    fence = LeftArmFence(_profile(), BASE_RIGHT_TO_BASE_LEFT)
    rotation = BASE_RIGHT_TO_BASE_LEFT[:3, :3]
    translation = BASE_RIGHT_TO_BASE_LEFT[:3, 3]
    point = np.array([0.163, 0.396, -0.180])
    in_left = rotation.T @ (point - translation)
    assert fence.to_right_base(in_left) == pytest.approx(point)


def test_a_transform_that_is_not_a_rigid_motion_is_refused():
    profile = _profile()
    with pytest.raises(SafetyAbort, match="4x4"):
        LeftArmFence(profile, np.eye(3))
    skewed = np.eye(4)
    skewed[0, 0] = 2.0
    with pytest.raises(SafetyAbort, match="正交"):
        LeftArmFence(profile, skewed)


def test_malformed_points_are_refused():
    fence = LeftArmFence(_profile(), BASE_RIGHT_TO_BASE_LEFT)
    with pytest.raises(SafetyAbort, match="三个有限数"):
        fence.to_right_base([0.0, 0.0])
    with pytest.raises(SafetyAbort, match="三个有限数"):
        fence.to_right_base([float("nan"), 0.0, 0.0])


def test_the_left_arm_refuses_to_borrow_the_right_arm_tool_calibration():
    """The profile's own evidence_id forbids exactly this transfer."""
    profile = _profile()
    assert "do not transfer to the left arm" in (
        profile.tool_mount_calibration.evidence_id
    )
    with pytest.raises(SafetyAbort, match="左臂工具标定"):
        open_left_arm(
            SimpleNamespace(connections=SimpleNamespace()),
            SimpleNamespace(
                tcp_z_m=0.151, moveit_link7_to_controller_flange_m=0.0172
            ),
            profile,
            take_control=False,
        )


def test_arrival_error_rejects_malformed_joints():
    assert arrival_error_deg([1.0] * 7, [1.5] * 7) == pytest.approx(0.5)
    with pytest.raises(SafetyAbort, match="7 个数"):
        arrival_error_deg([1.0] * 6, [1.0] * 7)
    with pytest.raises(SafetyAbort, match="非有限"):
        arrival_error_deg([float("nan")] * 7, [1.0] * 7)
