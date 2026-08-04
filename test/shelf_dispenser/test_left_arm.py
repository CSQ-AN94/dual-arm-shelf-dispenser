"""The left arm's fence must mean the same physical place as the right's."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.left_arm import arrival_error_deg, left_view, open_left_arm
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
    left = left_view(profile, BASE_RIGHT_TO_BASE_LEFT)
    rotation = BASE_RIGHT_TO_BASE_LEFT[:3, :3]
    translation = BASE_RIGHT_TO_BASE_LEFT[:3, 3]

    def verdict(view, point) -> bool:
        # The same code path for both, so this compares frames rather than two
        # people's idea of what the fence rule is.
        try:
            view.assert_tcp_point(point, label="点")
        except SafetyAbort:
            return False
        return True

    rng = np.random.default_rng(0)
    lower = np.asarray(profile.tcp_workspace.minimum) - 0.1
    upper = np.asarray(profile.tcp_workspace.maximum) + 0.1
    samples = [lower + rng.random(3) * (upper - lower) for _ in range(1000)]
    for zone in profile.allowed_tcp_zones:
        for corner in (zone.minimum, zone.maximum):
            samples.append(np.asarray(corner) + 1e-4)
            samples.append(np.asarray(corner) - 1e-4)

    disagreements = 0
    for in_right in samples:
        in_left = rotation.T @ (np.asarray(in_right) - translation)
        if verdict(left, in_left) != verdict(profile, in_right):
            disagreements += 1
    assert disagreements == 0, f"{disagreements}/{len(samples)} 个点两边判定不一致"


def test_the_point_conversion_round_trips():
    left = left_view(_profile(), BASE_RIGHT_TO_BASE_LEFT)
    rotation = BASE_RIGHT_TO_BASE_LEFT[:3, :3]
    translation = BASE_RIGHT_TO_BASE_LEFT[:3, 3]
    point = np.array([0.163, 0.396, -0.180])
    in_left = rotation.T @ (point - translation)
    assert left.in_fence_frame(in_left) == pytest.approx(point)
    # The right-arm view must be untouched by the left one existing.
    assert _profile().tcp_frame_transform is None


def test_a_transform_that_is_not_a_rigid_motion_is_refused():
    profile = _profile()
    with pytest.raises(SafetyAbort, match="4x4"):
        left_view(profile, np.eye(3))
    skewed = np.eye(4)
    skewed[0, 0] = 2.0
    with pytest.raises(SafetyAbort, match="正交"):
        left_view(profile, skewed)


def test_malformed_points_are_refused():
    left = left_view(_profile(), BASE_RIGHT_TO_BASE_LEFT)
    for bad in ([0.0, 0.0], [float("nan"), 0.0, 0.0]):
        with pytest.raises(SafetyAbort, match="TCP 坐标无效"):
            left.assert_tcp_point(bad, label="坏点")


def test_the_left_arm_uses_its_own_record_and_refuses_without_one():
    """The right record's evidence_id forbids the transfer; honour that."""
    profile = _profile()
    assert "do not transfer to the left arm" in (
        profile.tool_mount_calibration.evidence_id
    )
    assert profile.left_tool_mount_calibration is not None

    # With the left record removed, the entry must refuse rather than fall back.
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
