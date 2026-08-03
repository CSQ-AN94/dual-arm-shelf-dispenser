"""The left arm's fence has to describe the same physical volume, moved."""

from pathlib import Path

import numpy as np
import pytest

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.left_arm import arrival_error_deg, left_view, transform_box
from shelf_dispenser.safety import FenceBox, load_safety_profile

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


def test_transform_box_under_identity_is_the_same_box():
    box = FenceBox(id="b", minimum=(-1.0, -2.0, -3.0), maximum=(4.0, 5.0, 6.0))
    same = transform_box(box, np.eye(4))
    assert same.minimum == pytest.approx(box.minimum)
    assert same.maximum == pytest.approx(box.maximum)


def test_transform_box_translates_and_never_shrinks():
    box = FenceBox(id="b", minimum=(0.0, 0.0, 0.0), maximum=(1.0, 1.0, 1.0))
    shift = np.eye(4)
    shift[:3, 3] = [0.1, -0.2, 0.3]
    moved = transform_box(box, shift)
    assert moved.minimum == pytest.approx([0.1, -0.2, 0.3])
    assert moved.maximum == pytest.approx([1.1, 0.8, 1.3])

    # A rotated box is bounded, not rotated: the axis-aligned hull can only
    # grow, which is the safe direction for a keepout and a tolerable one for
    # an allowed zone.
    angle = np.deg2rad(30.0)
    spin = np.eye(4)
    spin[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    spun = transform_box(box, spin)
    original = np.asarray(box.maximum) - np.asarray(box.minimum)
    grown = np.asarray(spun.maximum) - np.asarray(spun.minimum)
    assert np.all(grown >= original - 1e-9)


def test_left_view_moves_the_real_fence_by_the_measured_offset():
    profile = load_safety_profile(PROFILES, "shelf_template", require_verified=False)
    left = left_view(profile, BASE_RIGHT_TO_BASE_LEFT)

    assert left.name.endswith("__left")
    assert len(left.allowed_tcp_zones) == len(profile.allowed_tcp_zones)
    assert len(left.keepout_boxes) == len(profile.keepout_boxes)

    # A box is centrally symmetric, so the hull's centre lands exactly on the
    # transformed centre even though the corners spread.  The corners are the
    # wrong thing to assert on: a degree of rotation inflates a 1.3 m box by
    # nearly 3 cm, which is the slack the docstring promises.
    def centre(box):
        return (np.asarray(box.minimum) + np.asarray(box.maximum)) / 2.0

    shift = centre(left.tcp_workspace) - centre(profile.tcp_workspace)
    assert shift[0] == pytest.approx(0.120, abs=0.005)

    span = lambda box: np.asarray(box.maximum) - np.asarray(box.minimum)
    inflation = span(left.tcp_workspace) - span(profile.tcp_workspace)
    assert np.all(inflation >= -1e-9)
    assert np.all(inflation < 0.05)

    for zone in left.allowed_tcp_zones:
        assert np.all(
            np.asarray(zone.maximum) > np.asarray(zone.minimum)
        ), zone.id


def test_left_view_refuses_a_transform_that_is_not_a_rigid_motion():
    profile = load_safety_profile(PROFILES, "shelf_template", require_verified=False)
    with pytest.raises(SafetyAbort, match="4x4"):
        left_view(profile, np.eye(3))
    skewed = np.eye(4)
    skewed[0, 0] = 2.0
    with pytest.raises(SafetyAbort, match="正交"):
        left_view(profile, skewed)


def test_a_point_keeps_its_verdict_across_the_two_frames():
    """The fence must mean the same physical place for either arm."""
    profile = load_safety_profile(PROFILES, "shelf_template", require_verified=False)
    left = left_view(profile, BASE_RIGHT_TO_BASE_LEFT)
    rotation = BASE_RIGHT_TO_BASE_LEFT[:3, :3]
    translation = BASE_RIGHT_TO_BASE_LEFT[:3, 3]

    rng = np.random.default_rng(0)
    lower = np.asarray(profile.tcp_workspace.minimum)
    upper = np.asarray(profile.tcp_workspace.maximum)
    agreed = 0
    for _ in range(200):
        in_right = lower + rng.random(3) * (upper - lower)
        # Same physical point, named in the left base frame.
        in_left = rotation.T @ (in_right - translation)
        if profile.tcp_workspace.contains(in_right, 0.0) == (
            left.tcp_workspace.contains(in_left, 0.0)
        ):
            agreed += 1
    assert agreed == 200


def test_arrival_error_rejects_malformed_joints():
    assert arrival_error_deg([1.0] * 7, [1.5] * 7) == pytest.approx(0.5)
    with pytest.raises(SafetyAbort, match="7 个数"):
        arrival_error_deg([1.0] * 6, [1.0] * 7)
    with pytest.raises(SafetyAbort, match="非有限"):
        arrival_error_deg([float("nan")] * 7, [1.0] * 7)
