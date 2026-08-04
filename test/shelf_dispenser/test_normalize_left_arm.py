"""The left arm now moves, under exactly the constraints its tool record allows."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts" / "normalize_left_arm.py"
PROFILES = ROOT / "shelf_dispenser" / "safety_profiles.json"


def test_the_left_tool_is_free_space_only():
    """Regression: it used to borrow the right arm's record, which forbids that.

    The same gripper part gives the same nominal geometry, but not the right
    arm's functional validation -- that came from shelf grasps that held, with
    the residual absorbed by a stop-short distance tuned against a real shelf.
    Nothing on the left arm has absorbed anything, so free space only.
    """
    from shelf_dispenser.core import SafetyAbort
    from shelf_dispenser.safety import load_safety_profile

    profile = load_safety_profile(PROFILES, "shelf_template", require_verified=True)
    left = profile.left_tool_mount_calibration
    assert left is not None and left.verified
    assert left.provenance == "nominal_unvalidated"
    assert left.grasping_allowed is False
    # Usable for motion...
    assert left.require_transforms()[0].shape == (4, 4)
    # ...and refused for grasping, by the record rather than by a caller.
    with pytest.raises(SafetyAbort, match="不足以支撑抓取"):
        left.require_grasping_transforms()

    # The right arm keeps its own, stronger record.
    assert profile.tool_mount_calibration.grasping_allowed is True
    assert (
        profile.tool_mount_calibration.evidence_id
        != left.evidence_id
    )


def test_the_entry_plans_the_left_group_against_the_converted_fence():
    """Both are silent when wrong: wrong group, or a fence 120 mm away."""
    source = ENTRY.read_text(encoding="utf-8")
    assert 'planning_group="left_arm"' in source
    assert "safety=left_profile" in source
    assert "left_view(" in source
    # The other arm goes in as live collision scene, not as the planned arm.
    assert "left_robot=right" in source
    # The dense re-check runs before anything executes.
    assert source.index("validate_planned_joints") < source.index(
        "execute_planned_joints"
    )
    # Nothing moves without the flag.
    assert "if not cli.execute:" in source
    assert source.index("if not cli.execute:") < source.index("SafeMotionPlanner(")


def test_the_planning_group_plumbing_is_still_in_place():
    plan_once = (ROOT / "shelf_dispenser" / "ros" / "plan_once.py").read_text(
        encoding="utf-8"
    )
    assert "arm_names(group)" in plan_once
    assert '"left_arm"' in plan_once

    planner = (ROOT / "shelf_dispenser" / "planner.py").read_text(encoding="utf-8")
    # Regression: the trajectory contract used to demand r_joint1..7 whatever
    # was planned, so every valid left-arm trajectory was rejected.
    assert "_normalize_planned_arm_trajectory" in planner
    assert 'prefix = "r" if planning_group == "right_arm" else "l"' in planner


def test_left_planning_is_gated_on_its_own_moveit_bridge():
    """Found by the runtime FK contract on 2026-08-04, gated here instead.

    The profile's T_moveit_from_profile bridges the *right* controller base to
    the MoveIt frame.  Using it for the left arm put MoveIt l_link7 and the
    SDK's left flange 127.2 mm and 179.9 deg apart at the same joint state, and
    the measured dual-arm transform's rotation is within a degree of identity,
    so composing the two cannot account for the half turn.
    """
    from shelf_dispenser.core import SafetyAbort
    from shelf_dispenser.left_arm import assert_left_bridge_measured
    from shelf_dispenser.safety import load_safety_profile

    profile = load_safety_profile(PROFILES, "shelf_template", require_verified=True)
    with pytest.raises(SafetyAbort, match="MoveIt 坐标桥"):
        assert_left_bridge_measured(profile)

    source = ENTRY.read_text(encoding="utf-8")
    # It has to refuse before the arm is opened and teleop is disturbed.
    assert source.index("assert_left_bridge_measured(profile)") < source.index(
        "open_left_arm("
    )
