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
    assert "collision_boxes=left_profile.moveit_collision_boxes()" in source
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


def test_the_joint_signs_cross_the_moveit_boundary_and_come_back():
    """Every valid left trajectory would be wrong if only one side mapped."""
    planner = (ROOT / "shelf_dispenser" / "planner.py").read_text(encoding="utf-8")
    assert '"start_joints_deg": to_moveit(start_joints_deg)' in planner
    assert '"goal_joints_deg": to_moveit(goal_joints_deg)' in planner
    # ...and the trajectory is mapped back before anything downstream sees it.
    assert 'plan["points_deg"] = [' in planner
    assert planner.index("_normalize_planned_arm_trajectory(plan") < planner.index(
        'plan["points_deg"] = ['
    )

    source = ENTRY.read_text(encoding="utf-8")
    assert "joint_signs=signs" in source
    assert "left_joint_signs(profile)" in source


def test_the_planner_is_given_the_left_tool_transform():
    """Without it SafeMotionPlanner falls back to the nominal +Z offset.

    The fallback is silent, and it makes the runtime FK contract compare the
    left arm against the right arm's tool -- a disagreement the model does not
    actually have.
    """
    source = ENTRY.read_text(encoding="utf-8")
    assert "link7_to_controller_flange=link7_to_flange" in source
    assert "left_tool_mount_calibration.require_transforms()" in source


def test_the_dense_recheck_knows_which_arm_it_is_checking():
    """Regression: it put the planned trajectory on r_joint* unconditionally.

    Validating a left-arm trajectory therefore drove the *right* arm along the
    left arm's path and parked the right arm's real position on the left arm's
    joints.  Every point collided, index 0 first, and the contact list was full
    of r_link pairs for a plan the right arm had no part in.
    """
    validate = (
        ROOT / "shelf_dispenser" / "ros" / "validate_path.py"
    ).read_text(encoding="utf-8")
    assert "arm_names(group)" in validate
    assert "request.group_name = group" in validate
    assert 'request.group_name = "right_arm"' not in validate
    assert "[*other_names, *planned_names]" in validate

    planner = (ROOT / "shelf_dispenser" / "planner.py").read_text(encoding="utf-8")
    assert '"planning_group": planning_group' in planner

    safe = (ROOT / "shelf_dispenser" / "safe_planner.py").read_text(encoding="utf-8")
    postcheck = safe[safe.index("validate_exact_path(") :]
    assert "planning_group=self.planning_group" in postcheck[:400]
    assert "joint_signs=self.joint_signs" in postcheck[:500]


def test_tool_guard_follows_the_planning_group():
    helper = (
        ROOT / "shelf_dispenser" / "ros" / "scene_helpers.py"
    ).read_text(encoding="utf-8")
    body = helper[helper.index("def attach_tool_guard(") :]
    assert 'prefix = "r" if planning_group == "right_arm" else "l"' in body
    assert 'else "l_hand_link"' in body
    assert "link_name=link7" in body
    assert 'link_name="r_link7"' not in body
    assert 'data.get("planning_group", "right_arm")' in body
