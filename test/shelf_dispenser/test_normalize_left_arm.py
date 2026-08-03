"""The left-arm entry must stay shut while its safety model is incomplete."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts" / "normalize_left_arm.py"


def test_execute_is_refused_and_says_why():
    """Both blockers are safety-model gaps, so the refusal has to name them."""
    result = subprocess.run(
        [sys.executable, str(ENTRY), "--execute"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
    )
    assert result.returncode == 2
    assert "左臂执行入口未开放" in result.stderr
    assert "工具链未实测" in result.stderr
    assert "LeftArmFence" in result.stderr


def test_the_entry_does_not_borrow_the_right_arm_calibration():
    """Regression: it used to pass the right arm's tool transforms straight in."""
    source = ENTRY.read_text(encoding="utf-8")
    assert "tool_mount_calibration.require_transforms()" not in source
    assert "LeftArmFence(" in source
    # The fence is constructed even on the report path, so a broken dual-arm
    # transform surfaces here rather than on the day motion opens.
    assert source.index("LeftArmFence(") < source.index("ArmJointReader(")


def test_the_planning_group_plumbing_is_still_in_place():
    """The blocked entry must not have quietly taken the left arm out of MoveIt."""
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
