#!/usr/bin/env python3
"""Static audit: this package must contain no robot motion API.

Runs offline, needs neither ROS nor a robot.  Writes motion_api_audit.txt when
invoked as a script so the audit is a deliverable artifact, not just a test.
"""

from __future__ import annotations

import pathlib
import sys

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Anything that can make the robot, gripper, lift or chassis move, plus the
# teleop-shutdown path this bridge must never take.
FORBIDDEN = [
    "rm_movej",
    "rm_movel",
    "rm_movep",
    "rm_movec",
    "rm_move_stop",
    "rm_set_arm_slow_stop",
    "rm_set_joint_",
    "rm_set_pos_",
    "rm_set_arm_",
    "rm_set_tool_voltage",
    "rm_set_gripper",
    "rm_set_rm_plus",
    "rm_write_single_register",
    "set_lift_height",
    "set_lift_speed",
    "chassis_move",
    "set_chassis",
    "FollowJointTrajectory",
    "follow_joint_trajectory",
    "execute_trajectory",
    "ExecuteTrajectory",
    "/execute",
    "pkill",
    "_stop_teleop",
]

def audit() -> tuple[list[str], list[str]]:
    """Scan every line, including comments and docstrings.

    No prose exemption: if the package needs to talk about a motion API it must
    do so without spelling the token, which keeps this audit unambiguous.
    """
    hits: list[str] = []
    scanned: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        scanned.append(str(path.relative_to(PACKAGE_ROOT)))
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for token in FORBIDDEN:
                if token in line:
                    hits.append(
                        f"{path.relative_to(PACKAGE_ROOT)}:{number}: {token}: "
                        f"{line.strip()}"
                    )
    return scanned, hits


def test_no_motion_api() -> None:
    _scanned, hits = audit()
    assert not hits, "motion API found in a read-only package:\n" + "\n".join(hits)


def main() -> int:
    scanned, hits = audit()
    lines = [
        "motion API static audit - grabber_robot_state_bridge",
        f"package root: {PACKAGE_ROOT}",
        f"files scanned: {len(scanned)}",
        *[f"  {name}" for name in scanned],
        f"forbidden tokens checked: {len(FORBIDDEN)}",
        *[f"  {token}" for token in FORBIDDEN],
        "",
        f"result: {'FAIL' if hits else 'PASS - no motion API present'}",
        *hits,
    ]
    text = "\n".join(lines) + "\n"
    out = sys.argv[1] if len(sys.argv) > 1 else "motion_api_audit.txt"
    pathlib.Path(out).write_text(text, encoding="utf-8")
    print(text)
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
