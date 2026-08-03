#!/usr/bin/env python3
"""Report the left arm against its taught pose.  Motion is not open yet.

The plumbing that kept the left arm unreachable is done and tested:

  * ``arm_worker.ArmProxy`` gives it its own process, so the RealMan SDK's
    process-global ``Algo`` cannot have the two arms overwrite each other.
  * ``ros/plan_once.py`` derives group, link and joint names from the planning
    group, so the collision-aware path serves ``left_arm``.
  * ``left_arm.LeftArmFence`` states the right arm's fence for a left-arm pose
    exactly, by converting the point rather than rewriting the boxes.

What is not done is the safety model, and it is not close enough to fudge:

  * The left tool has never been measured.  ``open_left_arm`` refuses to borrow
    the right arm's record because that record says, in its own evidence_id,
    not to: it is nominal, with its residual absorbed by a stop-short distance
    tuned on the right arm against a real shelf.
  * ``SafeMotionPlanner`` reads the fence through its own profile, so it needs
    a point-conversion hook before ``LeftArmFence`` can sit inside the dense
    re-check.  Handing it a rewritten profile instead is the bug
    ``LeftArmFence`` exists to undo -- bounding a rotated box grows it, and a
    grown *allowed* zone hands out space the fence never granted.

So this reports and stops.  A left arm moving under a fence nobody has verified
is worse than a left arm that does not move.

    python scripts/normalize_left_arm.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.left_arm import LeftArmFence, arrival_error_deg
from shelf_dispenser.arm import ArmJointReader
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config

LOG = logging.getLogger("normalize_left_arm")

BLOCKED = (
    "左臂执行入口未开放：\n"
    "  (1) 左臂工具链未实测——右臂那份 evidence_id 明确写着不得迁移；\n"
    "  (2) SafeMotionPlanner 尚未接入 LeftArmFence 的逐点折算。\n"
    "底层通路（独立进程、left_arm 规划组、精确围栏）已就绪并有测试覆盖。"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Refused for now; see the module docstring for the two blockers",
    )
    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if cli.execute:
        raise SafetyAbort(BLOCKED)

    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    target = profile.grasp_start_left_joints_deg
    if not target:
        raise SafetyAbort("shelf_template 未配置 grasp_start_left_joints_deg")

    # Constructed so a broken dual-arm transform is caught here rather than the
    # first time someone opens the motion path.
    LeftArmFence(profile, cfg.calibration.T_base_right_to_base_left)

    reader = ArmJointReader(cfg.connections.left_arm_ip, cfg.connections.arm_port)
    try:
        current = list(reader.joints_deg())
    finally:
        reader.close()
    error = arrival_error_deg(current, target)
    print(f"左臂 最大关节偏差 {error:7.2f}°")
    print(f"     当前 {' '.join(f'{v:7.1f}' for v in current)}")
    print(f"     目标 {' '.join(f'{v:7.1f}' for v in target)}")
    if error <= params.planned_start_tolerance_deg:
        print("\n已在示教位姿。")
    else:
        print(f"\n偏离示教位姿 {error:.2f}°，目前只能手动摆回。\n{BLOCKED}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
