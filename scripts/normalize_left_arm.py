#!/usr/bin/env python3
"""Put the left arm on its taught pose, through the right arm's safety chain.

The left arm was unreachable for months, for two reasons that had nothing to do
with the machinery being wrong:

  * The RealMan SDK's ``Algo`` is process-global, so a second ``RobotSession``
    in one process overwrites the first one's kinematics.  ``ArmProxy`` gives
    the left arm its own process.
  * Every fence box is authored in the right arm's base frame, 120 mm from the
    left arm's own.  ``left_view`` hands the profile the conversion, so the
    dense re-check reads left-arm poses correctly with no caller changed.

What this can and cannot do is decided by the tool record, not by this script.
The left tool is ``nominal_unvalidated``: the same RMG24 geometry as the right
arm, but with nothing on this arm having absorbed its error the way a tuned
stop-short distance did on the right.  That is enough for free space, where the
fence zones are generous, and not enough to close fingers on anything --
``require_grasping_transforms`` refuses.

The plan still carries the right arm as live collision scene, so MoveIt keeps
the two apart.  What it does not carry is a fresh RGB-D capture: this is a
transit through space the profile already calls a corridor, not a reach into a
shelf.

Defaults to a dry run.  Nothing moves without --execute.

    python scripts/normalize_left_arm.py               # 只报告
    python scripts/normalize_left_arm.py --execute     # 真运动
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.arm import ArmJointReader
from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.left_arm import (
    arrival_error_deg,
    left_joint_signs,
    left_view,
    open_left_arm,
)
from shelf_dispenser.planner import MoveItPlanner
from shelf_dispenser.safe_planner import PlanTarget, SafeMotionPlanner
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config

LOG = logging.getLogger("normalize_left_arm")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "left_arm_normalize")
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--target-joints",
        type=float,
        nargs=7,
        help=(
            "Plan to these seven degrees instead of the profile's taught pose. "
            "The taught one has never been through a planner, and the model "
            "reports it in collision with body_base_link; this is how a "
            "replacement gets tried before it is taught."
        ),
    )
    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run_dir = Path(cli.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    target = cli.target_joints or profile.grasp_start_left_joints_deg
    if not target:
        raise SafetyAbort("shelf_template 未配置 grasp_start_left_joints_deg")
    if cli.target_joints:
        LOG.info("使用命令行指定的目标，不是 profile 里的示教位姿")

    # Built before anything opens, so a broken dual-arm transform surfaces here
    # rather than midway through a motion.
    left_profile = left_view(profile)
    signs = left_joint_signs(profile)

    link7_to_flange, _ = profile.left_tool_mount_calibration.require_transforms()
    left = open_left_arm(cfg, params, profile, take_control=cli.execute)
    right = ArmJointReader(cfg.connections.right_arm_ip, cfg.connections.arm_port)
    moveit = None
    try:
        current = list(left.joints_deg())
        error = arrival_error_deg(current, target)
        print(f"左臂 最大关节偏差 {error:7.2f}°")
        print(f"     当前 {' '.join(f'{v:7.1f}' for v in current)}")
        print(f"     目标 {' '.join(f'{v:7.1f}' for v in target)}")
        if error <= params.planned_start_tolerance_deg:
            print("\n已在示教位姿，无需移动。")
            return 0
        if not cli.execute:
            print("\n干跑：未下发任何运动。加 --execute 才会移动左臂。")
            return 0

        # The planning services live in a headless move_group this owns for
        # the duration.  The right arm's flow starts one inside initialize();
        # a standalone entry has to start its own or every plan fails with
        # "service unavailable: /apply_planning_scene".
        moveit = MoveItPlanner(project_root=ROOT, run_dir=run_dir)
        moveit.start()
        planner = SafeMotionPlanner(
            moveit=moveit,
            robot=left,
            left_robot=right,  # the other arm, live, as collision scene
            safety=left_profile,
            params=params,
            report=lambda name, message: LOG.info("%s: %s", name, message),
            planning_group="left_arm",
            joint_signs=signs,
            # Without this it silently falls back to the nominal +Z offset, so
            # the runtime FK contract compares the left arm against the wrong
            # tool and reports a disagreement the model does not have.
            link7_to_controller_flange=link7_to_flange,
        )
        verified = planner.plan(
            name="normalize_left_arm",
            targets=[
                PlanTarget(
                    label="示教左臂位姿",
                    flange=np.asarray(
                        left.controller_flange_from_joints(list(target)),
                        dtype=float,
                    ),
                    goal_joints=tuple(map(float, target)),
                    goal_constraint="joints",
                )
            ],
            obstacle_points=[],
            collision_boxes=[],
        )
        points = [
            list(map(float, point)) for point in verified.trajectory["points_deg"]
        ]
        left.validate_planned_joints(points, measured_start=current)
        left.execute_planned_joints(points)
        reached = list(left.joints_deg())
        error = arrival_error_deg(reached, target)
        if error > params.planned_start_tolerance_deg:
            raise SafetyAbort(
                f"左臂到位偏差 {error:.2f}° 超过 "
                f"{params.planned_start_tolerance_deg:.2f}°"
            )
        print(f"\n已到达示教左臂位姿，偏差 {error:.2f}°。")
    finally:
        if moveit is not None:
            moveit.close()
        right.close()
        left.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
