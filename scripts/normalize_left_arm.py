#!/usr/bin/env python3
"""Put the left arm on its taught pose, through the same safety chain as the right.

The left arm has been driven before, by hand and by one-off scripts.  What it
has never had is the chain the right arm gets: a collision-aware MoveIt plan
against the live scene, a dense re-check of every trajectory point against the
electronic fence, and a joint-limit margin measured from the arm's own starting
excess.  Moving it by hand is how it ended up at a pose nobody taught, which is
how the profile ended up recording one.

Two things made that chain unreachable for the left arm, and both are fixed:

  * The RealMan SDK's Algo is process-global, so a second RobotSession in this
    process would overwrite the right arm's kinematics.  ArmProxy gives the left
    arm its own process (bottle_grasp/arm_worker.py).
  * Every fence box is expressed in the right arm's base frame, 120 mm from the
    left arm's own.  left_view moves the fence over once, using the measured
    dual-arm transform from config.yaml (bottle_grasp/left_arm.py).

Defaults to a dry run.  Nothing moves without --execute.

    python scripts/normalize_left_arm.py               # 只报告
    python scripts/normalize_left_arm.py --execute     # 真运动
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bottle_grasp.core import DemoParams, SafetyAbort
from bottle_grasp.left_arm import arrival_error_deg, left_view, open_left_arm
from bottle_grasp.planner import MoveItPlanner
from bottle_grasp.robot import ArmJointReader
from bottle_grasp.safe_planner import PlanTarget, SafeMotionPlanner
from bottle_grasp.safety import load_safety_profile
from utils.config import load_config

LOG = logging.getLogger("normalize_left_arm")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "bottle_grasp" / "safety_profiles.json"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "left_arm_normalize")
    )
    parser.add_argument("--execute", action="store_true")
    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run_dir = Path(cli.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    target = profile.grasp_start_left_joints_deg
    if not target:
        raise SafetyAbort("shelf_template 未配置 grasp_start_left_joints_deg")

    left_profile = left_view(
        profile, cfg.calibration.T_base_right_to_base_left
    )
    LOG.info(
        "左臂围栏已折算：工作区 x [%.3f, %.3f]",
        left_profile.tcp_workspace.minimum[0],
        left_profile.tcp_workspace.maximum[0],
    )

    left = open_left_arm(cfg, params, profile, take_control=cli.execute)
    right = ArmJointReader(cfg.connections.right_arm_ip, cfg.connections.arm_port)
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

        planner = SafeMotionPlanner(
            moveit=MoveItPlanner(project_root=ROOT, run_dir=run_dir),
            robot=left,
            left_robot=right,  # the other arm, captured live as collision scene
            safety=left_profile,
            params=params,
            report=lambda name, message: LOG.info("%s: %s", name, message),
            planning_group="left_arm",
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
        print(f"\n已到达示教左臂位姿，偏差 {error:.2f}°。")
        if error > params.planned_start_tolerance_deg:
            raise SafetyAbort(
                f"左臂到位偏差 {error:.2f}° 超过 "
                f"{params.planned_start_tolerance_deg:.2f}°"
            )
    finally:
        right.close()
        left.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
