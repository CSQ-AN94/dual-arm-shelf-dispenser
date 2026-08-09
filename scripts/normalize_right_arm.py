#!/usr/bin/env python3
"""Put the right arm on its taught pose, without touching the head servo.

``normalize_to_grasp_start`` re-centres the head first, because every 3-D point
the head camera produces is invalid if the servo has drifted.  That is right for
a run that uses the camera, and wrong for one that does not: a joint-space move
to a taught pose needs no perception at all, and on 2026-08-04 the servo would
not converge -- it oscillated between 488 and 539 around a 516 target, twenty
steps, overshooting by 22 each way -- which left the arm stranded off-pose with
nothing wrong with the arm.

So this is the right arm's counterpart to ``normalize_left_arm``: the same
MoveIt plan, dense fence re-check and joint-limit margin, and no camera.

Use ``normalize_to_grasp_start`` when the next thing is a camera-driven run;
use this when the next thing is not, or when the head is the thing that is
broken.

    python scripts/normalize_right_arm.py             # 只报告
    python scripts/normalize_right_arm.py --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.arm import ArmJointReader, RobotSession
from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.mobile_body import LiftSocketAdapter
from shelf_dispenser.planner import MoveItPlanner
from shelf_dispenser.safe_planner import PlanTarget, SafeMotionPlanner
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config

LOG = logging.getLogger("normalize_right_arm")
LIFT_TOLERANCE_MM = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "right_arm_normalize")
    )
    parser.add_argument("--speed", type=int, default=30)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--skip-lift",
        action="store_true",
        help="Leave the lift where it is; only move the arm",
    )
    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = Path(cli.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    target = profile.grasp_start_right_joints_deg
    if not target:
        raise SafetyAbort("shelf_template 未配置 grasp_start_right_joints_deg")

    lift = LiftSocketAdapter(cfg.connections.left_arm_ip, cfg.connections.arm_port)
    lift_now = lift.state()
    lift_error = abs(
        int(lift_now.height_mm) - int(profile.grasp_start_lift_height_mm)
    )
    link7_to_flange, flange_to_tcp = (
        profile.tool_mount_calibration.require_transforms()
    )
    robot = RobotSession(
        cfg.connections.right_arm_ip, cfg.connections.arm_port, threading.Event(),
        params.tcp_z_m, params.moveit_link7_to_controller_flange_m,
        take_control=cli.execute, tcp_transform=flange_to_tcp,
        link7_to_controller_flange=link7_to_flange,
    )
    left = ArmJointReader(cfg.connections.left_arm_ip, cfg.connections.arm_port)
    moveit = None
    try:
        current = list(robot.joints_deg())
        error = float(
            np.max(np.abs(np.asarray(current) - np.asarray(target, dtype=float)))
        )
        print(f"右臂 最大关节偏差 {error:7.2f}°")
        print(f"     当前 {' '.join(f'{v:7.1f}' for v in current)}")
        print(f"     目标 {' '.join(f'{v:7.1f}' for v in target)}")
        print(
            f"升降 {lift_now.height_mm} mm -> "
            f"{profile.grasp_start_lift_height_mm} mm  差 {lift_error} mm"
        )
        if not cli.execute:
            print("\n干跑：未下发任何运动。加 --execute 才会移动。")
            return 0

        if lift_error > LIFT_TOLERANCE_MM and not cli.skip_lift:
            after = lift.move_to(
                int(profile.grasp_start_lift_height_mm), speed=params.final_speed
            )
            print(f"升降到 {after.height_mm} mm")

        if error <= params.planned_start_tolerance_deg:
            print("\n右臂已在示教位姿。")
            return 0

        moveit = MoveItPlanner(project_root=ROOT, run_dir=run_dir)
        moveit.start()
        planner = SafeMotionPlanner(
            moveit=moveit, robot=robot, left_robot=left, safety=profile,
            params=params, report=lambda n, m: LOG.info("  %s: %s", n, m),
            link7_to_controller_flange=link7_to_flange,
        )
        verified = planner.plan(
            name="normalize_right_arm",
            targets=[
                PlanTarget(
                    label="示教右臂位姿",
                    flange=np.asarray(
                        robot.controller_flange_from_joints(list(target)),
                        dtype=float,
                    ),
                    goal_joints=tuple(map(float, target)),
                    goal_constraint="joints",
                )
            ],
            obstacle_points=[],
            collision_boxes=[],
            start_right_joints_deg=current,
        )
        points = [
            list(map(float, point)) for point in verified.trajectory["points_deg"]
        ]
        robot.validate_planned_joints(
            points, params.planned_joint_step_deg, profile,
            start_joints_deg=current,
        )
        robot.execute_planned_joints(
            points, cli.speed, params.planned_joint_step_deg,
            expected_start_joints_deg=current,
        )
        deadline = time.monotonic() + 2.0
        while True:
            reached = list(robot.joints_deg())
            error = float(
                np.max(np.abs(np.asarray(reached) - np.asarray(target, dtype=float)))
            )
            if (
                error <= params.planned_start_tolerance_deg
                or time.monotonic() >= deadline
            ):
                break
            time.sleep(0.05)
        print(f"\n已到达示教右臂位姿，偏差 {error:.2f}°。")
        if error > params.planned_start_tolerance_deg:
            raise SafetyAbort(f"右臂到位偏差 {error:.2f}° 超限")
    finally:
        if moveit is not None:
            moveit.close()
        left.close()
        robot.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
