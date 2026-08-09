#!/usr/bin/env python3
"""Pick and cross-layer place from a taught pose, with perception out of the loop.

Fourteen consecutive autonomous attempts have failed, and the last few narrowed
the cause to the two least stable measurements in the pipeline: the shelf-face
plane fit (three frames disagreeing by 70 mm on the same panel) and the grasp
point derived from a YOLO box (30 mm off the bottle's centre laterally).  The
grasp itself is fine -- the operator reached it by hand, and MoveIt agrees that
pose is collision-free.

So this drives the pose that was taught instead of one computed per run.  What
it keeps is every check that decides whether the arm may move: MoveIt planning
against the live scene, the dense fence re-check on every trajectory point,
joint limits and margin, gripper force feedback against a measured empty-close
baseline, and the lift transfer's own gates.  What it gives up is autonomy --
move the bottle and the taught pose has to be taught again.

The cross-layer part needs no second demonstration.  The lift carries the whole
torso, so the same arm configuration at 250 mm reaches the layer below the one
it grasped from at 647 mm.

    python scripts/demonstrated_cross_layer_cycle.py             # 干跑
    python scripts/demonstrated_cross_layer_cycle.py --execute
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

from shelf_dispenser.arm import (
    ArmJointReader,
    RobotSession,
    validate_holding_gripper_feedback,
)
from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.mobile_body import LiftSocketAdapter
from shelf_dispenser.planner import MoveItPlanner
from shelf_dispenser.safe_planner import PlanTarget, SafeMotionPlanner
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config

LOG = logging.getLogger("demonstrated_cycle")

RETREAT_M = 0.085
LOWER_LIFT_MM = 250


def plan_and_move(planner, robot, params, profile, label, joints, start, speed):
    """One taught-pose move, through the same chain every other move uses."""
    verified = planner.plan(
        name=label,
        targets=[
            PlanTarget(
                label=label,
                flange=np.asarray(
                    robot.controller_flange_from_joints(list(joints)), dtype=float
                ),
                goal_joints=tuple(map(float, joints)),
                goal_constraint="joints",
            )
        ],
        obstacle_points=[],
        collision_boxes=[],
        start_right_joints_deg=start,
    )
    points = [list(map(float, p)) for p in verified.trajectory["points_deg"]]
    robot.validate_planned_joints(
        points, params.planned_joint_step_deg, profile, start_joints_deg=start
    )
    robot.execute_planned_joints(
        points, speed, params.planned_joint_step_deg,
        expected_start_joints_deg=start,
    )
    deadline = time.monotonic() + 2.0
    while True:
        reached = list(robot.joints_deg())
        error = float(np.max(np.abs(np.asarray(reached) - np.asarray(joints))))
        if error <= params.planned_start_tolerance_deg or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    LOG.info("%s 到位，偏差 %.2f°", label, error)
    if error > params.planned_start_tolerance_deg:
        raise SafetyAbort(f"{label} 偏差 {error:.2f}° 超限")
    return reached


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "demonstrated_cycle")
    )
    parser.add_argument("--speed", type=int, default=30)
    parser.add_argument("--lift-speed", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = Path(cli.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    taught = profile.demonstrated_grasp_right_joints_deg
    if not taught:
        raise SafetyAbort(
            "profile 没有 demonstrated_grasp_right_joints_deg；"
            "先跑 scripts/record_demonstrated_grasp.py --write"
        )
    start_pose = list(profile.grasp_start_right_joints_deg)
    lift = LiftSocketAdapter(cfg.connections.left_arm_ip, cfg.connections.arm_port)

    print(f"起点位姿   {' '.join(f'{v:7.1f}' for v in start_pose)}")
    print(f"示教抓取位 {' '.join(f'{v:7.1f}' for v in taught)}")
    print(f"升降 {lift.state().height_mm} mm → {LOWER_LIFT_MM} mm（跨层）")
    if not cli.execute:
        print("\n干跑：未下发任何运动。加 --execute 才会动。")
        return 0

    link7_to_flange, flange_to_tcp = (
        profile.tool_mount_calibration.require_transforms()
    )
    stop = threading.Event()
    robot = RobotSession(
        cfg.connections.right_arm_ip, cfg.connections.arm_port, stop,
        params.tcp_z_m, params.moveit_link7_to_controller_flange_m,
        take_control=True, tcp_transform=flange_to_tcp,
        link7_to_controller_flange=link7_to_flange,
    )
    left = ArmJointReader(cfg.connections.left_arm_ip, cfg.connections.arm_port)
    moveit = MoveItPlanner(project_root=ROOT, run_dir=run_dir)
    try:
        moveit.start()
        planner = SafeMotionPlanner(
            moveit=moveit, robot=robot, left_robot=left, safety=profile,
            params=params, report=lambda n, m: LOG.info("  %s: %s", n, m),
            link7_to_controller_flange=link7_to_flange,
        )

        print("\n=== 1/6  空夹基线 ===")
        robot.open_gripper(params)
        baseline = robot.calibrate_empty_close(params)
        robot.open_gripper(params)
        print(f"  空夹闭合位 {baseline}")

        print("\n=== 2/6  进入示教抓取位 ===")
        here = list(robot.joints_deg())
        plan_and_move(planner, robot, params, profile,
                      "示教抓取位", taught, here, cli.speed)

        print("\n=== 3/6  合夹爪 ===")
        robot.close_gripper(params)
        feedback = validate_holding_gripper_feedback(
            robot.gripper_state(), params, empty_close_pos=baseline
        )
        print(f"  夹持确认 {feedback}")

        print("\n=== 4/6  退出并回到起点位（持瓶）===")
        tcp = np.asarray(robot.current_tcp(), dtype=float)
        retreat = tcp.copy()
        retreat[:3, 3] = tcp[:3, 3] - tcp[:3, 2] * RETREAT_M
        profile.assert_tcp_point(retreat[:3, 3], label="退出点")
        robot.move_linear(retreat, cli.speed)
        plan_and_move(planner, robot, params, profile,
                      "回起点位", start_pose, list(robot.joints_deg()), cli.speed)

        print(f"\n=== 5/6  升降 → {LOWER_LIFT_MM} mm（持瓶跨层）===")
        after = lift.move_to(LOWER_LIFT_MM, speed=cli.lift_speed)
        print(f"  升降到 {after.height_mm} mm")
        validate_holding_gripper_feedback(
            robot.gripper_state(), params, empty_close_pos=baseline
        )

        print("\n=== 6/6  下层放置 ===")
        plan_and_move(planner, robot, params, profile,
                      "下层放置位", taught, list(robot.joints_deg()), cli.speed)
        robot.open_gripper(params)
        print("  已松开")
        tcp = np.asarray(robot.current_tcp(), dtype=float)
        retreat = tcp.copy()
        retreat[:3, 3] = tcp[:3, 3] - tcp[:3, 2] * RETREAT_M
        profile.assert_tcp_point(retreat[:3, 3], label="放置退出点")
        robot.move_linear(retreat, cli.speed)
        plan_and_move(planner, robot, params, profile,
                      "回起点位", start_pose, list(robot.joints_deg()), cli.speed)
        print("\n完整跨层抓放完成。")
    finally:
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
