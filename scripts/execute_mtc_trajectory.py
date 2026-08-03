#!/usr/bin/env python3
"""Validate, then explicitly execute one live MTC pick-only or place-only export."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bottle_grasp.core import DemoParams, SafetyAbort
from bottle_grasp.mobile_body import LiftSocketAdapter
from bottle_grasp.mtc_execution import (
    execute_pick,
    execute_place,
    load_gripper_calibration_record,
)
from bottle_grasp.mtc_pick_contract import (
    load_pick_result,
    load_pick_scenario,
    load_pick_trajectory,
    load_place_trajectory,
    validate_execution_bundle,
    validate_place_execution_bundle,
)
from bottle_grasp.robot import ArmJointReader, RobotSession
from bottle_grasp.safety import load_safety_profile
from utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pick", "place"))
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "bottle_grasp" / "safety_profiles.json"),
    )
    parser.add_argument("--speed", type=int, default=100)
    parser.add_argument("--gripper-calibration-record", type=Path)
    parser.add_argument(
        "--record",
        type=Path,
        help="成功后执行证据 JSON；默认写在轨迹文件旁",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Connect hardware and execute; absent means artifact validation only",
    )
    parser.add_argument(
        "--allow-sdk-retiming",
        action="store_true",
        help=(
            "明确接受 RealMan SDK movej 重定时；不会保留 MTC/Pilz 的时间、"
            "速度或加速度"
        ),
    )
    cli = parser.parse_args(argv)
    if not 1 <= cli.speed <= 100:
        parser.error("--speed 必须在 1..100")

    result = load_pick_result(cli.result)
    scenario = load_pick_scenario(cli.scenario)
    trajectory = (
        load_pick_trajectory(cli.trajectory)
        if cli.mode == "pick"
        else load_place_trajectory(cli.trajectory)
    )
    validator = (
        validate_execution_bundle
        if cli.mode == "pick"
        else validate_place_execution_bundle
    )
    summary = validator(result, trajectory, scenario)
    if not cli.execute:
        print(f"{cli.mode} 三件套契约有效: {summary}")
        print("未指定 --execute：未连接机械臂、夹爪或升降机构。")
        return 0
    if not cli.allow_sdk_retiming:
        parser.error(
            "--execute 还必须显式提供 --allow-sdk-retiming；当前 SDK 桥不保留 "
            "MTC/Pilz timing"
        )
    calibration = None
    if cli.mode == "pick":
        if cli.gripper_calibration_record is None:
            parser.error("pick --execute 必须提供 --gripper-calibration-record")
        calibration = load_gripper_calibration_record(
            cli.gripper_calibration_record
        )

    cfg = load_config(cli.config)
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    link7_to_flange, flange_to_tcp = (
        profile.tool_mount_calibration.require_transforms()
    )
    stop_event = threading.Event()

    def stop(*_args) -> None:
        setattr(stop_event, "source", "signal")
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    robot = RobotSession(
        cfg.connections.right_arm_ip,
        cfg.connections.arm_port,
        stop_event,
        DemoParams().tcp_z_m,
        DemoParams().moveit_link7_to_controller_flange_m,
        take_control=True,
        tcp_transform=flange_to_tcp,
        link7_to_controller_flange=link7_to_flange,
    )
    left = ArmJointReader(
        cfg.connections.left_arm_ip, cfg.connections.arm_port
    )
    try:
        lift_state = LiftSocketAdapter(
            cfg.connections.left_arm_ip, cfg.connections.arm_port
        ).state()
        if cli.mode == "pick":
            profile.assert_grasp_start(
                right_joints_deg=robot.joints_deg(),
                left_joints_deg=left.joints_deg(),
                lift_height_mm=lift_state.height_mm,
                lift_mode=lift_state.mode,
                joint_tolerance_deg=(
                    DemoParams().planned_start_tolerance_deg
                ),
            )
        run = execute_pick if cli.mode == "pick" else execute_place
        completed = run(
            result,
            trajectory,
            scenario,
            robot=robot,
            left_reader=left,
            lift_state=lift_state,
            safety_profile=profile,
            speed=cli.speed,
            allow_sdk_retiming=True,
            **(
                {"empty_close_pos": int(calibration["empty_close_pos"])}
                if calibration is not None
                else {}
            ),
        )
    finally:
        left.close()
        robot.close()
    record = cli.record or cli.trajectory.with_suffix(
        cli.trajectory.suffix + ".execution.json"
    )
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps(
            {
                "schema_version": "grabber.mtc_execution.v1",
                "mode": cli.mode,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "speed_percent": cli.speed,
                "execution_backend": "realman_sdk_movej_retimed",
                "trajectory_timing_preserved": False,
                "result_path": str(cli.result.resolve()),
                "trajectory_path": str(cli.trajectory.resolve()),
                "scenario_path": str(cli.scenario.resolve()),
                "completion": completed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{cli.mode} 分阶段执行完成: {completed}")
    print(f"执行证据已写入: {record}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        raise SystemExit(main())
    except (SafetyAbort, OSError, ValueError, KeyError) as exc:
        print(f"拒绝执行 MTC 轨迹: {exc}", file=sys.stderr)
        raise SystemExit(2)
