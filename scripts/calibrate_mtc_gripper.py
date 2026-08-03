#!/usr/bin/env python3
"""Measure the empty-close baseline at the taught free-space carry pose."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.mobile_body import LiftSocketAdapter
from shelf_dispenser.mtc_execution import validate_hardware_preflight
from shelf_dispenser.arm import ArmJointReader, RobotSession
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser/safety_profiles.json"),
    )
    parser.add_argument(
        "--expected-lift-mm",
        type=int,
        help="可选复核值；省略时以 shelf_template 抓取初始高度为唯一来源",
    )
    parser.add_argument("--execute", action="store_true", required=True)
    cli = parser.parse_args(argv)
    cfg = load_config(cli.config)
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    expected_lift_mm = profile.grasp_start_lift_height_mm
    if expected_lift_mm is None:
        raise SafetyAbort("shelf_template 缺少抓取初始升降高度")
    if (
        cli.expected_lift_mm is not None
        and cli.expected_lift_mm != expected_lift_mm
    ):
        raise SafetyAbort(
            "--expected-lift-mm 与 shelf_template 抓取初始高度不一致"
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
    left = None
    try:
        left = ArmJointReader(
            cfg.connections.left_arm_ip, cfg.connections.arm_port
        )
        recovered = robot.recover_transient_joint_frame_loss()
        if recovered:
            print(
                "空夹标定前已清除并双读复核瞬态通信丢帧: "
                + ",".join(f"J{joint}" for joint in recovered)
            )
        validate_hardware_preflight(robot)
        actual = np.asarray(robot.joints_deg(), dtype=float)
        lift = LiftSocketAdapter(
            cfg.connections.left_arm_ip, cfg.connections.arm_port
        ).state()
        profile.assert_grasp_start(
            right_joints_deg=actual,
            left_joints_deg=left.joints_deg(),
            lift_height_mm=lift.height_mm,
            lift_mode=lift.mode,
            joint_tolerance_deg=DemoParams().planned_start_tolerance_deg,
        )
        baseline = robot.calibrate_empty_close(DemoParams())
        opened = robot.gripper_state()
    finally:
        if left is not None:
            left.close()
        robot.close()
    cli.record.parent.mkdir(parents=True, exist_ok=True)
    cli.record.write_text(
        json.dumps(
            {
                "schema_version": "grabber.gripper_calibration.v1",
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "empty_close_pos": int(baseline),
                "right_joints_deg": actual.tolist(),
                "lift_height_mm": int(lift.height_mm),
                "gripper_open_feedback": opened,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"空夹标定证据已写入: {cli.record}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SafetyAbort, OSError, ValueError, KeyError) as exc:
        print(f"拒绝空夹标定: {exc}", file=sys.stderr)
        raise SystemExit(2)
