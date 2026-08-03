#!/usr/bin/env python3
"""Execute the verified stationary 647→250 mm held-bottle lift transition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bottle_grasp.core import DemoParams, SafetyAbort
from bottle_grasp.mobile_body import LiftSocketAdapter, WooshChassisAdapter
from bottle_grasp.mtc_execution import (
    execute_lift_transfer,
    load_lift_transfer_contract,
)
from bottle_grasp.robot import ArmJointReader, RobotSession
from bottle_grasp.safety import load_safety_profile
from utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pick_execution_record", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "bottle_grasp/lift_transfer_647_to_250.json",
    )
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "bottle_grasp/safety_profiles.json"),
    )
    parser.add_argument("--speed", type=int, default=30)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--execute", action="store_true")
    cli = parser.parse_args(argv)
    contract = load_lift_transfer_contract(cli.contract)
    pick_record = json.loads(
        cli.pick_execution_record.read_text(encoding="utf-8")
    )
    if not cli.execute:
        print("升降安全契约和 pick 执行证据格式有效。")
        print("未指定 --execute：未连接机械臂、底盘或升降机构。")
        return 0

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
        completed = execute_lift_transfer(
            pick_record,
            contract,
            robot=robot,
            left_reader=left,
            lift=LiftSocketAdapter(
                cfg.connections.left_arm_ip, cfg.connections.arm_port
            ),
            chassis=WooshChassisAdapter(stop_event=stop_event),
            speed=cli.speed,
        )
    finally:
        left.close()
        robot.close()
    record = cli.record or cli.pick_execution_record.with_suffix(
        cli.pick_execution_record.suffix + ".lift.json"
    )
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps(
            {
                "schema_version": "grabber.mtc_lift_execution.v1",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "completion": completed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"升降执行完成: {completed}")
    print(f"执行证据已写入: {record}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SafetyAbort, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"拒绝执行升降: {exc}", file=sys.stderr)
        raise SystemExit(2)
