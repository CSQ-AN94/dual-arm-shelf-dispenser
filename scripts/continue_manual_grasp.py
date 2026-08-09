#!/usr/bin/env python3
"""Continue from an operator-confirmed manual grasp to the carry pose."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.arm import validate_holding_gripper_feedback
from shelf_dispenser.core import DemoParams, SafetyAbort, matrix_pose
from shelf_dispenser.mobile_body import LiftSocketAdapter
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config


def build_record(
    *,
    before_right: list[float],
    before_left: list[float],
    before_tcp: list[float],
    reached_right: list[float],
    tuck_error_deg: float,
    lift_height_mm: int,
    empty_close_pos: int,
    bottle_center_in_tcp_m: list[float],
    gripper_feedback: dict,
) -> dict:
    return {
        "schema_version": "grabber.mtc_execution.v1",
        "mode": "pick",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_backend": "operator_manual_grasp_then_moveit_tuck",
        "completion": {
            "scenario_id": "operator_confirmed_manual_grasp",
            "grasp_candidate_id": "operator_confirmed_manual_grasp",
            "right_start_deg": before_right,
            "left_start_deg": before_left,
            "lift_start_mm": int(lift_height_mm),
            "empty_close_pos": int(empty_close_pos),
            "final_right_joints_deg": before_right,
            "final_tcp_base_xyz_rpy_rad": before_tcp,
            "gripper_close_feedback": gripper_feedback,
            "bottle_center_in_tcp_m": bottle_center_in_tcp_m,
            "manual_confirmation": True,
            "post_pick_tuck": {
                "right_joints_deg": reached_right,
                "max_error_deg": float(tuck_error_deg),
                "recorded_at": datetime.now(timezone.utc).timestamp(),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument(
        "--bottle-center-in-tcp-m",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--empty-close-pos", type=int, required=True)
    parser.add_argument(
        "--expected-lift-mm",
        type=int,
        help="Expected stationary lift height; defaults to the taught pick height",
    )
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "manual_grasp_resume")
    )
    parser.add_argument("--port", type=int, default=8886)
    parser.add_argument("--execute", action="store_true")
    cli = parser.parse_args(argv)

    params = DemoParams()
    offset = np.asarray(cli.bottle_center_in_tcp_m, dtype=float)
    if (
        offset.shape != (3,)
        or not np.all(np.isfinite(offset))
        or float(np.linalg.norm(offset)) > 0.25
    ):
        raise SafetyAbort("瓶心相对 TCP 偏移无效")
    if not 0 <= cli.empty_close_pos <= params.gripper_open_position:
        raise SafetyAbort("空夹基线无效")
    if cli.record.exists():
        raise SafetyAbort(f"拒绝覆盖已有执行证据: {cli.record}")

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(cli.config)
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    demo = RunOrchestrator(
        SimpleNamespace(
            task_mode=None,
            execute=bool(cli.execute),
            plan_only=not cli.execute,
            config=cli.config,
            safety_config=cli.safety_config,
            safety_profile="shelf_template",
            stop_after_observation=False,
            confirm_before_grasp=False,
            place_back=False,
            return_home=False,
            resume_at_wrist=False,
            finish_from_current=True,
            host="127.0.0.1",
            port=cli.port,
            output_dir=cli.output_dir,
            observe_seconds=0.0,
        ),
        cfg,
    )
    lift = LiftSocketAdapter(
        cfg.connections.left_arm_ip, cfg.connections.arm_port
    )
    try:
        demo.initialize()
        lift_state = lift.state()
        expected_height = int(
            profile.grasp_start_lift_height_mm
            if cli.expected_lift_mm is None
            else cli.expected_lift_mm
        )
        if not 0 <= expected_height <= 2600:
            raise SafetyAbort("人工抓取接续的预期升降高度无效")
        if lift_state.mode != 0 or abs(lift_state.height_mm - expected_height) > 5:
            raise SafetyAbort(
                "人工抓取接续时升降不在静止抓取高度: "
                f"actual={lift_state.height_mm} mode={lift_state.mode}, "
                f"expected={expected_height}"
            )
        before_right = list(map(float, demo.robot.joints_deg()))
        before_left = list(map(float, demo.left_robot.joints_deg()))
        before_tcp_matrix = np.asarray(
            demo.robot.current_tcp()
            if cli.execute
            else demo.robot.tcp_from_joints(before_right),
            dtype=float,
        )
        feedback = validate_holding_gripper_feedback(
            demo.robot.gripper_state(),
            params,
            empty_close_pos=cli.empty_close_pos,
        )
        bottle_center = (
            before_tcp_matrix[:3, 3] + before_tcp_matrix[:3, :3] @ offset
        )
        profile.assert_tcp_point(bottle_center, label="人工确认的持瓶瓶心")
        if cli.execute:
            demo._set_held_bottle_guard(SimpleNamespace(point_base=bottle_center))
        print(
            "持瓶门禁通过；瓶体碰撞包络已附着。"
            f"当前右臂距收拢位最大关节差 "
            f"{np.max(np.abs(np.asarray(before_right) - np.asarray(profile.grasp_start_right_joints_deg))):.2f}°"
        )
        if not cli.execute:
            print("干跑：未发送运动命令。")
            return 0

        tuck_error = float(demo.normalize_to_grasp_start())
        reached = list(map(float, demo.robot.joints_deg()))
        validate_holding_gripper_feedback(
            demo.robot.gripper_state(),
            params,
            empty_close_pos=cli.empty_close_pos,
        )
        record = build_record(
            before_right=before_right,
            before_left=before_left,
            before_tcp=matrix_pose(before_tcp_matrix),
            reached_right=reached,
            tuck_error_deg=tuck_error,
            lift_height_mm=lift_state.height_mm,
            empty_close_pos=cli.empty_close_pos,
            bottle_center_in_tcp_m=offset.tolist(),
            gripper_feedback=feedback,
        )
        cli.record.parent.mkdir(parents=True, exist_ok=True)
        cli.record.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"持瓶收拢完成，执行证据已写入: {cli.record}")
        return 0
    finally:
        demo.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SafetyAbort, OSError, ValueError) as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
