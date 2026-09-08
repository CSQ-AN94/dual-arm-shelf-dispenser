#!/usr/bin/env python3
"""Place the currently held P01 bottle with three verified RealMan SDK movel legs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.arm import RobotSession, validate_holding_gripper_feedback
from shelf_dispenser.core import DemoParams, SafetyAbort, interpolate_poses, matrix_pose, pose_matrix
from shelf_dispenser.relative_place import product_geometry
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config


TABLE_BOUNDS = (-0.430, 0.250, 0.488, 0.888)
DEFAULT_TABLE_Z_M = -0.3105  # 2026-08-13 RGB-D patch at x=0.10, y=0.64.


def wait_for_stable_joints(robot, timeout_s: float = 6.0) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    window = []
    while time.monotonic() < deadline:
        joints = np.asarray(robot.joints_deg(), float)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise SafetyAbort("机械臂稳定性采样返回无效关节值")
        window.append(joints)
        window = window[-5:]
        if len(window) == 5 and float(np.max(np.ptp(window, axis=0))) <= 0.05:
            return joints
        time.sleep(0.25)
    raise SafetyAbort("机械臂在 6 秒内没有稳定，拒绝按瞬时起点规划")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--execute", action="store_true")
    result.add_argument("--home-only", action="store_true")
    result.add_argument("--config", default=str(ROOT / "config.yaml"))
    result.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser/safety_profiles.json"),
    )
    result.add_argument(
        "--pick-record",
        default="/home/rm/pick_p01_20260813_173304/pick_record.json",
    )
    result.add_argument("--target-x", type=float, default=0.10)
    result.add_argument("--target-y", type=float, default=0.60)
    result.add_argument("--table-z", type=float, default=DEFAULT_TABLE_Z_M)
    result.add_argument("--clearance", type=float, default=0.04)
    result.add_argument("--release-gap", type=float, default=0.002)
    result.add_argument("--speed", type=int, default=5)
    result.add_argument(
        "--output-dir",
        default="/home/rm/pick_p01_20260813_173304/direct_sdk_place",
    )
    return result


def held_transform(pick_record: dict) -> np.ndarray:
    completion = pick_record.get("completion")
    if not isinstance(completion, dict):
        raise SafetyAbort("抓取记录缺少 completion")
    final_pose = np.asarray(completion.get("final_tcp_base_xyz_rpy_rad"), float)
    offset = np.asarray(completion.get("bottle_center_in_tcp_m"), float)
    if final_pose.shape != (6,) or offset.shape != (3,):
        raise SafetyAbort("抓取记录缺少有效的 TCP/瓶心变换")
    transform = np.eye(4)
    transform[:3, :3] = pose_matrix(final_pose)[:3, :3].T
    transform[:3, 3] = offset
    return transform


def direct_place_geometry(
    current_tcp: np.ndarray,
    tcp_from_bottle: np.ndarray,
    *,
    target_x: float,
    target_y: float,
    table_z: float,
    clearance: float,
    release_gap: float,
) -> dict:
    geometry = product_geometry("P01")
    radius = float(geometry["radius_m"])
    height = float(geometry["height_m"])
    x_min, x_max, y_near, y_far = TABLE_BOUNDS
    edge_margin = 0.06
    if not (
        x_min + radius + edge_margin <= target_x <= x_max - radius - edge_margin
        and y_near + radius + edge_margin <= target_y <= y_far - radius - edge_margin
    ):
        raise SafetyAbort("放置点没有给瓶底和桌沿各留 60mm 净空")

    current_bottle = current_tcp @ tcp_from_bottle
    axis = current_bottle[:3, 2]
    tilt_deg = math.degrees(math.acos(float(np.clip(axis[2], -1.0, 1.0))))
    if tilt_deg > 8.0:
        raise SafetyAbort(f"当前瓶子倾斜 {tilt_deg:.1f}°，不适合保持姿态直放")
    contact_error = abs(float(current_bottle[1, 3]) + radius - y_near)
    if contact_error > 0.02:
        raise SafetyAbort(
            f"瓶边与已知桌近边不一致: error={contact_error * 1000:.1f}mm"
        )
    travel_xy = float(
        np.linalg.norm(current_bottle[:2, 3] - [target_x, target_y])
    )
    if travel_xy > 0.25:
        raise SafetyAbort(f"SDK 直放水平移动过长: {travel_xy:.3f}m")

    support = height / 2.0 * abs(float(axis[2])) + radius * float(
        np.linalg.norm(axis[:2])
    )
    final_bottle = current_bottle.copy()
    final_bottle[:3, 3] = [target_x, target_y, table_z + support + release_gap]
    final_tcp = final_bottle @ np.linalg.inv(tcp_from_bottle)
    preplace_tcp = final_tcp.copy()
    preplace_tcp[2, 3] += clearance
    lift_tcp = current_tcp.copy()
    lift_tcp[2, 3] += clearance

    sampled = []
    for start, goal in (
        (current_tcp, lift_tcp),
        (lift_tcp, preplace_tcp),
        (preplace_tcp, final_tcp),
    ):
        sampled.extend(
            interpolate_poses(matrix_pose(start), matrix_pose(goal), 0.02)
        )
    return {
        "current_bottle": current_bottle,
        "final_bottle": final_bottle,
        "command_tcps": (lift_tcp, preplace_tcp, final_tcp),
        "sampled_poses": sampled,
        "tilt_deg": tilt_deg,
        "contact_error_m": contact_error,
        "travel_xy_m": travel_xy,
        "support_m": support,
    }


def main() -> int:
    args = parser().parse_args()
    if args.home_only and not args.execute:
        raise SystemExit("--home-only 必须与 --execute 一起使用")
    if not (1 <= args.speed <= 20):
        raise SystemExit("--speed 必须在 1..20")
    if not (0.02 <= args.clearance <= 0.08):
        raise SystemExit("--clearance 必须在 0.02..0.08m")
    if not (0.0 <= args.release_gap <= 0.008):
        raise SystemExit("--release-gap 必须在 0..0.008m")

    config = load_config(args.config)
    safety = load_safety_profile(
        args.safety_config,
        "side_table_template",
        require_verified=True,
    )
    link7_flange, flange_tcp = safety.tool_mount_calibration.require_transforms()
    params = replace(DemoParams(), gripper_speed=min(5, args.speed))
    record = json.loads(Path(args.pick_record).read_text(encoding="utf-8"))
    completed_at = datetime.fromisoformat(
        str(record["completed_at_utc"]).replace("Z", "+00:00")
    )
    age_s = (datetime.now(timezone.utc) - completed_at).total_seconds()
    if age_s < 0.0 or (age_s > 7200.0 and not args.home_only):
        raise SafetyAbort(f"抓取记录年龄超出本次续放上限: {age_s:.0f}s")
    empty_close_pos = int(record["completion"]["empty_close_pos"])
    tcp_from_bottle = held_transform(record)

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    robot = RobotSession(
        config.connections.right_arm_ip,
        config.connections.arm_port,
        stop_event,
        params.tcp_z_m,
        model_flange_offset_m=params.moveit_link7_to_controller_flange_m,
        take_control=args.execute,
        tcp_transform=flange_tcp,
        link7_to_controller_flange=link7_flange,
    )
    released = False
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_direct_sdk_place.json"
    )
    result = {"status": "fault", "bottle_released": False}
    try:
        joints = wait_for_stable_joints(robot)
        expected = np.asarray(safety.side_table_delivery.transport_joints_deg, float)
        tuck_error = float(np.max(np.abs(joints - expected)))
        feedback = validate_holding_gripper_feedback(
            robot.gripper_state(), params, empty_close_pos=empty_close_pos
        )
        robot.empty_close_pos = empty_close_pos
        home_recovery = None
        if tuck_error > params.planned_start_tolerance_deg:
            if not args.execute:
                raise SafetyAbort(f"右臂不在持瓶收拢位: error={tuck_error:.2f}°")
            clear_rc = robot.arm.rm_clear_system_err()
            if clear_rc != 0:
                raise SafetyAbort(f"清除上次运动的一般报警失败: rc={clear_rc}")
            time.sleep(0.5)
            recovered = robot.recover_transient_joint_frame_loss()
            robot.assert_arm_healthy()
            robot.validate_planned_joints(
                [expected.tolist()],
                params.planned_joint_step_deg,
                safety,
                start_joints_deg=joints,
            )
            recovery_start = joints.copy()
            robot.execute_planned_joints(
                [expected.tolist()],
                args.speed,
                params.planned_joint_step_deg,
                expected_start_joints_deg=recovery_start,
            )
            joints = np.asarray(robot.joints_deg(), float)
            recovery_error = float(np.max(np.abs(joints - expected)))
            if recovery_error > params.planned_start_tolerance_deg:
                raise SafetyAbort(f"回到持瓶 home 后反馈超差: {recovery_error:.2f}°")
            feedback = validate_holding_gripper_feedback(
                robot.gripper_state(), params, empty_close_pos=empty_close_pos
            )
            home_recovery = {
                "start_joints_deg": recovery_start.tolist(),
                "target_joints_deg": expected.tolist(),
                "arrival_error_deg": recovery_error,
                "recovered_transient_joint_frame_loss": recovered,
            }
            tuck_error = recovery_error
        if args.home_only:
            result = {
                "status": "home_verified",
                "bottle_released": False,
                "home_recovery": home_recovery,
                "home_joints_deg": joints.tolist(),
                "home_error_deg": tuck_error,
                "gripper_feedback": feedback,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return 0
        current_tcp = (
            robot.current_tcp() if args.execute else robot.tcp_from_joints(joints)
        )
        geometry = direct_place_geometry(
            current_tcp,
            tcp_from_bottle,
            target_x=args.target_x,
            target_y=args.target_y,
            table_z=args.table_z,
            clearance=args.clearance,
            release_gap=args.release_gap,
        )
        safety.assert_tcp_path(pose[:3] for pose in geometry["sampled_poses"])
        ik = robot.plan_ik(geometry["sampled_poses"], params)
        robot.validate_planned_joints(
            ik,
            params.planned_joint_step_deg,
            safety,
            start_joints_deg=joints,
        )
        retreat_tcp = geometry["command_tcps"][-1].copy()
        retreat_tcp[2, 3] += 0.06
        retreat_poses = interpolate_poses(
            matrix_pose(geometry["command_tcps"][-1]),
            matrix_pose(retreat_tcp),
            0.02,
        )
        retreat_ik = robot.plan_ik(
            retreat_poses,
            params,
            seed_joints_deg=ik[-1],
        )
        robot.validate_planned_joints(
            retreat_ik,
            params.planned_joint_step_deg,
            safety,
            start_joints_deg=ik[-1],
        )
        result = {
            "status": "checked" if not args.execute else "executing",
            "bottle_released": False,
            "planner": "RealMan SDK IK + rm_movej only; no MoveIt",
            "target_bottle_center_base_m": geometry["final_bottle"][:3, 3].tolist(),
            "table_z_base_m": args.table_z,
            "bottle_tilt_deg": geometry["tilt_deg"],
            "near_edge_contact_error_m": geometry["contact_error_m"],
            "horizontal_travel_m": geometry["travel_xy_m"],
            "tuck_error_deg": tuck_error,
            "gripper_feedback_before": feedback,
            "home_recovery": home_recovery,
            "sdk_ik_waypoints": len(ik),
            "sdk_ik_min_abs_j4_deg": min(abs(float(point[3])) for point in ik),
            "sdk_ik_final_joints_deg": list(map(float, ik[-1])),
            "command_tcp_xyz_rpy": [
                matrix_pose(target) for target in geometry["command_tcps"]
            ],
        }
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if not args.execute:
            return 0

        recovered = robot.recover_transient_joint_frame_loss()
        result["recovered_transient_joint_frame_loss"] = recovered
        robot.assert_arm_healthy()
        validate_holding_gripper_feedback(
            robot.gripper_state(), params, empty_close_pos=empty_close_pos
        )
        robot.execute_planned_joints(
            ik,
            args.speed,
            params.planned_joint_step_deg,
            expected_start_joints_deg=joints,
        )
        place_actual = matrix_pose(robot.current_tcp())
        robot.open_gripper(params)
        released = True
        robot.execute_planned_joints(
            retreat_ik,
            args.speed,
            params.planned_joint_step_deg,
            expected_start_joints_deg=ik[-1],
        )
        result.update(
            {
                "status": "success",
                "bottle_released": True,
                "place_actual": place_actual,
                "retreat_actual": matrix_pose(robot.current_tcp()),
            }
        )
        return 0
    except Exception as exc:
        result.update(
            {
                "status": "safe_abort" if isinstance(exc, SafetyAbort) else "fault",
                "bottle_released": released,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 2
    finally:
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
