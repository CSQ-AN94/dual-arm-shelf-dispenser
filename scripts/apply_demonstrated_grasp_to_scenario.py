#!/usr/bin/env python3
"""Replace a pick-only scenario's computed grasp with the taught TCP pose."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.arm import RobotSession
from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config


def select_row_candidate(
    templates: dict,
    *,
    layer_id: str,
    arm_id: str,
    target_tcp_x_m: float,
    max_x_error_m: float,
    operation: str,
    pose_semantics: str,
) -> dict:
    if (
        not isinstance(templates, dict)
        or templates.get("schema_version") != "grabber.shelf_row_templates.v1"
        or templates.get("pose_frame") != "platform_base_link"
    ):
        raise SafetyAbort("行模板格式或坐标系无效")
    if not math.isfinite(float(target_tcp_x_m)):
        raise SafetyAbort("目标 TCP X 必须是有限数")
    if not 0.005 <= float(max_x_error_m) <= 0.10:
        raise SafetyAbort("行模板最大 X 匹配误差必须在 5..100 mm")

    matches = []
    for point in templates.get("points", []):
        if not isinstance(point, dict) or point.get("layer_id") != layer_id:
            continue
        operation_data = (point.get("operations") or {}).get(operation) or {}
        if operation_data.get("pose_semantics") != pose_semantics:
            continue
        candidate = (operation_data.get("arm_candidates") or {}).get(arm_id)
        if not isinstance(candidate, dict):
            continue
        pose = candidate.get("tcp_pose_moveit") or {}
        xyz = np.asarray(pose.get("xyz"), dtype=float)
        quaternion = np.asarray(pose.get("quat_xyzw"), dtype=float)
        joints = np.asarray(candidate.get("reference_joints_deg"), dtype=float)
        if (
            xyz.shape != (3,)
            or quaternion.shape != (4,)
            or joints.shape != (7,)
            or not np.all(np.isfinite(xyz))
            or not np.all(np.isfinite(quaternion))
            or not np.all(np.isfinite(joints))
            or np.linalg.norm(quaternion) < 1e-9
        ):
            raise SafetyAbort("行模板包含无效的抓取候选")
        matches.append(
            {
                "point_id": str(point.get("point_id", "")),
                "row_position_m": float(point["row_position_m"]),
                "candidate": candidate,
                "x_error_m": abs(float(xyz[0]) - float(target_tcp_x_m)),
            }
        )
    if not matches:
        raise SafetyAbort(f"行模板没有 {layer_id}/{arm_id} 抓取候选")
    selected = min(matches, key=lambda item: item["x_error_m"])
    if selected["x_error_m"] > float(max_x_error_m):
        raise SafetyAbort(
            "目标超出已学习横向范围: "
            f"nearest_error={selected['x_error_m'] * 1000:.1f} mm > "
            f"{float(max_x_error_m) * 1000:.1f} mm"
        )
    return selected


def select_row_pick_candidate(
    templates: dict,
    *,
    layer_id: str,
    arm_id: str,
    target_tcp_x_m: float,
    max_x_error_m: float,
) -> dict:
    return select_row_candidate(
        templates,
        layer_id=layer_id,
        arm_id=arm_id,
        target_tcp_x_m=target_tcp_x_m,
        max_x_error_m=max_x_error_m,
        operation="pick",
        pose_semantics="demonstrated_grasp_curve",
    )


def build_grasp_candidates(
    pose: dict,
    taught_rotation: Rotation,
    *,
    id_prefix: str,
) -> list[dict]:
    return [
        {
            "id": f"{id_prefix}{suffix}",
            "pose": {
                "xyz": pose["xyz"],
                "quat_xyzw": (
                    taught_rotation
                    * Rotation.from_euler("y", pitch_deg, degrees=True)
                ).as_quat().tolist(),
            },
        }
        for pitch_deg, suffix in (
            (0.0, ""),
            (3.0, "_pitch_plus_3"),
            (-3.0, "_pitch_minus_3"),
        )
    ]


def level_finger_axis(taught: Rotation) -> tuple[Rotation, float]:
    matrix = taught.as_matrix()
    approach = matrix[:, 2]
    finger = np.cross(np.array([0.0, 0.0, 1.0]), approach)
    norm = float(np.linalg.norm(finger))
    if norm < 1e-6:
        raise SafetyAbort("示教接近轴接近竖直，无法定义水平夹爪")
    finger /= norm
    if float(np.dot(finger, matrix[:, 1])) < 0.0:
        finger *= -1.0
    palm = np.cross(finger, approach)
    palm /= np.linalg.norm(palm)
    correction_deg = math.degrees(
        math.asin(min(1.0, abs(float(matrix[2, 1]))))
    )
    return Rotation.from_matrix(np.column_stack((palm, finger, approach))), correction_deg


def merge_live_translation_with_taught_orientation(
    live_pose: dict, taught_rotation: Rotation
) -> dict:
    """Keep the perceived bottle position; reuse only the taught wrist attitude."""
    xyz = np.asarray(live_pose.get("xyz"), dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise SafetyAbort("实时抓取 TCP 必须是三个有限坐标")
    return {
        "xyz": xyz.tolist(),
        "quat_xyzw": taught_rotation.as_quat().tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--row-templates", type=Path)
    parser.add_argument("--layer", choices=("upper", "lower"))
    parser.add_argument("--max-template-x-error-m", type=float, default=0.04)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    cli = parser.parse_args()

    scenario = yaml.safe_load(cli.scenario.read_text(encoding="utf-8"))
    if (
        not isinstance(scenario, dict)
        or scenario.get("mode") != "pick_only"
        or scenario.get("planning_arm_id") != "right_arm"
    ):
        raise SafetyAbort("示教抓取只接受 right_arm pick_only 场景")

    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    row_provenance = None
    if cli.row_templates is not None:
        if cli.layer is None:
            raise SafetyAbort("使用 --row-templates 时必须指定 --layer")
        template_bytes = cli.row_templates.read_bytes()
        templates = json.loads(template_bytes)
        target_x = float(scenario["source_grasp_pose"]["xyz"][0])
        selected = select_row_pick_candidate(
            templates,
            layer_id=cli.layer,
            arm_id="right_arm",
            target_tcp_x_m=target_x,
            max_x_error_m=cli.max_template_x_error_m,
        )
        candidate = selected["candidate"]
        original_grasp_xyz = np.asarray(
            scenario["source_grasp_pose"]["xyz"], dtype=float
        )
        template_grasp_xyz = np.asarray(
            candidate["tcp_pose_moveit"]["xyz"], dtype=float
        )
        raw_rotation = Rotation.from_quat(
            candidate["tcp_pose_moveit"]["quat_xyzw"]
        )
        taught_rotation, roll_correction_deg = level_finger_axis(raw_rotation)
        pose = merge_live_translation_with_taught_orientation(
            scenario["source_grasp_pose"], taught_rotation
        )
        joints = list(map(float, candidate["reference_joints_deg"]))
        profile_point = (
            np.linalg.inv(profile.T_moveit_from_profile)
            @ np.r_[pose["xyz"], 1.0]
        )[:3]
        profile.assert_tcp_point(profile_point, label="学习行抓取 TCP")
        row_provenance = {
            "template_sha256": hashlib.sha256(template_bytes).hexdigest(),
            "layer_id": cli.layer,
            "point_id": selected["point_id"],
            "row_position_m": selected["row_position_m"],
            "detected_tcp_x_m": target_x,
            "selected_tcp_x_m": float(template_grasp_xyz[0]),
            "x_error_m": selected["x_error_m"],
            "template_tcp_xyz_m": template_grasp_xyz.tolist(),
            "ignored_template_translation_m": (
                template_grasp_xyz - original_grasp_xyz
            ).tolist(),
            "finger_roll_correction_deg": roll_correction_deg,
            "execution_verified": bool(templates.get("execution_verified")),
        }
    else:
        joints = profile.demonstrated_grasp_right_joints_deg
        if joints is None:
            raise SafetyAbort("shelf_template 缺少示教抓取关节")
        cfg = load_config(cli.config)
        params = DemoParams()
        link7_to_flange, flange_to_tcp = (
            profile.tool_mount_calibration.require_transforms()
        )
        robot = RobotSession(
            cfg.connections.right_arm_ip,
            cfg.connections.arm_port,
            threading.Event(),
            params.tcp_z_m,
            params.moveit_link7_to_controller_flange_m,
            take_control=False,
            tcp_transform=flange_to_tcp,
            link7_to_controller_flange=link7_to_flange,
        )
        try:
            taught_profile = robot.tcp_from_joints(joints)
            taught = profile.pose_to_moveit(taught_profile)
        finally:
            robot.close()
        profile.assert_tcp_point(
            taught_profile[:3, 3], label="示教抓取 TCP"
        )
        taught_rotation = Rotation.from_matrix(taught[:3, :3])
        pose = {
            "xyz": taught[:3, 3].tolist(),
            "quat_xyzw": taught_rotation.as_quat().tolist(),
        }
    scenario["source_grasp_pose"] = pose
    candidate_prefix = (
        f"learned_row_{row_provenance['point_id']}"
        if row_provenance is not None
        else "demonstrated_body_grasp"
    )
    scenario["source_grasp_candidates"] = build_grasp_candidates(
        pose,
        taught_rotation,
        id_prefix=candidate_prefix,
    )
    scenario["source_grasp_reference_joints_deg"] = list(map(float, joints))
    if row_provenance is not None:
        approach = taught_rotation.as_matrix()[:, 2]
        scenario["source_approach_direction"] = approach.tolist()
        scenario["source_retreat_direction"] = (-approach).tolist()
        scenario["row_template_provenance"] = row_provenance

    output = cli.output or cli.scenario
    output.write_text(
        yaml.safe_dump(scenario, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"已写入示教抓取候选: {output}")
    print(f"MoveIt TCP: {[round(float(v), 4) for v in pose['xyz']]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SafetyAbort, OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
