#!/usr/bin/env python3
"""Record two-arm shelf-row anchors and expand them to a 1 cm pose map.

The tool never moves either arm.  ``record --live`` reads a settled joint
state and derives its TCP pose; ``generate`` interpolates TCP poses only.
Generated points are deliberately non-executable until IK, collision, fence,
and gripper-clearance validation have been recorded elsewhere.

Typical two-layer setup::

    # Recommended: one guided session, Enter records the current pose.
    python scripts/shelf_row_template_tool.py teach outputs/row_anchors.json

Low-level commands, useful for automation::

    python scripts/shelf_row_template_tool.py init row_anchors.json \
      --row-start-m 0.00 --row-end-m 0.60 --spacing-m 0.01 \
      --overlap-start-m 0.25 --overlap-end-m 0.35 \
      --layer upper:647 --layer lower:250

    python scripts/shelf_row_template_tool.py record row_anchors.json \
      --layer upper --position-m 0.00 --arm left_arm \
      --operation both --live

    python scripts/shelf_row_template_tool.py generate row_anchors.json \
      row_templates.json
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.live_arm import open_live_arm
from shelf_dispenser.mobile_body import LiftSocketAdapter
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config


ANCHOR_SCHEMA = "grabber.shelf_row_anchors.v1"
TEMPLATE_SCHEMA = "grabber.shelf_row_templates.v1"
ARMS = ("left_arm", "right_arm")
OPERATIONS = ("pick", "place")


def _finite_vector(value, size: int, label: str) -> list[float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise SafetyAbort(f"{label} 必须是 {size} 个有限数")
    return [float(item) for item in array]


def _finite_float(value, label: str) -> float:
    if isinstance(value, bool):
        raise SafetyAbort(f"{label} 必须是有限数")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SafetyAbort(f"{label} 必须是有限数") from exc
    if not math.isfinite(result):
        raise SafetyAbort(f"{label} 必须是有限数")
    return result


def _empty_arm_anchors() -> dict:
    return {operation: [] for operation in OPERATIONS}


def create_anchor_map(
    *,
    row_start_m: float,
    row_end_m: float,
    spacing_m: float,
    overlap_start_m: float,
    overlap_end_m: float,
    layers: Sequence[tuple[str, int]],
) -> dict:
    start = _finite_float(row_start_m, "row_start_m")
    end = _finite_float(row_end_m, "row_end_m")
    spacing = _finite_float(spacing_m, "spacing_m")
    overlap_start = _finite_float(overlap_start_m, "overlap_start_m")
    overlap_end = _finite_float(overlap_end_m, "overlap_end_m")
    if not start < end:
        raise SafetyAbort("row_start_m 必须小于 row_end_m")
    if not 0.001 <= spacing <= 0.10:
        raise SafetyAbort("spacing_m 必须在 0.001..0.10 m")
    steps = round((end - start) / spacing)
    if steps < 1 or not math.isclose(
        start + steps * spacing, end, abs_tol=1e-9, rel_tol=0.0
    ):
        raise SafetyAbort("整行宽度必须能被 spacing_m 整除")
    if not start < overlap_start <= overlap_end < end:
        raise SafetyAbort(
            "左右臂重叠带必须位于整行内部且起点不大于终点"
        )
    if not layers:
        raise SafetyAbort("至少需要配置一层")

    layer_payload = {}
    for raw_name, raw_height in layers:
        name = str(raw_name).strip()
        if not name or name in layer_payload:
            raise SafetyAbort(f"层 ID 为空或重复: {raw_name!r}")
        if isinstance(raw_height, bool) or not 0 <= int(raw_height) <= 2600:
            raise SafetyAbort(f"层 {name} 的升降高度必须在 0..2600 mm")
        layer_payload[name] = {
            "lift_height_mm": int(raw_height),
            "anchors": {arm: _empty_arm_anchors() for arm in ARMS},
        }
    return {
        "schema_version": ANCHOR_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pose_frame": "platform_base_link",
        "row_start_m": start,
        "row_end_m": end,
        "spacing_m": spacing,
        "arm_assignment": {
            "overlap_start_m": overlap_start,
            "overlap_end_m": overlap_end,
            "left_arm_max_m": overlap_end,
            "right_arm_min_m": overlap_start,
            "overlap_selection": "runtime_plan_both_choose_valid_best",
        },
        "gripper_contract": {
            "included_in_tcp_pose": False,
            "status": "separate_clearance_validation_required",
        },
        "layers": layer_payload,
    }


def _validate_map(payload: dict) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != ANCHOR_SCHEMA:
        raise SafetyAbort(f"锚点文件必须使用 {ANCHOR_SCHEMA}")
    if payload.get("pose_frame") != "platform_base_link":
        raise SafetyAbort("锚点位姿必须统一存储在 platform_base_link")
    start = _finite_float(payload.get("row_start_m"), "row_start_m")
    end = _finite_float(payload.get("row_end_m"), "row_end_m")
    spacing = _finite_float(payload.get("spacing_m"), "spacing_m")
    assignment = payload.get("arm_assignment")
    if not isinstance(assignment, dict):
        raise SafetyAbort("锚点文件缺少 arm_assignment")
    overlap_start = _finite_float(
        assignment.get("overlap_start_m"), "overlap_start_m"
    )
    overlap_end = _finite_float(
        assignment.get("overlap_end_m"), "overlap_end_m"
    )
    if not start < overlap_start <= overlap_end < end or spacing <= 0.0:
        raise SafetyAbort("整行范围、间距或左右臂分界无效")
    if (
        assignment.get("left_arm_max_m") != overlap_end
        or assignment.get("right_arm_min_m") != overlap_start
        or assignment.get("overlap_selection")
        != "runtime_plan_both_choose_valid_best"
    ):
        raise SafetyAbort("左右臂分区契约无效")
    gripper = payload.get("gripper_contract")
    if (
        not isinstance(gripper, dict)
        or gripper.get("included_in_tcp_pose") is not False
    ):
        raise SafetyAbort("夹爪宽度不得写入 TCP 位姿")
    layers = payload.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise SafetyAbort("锚点文件没有层")
    for layer_id, layer in layers.items():
        if not isinstance(layer, dict):
            raise SafetyAbort(f"层 {layer_id} 必须是对象")
        height = layer.get("lift_height_mm")
        if (
            isinstance(height, bool)
            or not isinstance(height, int)
            or not 0 <= height <= 2600
        ):
            raise SafetyAbort(f"层 {layer_id} 的升降高度无效")
        anchors = layer.get("anchors")
        if not isinstance(anchors, dict):
            raise SafetyAbort(f"层 {layer_id} 缺少 anchors")
        for arm in ARMS:
            by_operation = anchors.get(arm)
            if not isinstance(by_operation, dict):
                raise SafetyAbort(f"层 {layer_id} 缺少 {arm} 锚点")
            for operation in OPERATIONS:
                if not isinstance(by_operation.get(operation), list):
                    raise SafetyAbort(
                        f"层 {layer_id} 的 {arm}/{operation} 锚点必须是列表"
                    )


def _arms_for_position(payload: dict, position_m: float) -> tuple[str, ...]:
    assignment = payload["arm_assignment"]
    overlap_start = float(assignment["overlap_start_m"])
    overlap_end = float(assignment["overlap_end_m"])
    if position_m < overlap_start:
        return ("left_arm",)
    if position_m > overlap_end:
        return ("right_arm",)
    return ARMS


def add_anchor(
    payload: dict,
    *,
    layer_id: str,
    position_m: float,
    arm_id: str,
    operations: Sequence[str],
    xyz_moveit: Sequence[float],
    quat_xyzw_moveit: Sequence[float],
    reference_joints_deg: Sequence[float],
    source: str,
    replace_existing: bool = False,
) -> None:
    _validate_map(payload)
    if layer_id not in payload["layers"]:
        raise SafetyAbort(f"未知层: {layer_id}")
    position = _finite_float(position_m, "position_m")
    if not float(payload["row_start_m"]) <= position <= float(payload["row_end_m"]):
        raise SafetyAbort("锚点位置超出整行范围")
    eligible_arms = _arms_for_position(payload, position)
    if arm_id not in eligible_arms:
        raise SafetyAbort(
            f"位置 {position:.3f} m 只允许 {','.join(eligible_arms)}，"
            f"不能录入 {arm_id}"
        )
    requested = tuple(dict.fromkeys(operations))
    if not requested or any(item not in OPERATIONS for item in requested):
        raise SafetyAbort("operation 只允许 pick/place/both")
    xyz = _finite_vector(xyz_moveit, 3, "TCP xyz")
    quaternion = np.asarray(
        _finite_vector(quat_xyzw_moveit, 4, "TCP quaternion"), dtype=float
    )
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise SafetyAbort("TCP quaternion 不能为零")
    quaternion /= norm
    joints = _finite_vector(reference_joints_deg, 7, "参考关节角")
    anchor = {
        "position_m": position,
        "tcp_pose_moveit": {
            "xyz": xyz,
            "quat_xyzw": quaternion.tolist(),
        },
        "reference_joints_deg": joints,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
    }
    for operation in requested:
        target = payload["layers"][layer_id]["anchors"][arm_id][operation]
        duplicate = next(
            (
                index
                for index, item in enumerate(target)
                if math.isclose(
                    float(item.get("position_m")), position, abs_tol=1e-9
                )
            ),
            None,
        )
        if duplicate is not None and not replace_existing:
            raise SafetyAbort(
                f"{layer_id}/{arm_id}/{operation} 已有 {position:.3f} m 锚点；"
                "使用 --replace 才能覆盖"
            )
        if duplicate is None:
            target.append(dict(anchor))
        else:
            target[duplicate] = dict(anchor)
        target.sort(key=lambda item: float(item["position_m"]))


def _validated_anchors(raw: list, label: str) -> list[dict]:
    anchors = []
    positions = set()
    for item in raw:
        if not isinstance(item, dict):
            raise SafetyAbort(f"{label} 锚点必须是对象")
        position = _finite_float(item.get("position_m"), f"{label} position_m")
        key = round(position, 9)
        if key in positions:
            raise SafetyAbort(f"{label} 有重复锚点 {position:.3f} m")
        positions.add(key)
        pose = item.get("tcp_pose_moveit")
        if not isinstance(pose, dict):
            raise SafetyAbort(f"{label} 缺少 tcp_pose_moveit")
        xyz = _finite_vector(pose.get("xyz"), 3, f"{label} xyz")
        quat = np.asarray(
            _finite_vector(pose.get("quat_xyzw"), 4, f"{label} quaternion"),
            dtype=float,
        )
        norm = float(np.linalg.norm(quat))
        if norm < 1e-9:
            raise SafetyAbort(f"{label} quaternion 不能为零")
        joints = _finite_vector(
            item.get("reference_joints_deg"), 7, f"{label} 参考关节角"
        )
        anchors.append(
            {
                **item,
                "position_m": position,
                "tcp_pose_moveit": {
                    "xyz": xyz,
                    "quat_xyzw": (quat / norm).tolist(),
                },
                "reference_joints_deg": joints,
            }
        )
    return sorted(anchors, key=lambda item: item["position_m"])


def _learn_spatial_anchor_positions(
    anchors: list[dict],
    *,
    canonical_start_m: float,
    canonical_end_m: float,
    label: str,
) -> list[dict]:
    """Order noisy demonstrations by measured shelf-row position.

    ``position_m`` is an operator prompt, not metrology: a ruler and the open
    gripper cannot identify the bottle centre exactly, and demonstrations from
    separate sessions may disagree about the prompt labels.  The shelf row is
    the X axis of ``platform_base_link``, so use measured TCP X to establish
    order and normalize the observed span onto this arm's canonical region.
    Every demonstration remains a breakpoint in the learned curve.
    """
    if len(anchors) < 2:
        raise SafetyAbort(f"{label} 至少需要两个锚点")
    nominal = np.asarray([item["position_m"] for item in anchors], dtype=float)
    row_x = np.asarray(
        [item["tcp_pose_moveit"]["xyz"][0] for item in anchors], dtype=float
    )
    votes = [
        np.sign((nominal[j] - nominal[i]) * (row_x[j] - row_x[i]))
        for i in range(len(anchors))
        for j in range(i + 1, len(anchors))
        if not math.isclose(nominal[i], nominal[j], abs_tol=1e-12)
        and not math.isclose(row_x[i], row_x[j], abs_tol=1e-12)
    ]
    vote_total = float(np.sum(votes))
    if not votes or math.isclose(vote_total, 0.0, abs_tol=1e-12):
        raise SafetyAbort(f"{label} 无法从示教数据判定货架行方向")
    direction = 1.0 if vote_total > 0.0 else -1.0
    spatial = direction * row_x
    order = np.argsort(spatial, kind="stable")
    ordered_spatial = spatial[order]
    span = float(ordered_spatial[-1] - ordered_spatial[0])
    if span < 1e-4:
        raise SafetyAbort(f"{label} 实际 TCP 横向覆盖不足 0.1 mm")
    if np.any(np.diff(ordered_spatial) < 1e-5):
        raise SafetyAbort(f"{label} 有无法区分的 TCP 横向重复点")

    canonical_span = float(canonical_end_m - canonical_start_m)
    learned = []
    for index in order:
        item = dict(anchors[int(index)])
        item["recorded_position_m"] = float(item["position_m"])
        item["position_m"] = canonical_start_m + canonical_span * (
            float(spatial[int(index)] - ordered_spatial[0]) / span
        )
        learned.append(item)
    return learned


def _interpolate_anchor(anchors: list[dict], position_m: float, label: str) -> dict:
    if len(anchors) < 2:
        raise SafetyAbort(f"{label} 至少需要两个锚点")
    positions = [float(item["position_m"]) for item in anchors]
    if position_m < positions[0] - 1e-9 or position_m > positions[-1] + 1e-9:
        raise SafetyAbort(
            f"{label} 锚点未覆盖 {position_m:.3f} m；"
            f"当前范围 {positions[0]:.3f}..{positions[-1]:.3f} m"
        )
    right_index = min(max(bisect_right(positions, position_m), 1), len(anchors) - 1)
    left = anchors[right_index - 1]
    right = anchors[right_index]
    left_position = float(left["position_m"])
    right_position = float(right["position_m"])
    if math.isclose(position_m, left_position, abs_tol=1e-9):
        fraction = 0.0
    elif math.isclose(position_m, right_position, abs_tol=1e-9):
        fraction = 1.0
    else:
        fraction = (position_m - left_position) / (right_position - left_position)
    left_xyz = np.asarray(left["tcp_pose_moveit"]["xyz"], dtype=float)
    right_xyz = np.asarray(right["tcp_pose_moveit"]["xyz"], dtype=float)
    xyz = left_xyz + fraction * (right_xyz - left_xyz)
    rotations = Rotation.from_quat(
        [
            left["tcp_pose_moveit"]["quat_xyzw"],
            right["tcp_pose_moveit"]["quat_xyzw"],
        ]
    )
    quaternion = Slerp([0.0, 1.0], rotations)([fraction]).as_quat()[0]
    nearest = left if fraction <= 0.5 else right
    return {
        "tcp_pose_moveit": {
            "xyz": xyz.tolist(),
            "quat_xyzw": quaternion.tolist(),
        },
        # Never interpolate joints.  This is only an IK branch reference.
        "reference_joints_deg": list(nearest["reference_joints_deg"]),
        "source_anchor_positions_m": [
            float(left.get("recorded_position_m", left_position)),
            float(right.get("recorded_position_m", right_position)),
        ],
        "source_learned_spatial_positions_m": [left_position, right_position],
        "learning_method": "tcp_x_spatial_order_piecewise_xyz_slerp",
        "execution_verified": False,
        "requires_ik_collision_fence_validation": True,
        "requires_gripper_clearance_validation": True,
    }


def generate_templates(payload: dict) -> dict:
    _validate_map(payload)
    start = float(payload["row_start_m"])
    end = float(payload["row_end_m"])
    spacing = float(payload["spacing_m"])
    step_count = round((end - start) / spacing)
    if not math.isclose(start + step_count * spacing, end, abs_tol=1e-9):
        raise SafetyAbort("整行宽度不能按 spacing_m 精确生成")
    positions = [round(start + index * spacing, 9) for index in range(step_count + 1)]
    points = []
    for layer_id, layer in payload["layers"].items():
        cached = {}
        for arm in ARMS:
            if arm == "left_arm":
                arm_start = start
                arm_end = float(payload["arm_assignment"]["overlap_end_m"])
            else:
                arm_start = float(payload["arm_assignment"]["overlap_start_m"])
                arm_end = end
            for operation in OPERATIONS:
                label = f"{layer_id}/{arm}/{operation}"
                cached[(arm, operation)] = _learn_spatial_anchor_positions(
                    _validated_anchors(
                        layer["anchors"][arm][operation], label
                    ),
                    canonical_start_m=arm_start,
                    canonical_end_m=arm_end,
                    label=label,
                )
        for position in positions:
            eligible_arms = _arms_for_position(payload, position)
            operations = {}
            for operation in OPERATIONS:
                candidates = {
                    arm: _interpolate_anchor(
                        cached[(arm, operation)],
                        position,
                        f"{layer_id}/{arm}/{operation}",
                    )
                    for arm in eligible_arms
                }
                operations[operation] = {
                    "arm_selection": (
                        "runtime_plan_both_choose_valid_best"
                        if len(candidates) > 1
                        else "fixed_by_reach_region"
                    ),
                    "pose_semantics": (
                        "demonstrated_grasp_curve"
                        if operation == "pick"
                        else "grasp_curve_place_initial_guess_only"
                    ),
                    "release_allowed_without_runtime_adjustment": False,
                    "required_runtime_adjustment": (
                        None
                        if operation == "pick"
                        else "fresh_support_surface_height_and_collision_checked_place_plan"
                    ),
                    "arm_candidates": candidates,
                }
            points.append(
                {
                    "point_id": f"{layer_id}_{round(position * 100):03d}cm",
                    "layer_id": layer_id,
                    "row_position_m": position,
                    "lift_height_mm": int(layer["lift_height_mm"]),
                    "eligible_arms": list(eligible_arms),
                    "operations": operations,
                }
            )
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": TEMPLATE_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(canonical).hexdigest(),
        "pose_frame": "platform_base_link",
        "execution_verified": False,
        "learning_contract": {
            "operator_position_m_is_exact_metrology": False,
            "spatial_order_source": "tcp_pose_moveit.xyz[0]",
            "spatial_frame": "platform_base_link",
            "interpolation": "piecewise_linear_xyz_and_slerp_quaternion",
            "all_demonstrations_are_curve_breakpoints": True,
        },
        "gripper_contract": dict(payload["gripper_contract"]),
        "point_count": len(points),
        "points": points,
    }


def _write_json(path: Path, payload: dict, *, force: bool) -> None:
    if path.exists() and not force:
        raise SafetyAbort(f"输出已存在: {path}；使用 --force 才能覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyAbort(f"无法读取锚点文件 {path}: {exc}") from exc
    _validate_map(payload)
    return payload


def _parse_layer(value: str) -> tuple[str, int]:
    try:
        name, height = value.rsplit(":", 1)
        return name.strip(), int(height)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "层必须写成 ID:升降毫米，例如 upper:647"
        ) from exc


def _settled_joints(
    reader, *, samples: int, gap_s: float, max_drift_deg: float
) -> list[float]:
    if samples < 2 or gap_s < 0.0 or max_drift_deg <= 0.0:
        raise SafetyAbort("稳定采样参数无效")
    readings = []
    for index in range(samples):
        readings.append(reader.joints_deg())
        if index + 1 < samples:
            time.sleep(gap_s)
    values = np.asarray(readings, dtype=float)
    if values.shape != (samples, 7) or not np.all(np.isfinite(values)):
        raise SafetyAbort("机械臂稳定采样没有得到完整的七关节数据")
    drift = float(np.max(np.ptp(values, axis=0)))
    if drift > max_drift_deg:
        raise SafetyAbort(
            "机械臂仍在运动："
            f"采样漂移 {drift:.3f}° > {max_drift_deg:.3f}°"
        )
    return values.mean(axis=0).tolist()


def _capture_live(cli, payload: dict, arm_id: str, expected_lift_mm: int):
    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=False
    )
    lift = LiftSocketAdapter(
        cfg.connections.left_arm_ip, cfg.connections.arm_port
    ).state()
    if lift.mode != 0 or abs(int(lift.height_mm) - int(expected_lift_mm)) > 5:
        raise SafetyAbort(
            "升降未静止在该层录入高度: "
            f"actual={lift.height_mm} mm mode={lift.mode}, "
            f"expected={expected_lift_mm} mm"
        )
    robot, view = open_live_arm(cfg, params, profile, arm_id, take_control=False)
    try:
        joints = _settled_joints(
            robot,
            samples=cli.samples,
            gap_s=cli.sample_gap_s,
            max_drift_deg=cli.max_drift_deg,
        )
        tcp_arm = np.asarray(robot.tcp_from_joints(joints), dtype=float)
    finally:
        robot.close()
    view.assert_tcp_point(tcp_arm[:3, 3], label=f"{arm_id} 示教 TCP")
    tcp_moveit = view.pose_to_moveit(tcp_arm)
    quaternion = Rotation.from_matrix(tcp_moveit[:3, :3]).as_quat()
    return tcp_moveit[:3, 3].tolist(), quaternion.tolist(), joints


def _command_live_gripper(cli, arm_id: str, *, close: bool = False) -> dict:
    """Open or shut one gripper on operator request; never move joints.

    Closing uses ``close_empty_gripper``: ``close_gripper`` is the *grasp*, and
    it deliberately fails on an empty hand.  This is the retract action.
    """
    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=False
    )
    robot, view = open_live_arm(cfg, params, profile, arm_id, take_control=True)
    try:
        if arm_id == "right_arm":
            robot.recover_transient_joint_frame_loss()
        robot.assert_arm_healthy()
        joints = robot.joints_deg()
        tcp_arm = np.asarray(robot.tcp_from_joints(joints), dtype=float)
        view.assert_tcp_point(tcp_arm[:3, 3], label=f"{arm_id} 夹爪动作前 TCP")
        # ArmProxy crosses a JSON boundary; its worker builds the same
        # default DemoParams on the far side, so never serialize this one.
        args = () if arm_id == "left_arm" else (params,)
        return (
            robot.close_empty_gripper(*args) if close
            else robot.open_gripper(*args)
        )
    finally:
        robot.close()


def _teaching_positions(payload: dict, arm_id: str, count: int) -> list[float]:
    if not 2 <= count <= 9:
        raise SafetyAbort("每臂每层的示教锚点数必须在 2..9")
    assignment = payload["arm_assignment"]
    if arm_id == "left_arm":
        start = float(payload["row_start_m"])
        end = float(assignment["overlap_end_m"])
    else:
        start = float(assignment["overlap_start_m"])
        end = float(payload["row_end_m"])
    return [
        round(float(value), 6)
        for value in np.linspace(start, end, count)
    ]


def _anchor_recorded(
    payload: dict, *, layer_id: str, arm_id: str, position_m: float
) -> bool:
    for operation in OPERATIONS:
        anchors = payload["layers"][layer_id]["anchors"][arm_id][operation]
        if not any(
            math.isclose(
                float(item.get("position_m")), position_m, abs_tol=1e-9
            )
            for item in anchors
        ):
            return False
    return True


def _default_templates_path(anchors: Path) -> Path:
    name = anchors.stem
    if name.endswith("_anchors"):
        name = name[: -len("_anchors")]
    return anchors.with_name(name + "_templates.json")


def _choose_layer_order(payload: dict) -> tuple[str, ...] | None:
    layer_ids = tuple(payload["layers"])
    if len(layer_ids) == 1:
        return layer_ids
    print("\n请选择先录入哪一层（之后自动继续其他层）：")
    for index, layer_id in enumerate(layer_ids, 1):
        height = payload["layers"][layer_id]["lift_height_mm"]
        print(f"  {index}. {layer_id} ({height} mm)")
    while True:
        try:
            choice = input(
                f"输入 1..{len(layer_ids)} 或层名；"
                f"直接 Enter 默认 {layer_ids[0]}；q 退出 > "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已保存当前进度并退出。")
            return None
        if choice.lower() == "q":
            print("已保存当前进度并退出。")
            return None
        if not choice:
            selected = layer_ids[0]
            break
        if choice in layer_ids:
            selected = choice
            break
        if choice.isdigit() and 1 <= int(choice) <= len(layer_ids):
            selected = layer_ids[int(choice) - 1]
            break
        print("无法识别层，请输入序号或层名。")
    print(f"将先录入 {selected} 层。")
    return (selected, *(item for item in layer_ids if item != selected))


def _teach(cli) -> int:
    if cli.anchors.exists():
        payload = _read_json(cli.anchors)
        print(f"继续已有录入文件: {cli.anchors}")
    else:
        layers = cli.layer or (("upper", 647), ("lower", 250))
        payload = create_anchor_map(
            row_start_m=cli.row_start_m,
            row_end_m=cli.row_end_m,
            spacing_m=cli.spacing_m,
            overlap_start_m=cli.overlap_start_m,
            overlap_end_m=cli.overlap_end_m,
            layers=layers,
        )
        _write_json(cli.anchors, payload, force=False)
        print(f"已创建录入文件: {cli.anchors}")

    layer_order = _choose_layer_order(payload)
    if layer_order is None:
        return 0
    tasks = [
        (layer_id, arm_id, position)
        for arm_id in ("right_arm", "left_arm")
        for layer_id in layer_order
        for position in _teaching_positions(
            payload, arm_id, cli.anchors_per_arm
        )
    ]
    pending = [
        task
        for task in tasks
        if not _anchor_recorded(
            payload,
            layer_id=task[0],
            arm_id=task[1],
            position_m=task[2],
        )
    ]
    print(
        "\n录入说明：工具不会移动机械臂或升降。"
        "你负责手动摆位，"
        "回车后工具只读当前状态。"
    )
    print(
        "位姿不包含夹爪宽度；"
        "每个点仍需后续 IK、碰撞和夹爪净空验证。"
    )
    print(
        f"整行 {payload['row_start_m']:.2f}..{payload['row_end_m']:.2f} m，"
        f"重叠区 {payload['arm_assignment']['overlap_start_m']:.2f}.."
        f"{payload['arm_assignment']['overlap_end_m']:.2f} m；"
        f"共 {len(tasks)} 个锚点，待录 {len(pending)} 个。"
    )
    if not pending:
        print("所有锚点已经录完，将直接重新生成点位图。")

    completed = len(tasks) - len(pending)
    skipped = 0
    for layer_id, arm_id, position in pending:
        layer = payload["layers"][layer_id]
        lift_height = int(layer["lift_height_mm"])
        arm_label = "左臂" if arm_id == "left_arm" else "右臂"
        while True:
            print(
                f"\n[{completed + 1}/{len(tasks)}] {layer_id} 层 "
                f"({lift_height} mm) | {arm_label} | "
                f"整行位置 {position:.3f} m"
            )
            print(
                "请把升降停到上述高度，手动摆好夹取/放置接触位，"
                "松开拖动按钮并等待静止。"
            )
            try:
                action = input(
                    "按 Enter 记录；输入 o 打开当前夹爪；"
                    "输入 s 跳过；输入 q 保存进度并退出 > "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n收到中断，当前成功记录已经保存。")
                return 0
            if action == "q":
                print(f"已保存当前进度: {cli.anchors}")
                return 0
            if action == "s":
                print("已跳过；未写入任何数据。")
                skipped += 1
                completed += 1
                break
            if action == "o":
                print(
                    "即将只打开当前指定机械臂的夹爪，不移动关节。"
                    "请确认手指四周有足够空间且拖动按钮已松开。"
                )
                try:
                    state = _command_live_gripper(cli, arm_id)
                except SafetyAbort as exc:
                    print(f"✗ 夹爪打开失败：{exc}")
                    print(
                        "失败后没有记录点位；"
                        "检查原因后可再次输入 o。"
                    )
                    continue
                print(
                    "✓ 夹爪已打开并通过反馈检查："
                    f"pos={int(state['pos'][0])}。"
                    "现在可放入瓶子、调整姿态，再按 Enter 记录。"
                )
                continue
            if action:
                print("无法识别，请只按 Enter，或输入 o/s/q。")
                continue
            print(
                f"正在只读采样 {cli.samples} 次，"
                f"约需 {(cli.samples - 1) * cli.sample_gap_s:.1f} 秒……"
            )
            try:
                xyz, quaternion, joints = _capture_live(
                    cli, payload, arm_id, lift_height
                )
                add_anchor(
                    payload,
                    layer_id=layer_id,
                    position_m=position,
                    arm_id=arm_id,
                    operations=OPERATIONS,
                    xyz_moveit=xyz,
                    quat_xyzw_moveit=quaternion,
                    reference_joints_deg=joints,
                    source="guided_live_settled_fk_read_only",
                )
                _write_json(cli.anchors, payload, force=True)
            except SafetyAbort as exc:
                print(f"✗ 记录失败：{exc}")
                print(
                    "请按提示修正后在同一点重试；"
                    "失败数据没有写入。"
                )
                continue
            completed += 1
            print(
                "✓ 记录成功并已保存："
                f"TCP(MoveIt)={[round(value, 4) for value in xyz]}"
            )
            break

    if skipped:
        print(
            f"\n有 {skipped} 个锚点被跳过，暂不生成点位图；"
            "再次运行同一条 teach 命令即可续录。"
        )
        return 0

    generated = generate_templates(payload)
    output = cli.templates_output or _default_templates_path(cli.anchors)
    _write_json(output, generated, force=True)
    print(
        "\n✓ 全部锚点录入完成，已生成 "
        f"{generated['point_count']} 个1cm点位: "
        f"{output}"
    )
    print(
        "注意：这是录入成功，不是抓取执行验证成功；"
        "所有生成点仍为 execution_verified=false。"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    teach = commands.add_parser(
        "teach", help="交互式引导录入整行锚点（推荐）"
    )
    teach.add_argument("anchors", type=Path)
    teach.add_argument("--row-start-m", type=float, default=0.00)
    teach.add_argument("--row-end-m", type=float, default=0.60)
    teach.add_argument("--spacing-m", type=float, default=0.01)
    teach.add_argument("--overlap-start-m", type=float, default=0.25)
    teach.add_argument("--overlap-end-m", type=float, default=0.35)
    teach.add_argument("--layer", type=_parse_layer, action="append")
    teach.add_argument(
        "--anchors-per-arm",
        type=int,
        default=8,
        help="每臂每层的锚点数（默认臂区间内每 5 cm 一个）",
    )
    teach.add_argument("--templates-output", type=Path)
    teach.add_argument("--config", default=str(ROOT / "config.yaml"))
    teach.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    teach.add_argument("--samples", type=int, default=7)
    teach.add_argument("--sample-gap-s", type=float, default=1.5)
    teach.add_argument("--max-drift-deg", type=float, default=0.5)

    for name, help_text in (
        ("open-gripper", "只打开一个夹爪，不移动关节"),
        ("close-gripper", "只收拢一个夹爪，不判定抓取，不移动关节"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--arm", choices=ARMS, default="left_arm")
        sub.add_argument("--config", default=str(ROOT / "config.yaml"))
        sub.add_argument(
            "--safety-config",
            default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
        )

    init = commands.add_parser("init", help="创建空的两臂/多层锚点文件")
    init.add_argument("output", type=Path)
    init.add_argument("--row-start-m", type=float, required=True)
    init.add_argument("--row-end-m", type=float, required=True)
    init.add_argument("--spacing-m", type=float, default=0.01)
    init.add_argument("--overlap-start-m", type=float, required=True)
    init.add_argument("--overlap-end-m", type=float, required=True)
    init.add_argument("--layer", type=_parse_layer, action="append", required=True)
    init.add_argument("--force", action="store_true")

    record = commands.add_parser("record", help="只读录入一个示教锚点")
    record.add_argument("anchors", type=Path)
    record.add_argument("--layer", required=True)
    record.add_argument("--position-m", type=float, required=True)
    record.add_argument("--arm", choices=ARMS, required=True)
    record.add_argument(
        "--operation", choices=("pick", "place", "both"), default="both"
    )
    source = record.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--live", action="store_true", help="连接机械臂只读当前姿态"
    )
    source.add_argument("--xyz-moveit", type=float, nargs=3)
    record.add_argument("--quat-moveit", type=float, nargs=4)
    record.add_argument("--joints-deg", type=float, nargs=7)
    record.add_argument("--config", default=str(ROOT / "config.yaml"))
    record.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    record.add_argument("--samples", type=int, default=7)
    record.add_argument("--sample-gap-s", type=float, default=1.5)
    record.add_argument("--max-drift-deg", type=float, default=0.5)
    record.add_argument("--replace", action="store_true")

    generate = commands.add_parser("generate", help="生成 1cm pick/place 点位图")
    generate.add_argument("anchors", type=Path)
    generate.add_argument("output", type=Path)
    generate.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    cli = _build_parser().parse_args(argv)
    if cli.command == "teach":
        return _teach(cli)
    if cli.command in ("open-gripper", "close-gripper"):
        close = cli.command == "close-gripper"
        verb = "收拢" if close else "打开"
        print(
            f"即将只{verb} {cli.arm} 的夹爪，不移动关节。"
            "请确认手指之间和四周没有不该碰的东西。"
        )
        state = _command_live_gripper(cli, cli.arm, close=close)
        print(f"✓ {cli.arm} 夹爪已{verb}：pos={int(state['pos'][0])}")
        return 0

    if cli.command == "init":
        payload = create_anchor_map(
            row_start_m=cli.row_start_m,
            row_end_m=cli.row_end_m,
            spacing_m=cli.spacing_m,
            overlap_start_m=cli.overlap_start_m,
            overlap_end_m=cli.overlap_end_m,
            layers=cli.layer,
        )
        _write_json(cli.output, payload, force=cli.force)
        print(f"空锚点文件已写入 {cli.output}")
        return 0

    if cli.command == "record":
        payload = _read_json(cli.anchors)
        layer = payload["layers"].get(cli.layer)
        if layer is None:
            raise SafetyAbort(f"未知层: {cli.layer}")
        if cli.live:
            xyz, quaternion, joints = _capture_live(
                cli, payload, cli.arm, int(layer["lift_height_mm"])
            )
            source = "live_settled_fk_read_only"
        else:
            if cli.quat_moveit is None or cli.joints_deg is None:
                raise SafetyAbort(
                    "离线录入 --xyz-moveit 时还必须提供 "
                    "--quat-moveit 和 --joints-deg"
                )
            xyz, quaternion, joints = (
                cli.xyz_moveit,
                cli.quat_moveit,
                cli.joints_deg,
            )
            source = "offline_operator_supplied"
        operations = OPERATIONS if cli.operation == "both" else (cli.operation,)
        add_anchor(
            payload,
            layer_id=cli.layer,
            position_m=cli.position_m,
            arm_id=cli.arm,
            operations=operations,
            xyz_moveit=xyz,
            quat_xyzw_moveit=quaternion,
            reference_joints_deg=joints,
            source=source,
            replace_existing=cli.replace,
        )
        _write_json(cli.anchors, payload, force=True)
        print(
            f"已录入 {cli.layer} {cli.position_m:.3f}m "
            f"{cli.arm} {','.join(operations)}；未执行任何运动"
        )
        return 0

    payload = _read_json(cli.anchors)
    generated = generate_templates(payload)
    _write_json(cli.output, generated, force=cli.force)
    print(
        f"已生成 {generated['point_count']} 个抓取曲线点位；"
        "place 仅为需现场高度修正的初值，"
        "全部仍为 execution_verified=false"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
