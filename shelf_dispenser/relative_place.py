"""Build a deterministic place route by retargeting one successful teaching.

The interface deliberately accepts only live endpoints plus a demonstrated
record.  Shelf perception decides the target, grasp execution decides the
current TCP, and this module owns every placement-specific interpolation
between them.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np

from .core import SafetyAbort
from utils.items_info import get_item_info


GATE_DISTANCE_M = 0.085
MIN_PREPLACE_DISTANCE_M = 0.040
MAX_PREPLACE_DISTANCE_M = 0.180
MAX_SOURCE_CORRECTION_M = 0.350
MAX_TARGET_CORRECTION_M = 0.120


def _finite_vector(value, length: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise SafetyAbort(f"{label} 必须包含 {length} 个有限数")
    return vector


def product_geometry(product_code: str) -> dict:
    """Return planning geometry from the single recorded product catalog."""
    item = get_item_info(product_code)
    if item is None or item["product_code"] != product_code:
        raise SafetyAbort(f"未知商品编号: {product_code}")
    return {
        "product_code": product_code,
        "height_m": float(item["total_height_cm"]) / 100.0,
        "graspable_height_m": float(item["graspable_height_cm"]) / 100.0,
        "radius_m": float(item["conservative_max_width_cm"]) / 200.0,
    }


def _teaching_identity(demonstration: dict) -> tuple[str, str]:
    match = re.fullmatch(
        r"place_([AB][1-4])_(left|right)", str(demonstration.get("label", ""))
    )
    arm_id = str(demonstration.get("arm_id", ""))
    if match is None or arm_id != f"{match.group(2)}_arm":
        raise SafetyAbort("放置示教标签与 arm_id 不一致")
    return match.group(1), arm_id


def _anchor_indices(xyz: np.ndarray) -> tuple[int, int]:
    end = xyz[-1]
    gate_index = 0
    for index in range(len(xyz) - 2, -1, -1):
        if float(np.linalg.norm(end - xyz[index])) >= GATE_DISTANCE_M:
            gate_index = index
            break
    if gate_index <= 0:
        raise SafetyAbort("放置示教末段不足 85 mm，无法提取槽口路标")
    peak_z = float(np.max(xyz[: gate_index + 1, 2]))
    raised_z = float(xyz[0, 2]) + 0.9 * (peak_z - float(xyz[0, 2]))
    clearance_index = next(
        index
        for index in range(gate_index + 1)
        if float(xyz[index, 2]) >= raised_z
    )
    if clearance_index >= gate_index:
        raise SafetyAbort("放置示教没有分离的抬升与槽口阶段")
    return clearance_index, gate_index


def _pose(xyz: Sequence[float], quaternion: Sequence[float]) -> dict:
    return {
        "xyz": list(map(float, xyz)),
        "quat_xyzw": list(map(float, quaternion)),
    }


def build_relative_place_route(
    demonstration: dict,
    *,
    current_tcp_moveit_xyz,
    target_tcp_moveit_xyz,
    target_tcp_moveit_quat_xyzw,
    contact_distance_m: float,
    minimum_transit_raise_m: float = 0.0,
) -> dict:
    """Retarget four taught stages while keeping start and goal exact.

    The held and placed endpoints come from this run.  The two middle anchors
    keep the taught route shape, with corrections blended from source to goal.
    The returned pre-place pose is consumed directly by MTC; it is not rebuilt
    from a fixed global angle.
    """
    slot_id, arm_id = _teaching_identity(demonstration)
    if (
        demonstration.get("schema_version")
        != "grabber.demonstrated_joint_trajectory.v1"
        or demonstration.get("pose_frame") != "platform_base_link"
        or demonstration.get("usable_for_planning_constraints") is not True
        or demonstration.get("fence_violations")
    ):
        raise SafetyAbort(f"{slot_id} 放置示教未通过规划约束门禁")
    samples = demonstration.get("samples") or []
    if len(samples) < 4:
        raise SafetyAbort(f"{slot_id} 放置示教样本不足")
    xyz = np.asarray(
        [sample["tcp_pose_moveit"]["xyz"] for sample in samples], dtype=float
    )
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise SafetyAbort(f"{slot_id} 放置示教 TCP 数据无效")

    current = _finite_vector(current_tcp_moveit_xyz, 3, "当前 TCP")
    target = _finite_vector(target_tcp_moveit_xyz, 3, "目标 TCP")
    quaternion = _finite_vector(
        target_tcp_moveit_quat_xyzw, 4, "目标 TCP 四元数"
    )
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm < 1e-9:
        raise SafetyAbort("目标 TCP 四元数长度为零")
    quaternion /= quaternion_norm
    if (
        not math.isfinite(float(contact_distance_m))
        or not 0.003 <= float(contact_distance_m) <= 0.030
        or not math.isfinite(float(minimum_transit_raise_m))
        or not 0.0 <= float(minimum_transit_raise_m) <= 0.5
    ):
        raise SafetyAbort("放置接触或抬升距离无效")

    source_delta = current - xyz[0]
    target_delta = target - xyz[-1]
    if float(np.linalg.norm(source_delta)) > MAX_SOURCE_CORRECTION_M:
        raise SafetyAbort("本次持瓶起点偏离示教超过 350 mm，拒绝外推")
    if float(np.linalg.norm(target_delta)) > MAX_TARGET_CORRECTION_M:
        raise SafetyAbort("本次目标偏离示教超过 120 mm，拒绝外推")

    clearance_index, gate_index = _anchor_indices(xyz)
    taught = xyz[[0, clearance_index, gate_index, len(xyz) - 1]]
    weights = np.asarray([0.0, 0.35, 0.80, 1.0])[:, None]
    corrections = (1.0 - weights) * source_delta + weights * target_delta
    adapted = taught + corrections
    adapted[0] = current
    adapted[-1] = target

    transit_raise = max(
        float(minimum_transit_raise_m),
        0.0,
        float(adapted[1, 2] - current[2]),
    )
    clearance = current.copy()
    clearance[2] += transit_raise

    contact_direction = np.asarray([0.0, 0.0, -1.0])
    precontact = target - contact_direction * float(contact_distance_m)
    gate = adapted[2]
    insert = precontact - gate
    preplace_offset = float(np.linalg.norm(insert))
    if not MIN_PREPLACE_DISTANCE_M <= preplace_offset <= MAX_PREPLACE_DISTANCE_M:
        raise SafetyAbort(
            f"{slot_id} 重定向后的槽口距离 {preplace_offset * 1000:.1f} mm "
            "超出 40..180 mm"
        )
    insert /= preplace_offset
    if float(insert[2]) > -0.15:
        raise SafetyAbort(f"{slot_id} 示教入槽方向下降分量不足")

    gate_joints = _finite_vector(
        samples[gate_index].get("joints_deg"), 7, "槽口参考关节"
    )
    placed_joints = _finite_vector(
        samples[-1].get("joints_deg"), 7, "终点参考关节"
    )
    waypoints = [
        {"name": "held", "pose": _pose(current, quaternion)},
        {"name": "clearance", "pose": _pose(clearance, quaternion)},
        {"name": "gate", "pose": _pose(gate, quaternion)},
        {"name": "placed", "pose": _pose(target, quaternion)},
    ]
    return {
        "schema_version": "grabber.relative_place_route.v1",
        "slot_id": slot_id,
        "arm_id": arm_id,
        "anchor_indices": [0, clearance_index, gate_index, len(xyz) - 1],
        "waypoints": waypoints,
        "transit_raise_m": transit_raise,
        "preplace_pose": waypoints[2]["pose"],
        "preplace_offset_m": preplace_offset,
        "preplace_reference_joints_deg": gate_joints.tolist(),
        "place_reference_joints_deg": placed_joints.tolist(),
        "insert_direction": insert.tolist(),
        "contact_direction": contact_direction.tolist(),
        "contact_distance_m": float(contact_distance_m),
        "retreat_direction": (-insert).tolist(),
        "max_endpoint_correction_m": max(
            float(np.linalg.norm(source_delta)),
            float(np.linalg.norm(target_delta)),
        ),
    }
