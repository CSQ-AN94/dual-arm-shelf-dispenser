"""Fail-closed validation for the plan-only MTC pick trajectory export."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json
import math
import numpy as np
import yaml

from .core import DemoParams, SafetyAbort
from .arm import (
    validate_holding_gripper_feedback,
    validate_open_gripper_feedback,
)

EXPECTED_JOINTS = tuple(f"r_joint{index}" for index in range(1, 8))
EXPECTED_LEFT_JOINTS = tuple(f"l_joint{index}" for index in range(1, 8))
EXPECTED_PHASES = ("pregrasp", "approach", "attach", "retreat")
PLAN_ONLY_BLOCK_REASON = "EXTERNAL_GATED_PICK_EXECUTOR_REQUIRED"
LEFT_PICK_PLAN_ONLY_BLOCK_REASON = "LEFT_TOOL_CALIBRATION_REQUIRED"
PLACE_PLAN_ONLY_BLOCK_REASON = "EXTERNAL_GATED_PLACE_EXECUTOR_REQUIRED"
PICK_JOINTS_BY_ARM = {
    "right_arm": EXPECTED_JOINTS,
    "left_arm": EXPECTED_LEFT_JOINTS,
}
PICK_BLOCK_REASON_BY_ARM = {
    "right_arm": PLAN_ONLY_BLOCK_REASON,
    "left_arm": LEFT_PICK_PLAN_ONLY_BLOCK_REASON,
}
EXPECTED_PLACE_PHASES = ("transport", "approach", "release", "retreat")
EXPECTED_FULL_TRANSFER_PHASES = (
    "pregrasp",
    "approach",
    "attach",
    "source_retreat",
    "platform_lower",
    "transport",
    "place",
    "release",
    "target_retreat",
)
MAX_PLANNING_JOINT_STATE_AGE_S = 0.5


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise SafetyAbort(f"{label}必须是带时区的 ISO 时间")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SafetyAbort(f"{label}不是有效 ISO 时间") from exc
    if parsed.tzinfo is None:
        raise SafetyAbort(f"{label}必须带时区")
    return parsed


def _assert_fresh(value: Any, *, label: str, now: datetime, max_age_s: float) -> None:
    age_s = (now - _timestamp(value, label)).total_seconds()
    if not math.isfinite(age_s) or age_s < -60 or age_s > max_age_s:
        raise SafetyAbort(
            f"{label}不新鲜: age={age_s:.1f}s, limit={max_age_s:.1f}s"
        )


def load_pick_trajectory(path: str | Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SafetyAbort(f"MTC pick 轨迹文件无法读取: {exc}") from exc
    validate_pick_trajectory(payload)
    return payload


def _read_json(path: str | Path, label: str) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SafetyAbort(f"{label}无法读取: {exc}") from exc
    if not isinstance(payload, dict):
        raise SafetyAbort(f"{label}顶层必须是对象")
    return payload


def load_pick_result(path: str | Path) -> dict:
    return _read_json(path, "MTC 规划结果")


def load_pick_scenario(path: str | Path) -> dict:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SafetyAbort(f"MTC 场景无法读取: {exc}") from exc
    if not isinstance(payload, dict):
        raise SafetyAbort("MTC 场景顶层必须是对象")
    return payload


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SafetyAbort(f"{label}必须是有限数")
    result = float(value)
    if not math.isfinite(result):
        raise SafetyAbort(f"{label}必须是有限数")
    return result


def _validate_optional_accelerations(point: dict, index: int, mode: str) -> None:
    values = np.asarray(point.get("accelerations_deg_s2", []), dtype=float)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise SafetyAbort(f"MTC {mode} 轨迹点 {index} 加速度维度或数值无效")


def _validate_selected_arm_start_state(
    start: Any, *, arm_id: str, label: str
) -> dict:
    if not isinstance(start, dict):
        raise SafetyAbort(f"MTC {label} 结果缺少 start_state")
    age_s = _finite(
        start.get("joint_state_age_s_at_planning"),
        f"MTC {label} 所选{label} /joint_states 新鲜度",
    )
    stamp_ns = start.get("joint_state_stamp_ns")
    if (
        start.get("selected_arm") != arm_id
        or start.get("selected_arm_complete") is not True
        or age_s < 0.0
        or age_s > MAX_PLANNING_JOINT_STATE_AGE_S
        or isinstance(stamp_ns, bool)
        or not isinstance(stamp_ns, int)
        or stamp_ns <= 0
    ):
        raise SafetyAbort(
            f"MTC 所选{label}没有完整且新鲜的 /joint_states 规划起点证据"
        )
    return start


def validate_execution_bundle(
    result: dict,
    trajectory: dict,
    scenario: dict,
    *,
    params: DemoParams | None = None,
) -> dict:
    """Validate one plan-only source bundle for the supervised bridge."""
    validate_pick_trajectory(trajectory)
    if trajectory.get("arm_id") != "right_arm":
        raise SafetyAbort("MTC pick 真机执行桥只支持右臂")
    params = params or DemoParams()
    required_result = {
        "plan_only": True,
        "solved": True,
        "mode": "pick_only",
        "selected_arm": "right_arm",
        "execution_eligible": False,
        "execution_block_reason": PLAN_ONLY_BLOCK_REASON,
        "fixture_source": False,
    }
    for key, expected in required_result.items():
        if result.get(key) != expected:
            raise SafetyAbort(f"MTC 结果字段 {key} 必须是 {expected!r}")
    required_scenario = {
        "mode": "pick_only",
        "planning_arm_id": "right_arm",
        "fixture_source": False,
        "start_state_source": "current_state",
        "spawn_scene_objects": True,
    }
    for key, expected in required_scenario.items():
        if scenario.get(key) != expected:
            raise SafetyAbort(f"MTC 场景字段 {key} 必须是 {expected!r}")

    scenario_id = trajectory["scenario_id"]
    candidate_id = trajectory["grasp_candidate_id"]
    if (
        result.get("scenario_id") != scenario_id
        or scenario.get("scenario_id") != scenario_id
    ):
        raise SafetyAbort("result/trajectory/scenario 的 scenario_id 不一致")
    if result.get("selected_grasp_candidate") != candidate_id:
        raise SafetyAbort("MTC 结果与轨迹的抓取候选不一致")
    branch_id = f"right_arm__{candidate_id}"
    if result.get("selected_solution_id") != f"{branch_id}#execution_safe":
        raise SafetyAbort("MTC 结果不是通过执行资格审计的抓取解")
    if result.get("solved_by_arm", {}).get(branch_id) is not True:
        raise SafetyAbort("所选右臂抓取候选没有完整解")
    count = result.get("complete_solution_count_by_arm", {}).get(branch_id)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise SafetyAbort("所选右臂抓取候选完整解数量无效")

    for key in ("target_captured_at_utc", "scene_captured_at_utc"):
        if scenario.get(key) != trajectory.get(key):
            raise SafetyAbort(f"MTC 场景与轨迹的 {key} 不一致")
    if _finite(
        scenario.get("freshness_max_age_s"), "场景 freshness_max_age_s"
    ) != float(trajectory["freshness_max_age_s"]):
        raise SafetyAbort("MTC 场景与轨迹的新鲜度上限不一致")
    if (
        not isinstance(scenario.get("scene_version"), str)
        or not scenario["scene_version"]
        or result.get("scene_version") != scenario["scene_version"]
    ):
        raise SafetyAbort("MTC 结果与场景的 scene_version 不一致")
    provenance = scenario.get("localization_provenance")
    if not isinstance(provenance, dict) or provenance.get("profile") != "shelf_template":
        raise SafetyAbort("MTC 场景不是 shelf_template 实时定位产物")
    scene_provenance = scenario.get("scene_provenance")
    if not isinstance(scene_provenance, dict):
        raise SafetyAbort("MTC 场景缺少实时 scene_provenance")
    if not isinstance(scenario.get("obstacle_voxels"), list):
        raise SafetyAbort("MTC 场景缺少非目标实时障碍")
    shelf_ids = {
        item.get("id")
        for item in scenario.get("shelf_boxes", [])
        if isinstance(item, dict)
    }
    if not {"fence_shelf_bottom", "fence_shelf_top", "fence_shelf_back"}.issubset(
        shelf_ids
    ):
        raise SafetyAbort("MTC 场景缺少货架底/顶/背几何")
    candidates = scenario.get("source_grasp_candidates")
    if not isinstance(candidates, list) or candidate_id not in {
        item.get("id") for item in candidates if isinstance(item, dict)
    }:
        raise SafetyAbort("MTC 场景不包含所选抓取候选")
    workspace = scenario.get("tcp_path_workspace")
    if not isinstance(workspace, dict) or not workspace.get("id"):
        raise SafetyAbort("MTC pick 场景缺少 TCP 路径工作区约束")
    workspace_size = np.asarray(workspace.get("size"), dtype=float)
    workspace_pose = workspace.get("pose")
    if (
        workspace_size.shape != (3,)
        or not np.all(np.isfinite(workspace_size))
        or np.any(workspace_size <= 0)
        or not isinstance(workspace_pose, dict)
        or np.asarray(workspace_pose.get("xyz"), dtype=float).shape != (3,)
    ):
        raise SafetyAbort("MTC pick TCP 路径工作区约束无效")

    start = _validate_selected_arm_start_state(
        result.get("start_state"), arm_id="right_arm", label="右臂"
    )
    joints = start.get("joints")
    if not isinstance(joints, dict):
        raise SafetyAbort("MTC start_state.joints 必须是对象")

    def ordered(names: tuple[str, ...], label: str) -> np.ndarray:
        try:
            values = np.asarray([joints[name] for name in names], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetyAbort(f"MTC start_state 缺少{label}关节") from exc
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise SafetyAbort(f"MTC start_state {label}关节无效")
        return values

    right_deg = np.degrees(ordered(EXPECTED_JOINTS, "右臂"))
    left_deg = np.degrees(ordered(EXPECTED_LEFT_JOINTS, "左臂"))
    planned = np.asarray(trajectory["points"][0]["positions_deg"], dtype=float)
    error = float(np.max(np.abs(right_deg - planned)))
    if error > params.planned_start_tolerance_deg:
        raise SafetyAbort(
            "MTC 结果 start_state 与导出轨迹起点不一致: "
            f"最大差={error:.2f}°"
        )
    platform_m = _finite(joints.get("platform_joint"), "MTC platform_joint")
    if not 0.0 <= platform_m <= 1.0:
        raise SafetyAbort("MTC platform_joint 超出 0..1 m")
    return {
        "scenario_id": scenario_id,
        "grasp_candidate_id": candidate_id,
        "right_start_deg": right_deg.tolist(),
        "left_start_deg": left_deg.tolist(),
        "lift_start_mm": int(round(platform_m * 1000.0)),
    }


def validate_pick_trajectory(payload: dict) -> None:
    """Validate the stable offline contract; this never authorizes motion."""
    if not isinstance(payload, dict):
        raise SafetyAbort("MTC pick 轨迹顶层必须是对象")
    required = {
        "schema_version": "grabber.mtc_pick.v2",
        "plan_only": True,
        "execution_supported": False,
        "mode": "pick_only",
        "joint_units": "degrees",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise SafetyAbort(f"MTC pick 契约字段 {key} 必须是 {expected!r}")
    for key in ("scenario_id", "grasp_candidate_id"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise SafetyAbort(f"MTC pick 契约字段 {key} 必须是非空字符串")

    arm_id = payload.get("arm_id")
    if arm_id not in PICK_JOINTS_BY_ARM:
        raise SafetyAbort("MTC pick arm_id 只允许 right_arm/left_arm")
    expected_block_reason = PICK_BLOCK_REASON_BY_ARM[arm_id]
    if payload.get("execution_block_reason") != expected_block_reason:
        raise SafetyAbort(
            "MTC pick 契约字段 execution_block_reason 必须是 "
            f"{expected_block_reason!r}"
        )
    names = tuple(payload.get("joint_names") or ())
    if names != PICK_JOINTS_BY_ARM[arm_id]:
        raise SafetyAbort(f"MTC pick {arm_id} 关节名称或顺序无效: {names!r}")
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise SafetyAbort("MTC pick 轨迹至少需要两个点")
    last_time = -1.0
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise SafetyAbort(f"MTC pick 轨迹点 {index} 必须是对象")
        _validate_optional_accelerations(point, index, "pick")
        positions = np.asarray(point.get("positions_deg"), dtype=float)
        velocities = np.asarray(point.get("velocities_deg_s"), dtype=float)
        time_s = point.get("time_from_start_s")
        if (
            positions.shape != (7,)
            or velocities.shape != (7,)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(velocities))
            or isinstance(time_s, bool)
            or not isinstance(time_s, (int, float))
            or not math.isfinite(float(time_s))
            or float(time_s) < 0.0
            or (index > 0 and float(time_s) <= last_time)
        ):
            raise SafetyAbort(f"MTC pick 轨迹点 {index} 维度、数值或时间无效")
        last_time = float(time_s)

    phases = payload.get("phase_boundaries")
    if not isinstance(phases, list) or tuple(
        item.get("name") for item in phases if isinstance(item, dict)
    ) != EXPECTED_PHASES:
        raise SafetyAbort("MTC pick 阶段必须严格为 pregrasp/approach/attach/retreat")
    try:
        bounds = {
            item["name"]: (int(item["start_index"]), int(item["end_index"]))
            for item in phases
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise SafetyAbort("MTC pick 阶段边界格式无效") from exc
    pregrasp, approach, attach, retreat = (
        bounds[name] for name in EXPECTED_PHASES
    )
    if not (
        pregrasp[0] == 0
        and pregrasp[1] == approach[0]
        and pregrasp[1] < approach[1]
        and attach == (approach[1], approach[1])
        and retreat[0] == attach[1]
        and retreat[0] < retreat[1] == len(points) - 1
    ):
        raise SafetyAbort(f"MTC pick 阶段边界不连续或顺序无效: {bounds!r}")

    events = payload.get("gripper_events")
    expected_events = [
        (
            "open_before_motion",
            0,
            "RobotSession.open_gripper",
        ),
        (
            "close_at_attach",
            attach[0],
            "RobotSession.close_gripper",
        ),
    ]
    actual_events = []
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict) or event.get("feedback_required") is not True:
                raise SafetyAbort("MTC pick 夹爪事件必须要求反馈")
            actual_events.append(
                (
                    event.get("name"),
                    event.get("point_index"),
                    event.get("operation"),
                )
            )
    if actual_events != expected_events:
        raise SafetyAbort("MTC pick 夹爪打开/闭合事件与阶段边界不一致")

    max_age_s = payload.get("freshness_max_age_s")
    if (
        isinstance(max_age_s, bool)
        or not isinstance(max_age_s, (int, float))
        or not math.isfinite(float(max_age_s))
        or float(max_age_s) <= 0
    ):
        raise SafetyAbort("MTC pick freshness_max_age_s 必须是正有限数")
    _timestamp(payload.get("target_captured_at_utc"), "目标时间")
    _timestamp(payload.get("scene_captured_at_utc"), "场景时间")


def validate_full_transfer_trajectory(payload: dict) -> None:
    """Validate the plan-only pick/place replay contract."""
    required = {
        "schema_version": "grabber.mtc_full_transfer.v1",
        "plan_only": True,
        "execution_supported": False,
        "execution_block_reason": "PLAN_ONLY_FULL_TRANSFER",
        "mode": "full_transfer",
        "joint_units": "degrees",
    }
    if not isinstance(payload, dict):
        raise SafetyAbort("MTC full-transfer 轨迹顶层必须是对象")
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise SafetyAbort(
                f"MTC full-transfer 契约字段 {key} 必须是 {expected!r}"
            )
    names = tuple(payload.get("joint_names") or ())
    if names not in (EXPECTED_JOINTS, EXPECTED_LEFT_JOINTS):
        raise SafetyAbort("MTC full-transfer 必须只含一侧七个有序关节")
    expected_arm = "right_arm" if names == EXPECTED_JOINTS else "left_arm"
    if payload.get("arm_id") != expected_arm:
        raise SafetyAbort("MTC full-transfer arm_id 与关节侧不一致")
    for key in ("scenario_id", "grasp_candidate_id"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise SafetyAbort(f"MTC full-transfer {key} 必须是非空字符串")
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise SafetyAbort("MTC full-transfer 轨迹至少需要两个点")
    last_time = -1.0
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise SafetyAbort(f"MTC full-transfer 轨迹点 {index} 必须是对象")
        _validate_optional_accelerations(point, index, "full-transfer")
        try:
            positions = np.asarray(point["positions_deg"], dtype=float)
            velocities = np.asarray(point["velocities_deg_s"], dtype=float)
            time_s = float(point["time_from_start_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetyAbort(
                f"MTC full-transfer 轨迹点 {index} 格式无效"
            ) from exc
        if (
            positions.shape != (7,)
            or velocities.shape != (7,)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(velocities))
            or not math.isfinite(time_s)
            or time_s < 0.0
            or (index > 0 and time_s <= last_time)
        ):
            raise SafetyAbort(
                f"MTC full-transfer 轨迹点 {index} 维度、数值或时间无效"
            )
        last_time = time_s

    phases = payload.get("phase_boundaries")
    if not isinstance(phases, list) or tuple(
        item.get("name") for item in phases if isinstance(item, dict)
    ) != EXPECTED_FULL_TRANSFER_PHASES:
        raise SafetyAbort("MTC full-transfer 阶段顺序无效")
    try:
        bounds = {
            item["name"]: (int(item["start_index"]), int(item["end_index"]))
            for item in phases
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise SafetyAbort("MTC full-transfer 阶段边界格式无效") from exc
    (
        pregrasp,
        approach,
        attach,
        source_retreat,
        platform_lower,
        transport,
        place,
        release,
        retreat,
    ) = (bounds[name] for name in EXPECTED_FULL_TRANSFER_PHASES)
    if not (
        pregrasp[0] == 0
        and pregrasp[1] == approach[0] < approach[1]
        and attach == (approach[1], approach[1])
        and source_retreat[0] == attach[1] < source_retreat[1]
        and platform_lower[0] == source_retreat[1] < platform_lower[1]
        and transport[0] == platform_lower[1] < transport[1]
        and place[0] == transport[1] < place[1]
        and release == (place[1], place[1])
        and retreat[0] == release[1] < retreat[1] == len(points) - 1
    ):
        raise SafetyAbort(f"MTC full-transfer 阶段边界不连续: {bounds!r}")
    expected_events = [
        ("open_before_motion", 0),
        ("close_at_attach", attach[0]),
        ("open_at_release", release[0]),
    ]
    events = payload.get("gripper_events")
    actual_events = (
        [(item.get("name"), item.get("point_index")) for item in events]
        if isinstance(events, list) and all(isinstance(item, dict) for item in events)
        else []
    )
    if actual_events != expected_events:
        raise SafetyAbort("MTC full-transfer 夹爪事件与阶段边界不一致")


def validate_pre_motion_gate(
    payload: dict,
    *,
    current_state: dict,
    gripper_open_feedback: dict,
    now: datetime | None = None,
    params: DemoParams | None = None,
) -> None:
    """Validate the gates required immediately before the first arm command."""
    validate_pick_trajectory(payload)
    params = params or DemoParams()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise SafetyAbort("门禁当前时间必须带时区")
    max_age_s = float(payload["freshness_max_age_s"])
    _assert_fresh(
        payload["target_captured_at_utc"],
        label="目标时间",
        now=now,
        max_age_s=max_age_s,
    )
    _assert_fresh(
        payload["scene_captured_at_utc"],
        label="场景时间",
        now=now,
        max_age_s=max_age_s,
    )
    if not isinstance(current_state, dict):
        raise SafetyAbort("实时关节状态必须是对象")
    _assert_fresh(
        current_state.get("captured_at_utc"),
        label="实时关节时间",
        now=now,
        max_age_s=max_age_s,
    )
    names = list(current_state.get("joint_names") or [])
    positions = np.asarray(current_state.get("positions_deg"), dtype=float)
    if (
        len(names) != 7
        or len(set(names)) != 7
        or set(names) != set(EXPECTED_JOINTS)
        or positions.shape != (7,)
        or not np.all(np.isfinite(positions))
    ):
        raise SafetyAbort("实时关节状态必须只含七个右臂关节")
    columns = {name: index for index, name in enumerate(names)}
    current = np.asarray(
        [positions[columns[name]] for name in EXPECTED_JOINTS],
        dtype=float,
    )
    planned = np.asarray(payload["points"][0]["positions_deg"], dtype=float)
    error = float(np.max(np.abs(current - planned)))
    if error > params.planned_start_tolerance_deg:
        raise SafetyAbort(
            "实时右臂关节与 MTC 轨迹起点不匹配: "
            f"最大差={error:.2f}°, 上限={params.planned_start_tolerance_deg:.2f}°"
        )
    validate_open_gripper_feedback(gripper_open_feedback, params)


def validate_attach_gate(
    payload: dict,
    *,
    point_index: int,
    gripper_close_feedback: dict,
    empty_close_pos: int,
    params: DemoParams | None = None,
) -> None:
    """Validate close feedback exactly at the exported attach boundary."""
    validate_pick_trajectory(payload)
    params = params or DemoParams()
    attach = next(
        item for item in payload["phase_boundaries"] if item["name"] == "attach"
    )
    if point_index != attach["start_index"]:
        raise SafetyAbort("夹爪闭合不在 attach 边界，拒绝进入 retreat")
    validate_holding_gripper_feedback(
        gripper_close_feedback,
        params,
        empty_close_pos=empty_close_pos,
    )


def validate_place_trajectory(payload: dict) -> None:
    """Validate the blocked place-only export; this never authorizes motion."""
    required = {
        "schema_version": "grabber.mtc_place.v1",
        "plan_only": True,
        "execution_supported": False,
        "execution_block_reason": PLACE_PLAN_ONLY_BLOCK_REASON,
        "mode": "place_only",
        "arm_id": "right_arm",
        "joint_units": "degrees",
    }
    if not isinstance(payload, dict):
        raise SafetyAbort("MTC place 轨迹顶层必须是对象")
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise SafetyAbort(f"MTC place 契约字段 {key} 必须是 {expected!r}")
    if not isinstance(payload.get("scenario_id"), str) or not payload["scenario_id"]:
        raise SafetyAbort("MTC place scenario_id 必须是非空字符串")
    if tuple(payload.get("joint_names") or ()) != EXPECTED_JOINTS:
        raise SafetyAbort("MTC place 只允许有序右臂关节")
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise SafetyAbort("MTC place 轨迹至少需要两个点")
    last_time = -1.0
    for index, point in enumerate(points):
        if isinstance(point, dict):
            _validate_optional_accelerations(point, index, "place")
        positions = np.asarray(
            point.get("positions_deg") if isinstance(point, dict) else None,
            dtype=float,
        )
        velocities = np.asarray(
            point.get("velocities_deg_s") if isinstance(point, dict) else None,
            dtype=float,
        )
        time_s = point.get("time_from_start_s") if isinstance(point, dict) else None
        if (
            positions.shape != (7,)
            or velocities.shape != (7,)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(velocities))
            or isinstance(time_s, bool)
            or not isinstance(time_s, (int, float))
            or not math.isfinite(float(time_s))
            or float(time_s) < 0.0
            or (index > 0 and float(time_s) <= last_time)
        ):
            raise SafetyAbort(f"MTC place 轨迹点 {index} 维度、数值或时间无效")
        last_time = float(time_s)
    phases = payload.get("phase_boundaries")
    if not isinstance(phases, list) or tuple(
        item.get("name") for item in phases if isinstance(item, dict)
    ) != EXPECTED_PLACE_PHASES:
        raise SafetyAbort("MTC place 阶段必须严格为 transport/approach/release/retreat")
    try:
        bounds = {
            item["name"]: (int(item["start_index"]), int(item["end_index"]))
            for item in phases
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise SafetyAbort("MTC place 阶段边界格式无效") from exc
    transport, approach, release, retreat = (
        bounds[name] for name in EXPECTED_PLACE_PHASES
    )
    if not (
        transport[0] == 0
        and transport[1] == approach[0]
        and transport[1] < approach[1]
        and release == (approach[1], approach[1])
        and retreat[0] == release[1]
        and retreat[0] < retreat[1] == len(points) - 1
    ):
        raise SafetyAbort(f"MTC place 阶段边界不连续或顺序无效: {bounds!r}")
    expected_events = [
        ("hold_before_motion", 0, "validate_holding_gripper_feedback"),
        ("open_at_release", release[0], "RobotSession.open_gripper"),
    ]
    actual_events = []
    for event in payload.get("gripper_events") or []:
        if not isinstance(event, dict) or event.get("feedback_required") is not True:
            raise SafetyAbort("MTC place 夹爪事件必须要求反馈")
        actual_events.append(
            (event.get("name"), event.get("point_index"), event.get("operation"))
        )
    if actual_events != expected_events:
        raise SafetyAbort("MTC place 夹持/释放事件与阶段边界不一致")
    max_age_s = _finite(payload.get("freshness_max_age_s"), "MTC place freshness")
    if max_age_s <= 0.0:
        raise SafetyAbort("MTC place freshness 必须为正数")
    _timestamp(payload.get("scene_captured_at_utc"), "放置场景时间")


def load_place_trajectory(path: str | Path) -> dict:
    payload = _read_json(path, "MTC place 轨迹")
    validate_place_trajectory(payload)
    return payload


def validate_place_execution_bundle(
    result: dict,
    trajectory: dict,
    scenario: dict,
    *,
    params: DemoParams | None = None,
) -> dict:
    validate_place_trajectory(trajectory)
    params = params or DemoParams()
    for key, expected in {
        "plan_only": True,
        "solved": True,
        "mode": "place_only",
        "selected_arm": "right_arm",
        "execution_eligible": False,
        "execution_block_reason": PLACE_PLAN_ONLY_BLOCK_REASON,
        "fixture_source": False,
    }.items():
        if result.get(key) != expected:
            raise SafetyAbort(f"MTC place 结果字段 {key} 必须是 {expected!r}")
    for key, expected in {
        "mode": "place_only",
        "planning_arm_id": "right_arm",
        "fixture_source": False,
        "start_state_source": "current_state",
        "spawn_scene_objects": True,
    }.items():
        if scenario.get(key) != expected:
            raise SafetyAbort(f"MTC place 场景字段 {key} 必须是 {expected!r}")
    scenario_id = trajectory["scenario_id"]
    if result.get("scenario_id") != scenario_id or scenario.get("scenario_id") != scenario_id:
        raise SafetyAbort("place result/trajectory/scenario 的 scenario_id 不一致")
    if result.get("solved_by_arm", {}).get("right_arm__place") is not True:
        raise SafetyAbort("右臂 place-only 分支没有完整解")
    if result.get("selected_solution_id") != "right_arm__place#execution_safe":
        raise SafetyAbort("MTC place 结果不是通过执行资格审计的放置解")
    count = result.get("complete_solution_count_by_arm", {}).get("right_arm__place")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise SafetyAbort("右臂 place-only 完整解数量无效")
    if scenario.get("scene_captured_at_utc") != trajectory.get("scene_captured_at_utc"):
        raise SafetyAbort("place 场景与轨迹的采集时间不一致")
    if _finite(scenario.get("freshness_max_age_s"), "place freshness") != float(
        trajectory["freshness_max_age_s"]
    ):
        raise SafetyAbort("place 场景与轨迹的新鲜度上限不一致")
    if result.get("scene_version") != scenario.get("scene_version"):
        raise SafetyAbort("place 结果与场景的 scene_version 不一致")
    provenance = scenario.get("placement_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("support_source")
        != "verified_shelf_geometry_operator_obstacle_confirmation"
    ):
        raise SafetyAbort("place 场景缺少已确认的货架遮挡补全来源")
    if not isinstance(scenario.get("obstacle_voxels"), list):
        raise SafetyAbort("place 场景缺少非目标实时障碍")
    shelf_ids = {
        item.get("id")
        for item in scenario.get("shelf_boxes", [])
        if isinstance(item, dict)
    }
    if not {"fence_shelf_bottom", "fence_shelf_top", "fence_shelf_back"}.issubset(
        shelf_ids
    ):
        raise SafetyAbort("place 场景缺少货架底/顶/背几何")
    start = _validate_selected_arm_start_state(
        result.get("start_state"), arm_id="right_arm", label="右臂"
    )
    joints = start.get("joints") if isinstance(start, dict) else None
    if not isinstance(joints, dict):
        raise SafetyAbort("MTC place 结果缺少实时 start_state 关节")
    try:
        right = np.degrees(
            np.asarray([joints[name] for name in EXPECTED_JOINTS], dtype=float)
        )
        left = np.degrees(
            np.asarray([joints[name] for name in EXPECTED_LEFT_JOINTS], dtype=float)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SafetyAbort("MTC place start_state 缺少左右臂关节") from exc
    if right.shape != (7,) or left.shape != (7,) or not np.all(np.isfinite(right)) or not np.all(np.isfinite(left)):
        raise SafetyAbort("MTC place start_state 关节无效")
    held_right = np.asarray(provenance.get("held_right_joints_deg"), dtype=float)
    if (
        held_right.shape != (7,)
        or not np.all(np.isfinite(held_right))
        or float(np.max(np.abs(right - held_right)))
        > params.planned_start_tolerance_deg
    ):
        raise SafetyAbort("MTC place 起点已偏离升降后持瓶右臂快照")
    home = np.asarray(scenario.get("post_place_home_joints_deg"), dtype=float)
    endpoint = np.asarray(trajectory["points"][-1]["positions_deg"], dtype=float)
    if (
        home.shape != (7,)
        or not np.all(np.isfinite(home))
        or float(np.max(np.abs(home - endpoint)))
        > params.planned_start_tolerance_deg
    ):
        raise SafetyAbort("MTC place 轨迹未回到场景指定的安全 home 位")
    planned = np.asarray(trajectory["points"][0]["positions_deg"], dtype=float)
    if float(np.max(np.abs(right - planned))) > params.planned_start_tolerance_deg:
        raise SafetyAbort("MTC place start_state 与轨迹起点不一致")
    platform_m = _finite(joints.get("platform_joint"), "MTC platform_joint")
    if not 0.0 <= platform_m <= 1.0:
        raise SafetyAbort("MTC platform_joint 超出 0..1 m")
    return {
        "scenario_id": scenario_id,
        "right_start_deg": right.tolist(),
        "left_start_deg": left.tolist(),
        "lift_start_mm": int(round(platform_m * 1000.0)),
    }


def validate_place_pre_motion_gate(
    payload: dict,
    *,
    current_state: dict,
    gripper_holding_feedback: dict,
    now: datetime | None = None,
    params: DemoParams | None = None,
) -> None:
    validate_place_trajectory(payload)
    params = params or DemoParams()
    now = now or datetime.now(timezone.utc)
    max_age_s = float(payload["freshness_max_age_s"])
    _assert_fresh(
        payload["scene_captured_at_utc"],
        label="放置场景时间",
        now=now,
        max_age_s=max_age_s,
    )
    _assert_fresh(
        current_state.get("captured_at_utc") if isinstance(current_state, dict) else None,
        label="实时关节时间",
        now=now,
        max_age_s=max_age_s,
    )
    names = list(current_state.get("joint_names") or [])
    positions = np.asarray(current_state.get("positions_deg"), dtype=float)
    if len(names) != 7 or len(set(names)) != 7 or set(names) != set(EXPECTED_JOINTS) or positions.shape != (7,) or not np.all(np.isfinite(positions)):
        raise SafetyAbort("place 实时关节状态必须只含七个右臂关节")
    columns = {name: index for index, name in enumerate(names)}
    current = np.asarray([positions[columns[name]] for name in EXPECTED_JOINTS])
    planned = np.asarray(payload["points"][0]["positions_deg"], dtype=float)
    if float(np.max(np.abs(current - planned))) > params.planned_start_tolerance_deg:
        raise SafetyAbort("place 实时右臂关节与轨迹起点不匹配")
    try:
        state = int(gripper_holding_feedback["dof_state"][0])
        pos = int(gripper_holding_feedback["pos"][0])
        current = int(gripper_holding_feedback["current"][0])
        speed = int(gripper_holding_feedback["speed"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SafetyAbort("place 夹爪反馈缺失或格式无效") from exc
    if (
        state != 3
        or speed != 0
        or current < max(20, params.gripper_force // 2)
        or not 0 <= pos <= params.gripper_open_position
    ):
        raise SafetyAbort(
            "place 前夹爪未处于稳定力控夹持: "
            f"state={state}, pos={pos}, current={current}, speed={speed}"
        )


def validate_place_release_gate(
    payload: dict,
    *,
    point_index: int,
    gripper_open_feedback: dict,
    params: DemoParams | None = None,
) -> None:
    validate_place_trajectory(payload)
    release = next(
        item for item in payload["phase_boundaries"] if item["name"] == "release"
    )
    if point_index != release["start_index"]:
        raise SafetyAbort("夹爪打开不在 release 边界")
    validate_open_gripper_feedback(gripper_open_feedback, params or DemoParams())
