"""Fail-closed RealMan execution of validated MTC pick/place exports."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .core import DemoParams, SafetyAbort, matrix_pose
from .environment_guard import LeftArmStabilityGuard
from .mtc_pick_contract import (
    EXPECTED_JOINTS,
    validate_attach_gate,
    validate_execution_bundle,
    validate_place_execution_bundle,
    validate_place_pre_motion_gate,
    validate_place_release_gate,
    validate_pre_motion_gate,
)
from .arm import validate_holding_gripper_feedback


def _phase_index(trajectory: dict, name: str) -> int:
    return int(
        next(
            item
            for item in trajectory["phase_boundaries"]
            if item["name"] == name
        )["start_index"]
    )


def _positions(trajectory: dict) -> list[list[float]]:
    return [list(map(float, point["positions_deg"])) for point in trajectory["points"]]


def _validate_connected_segment_fit(robot, start, points) -> None:
    if os.environ.get("BOTTLE_GRASP_CONTINUOUS_TRAJECTORY", "1") == "0":
        return
    start_array = np.asarray(start, dtype=float)
    original = []
    last = start_array
    for point in points:
        candidate = np.asarray(point, dtype=float)
        if float(np.max(np.abs(candidate - last))) <= 1e-9:
            continue
        original.append(candidate.tolist())
        last = candidate
    if original:
        robot._compress_connected_joint_path(start_array.tolist(), original)


def _current_state(robot) -> dict:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "joint_names": list(EXPECTED_JOINTS),
        "positions_deg": list(map(float, robot.joints_deg())),
    }


def _assert_lift_matches(lift_state, expected_mm: int) -> None:
    if (
        int(lift_state.mode) != 0
        or abs(int(lift_state.height_mm) - int(expected_mm)) > 5
    ):
        raise SafetyAbort(
            "升降高度/状态与 MTC 起点不一致: "
            f"actual={lift_state.height_mm} mm mode={lift_state.mode}, "
            f"planned={expected_mm} mm"
        )


def _assert_left(left_reader, expected: Sequence[float], tolerance_deg: float) -> None:
    actual = np.asarray(left_reader.joints_deg(), dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    if (
        actual.shape != (7,)
        or expected_array.shape != (7,)
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(expected_array))
    ):
        raise SafetyAbort("左臂规划快照或实时反馈无效")
    error = float(np.max(np.abs(actual - expected_array)))
    if error > tolerance_deg:
        raise SafetyAbort(
            "左臂已偏离 MTC 碰撞场快照: "
            f"最大关节差={error:.2f}°，上限={tolerance_deg:.2f}°"
        )


def _scenario_pose_matrix(pose: dict, *, label: str) -> np.ndarray:
    try:
        xyz = np.asarray(pose["xyz"], dtype=float)
        quaternion = np.asarray(pose["quat_xyzw"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise SafetyAbort(f"{label}位姿格式无效") from exc
    if (
        xyz.shape != (3,)
        or quaternion.shape != (4,)
        or not np.all(np.isfinite(xyz))
        or not np.all(np.isfinite(quaternion))
        or float(np.linalg.norm(quaternion)) < 1e-9
    ):
        raise SafetyAbort(f"{label}位姿必须是有限 xyz/quaternion")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    result[:3, 3] = xyz
    return result


def _assert_planned_tcp_matches(
    robot,
    safety_profile,
    joints_deg: Sequence[float],
    expected_moveit: np.ndarray,
    *,
    label: str,
    params: DemoParams,
    max_finger_tilt_deg: float | None = None,
) -> None:
    moveit_from_profile = np.asarray(
        safety_profile.T_moveit_from_profile, dtype=float
    )
    if (
        moveit_from_profile.shape != (4, 4)
        or not np.all(np.isfinite(moveit_from_profile))
        or not np.allclose(
            moveit_from_profile[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9
        )
        or not np.allclose(
            moveit_from_profile[:3, :3].T @ moveit_from_profile[:3, :3],
            np.eye(3),
            atol=1e-6,
        )
        or not np.isclose(
            np.linalg.det(moveit_from_profile[:3, :3]), 1.0, atol=1e-6
        )
    ):
        raise SafetyAbort("shelf_template 坐标桥无效")
    expected_profile = np.linalg.inv(moveit_from_profile) @ expected_moveit
    actual_profile = np.asarray(robot.tcp_from_joints(joints_deg), dtype=float)
    if actual_profile.shape != (4, 4) or not np.all(np.isfinite(actual_profile)):
        raise SafetyAbort(f"{label} RealMan FK 无效")
    position_error = float(
        np.linalg.norm(actual_profile[:3, 3] - expected_profile[:3, 3])
    )
    relative = expected_profile[:3, :3].T @ actual_profile[:3, :3]
    orientation_error = math.degrees(
        math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    )
    if (
        position_error > params.moveit_endpoint_position_tolerance_m
        or orientation_error > params.moveit_endpoint_orientation_tolerance_deg
    ):
        raise SafetyAbort(
            f"{label} MTC/RealMan FK 不一致: "
            f"位置差={position_error * 1000.0:.1f} mm，"
            f"姿态差={orientation_error:.1f}°"
        )
    if max_finger_tilt_deg is not None:
        finger_tilt = math.degrees(
            math.asin(float(np.clip(abs(actual_profile[2, 1]), 0.0, 1.0)))
        )
        if finger_tilt > float(max_finger_tilt_deg):
            raise SafetyAbort(
                f"{label} 左右滚转超限: finger_tilt={finger_tilt:.2f}°，"
                f"上限={float(max_finger_tilt_deg):.2f}°"
            )


def _assert_target_fits_gripper(scenario: dict, safety_profile) -> float | None:
    """Refuse a target the installed fingers cannot span.

    Planning happily solves for a bottle wider than the gripper -- the target
    is a collision cylinder, not a graspable width -- so the check belongs
    here, before any gripper command.

    The limit is a property of this robot's installed tool, so it lives on the
    safety profile.  A profile that has not measured it leaves the check off
    rather than enforcing a guess: the 0.065 m implied by
    dual_rm_75b_description cannot be reconciled with grasps that demonstrably
    held, and a wrong limit here refuses work the hardware can actually do.
    """
    limit = getattr(safety_profile, "gripper_max_opening_m", None)
    if limit is None:
        return None
    radius = (scenario.get("bottle") or {}).get("radius_m")
    if (
        isinstance(radius, bool)
        or not isinstance(radius, (int, float))
        or not math.isfinite(float(radius))
        or float(radius) <= 0.0
    ):
        raise SafetyAbort("MTC pick 场景 bottle.radius_m 必须是正有限数")
    diameter = 2.0 * float(radius)
    if diameter >= float(limit):
        raise SafetyAbort(
            "抓取目标宽于夹爪可张开量: "
            f"直径 {diameter * 1000:.1f} mm ≥ 最大张开 "
            f"{float(limit) * 1000:.1f} mm"
        )
    return diameter


def _pick_candidate_pose(scenario: dict, candidate_id: str) -> np.ndarray:
    candidate = next(
        (
            item
            for item in scenario.get("source_grasp_candidates") or []
            if isinstance(item, dict) and item.get("id") == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise SafetyAbort("MTC pick 场景缺少所选抓取候选")
    pose = _scenario_pose_matrix(candidate.get("pose"), label="MTC pick 候选")
    reference = _scenario_pose_matrix(
        scenario.get("source_grasp_pose"), label="MTC pick 零滚转基准"
    )
    # The finger line must point along the authored opening axis, not merely
    # lie on it.  Antiparallel is the same grasp only if the finger centre is
    # exactly on the tool axis, and the link7->TCP transform making that claim
    # is nominal rather than measured, so it is refused with everything else.
    authored_roll_error = math.degrees(
        math.acos(
            float(
                np.clip(reference[:3, 1] @ pose[:3, 1], -1.0, 1.0)
            )
        )
    )
    if authored_roll_error > 1e-3:
        raise SafetyAbort(
            "MTC pick 抓取姿态不是水平正对瓶身，包含左右滚转: "
            f"finger_axis_error={authored_roll_error:.3f}°"
        )
    direction = np.asarray(scenario.get("source_approach_direction"), dtype=float)
    if (
        direction.shape != (3,)
        or not np.all(np.isfinite(direction))
        or float(np.linalg.norm(direction)) < 1e-9
    ):
        raise SafetyAbort("MTC pick source_approach_direction 无效")
    direction /= float(np.linalg.norm(direction))
    approach_error = math.degrees(
        math.acos(float(np.clip(pose[:3, 2] @ direction, -1.0, 1.0)))
    )
    finger_tilt = math.degrees(
        math.asin(float(np.clip(abs(pose[2, 1]), 0.0, 1.0)))
    )
    if approach_error > 5.0 or finger_tilt > 5.0:
        raise SafetyAbort(
            "MTC pick 抓取姿态不是水平正对瓶身: "
            f"approach={approach_error:.1f}°，finger_tilt={finger_tilt:.1f}°"
        )
    return pose


def load_gripper_calibration_record(
    path: str | Path, *, max_age_s: float = 900.0
) -> dict:
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SafetyAbort(f"空夹标定证据无法读取: {exc}") from exc
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != "grabber.gripper_calibration.v1"
    ):
        raise SafetyAbort("空夹标定证据格式无效")
    try:
        captured = datetime.fromisoformat(
            str(record["captured_at_utc"]).replace("Z", "+00:00")
        )
        baseline = int(record["empty_close_pos"])
        lift_height_mm = int(record["lift_height_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SafetyAbort("空夹标定证据字段无效") from exc
    joints = np.asarray(record.get("right_joints_deg"), dtype=float)
    if captured.tzinfo is None:
        raise SafetyAbort("空夹标定证据时间缺少时区")
    age_s = (datetime.now(timezone.utc) - captured).total_seconds()
    if (
        age_s < -60.0
        or age_s > max_age_s
        or baseline < 0
        or baseline > DemoParams().gripper_open_position
        or not 0 <= lift_height_mm <= 2600
        or joints.shape != (7,)
        or not np.all(np.isfinite(joints))
    ):
        raise SafetyAbort(
            f"空夹标定证据过期或基线无效: "
            f"age={age_s:.1f}s, baseline={baseline}"
        )
    return record


def _assert_chassis_still(state, *, label: str) -> None:
    if (
        state.control_mode != "kAuto"
        or state.robot_state != "kIdle"
        or abs(float(state.linear_mps)) > 0.01
        or abs(float(state.angular_radps)) > 0.02
    ):
        raise SafetyAbort(
            f"{label}底盘不是 kAuto/kIdle 零速: "
            f"mode={state.control_mode}, state={state.robot_state}, "
            f"v={state.linear_mps}, w={state.angular_radps}"
        )


def execute_lift_transfer(
    pick_record: dict,
    *,
    profile,
    target_height_mm: int,
    robot,
    left_reader,
    lift,
    chassis,
    speed: int = 30,
    params: DemoParams | None = None,
) -> dict:
    """Lower a held bottle, with both arms on the taught start pose.

    The taught pose is read from the safety profile, not from a side file that
    holds a copy of the joint angles.  A copy is how the lift ended up driving
    the arm to a pose taught weeks earlier for a different task: nothing checks
    a copy against its source, so it goes stale silently.  One source, and the
    checks below are all against live readings.
    """
    params = params or DemoParams()
    if not 1 <= int(speed) <= 30:
        raise SafetyAbort("升降速度必须在 1..30")
    if (
        not isinstance(pick_record, dict)
        or pick_record.get("schema_version") != "grabber.mtc_execution.v1"
        or pick_record.get("mode") != "pick"
        or not isinstance(pick_record.get("completion"), dict)
    ):
        raise SafetyAbort("升降只接受已完成的 MTC pick 执行证据")
    source_height_mm = profile.grasp_start_lift_height_mm
    if source_height_mm is None:
        raise SafetyAbort(f"profile {profile.name} 未配置 grasp_start_lift_height_mm")
    if not 0 <= int(target_height_mm) < int(source_height_mm):
        raise SafetyAbort(
            f"升降目标高度 {target_height_mm} 必须低于起始高度 {source_height_mm}"
        )
    completion = pick_record["completion"]
    expected_right = np.asarray(
        profile.grasp_start_right_joints_deg, dtype=float
    )
    expected_left = np.asarray(profile.grasp_start_left_joints_deg, dtype=float)
    if expected_right.shape != (7,) or expected_left.shape != (7,):
        raise SafetyAbort(f"profile {profile.name} 的抓取起点双臂关节角不完整")
    # A pick ends where its Cartesian retreat ends, which is on the shelf IK
    # branch, not tucked.  Tucking is a separate validated move that records
    # itself here, so accept its evidence in place of the trajectory's own
    # endpoint -- but only that, never a bare claim.
    tuck = completion.get("post_pick_tuck")
    recorded_right = np.asarray(
        tuck.get("right_joints_deg")
        if isinstance(tuck, dict)
        else completion.get("final_right_joints_deg"),
        dtype=float,
    )
    if (
        recorded_right.shape != (7,)
        or float(np.max(np.abs(recorded_right - expected_right)))
        > params.planned_start_tolerance_deg
    ):
        raise SafetyAbort("pick 执行终点不是示教的抓取起点位姿")
    if int(completion.get("lift_start_mm", -1)) != int(source_height_mm):
        raise SafetyAbort("pick 执行高度与示教的起始升降高度不一致")

    validate_hardware_preflight(robot)
    actual_right = np.asarray(robot.joints_deg(), dtype=float)
    if (
        actual_right.shape != (7,)
        or not np.all(np.isfinite(actual_right))
        or float(np.max(np.abs(actual_right - expected_right)))
        > params.planned_start_tolerance_deg
    ):
        raise SafetyAbort("实时右臂不在示教的抓取起点位姿")
    _assert_left(
        left_reader, expected_left, params.planned_start_tolerance_deg
    )
    baseline = int(completion.get("empty_close_pos", -1))
    if baseline < 0:
        raise SafetyAbort("pick 执行证据缺少空夹基线")
    validate_holding_gripper_feedback(
        robot.gripper_state(), params, empty_close_pos=baseline
    )

    before_lift = lift.state()
    _assert_lift_matches(before_lift, int(source_height_mm))
    before_chassis = chassis.state()
    _assert_chassis_still(before_chassis, label="升降前")
    after_lift = lift.move_to(int(target_height_mm), speed=int(speed))
    _assert_lift_matches(after_lift, int(target_height_mm))
    after_chassis = chassis.state()
    _assert_chassis_still(after_chassis, label="升降后")
    translation = math.hypot(
        float(after_chassis.x_m) - float(before_chassis.x_m),
        float(after_chassis.y_m) - float(before_chassis.y_m),
    )
    yaw = abs(
        math.atan2(
            math.sin(float(after_chassis.yaw_rad) - float(before_chassis.yaw_rad)),
            math.cos(float(after_chassis.yaw_rad) - float(before_chassis.yaw_rad)),
        )
    )
    if translation > 0.01 or yaw > math.radians(1.0):
        raise SafetyAbort(
            "升降期间底盘发生漂移: "
            f"translation={translation:.4f}m, yaw={math.degrees(yaw):.2f}°"
        )
    actual_right = np.asarray(robot.joints_deg(), dtype=float)
    if float(np.max(np.abs(actual_right - expected_right))) > (
        params.planned_start_tolerance_deg
    ):
        raise SafetyAbort("升降后右臂偏离示教的抓取起点位姿")
    _assert_left(
        left_reader, expected_left, params.planned_start_tolerance_deg
    )
    feedback = robot.gripper_state()
    validate_holding_gripper_feedback(
        feedback, params, empty_close_pos=baseline
    )
    held_tcp = matrix_pose(robot.current_tcp())
    return {
        "source_height_mm": int(source_height_mm),
        "target_height_mm": int(after_lift.height_mm),
        "right_joints_deg": actual_right.tolist(),
        "left_joints_deg": expected_left.tolist(),
        "held_tcp_base_xyz_rpy_rad": held_tcp,
        "gripper_holding_feedback": feedback,
        "taught_pose_profile": profile.name,
    }


def validate_hardware_preflight(robot) -> None:
    robot.assert_arm_healthy()
    robot.current_tcp()
    status = robot.controller_fence_status()
    state = status.get("state") if isinstance(status, dict) else None
    if not isinstance(state, dict) or "enable_state" not in state:
        raise SafetyAbort(f"控制器原生电子围栏状态无效: {status}")
    if bool(state["enable_state"]):
        raise SafetyAbort(
            "控制器原生电子围栏仍启用；拒绝叠加未知旧围栏执行: "
            f"{status}"
        )


def _execute_segment(
    robot,
    left_reader,
    *,
    points: Sequence[Sequence[float]],
    expected_start: Sequence[float],
    expected_left: Sequence[float],
    speed: int,
    params: DemoParams,
) -> None:
    _assert_left(left_reader, expected_left, params.planned_start_tolerance_deg)
    guard = LeftArmStabilityGuard(
        left_reader=left_reader,
        stop_event=robot.stop_event,
        tolerance_deg=params.planned_start_tolerance_deg,
    )
    guard.start()
    motion_error: BaseException | None = None
    try:
        robot.execute_planned_joints(
            points,
            speed,
            params.planned_joint_step_deg,
            expected_start_joints_deg=expected_start,
            start_tolerance_deg=params.planned_start_tolerance_deg,
            tracking_tolerance_deg=params.planned_tracking_tolerance_deg,
        )
    except BaseException as exc:
        motion_error = exc
    finally:
        guard.close()
    if motion_error is not None:
        raise motion_error
    _assert_left(left_reader, expected_left, params.planned_start_tolerance_deg)


def execute_pick(
    result: dict,
    trajectory: dict,
    scenario: dict,
    *,
    robot,
    left_reader,
    lift_state,
    safety_profile,
    empty_close_pos: int,
    speed: int = 100,
    allow_sdk_retiming: bool = False,
    params: DemoParams | None = None,
) -> dict:
    """Execute current→attach, close with feedback, then retreat→carry."""
    params = params or DemoParams()
    if not 1 <= int(speed) <= 100:
        raise SafetyAbort("机械臂速度必须在 1..100")
    summary = validate_execution_bundle(result, trajectory, scenario, params=params)
    if not allow_sdk_retiming:
        raise SafetyAbort(
            "RealMan SDK movej 会重定时 MTC/Pilz 轨迹；必须明确允许 SDK 重定时，"
            "且不得把该模式当作保留 time_from_start/速度/加速度的执行"
        )
    _assert_lift_matches(lift_state, summary["lift_start_mm"])
    validate_hardware_preflight(robot)
    _assert_left(
        left_reader,
        summary["left_start_deg"],
        params.planned_start_tolerance_deg,
    )

    if not 0 <= int(empty_close_pos) <= params.gripper_open_position:
        raise SafetyAbort("空夹标定基线无效")
    points = _positions(trajectory)
    current_state = _current_state(robot)
    robot.validate_planned_joints(
        points,
        params.planned_joint_step_deg,
        safety_profile,
        start_joints_deg=current_state["positions_deg"],
    )

    attach = _phase_index(trajectory, "attach")
    _validate_connected_segment_fit(
        robot, current_state["positions_deg"], points[: attach + 1]
    )
    _validate_connected_segment_fit(robot, points[attach], points[attach:])
    attach_pose = _pick_candidate_pose(
        scenario, summary["grasp_candidate_id"]
    )
    _assert_planned_tcp_matches(
        robot,
        safety_profile,
        points[attach],
        attach_pose,
        label="MTC pick attach",
        params=params,
        max_finger_tilt_deg=0.25,
    )
    _assert_target_fits_gripper(scenario, safety_profile)
    open_feedback = robot.open_gripper(params)
    validate_pre_motion_gate(
        trajectory,
        current_state=_current_state(robot),
        gripper_open_feedback=open_feedback,
        params=params,
    )
    _execute_segment(
        robot,
        left_reader,
        points=points[: attach + 1],
        expected_start=points[0],
        expected_left=summary["left_start_deg"],
        speed=int(speed),
        params=params,
    )
    _assert_planned_tcp_matches(
        robot,
        safety_profile,
        robot.joints_deg(),
        attach_pose,
        label="MTC pick 实际 attach 到位",
        params=params,
        max_finger_tilt_deg=0.25,
    )
    close_feedback = robot.close_gripper(params)
    validate_attach_gate(
        trajectory,
        point_index=attach,
        gripper_close_feedback=close_feedback,
        empty_close_pos=empty_close_pos,
        params=params,
    )
    _execute_segment(
        robot,
        left_reader,
        points=points[attach:],
        expected_start=points[attach],
        expected_left=summary["left_start_deg"],
        speed=int(speed),
        params=params,
    )
    return {
        **summary,
        "execution_backend": "realman_sdk_movej_retimed",
        "trajectory_timing_preserved": False,
        "empty_close_pos": int(empty_close_pos),
        "final_right_joints_deg": points[-1],
        "final_tcp_base_xyz_rpy_rad": matrix_pose(robot.current_tcp()),
        "gripper_close_feedback": close_feedback,
    }


def execute_place(
    result: dict,
    trajectory: dict,
    scenario: dict,
    *,
    robot,
    left_reader,
    lift_state,
    safety_profile,
    speed: int = 100,
    allow_sdk_retiming: bool = False,
    params: DemoParams | None = None,
) -> dict:
    """Execute carry→release, open with feedback, then retreat."""
    params = params or DemoParams()
    if not 1 <= int(speed) <= 100:
        raise SafetyAbort("机械臂速度必须在 1..100")
    summary = validate_place_execution_bundle(
        result, trajectory, scenario, params=params
    )
    if not allow_sdk_retiming:
        raise SafetyAbort(
            "RealMan SDK movej 会重定时 MTC/Pilz 轨迹；必须明确允许 SDK 重定时，"
            "且不得把该模式当作保留 time_from_start/速度/加速度的执行"
        )
    _assert_lift_matches(lift_state, summary["lift_start_mm"])
    validate_hardware_preflight(robot)
    _assert_left(
        left_reader,
        summary["left_start_deg"],
        params.planned_start_tolerance_deg,
    )

    current_state = _current_state(robot)
    points = _positions(trajectory)
    robot.validate_planned_joints(
        points,
        params.planned_joint_step_deg,
        safety_profile,
        start_joints_deg=current_state["positions_deg"],
    )

    release = _phase_index(trajectory, "release")
    _validate_connected_segment_fit(
        robot, current_state["positions_deg"], points[: release + 1]
    )
    _validate_connected_segment_fit(robot, points[release], points[release:])
    _assert_planned_tcp_matches(
        robot,
        safety_profile,
        points[release],
        _scenario_pose_matrix(
            scenario.get("target_place_pose"), label="MTC place 目标"
        ),
        label="MTC place release",
        params=params,
    )
    validate_place_pre_motion_gate(
        trajectory,
        current_state=_current_state(robot),
        gripper_holding_feedback=robot.gripper_state(),
        params=params,
    )
    _execute_segment(
        robot,
        left_reader,
        points=points[: release + 1],
        expected_start=points[0],
        expected_left=summary["left_start_deg"],
        speed=int(speed),
        params=params,
    )
    open_feedback = robot.open_gripper(params)
    validate_place_release_gate(
        trajectory,
        point_index=release,
        gripper_open_feedback=open_feedback,
        params=params,
    )
    _execute_segment(
        robot,
        left_reader,
        points=points[release:],
        expected_start=points[release],
        expected_left=summary["left_start_deg"],
        speed=int(speed),
        params=params,
    )
    return {
        **summary,
        "execution_backend": "realman_sdk_movej_retimed",
        "trajectory_timing_preserved": False,
        "final_right_joints_deg": points[-1],
        "gripper_open_feedback": open_feedback,
    }
