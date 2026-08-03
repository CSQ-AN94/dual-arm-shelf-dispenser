from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from bottle_grasp.core import SafetyAbort
from bottle_grasp.mtc_execution import (
    execute_lift_transfer,
    execute_pick,
    execute_place,
    load_gripper_calibration_record,
)
from bottle_grasp.mtc_pick_contract import (
    EXPECTED_JOINTS,
    EXPECTED_LEFT_JOINTS,
    validate_attach_gate,
    validate_execution_bundle,
    validate_place_execution_bundle,
    validate_place_pre_motion_gate,
    validate_place_release_gate,
    validate_place_trajectory,
    validate_pick_trajectory,
    validate_pre_motion_gate,
)


def _trajectory(now: datetime) -> dict:
    points = [
        {
            "time_from_start_s": float(index),
            "positions_deg": [index] * 7,
            "velocities_deg_s": [1.0] * 7,
            "accelerations_deg_s2": [0.5] * 7,
        }
        for index in range(6)
    ]
    return {
        "schema_version": "grabber.mtc_pick.v2",
        "plan_only": True,
        "execution_supported": False,
        "execution_block_reason": "EXTERNAL_GATED_PICK_EXECUTOR_REQUIRED",
        "mode": "pick_only",
        "scenario_id": "test",
        "arm_id": "right_arm",
        "grasp_candidate_id": "wrist_roll_180",
        "target_captured_at_utc": now.isoformat(),
        "scene_captured_at_utc": now.isoformat(),
        "freshness_max_age_s": 45.0,
        "joint_units": "degrees",
        "joint_names": list(EXPECTED_JOINTS),
        "points": points,
        "phase_boundaries": [
            {"name": "pregrasp", "start_index": 0, "end_index": 2},
            {"name": "approach", "start_index": 2, "end_index": 4},
            {"name": "attach", "start_index": 4, "end_index": 4},
            {"name": "retreat", "start_index": 4, "end_index": 5},
        ],
        "gripper_events": [
            {
                "name": "open_before_motion",
                "point_index": 0,
                "operation": "RobotSession.open_gripper",
                "feedback_required": True,
            },
            {
                "name": "close_at_attach",
                "point_index": 4,
                "operation": "RobotSession.close_gripper",
                "feedback_required": True,
            },
        ],
    }


def _left_plan_only_trajectory(now: datetime) -> dict:
    trajectory = _trajectory(now)
    trajectory["arm_id"] = "left_arm"
    trajectory["joint_names"] = list(EXPECTED_LEFT_JOINTS)
    trajectory["execution_block_reason"] = "LEFT_TOOL_CALIBRATION_REQUIRED"
    return trajectory


def test_left_pick_export_is_valid_plan_only_but_execution_bridge_rejects_it():
    trajectory = _left_plan_only_trajectory(datetime.now(timezone.utc))
    validate_pick_trajectory(trajectory)
    with pytest.raises(SafetyAbort, match="执行桥只支持右臂"):
        validate_execution_bundle(
            _execution_result(trajectory),
            trajectory,
            _execution_scenario(trajectory),
        )


def _feedback(*, state: int, pos: int, current: int = 120) -> dict:
    return {
        "dof_state": [state],
        "pos": [pos],
        "current": [current],
        "speed": [0],
    }


class _Robot:
    def __init__(self, joints, *, holding=False, planned_tcp=None):
        self.q = list(joints)
        self.planned_tcp = (
            np.eye(4)
            if planned_tcp is None
            else np.asarray(planned_tcp, dtype=float)
        )
        self.stop_event = threading.Event()
        self.events = []
        self.feedback = (
            _feedback(state=3, pos=402)
            if holding
            else _feedback(state=2, pos=900)
        )

    def assert_arm_healthy(self):
        return {}

    def current_tcp(self):
        return np.eye(4)

    def tcp_from_joints(self, _joints):
        return self.planned_tcp.copy()

    def controller_fence_status(self):
        return {"state": {"enable_state": False}}

    def joints_deg(self):
        return list(self.q)

    def calibrate_empty_close(self, _params):
        self.events.append("calibrate")
        self.feedback = _feedback(state=2, pos=900)
        return 394

    def gripper_state(self):
        return self.feedback

    def validate_planned_joints(self, points, *_args, **_kwargs):
        self.events.append(("validate", len(points)))
        return len(points)

    @staticmethod
    def _compress_connected_joint_path(_start, points):
        return list(points)

    def execute_planned_joints(
        self, points, _speed, _step, *, expected_start_joints_deg, **_kwargs
    ):
        np.testing.assert_allclose(self.q, expected_start_joints_deg)
        self.events.append(("move", len(points)))
        self.q = list(points[-1])

    def close_gripper(self, _params):
        self.events.append("close")
        self.feedback = _feedback(state=3, pos=402)
        return self.feedback

    def open_gripper(self, _params):
        self.events.append("open")
        self.feedback = _feedback(state=2, pos=900)
        return self.feedback

    def hold(self):
        self.events.append("hold")


class _Left:
    def __init__(self, joints):
        self.q = list(joints)

    def joints_deg(self):
        return list(self.q)


class _Lift:
    def __init__(self, height):
        self.height = height
        self.moves = []

    def state(self):
        return SimpleNamespace(height_mm=self.height, mode=0)

    def move_to(self, height, *, speed):
        self.moves.append((height, speed))
        self.height = height
        return self.state()


class _Chassis:
    def state(self):
        return SimpleNamespace(
            x_m=1.0,
            y_m=2.0,
            yaw_rad=0.3,
            linear_mps=0.0,
            angular_radps=0.0,
            control_mode="kAuto",
            robot_state="kIdle",
        )


def test_pick_contract_and_temporal_gripper_gates():
    now = datetime.now(timezone.utc)
    trajectory = _trajectory(now)
    validate_pick_trajectory(trajectory)
    validate_pre_motion_gate(
        trajectory,
        current_state={
            "captured_at_utc": now.isoformat(),
            "joint_names": list(EXPECTED_JOINTS),
            "positions_deg": [0.0] * 7,
        },
        gripper_open_feedback=_feedback(state=2, pos=900),
        now=now,
    )
    validate_attach_gate(
        trajectory,
        point_index=4,
        gripper_close_feedback=_feedback(state=3, pos=402),
        empty_close_pos=394,
    )


def test_pick_gate_rejects_wrong_joints_start_freshness_and_attach_timing():
    now = datetime.now(timezone.utc)
    trajectory = _trajectory(now)

    wrong_joints = deepcopy(trajectory)
    wrong_joints["joint_names"][-1] = "l_joint7"
    with pytest.raises(SafetyAbort, match="right_arm"):
        validate_pick_trajectory(wrong_joints)

    with pytest.raises(SafetyAbort, match="起点不匹配"):
        validate_pre_motion_gate(
            trajectory,
            current_state={
                "captured_at_utc": now.isoformat(),
                "joint_names": list(EXPECTED_JOINTS),
                "positions_deg": [2.0] * 7,
            },
            gripper_open_feedback=_feedback(state=2, pos=900),
            now=now,
        )

    stale = _trajectory(now - timedelta(seconds=60))
    with pytest.raises(SafetyAbort, match="不新鲜"):
        validate_pre_motion_gate(
            stale,
            current_state={
                "captured_at_utc": now.isoformat(),
                "joint_names": list(EXPECTED_JOINTS),
                "positions_deg": [0.0] * 7,
            },
            gripper_open_feedback=_feedback(state=2, pos=900),
            now=now,
        )

    with pytest.raises(SafetyAbort, match="attach"):
        validate_attach_gate(
            trajectory,
            point_index=3,
            gripper_close_feedback=_feedback(state=3, pos=402),
            empty_close_pos=394,
        )
    with pytest.raises(SafetyAbort, match="空夹"):
        validate_attach_gate(
            trajectory,
            point_index=4,
            gripper_close_feedback=_feedback(state=3, pos=395),
            empty_close_pos=394,
        )

    bad_acceleration = deepcopy(trajectory)
    bad_acceleration["points"][2]["accelerations_deg_s2"] = [0.5] * 6
    with pytest.raises(SafetyAbort, match="加速度"):
        validate_pick_trajectory(bad_acceleration)

    missing_acceleration = deepcopy(trajectory)
    del missing_acceleration["points"][2]["accelerations_deg_s2"]
    with pytest.raises(SafetyAbort, match="加速度"):
        validate_pick_trajectory(missing_acceleration)


def _execution_result(trajectory: dict) -> dict:
    candidate = trajectory["grasp_candidate_id"]
    joints = {
        **{
            name: float(np.radians(value))
            for name, value in zip(
                EXPECTED_JOINTS, trajectory["points"][0]["positions_deg"]
            )
        },
        **{f"l_joint{index}": float(np.radians(20 + index)) for index in range(1, 8)},
        "platform_joint": 0.718,
    }
    return {
        "scenario_id": trajectory["scenario_id"],
        "mode": "pick_only",
        "plan_only": True,
        "solved": True,
        "selected_arm": "right_arm",
        "selected_grasp_candidate": candidate,
        "selected_solution_id": f"right_arm__{candidate}#execution_safe",
        "execution_eligible": False,
        "execution_block_reason": "EXTERNAL_GATED_PICK_EXECUTOR_REQUIRED",
        "fixture_source": False,
        "scene_version": "live-scene",
        "start_state": {
            "all_zero": False,
            "selected_arm": "right_arm",
            "selected_arm_complete": True,
            "joint_state_stamp_ns": 1_700_000_000_000_000_000,
            "joint_state_age_s_at_planning": 0.05,
            "joints": joints,
        },
        "solved_by_arm": {f"right_arm__{candidate}": True},
        "complete_solution_count_by_arm": {f"right_arm__{candidate}": 1},
    }


def _execution_scenario(trajectory: dict) -> dict:
    return {
        "scenario_id": trajectory["scenario_id"],
        "mode": "pick_only",
        "planning_arm_id": "right_arm",
        "fixture_source": False,
        "start_state_source": "current_state",
        "spawn_scene_objects": True,
        "target_captured_at_utc": trajectory["target_captured_at_utc"],
        "scene_captured_at_utc": trajectory["scene_captured_at_utc"],
        "freshness_max_age_s": trajectory["freshness_max_age_s"],
        "scene_version": "live-scene",
        "localization_provenance": {"profile": "shelf_template"},
        "scene_provenance": {"sha256": "scene"},
        "obstacle_voxels": [],
        "shelf_boxes": [
            {"id": "fence_shelf_bottom"},
            {"id": "fence_shelf_top"},
            {"id": "fence_shelf_back"},
        ],
        # A width the installed RMG24 can actually span; the executor refuses
        # anything at or beyond gripper_max_opening_m.
        "bottle": {"id": "bottle", "radius_m": 0.025},
        "source_grasp_pose": {
            "xyz": [0.0, 0.0, 0.0],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "source_approach_direction": [0.0, 0.0, 1.0],
        "source_grasp_candidates": [
            {
                "id": trajectory["grasp_candidate_id"],
                "pose": {
                    "xyz": [0.0, 0.0, 0.0],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ],
        "tcp_path_workspace": {
            "id": "tcp_path_workspace",
            "size": [1.0, 1.0, 1.0],
            "pose": {"xyz": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]},
        },
    }


def test_execution_bundle_binds_plan_only_artifacts():
    now = datetime.now(timezone.utc)
    trajectory = _trajectory(now)
    trajectory["points"] = [
        {**point, "positions_deg": [10.0 + index] * 7}
        for index, point in enumerate(trajectory["points"])
    ]
    result = _execution_result(trajectory)
    scenario = _execution_scenario(trajectory)
    summary = validate_execution_bundle(result, trajectory, scenario)
    assert summary["lift_start_mm"] == 718
    unchecked = deepcopy(result)
    unchecked["selected_solution_id"] = "right_arm#best"
    with pytest.raises(SafetyAbort, match="执行资格审计"):
        validate_execution_bundle(unchecked, trajectory, scenario)

    robot = _Robot(trajectory["points"][0]["positions_deg"])
    completed = execute_pick(
        result,
        trajectory,
        scenario,
        robot=robot,
        left_reader=_Left(summary["left_start_deg"]),
        lift_state=SimpleNamespace(height_mm=718, mode=0),
        safety_profile=SimpleNamespace(T_moveit_from_profile=np.eye(4)),
        empty_close_pos=394,
        allow_sdk_retiming=True,
    )
    assert robot.events == [
        ("validate", 6),
        "open",
        ("move", 5),
        "close",
        ("move", 2),
    ]
    assert completed["final_right_joints_deg"] == [15.0] * 7


def test_mtc_execution_refuses_to_drop_timing_without_explicit_acknowledgement():
    now = datetime.now(timezone.utc)
    trajectory = _trajectory(now)
    result = _execution_result(trajectory)
    scenario = _execution_scenario(trajectory)
    robot = _Robot(trajectory["points"][0]["positions_deg"])

    with pytest.raises(SafetyAbort, match="SDK.*重定时"):
        execute_pick(
            result,
            trajectory,
            scenario,
            robot=robot,
            left_reader=_Left([21, 22, 23, 24, 25, 26, 27]),
            lift_state=SimpleNamespace(height_mm=718, mode=0),
            safety_profile=SimpleNamespace(T_moveit_from_profile=np.eye(4)),
            empty_close_pos=394,
        )

    assert robot.events == []


def test_execution_bundle_requires_selected_arm_joint_state_freshness_proof():
    now = datetime.now(timezone.utc)
    trajectory = _trajectory(now)
    result = _execution_result(trajectory)

    # platform_joint makes the old global all_zero bit false even though it
    # says nothing about whether the seven selected-arm values were observed.
    result["start_state"]["joints"].update(
        {name: 0.0 for name in EXPECTED_JOINTS}
    )
    result["start_state"].pop("selected_arm_complete")
    trajectory["points"][0]["positions_deg"] = [0.0] * 7

    with pytest.raises(SafetyAbort, match="所选右臂.*新鲜"):
        validate_execution_bundle(
            result,
            trajectory,
            _execution_scenario(trajectory),
        )

    result = _execution_result(trajectory)
    result["start_state"]["all_zero"] = True
    result["start_state"]["joints"].update(
        {name: 0.0 for name in (*EXPECTED_JOINTS, *EXPECTED_LEFT_JOINTS)}
    )
    result["start_state"]["joints"]["platform_joint"] = 0.0
    summary = validate_execution_bundle(
        result,
        trajectory,
        _execution_scenario(trajectory),
    )
    assert summary["right_start_deg"] == [0.0] * 7


def test_pick_executor_rejects_wrong_lift_before_gripper_or_arm_commands():
    now = datetime.now(timezone.utc)
    trajectory = _trajectory(now)
    result = _execution_result(trajectory)
    scenario = _execution_scenario(trajectory)
    summary = validate_execution_bundle(result, trajectory, scenario)
    robot = _Robot(trajectory["points"][0]["positions_deg"])

    with pytest.raises(SafetyAbort, match="升降高度"):
        execute_pick(
            result,
            trajectory,
            scenario,
            robot=robot,
            left_reader=_Left(summary["left_start_deg"]),
            lift_state=SimpleNamespace(height_mm=250, mode=0),
            safety_profile=SimpleNamespace(T_moveit_from_profile=np.eye(4)),
            empty_close_pos=394,
            allow_sdk_retiming=True,
        )

    assert robot.events == []


def test_pick_executor_rejects_a_target_wider_than_the_gripper():
    """Planning treats the bottle as a collision cylinder, not a graspable width.

    The limit is a per-robot measurement carried on the safety profile.  A
    profile that has not measured it performs no check at all, so the value is
    set explicitly here rather than relying on a default.
    """
    now = datetime.now(timezone.utc)
    trajectory = _trajectory(now)
    result = _execution_result(trajectory)
    scenario = _execution_scenario(trajectory)
    scenario["bottle"]["radius_m"] = 0.033
    summary = validate_execution_bundle(result, trajectory, scenario)
    robot = _Robot(trajectory["points"][0]["positions_deg"])
    measured = SimpleNamespace(
        T_moveit_from_profile=np.eye(4), gripper_max_opening_m=0.065
    )

    with pytest.raises(SafetyAbort, match="宽于夹爪可张开量"):
        execute_pick(
            result,
            trajectory,
            scenario,
            robot=robot,
            left_reader=_Left(summary["left_start_deg"]),
            lift_state=SimpleNamespace(height_mm=718, mode=0),
            safety_profile=measured,
            empty_close_pos=394,
            allow_sdk_retiming=True,
        )

    assert "open" not in robot.events


def test_pick_executor_skips_the_width_check_when_it_is_unmeasured():
    """An unmeasured profile must not enforce a guessed opening.

    The 65 mm implied by the robot description cannot be reconciled with
    grasps that demonstrably held, so refusing on it would block real work.
    """
    now = datetime.now(timezone.utc)
    trajectory = _trajectory(now)
    result = _execution_result(trajectory)
    scenario = _execution_scenario(trajectory)
    scenario["bottle"]["radius_m"] = 0.033
    summary = validate_execution_bundle(result, trajectory, scenario)
    robot = _Robot(trajectory["points"][0]["positions_deg"])

    execute_pick(
        result,
        trajectory,
        scenario,
        robot=robot,
        left_reader=_Left(summary["left_start_deg"]),
        lift_state=SimpleNamespace(height_mm=718, mode=0),
        safety_profile=SimpleNamespace(T_moveit_from_profile=np.eye(4)),
        empty_close_pos=394,
        allow_sdk_retiming=True,
    )

    assert "open" in robot.events


def test_pick_executor_rejects_vertical_fingers_before_opening():
    now = datetime.now(timezone.utc)
    trajectory = _trajectory(now)
    result = _execution_result(trajectory)
    scenario = _execution_scenario(trajectory)
    scenario["source_approach_direction"] = [0.0, -1.0, 0.0]
    scenario["source_grasp_candidates"][0]["pose"]["quat_xyzw"] = [
        2**-0.5,
        0.0,
        0.0,
        2**-0.5,
    ]
    summary = validate_execution_bundle(result, trajectory, scenario)
    robot = _Robot(trajectory["points"][0]["positions_deg"])

    with pytest.raises(SafetyAbort, match="姿态不是水平正对瓶身"):
        execute_pick(
            result,
            trajectory,
            scenario,
            robot=robot,
            left_reader=_Left(summary["left_start_deg"]),
            lift_state=SimpleNamespace(height_mm=718, mode=0),
            safety_profile=SimpleNamespace(T_moveit_from_profile=np.eye(4)),
            empty_close_pos=394,
            allow_sdk_retiming=True,
        )

    assert robot.events == [("validate", 6)]


def test_pick_executor_rejects_mtc_realman_fk_mismatch_before_opening():
    now = datetime.now(timezone.utc)
    trajectory = _trajectory(now)
    result = _execution_result(trajectory)
    scenario = _execution_scenario(trajectory)
    summary = validate_execution_bundle(result, trajectory, scenario)
    wrong_tcp = np.eye(4)
    wrong_tcp[0, 3] = 0.10
    robot = _Robot(
        trajectory["points"][0]["positions_deg"], planned_tcp=wrong_tcp
    )

    with pytest.raises(SafetyAbort, match="MTC/RealMan FK 不一致"):
        execute_pick(
            result,
            trajectory,
            scenario,
            robot=robot,
            left_reader=_Left(summary["left_start_deg"]),
            lift_state=SimpleNamespace(height_mm=718, mode=0),
            safety_profile=SimpleNamespace(T_moveit_from_profile=np.eye(4)),
            empty_close_pos=394,
            allow_sdk_retiming=True,
        )

    assert robot.events == [("validate", 6)]


class _Profile:
    """The taught start pose, as the safety profile carries it."""

    name = "shelf_template"
    grasp_start_lift_height_mm = 647
    grasp_start_right_joints_deg = [5.0] * 7
    grasp_start_left_joints_deg = list(range(21, 28))


def test_lift_transfer_reads_the_taught_pose_from_the_profile():
    """No side file holding a copy of the joint angles that can go stale."""
    robot = _Robot([5.0] * 7, holding=True)
    lift = _Lift(647)
    completed = execute_lift_transfer(
        {
            "schema_version": "grabber.mtc_execution.v1",
            "mode": "pick",
            "completion": {
                "final_right_joints_deg": [5.0] * 7,
                "lift_start_mm": 647,
                "empty_close_pos": 394,
            },
        },
        profile=_Profile(),
        target_height_mm=250,
        robot=robot,
        left_reader=_Left(list(range(21, 28))),
        lift=lift,
        chassis=_Chassis(),
    )
    assert lift.moves == [(250, 30)]
    assert completed["target_height_mm"] == 250
    assert completed["source_height_mm"] == 647
    assert completed["taught_pose_profile"] == "shelf_template"


def test_lift_transfer_refuses_a_target_that_is_not_below_the_start():
    record = {
        "schema_version": "grabber.mtc_execution.v1",
        "mode": "pick",
        "completion": {
            "final_right_joints_deg": [5.0] * 7,
            "lift_start_mm": 647,
            "empty_close_pos": 394,
        },
    }
    for bad in (647, 700, -1):
        with pytest.raises(SafetyAbort, match="必须低于起始高度"):
            execute_lift_transfer(
                record,
                profile=_Profile(),
                target_height_mm=bad,
                robot=_Robot([5.0] * 7, holding=True),
                left_reader=_Left(list(range(21, 28))),
                lift=_Lift(647),
                chassis=_Chassis(),
            )


def test_lift_transfer_accepts_a_recorded_post_pick_tuck():
    """A pick ends at its retreat; the tuck is a later move that records itself."""
    completion = {
        "final_right_joints_deg": [130.0] * 7,  # the shelf-branch retreat
        "lift_start_mm": 647,
        "empty_close_pos": 394,
    }
    record = {
        "schema_version": "grabber.mtc_execution.v1",
        "mode": "pick",
        "completion": completion,
    }

    def run(lift=None):
        return execute_lift_transfer(
            record,
            profile=_Profile(),
            target_height_mm=250,
            robot=_Robot([5.0] * 7, holding=True),
            left_reader=_Left(list(range(21, 28))),
            lift=lift or _Lift(647),
            chassis=_Chassis(),
        )

    with pytest.raises(SafetyAbort, match="抓取起点"):
        run()

    completion["post_pick_tuck"] = {
        "right_joints_deg": [5.0] * 7,
        "max_error_deg": 0.18,
    }
    lift = _Lift(647)
    run(lift)
    assert lift.moves == [(250, 30)]

    # A tuck stamp that disagrees with the taught pose is still refused.
    completion["post_pick_tuck"]["right_joints_deg"] = [40.0] * 7
    with pytest.raises(SafetyAbort, match="抓取起点"):
        run()


def test_gripper_calibration_record_must_be_fresh(tmp_path):
    path = tmp_path / "gripper.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "grabber.gripper_calibration.v1",
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "empty_close_pos": 394,
                "right_joints_deg": [1.0] * 7,
                "lift_height_mm": 647,
            }
        ),
        encoding="utf-8",
    )
    assert load_gripper_calibration_record(path)["empty_close_pos"] == 394
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["captured_at_utc"] = "2020-01-01T00:00:00+00:00"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SafetyAbort, match="过期"):
        load_gripper_calibration_record(path)

def test_place_contract_bundle_and_temporal_gates():
    now = datetime.now(timezone.utc)
    points = [
        {
            "time_from_start_s": float(index),
            "positions_deg": [10.0 + index] * 7,
            "velocities_deg_s": [1.0] * 7,
            "accelerations_deg_s2": [0.0] * 7,
        }
        for index in range(5)
    ]
    trajectory = {
        "schema_version": "grabber.mtc_place.v1",
        "plan_only": True,
        "execution_supported": False,
        "execution_block_reason": "EXTERNAL_GATED_PLACE_EXECUTOR_REQUIRED",
        "mode": "place_only",
        "scenario_id": "place-test",
        "arm_id": "right_arm",
        "scene_captured_at_utc": now.isoformat(),
        "freshness_max_age_s": 45.0,
        "joint_units": "degrees",
        "joint_names": list(EXPECTED_JOINTS),
        "points": points,
        "phase_boundaries": [
            {"name": "transport", "start_index": 0, "end_index": 1},
            {"name": "approach", "start_index": 1, "end_index": 3},
            {"name": "release", "start_index": 3, "end_index": 3},
            {"name": "retreat", "start_index": 3, "end_index": 4},
        ],
        "gripper_events": [
            {
                "name": "hold_before_motion",
                "point_index": 0,
                "operation": "validate_holding_gripper_feedback",
                "feedback_required": True,
            },
            {
                "name": "open_at_release",
                "point_index": 3,
                "operation": "RobotSession.open_gripper",
                "feedback_required": True,
            },
        ],
    }
    joints = {
        **{
            name: float(np.radians(10.0))
            for name in EXPECTED_JOINTS
        },
        **{f"l_joint{index}": float(np.radians(index)) for index in range(1, 8)},
        "platform_joint": 0.258,
    }
    result = {
        "scenario_id": "place-test",
        "mode": "place_only",
        "plan_only": True,
        "solved": True,
        "selected_arm": "right_arm",
        "selected_solution_id": "right_arm__place#execution_safe",
        "execution_eligible": False,
        "execution_block_reason": "EXTERNAL_GATED_PLACE_EXECUTOR_REQUIRED",
        "fixture_source": False,
        "scene_version": "live-place",
        "start_state": {
            "all_zero": False,
            "selected_arm": "right_arm",
            "selected_arm_complete": True,
            "joint_state_stamp_ns": 1_700_000_000_000_000_000,
            "joint_state_age_s_at_planning": 0.05,
            "joints": joints,
        },
        "solved_by_arm": {"right_arm__place": True},
        "complete_solution_count_by_arm": {"right_arm__place": 1},
    }
    scenario = {
        "scenario_id": "place-test",
        "mode": "place_only",
        "planning_arm_id": "right_arm",
        "fixture_source": False,
        "start_state_source": "current_state",
        "spawn_scene_objects": True,
        "scene_captured_at_utc": now.isoformat(),
        "freshness_max_age_s": 45.0,
        "scene_version": "live-place",
        "placement_provenance": {
            "support_source": (
                "verified_shelf_geometry_operator_obstacle_confirmation"
            ),
            "held_right_joints_deg": [10.0] * 7,
        },
        "obstacle_voxels": [],
        "shelf_boxes": [
            {"id": "fence_shelf_bottom"},
            {"id": "fence_shelf_top"},
            {"id": "fence_shelf_back"},
        ],
        "post_place_home_joints_deg": [14.0] * 7,
        "target_place_pose": {
            "xyz": [0.0, 0.0, 0.0],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    validate_place_trajectory(trajectory)
    validate_place_execution_bundle(result, trajectory, scenario)
    unchecked = deepcopy(result)
    unchecked["selected_solution_id"] = "right_arm__place#best"
    with pytest.raises(SafetyAbort, match="执行资格审计"):
        validate_place_execution_bundle(unchecked, trajectory, scenario)
    validate_place_pre_motion_gate(
        trajectory,
        current_state={
            "captured_at_utc": now.isoformat(),
            "joint_names": list(EXPECTED_JOINTS),
            "positions_deg": [10.0] * 7,
        },
        gripper_holding_feedback=_feedback(state=3, pos=402),
        now=now,
    )
    validate_place_release_gate(
        trajectory,
        point_index=3,
        gripper_open_feedback=_feedback(state=2, pos=900),
    )
    robot = _Robot([10.0] * 7, holding=True)
    completed = execute_place(
        result,
        trajectory,
        scenario,
        robot=robot,
        left_reader=_Left(list(range(1, 8))),
        lift_state=SimpleNamespace(height_mm=258, mode=0),
        safety_profile=SimpleNamespace(T_moveit_from_profile=np.eye(4)),
        allow_sdk_retiming=True,
    )
    assert robot.events == [
        ("validate", 5),
        ("move", 4),
        "open",
        ("move", 2),
    ]
    assert completed["final_right_joints_deg"] == [14.0] * 7
