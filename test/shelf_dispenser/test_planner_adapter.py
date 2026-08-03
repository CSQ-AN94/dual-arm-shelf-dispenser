"""MoveIt subprocess adapter error normalization; no ROS required."""

import subprocess

import numpy as np
import pytest

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.planner import MoveItPlanner


def _planner(tmp_path):
    return MoveItPlanner(tmp_path, tmp_path)


def _plan(planner, *, minimum_link7_z=None):
    return planner.plan(
        name="probe",
        start_joints_deg=[0.0] * 7,
        start_left_joints_deg=[0.0] * 7,
        goal_joints_deg=[1.0] * 7,
        target_flange=np.eye(4),
        obstacles=[],
        boxes=[],
        workspace={"min": [-1, -1, -1], "max": [1, 1, 1]},
        planning_frame="platform_base_link",
        tool_guard={"xy": 0.1, "length": 0.2, "center_z": 0.1},
        voxel_size=0.05,
        minimum_link7_z=minimum_link7_z,
    )


def test_moveit_helper_timeout_becomes_safety_abort(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(SafetyAbort, match="规划 probe 超时"):
        _plan(_planner(tmp_path))


def test_moveit_malformed_output_becomes_safety_abort(tmp_path, monkeypatch):
    output_path = tmp_path / "probe_plan.json"

    def malformed(*args, **kwargs):
        output_path.write_text("not json", encoding="utf-8")
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(subprocess, "run", malformed)

    with pytest.raises(SafetyAbort, match="结果无法解析"):
        _plan(_planner(tmp_path))


def test_moveit_success_requires_live_tool_guard_and_fk_evidence(
    tmp_path, monkeypatch
):
    planner = _planner(tmp_path)
    payload = {
        "success": True,
        "error_code": 1,
        "planning_time": 0.1,
        "joint_names": [f"r_joint{i}" for i in range(1, 8)],
        "points_deg": [[1.0] * 7],
        "start_link7_fk": {
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "endpoint_link7_fk": {
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "attached_object_ids": [],
        "world_collision_ids": [],
    }
    monkeypatch.setattr(planner, "_run_json_helper", lambda **_: payload)

    with pytest.raises(SafetyAbort, match="bottle_tool_guard"):
        _plan(planner)


def test_moveit_request_records_explicit_goal_constraint(tmp_path, monkeypatch):
    planner = _planner(tmp_path)
    payload = {
        "success": True,
        "error_code": 1,
        "planning_time": 0.1,
        "joint_names": [f"r_joint{i}" for i in range(1, 8)],
        "points_deg": [[1.0] * 7],
        "start_link7_fk": {
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "endpoint_link7_fk": {
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "attached_object_ids": ["bottle_tool_guard"],
        "world_collision_ids": [],
    }
    monkeypatch.setattr(planner, "_run_json_helper", lambda **_: payload)

    _plan(planner)

    request = __import__("json").loads(
        (tmp_path / "probe_request.json").read_text(encoding="utf-8")
    )
    assert request["goal_constraint"] == "pose"


def test_moveit_request_records_link7_vertical_path_floor(
    tmp_path, monkeypatch
):
    planner = _planner(tmp_path)
    payload = {
        "success": True,
        "error_code": 1,
        "planning_time": 0.1,
        "joint_names": [f"r_joint{i}" for i in range(1, 8)],
        "points_deg": [[1.0] * 7],
        "start_link7_fk": {
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "endpoint_link7_fk": {
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "attached_object_ids": ["bottle_tool_guard"],
        "world_collision_ids": [],
    }
    monkeypatch.setattr(planner, "_run_json_helper", lambda **_: payload)

    _plan(planner, minimum_link7_z=-0.25)

    request = __import__("json").loads(
        (tmp_path / "probe_request.json").read_text(encoding="utf-8")
    )
    assert request["minimum_link7_z"] == -0.25


def test_moveit_trajectory_columns_are_reordered_by_joint_names(
    tmp_path, monkeypatch
):
    planner = _planner(tmp_path)
    names = [f"r_joint{i}" for i in (7, 2, 5, 1, 4, 6, 3)]
    payload = {
        "success": True,
        "error_code": 1,
        "planning_time": 0.1,
        "joint_names": names,
        "points_deg": [[float(name.removeprefix("r_joint")) for name in names]],
        "start_link7_fk": {
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "endpoint_link7_fk": {
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "attached_object_ids": ["bottle_tool_guard"],
        "world_collision_ids": [],
    }
    monkeypatch.setattr(planner, "_run_json_helper", lambda **_: payload)

    plan = _plan(planner)

    assert plan["joint_names"] == [f"r_joint{i}" for i in range(1, 8)]
    assert plan["points_deg"] == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]


@pytest.mark.parametrize(
    "names",
    [
        [f"r_joint{i}" for i in range(1, 7)],
        ["r_joint1"] * 7,
        [*[f"r_joint{i}" for i in range(1, 7)], "torso_joint"],
    ],
)
def test_moveit_trajectory_rejects_missing_duplicate_or_extra_joint(
    tmp_path, monkeypatch, names
):
    planner = _planner(tmp_path)
    payload = {
        "success": True,
        "error_code": 1,
        "planning_time": 0.1,
        "joint_names": names,
        "points_deg": [[0.0] * len(names)],
        "start_link7_fk": {
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "endpoint_link7_fk": {
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        "attached_object_ids": ["bottle_tool_guard"],
        "world_collision_ids": [],
    }
    monkeypatch.setattr(planner, "_run_json_helper", lambda **_: payload)

    with pytest.raises(SafetyAbort, match="轨迹关节集合"):
        _plan(planner)


def test_live_scene_contract_requires_dynamic_rgbd_voxels():
    result = {
        "attached_object_ids": ["bottle_tool_guard"],
        "world_collision_ids": ["fence_table_top"],
    }

    with pytest.raises(SafetyAbort, match="rgbd_voxels"):
        MoveItPlanner._assert_live_scene_contract(
            result,
            obstacles=[[0.1, 0.2, 0.3]],
            boxes=[{"id": "fence_table_top"}],
            tool_guard={"xy": 0.1},
        )
