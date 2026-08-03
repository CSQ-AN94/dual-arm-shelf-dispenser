"""Regression fixtures for the offline bottle-grasp run summarizer.

The fixtures preserve the two real log dialects emitted before and after
controller-continuous trajectory execution was introduced.  They deliberately
exercise the script without importing the robot, camera, MoveIt, or ROS stack.
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_bottle_grasp_runs.py"
FIXTURES = Path(__file__).with_name("fixtures")
SPEC = importlib.util.spec_from_file_location(
    "analyze_bottle_grasp_runs", SCRIPT_PATH
)
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZER
SPEC.loader.exec_module(ANALYZER)


def _write_run(
    tmp_path: Path,
    fixture_name: str,
    *,
    result: dict,
    journal: list[dict],
    plan_payloads: list[dict] = (),
    validation_payloads: list[dict] = (),
) -> Path:
    run_dir = tmp_path / "copied_run"
    run_dir.mkdir()
    shutil.copyfile(FIXTURES / fixture_name, run_dir / "run.log")
    (run_dir / "task_result.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "task_journal.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in journal)
        + "\n",
        encoding="utf-8",
    )
    for index, payload in enumerate(plan_payloads, 1):
        (run_dir / f"transfer_{index:02d}_plan.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    for index, payload in enumerate(validation_payloads, 1):
        (run_dir / f"transfer_{index:02d}_validation.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return run_dir


def test_legacy_dense_joint_fixture_preserves_execution_and_safety_metrics(
    tmp_path,
):
    run_dir = _write_run(
        tmp_path,
        "analyze_run_legacy_dense.fixture",
        result={
            "status": "done",
            "phase": "done",
            "object_state": "empty",
        },
        journal=[
            {"timestamp": "2026-07-18T12:32:00+00:00", "phase": "start"},
            {
                "timestamp": "2026-07-18T12:32:06+00:00",
                "phase": "scene_sync",
            },
            {
                "timestamp": "2026-07-18T12:32:10+00:00",
                "phase": "move_to_observation",
            },
            {"timestamp": "2026-07-18T12:32:31+00:00", "phase": "done"},
        ],
        plan_payloads=[{"success": True, "planning_time": 1.234}],
        validation_payloads=[{"success": True, "checked_states": 146}],
    )

    summary = ANALYZER.summarize_run(run_dir)

    assert summary.failure_category == "none"
    assert summary.localization_s == pytest.approx(5.25)
    assert summary.localization_count == 1
    assert summary.scene_sync_s == pytest.approx(4.0)
    assert summary.scene_sync_count == 1
    assert summary.candidate_and_ik_wall_s == pytest.approx(2.0)
    assert summary.candidate_endpoint_count == 9
    assert summary.candidate_ik_viable_count == 4
    assert summary.candidate_and_moveit_wall_s == pytest.approx(3.0)
    assert summary.moveit_and_collision_wall_s == pytest.approx(3.0)
    assert summary.moveit_attempt_count == 1
    assert summary.collision_review_wall_s == pytest.approx(1.0)
    assert summary.execution_s == pytest.approx(12.7)
    assert summary.execution_control_points == 146
    assert summary.dense_execution_points == 146
    assert summary.moveit_reported_planning_s == pytest.approx(1.234)
    assert summary.moveit_plan_count == 1
    assert summary.collision_validation_count == 1
    assert summary.collision_checked_states == 146


def test_control_point_fixture_separates_commands_from_dense_review_and_classifies_terminal_failure(
    tmp_path,
):
    run_dir = _write_run(
        tmp_path,
        "analyze_run_control_points.fixture",
        result={
            "status": "safe_abort",
            "phase": "move_to_observation",
            "object_state": "empty",
            "error": (
                "moveit_observation 在 1 次安全规划后仍无可执行轨迹；"
                "最后拒绝: SDK围栏=拒绝(轨迹离开允许区)；MoveIt密集复核=通过"
            ),
        },
        journal=[
            {"timestamp": "2026-07-23T06:00:00Z", "phase": "start"},
            {
                "timestamp": "2026-07-23T06:00:03Z",
                "phase": "scene_sync",
            },
            {
                "timestamp": "2026-07-23T06:00:05Z",
                "phase": "move_to_observation_staging",
            },
            {
                "timestamp": "2026-07-23T06:00:16Z",
                "phase": "move_to_observation",
            },
            {"timestamp": "2026-07-23T06:00:21Z", "phase": "aborted"},
        ],
        plan_payloads=[
            {"success": True, "planning_time": 0.2},
            {"success": True, "planning_time": 0.4},
        ],
        validation_payloads=[
            {"success": True, "checked_states": 80},
            {"success": True, "checked_states": 180},
        ],
    )

    summary = ANALYZER.summarize_run(run_dir)

    assert summary.failure_category == "collision_disagreement"
    assert summary.localization_s == pytest.approx(2.5)
    assert summary.scene_sync_s == pytest.approx(2.0)
    assert summary.candidate_and_ik_wall_s == pytest.approx(1.0)
    assert summary.candidate_endpoint_count == 8
    assert summary.candidate_ik_viable_count == 3
    assert summary.moveit_attempt_count == 2
    assert summary.moveit_and_collision_wall_s == pytest.approx(6.0)
    assert summary.collision_review_wall_s == pytest.approx(2.0)
    assert summary.execution_s == pytest.approx(4.2)
    assert summary.execution_control_points == 10
    assert summary.dense_execution_points == 80
    assert summary.moveit_reported_planning_s == pytest.approx(0.6)
    assert summary.moveit_plan_count == 2
    assert summary.collision_validation_count == 2
    assert summary.collision_checked_states == 260


def test_log_only_fixture_falls_back_to_logged_moveit_and_collision_metrics(tmp_path):
    run_dir = _write_run(
        tmp_path,
        "analyze_run_legacy_dense.fixture",
        result={"status": "done", "phase": "done", "object_state": "empty"},
        journal=[],
    )

    summary = ANALYZER.summarize_run(run_dir)

    assert summary.moveit_reported_planning_s == pytest.approx(1.234)
    assert summary.moveit_plan_count == 1
    assert summary.collision_validation_count == 1
    assert summary.collision_checked_states == 146
