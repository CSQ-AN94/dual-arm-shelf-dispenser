"""Run-manifest regression tests: filesystem and local git only, no hardware."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.run_manifest import (
    build_run_manifest,
    manifest_profile_expectations,
    write_run_manifest,
)
from shelf_dispenser.safety import load_safety_profile


def _git_runner_with(status: bytes, diff: bytes):
    def runner(_root: Path, args):
        values = {
            ("rev-parse", "--verify", "HEAD"): b"1" * 40 + b"\n",
            ("status", "--porcelain=v1", "-z", "--untracked-files=all"): status,
            ("diff", "--binary", "--no-ext-diff", "HEAD"): diff,
        }
        return values[tuple(args)]

    return runner


def _args(config_path: Path, safety_path: Path):
    return SimpleNamespace(
        config=str(config_path),
        safety_config=str(safety_path),
        safety_profile="shelf_template",
        delivery_safety_profile="side_table_delivery",
        task_mode="from-start",
        execute=True,
        plan_only=False,
        stop_after_observation=True,
        confirm_before_grasp=False,
        dispense=True,
        target_product="coke_bottle",
        visual_servo=False,
        visual_servo_mode="shadow",
        visual_servo_max_corrections=2,
        visual_servo_step_mm=8.0,
        visual_servo_total_mm=15.0,
        visual_servo_convergence_mm=4.0,
        commissioning_speed=10,
        port=8879,
    )


def _config(model_path: Path):
    return SimpleNamespace(
        calibration=SimpleNamespace(
            active_arm="right",
            T_base_right_to_camera_head=[[1.0, 0.0], [0.0, 1.0]],
        ),
        vision=SimpleNamespace(model_path=str(model_path)),
    )


def _params():
    return SimpleNamespace(
        transit_speed=10,
        travel_speed=10,
        final_speed=10,
        gripper_speed=100,
    )


def test_manifest_captures_effective_switches_and_artifact_digests(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_bytes = b"vision:\n  model_path: model.pt\n"
    config_path.write_bytes(config_bytes)
    safety_path = tmp_path / "safety.json"
    safety_path.write_text('{"profiles": {}}', encoding="utf-8")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"trained-model")

    manifest = build_run_manifest(
        args=_args(config_path, safety_path),
        config=_config(model_path),
        project_root=tmp_path,
        params=_params(),
        environ={"BOTTLE_GRASP_CONTINUOUS_TRAJECTORY": "0"},
        git_runner=_git_runner_with(b"", b""),
        created_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    assert manifest["git"]["commit_sha"] == "1" * 40
    assert manifest["git"]["dirty"] is False
    assert manifest["profiles"]["source"] == "shelf_template"
    assert manifest["profiles"]["delivery"] == "side_table_delivery"
    assert manifest["artifacts"]["config"]["sha256"] == hashlib.sha256(
        config_bytes
    ).hexdigest()
    assert manifest["artifacts"]["calibration"]["source"] == "effective_config"
    assert manifest["artifacts"]["calibration"]["sha256"]
    assert manifest["artifacts"]["model"]["sha256"] == hashlib.sha256(
        b"trained-model"
    ).hexdigest()
    assert manifest["visual_loop"]["mode"] == "shadow"
    assert manifest["execution"] == {
        "trajectory_mode": "blocking",
        "continuous_trajectory": False,
        "continuous_trajectory_env": "0",
        "commissioning_speed_cap_percent": 10,
    }
    assert manifest["speeds"]["effective_percent"] == {
        "transit_percent": 10,
        "travel_percent": 10,
        "final_percent": 10,
        "gripper_percent": 100,
    }
    assert manifest["stage_entry"] == {
        "task_mode": "from-start",
        "stop_after_observation": True,
        "confirm_before_grasp": False,
    }
    assert manifest["command_variables"]["arguments"]["commissioning_speed"] == 10
    assert (
        manifest["command_variables"]["environment"]
        ["BOTTLE_GRASP_CONTINUOUS_TRAJECTORY"]
        == "0"
    )


def test_dirty_digest_changes_when_an_untracked_file_content_changes(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("vision: {}\n", encoding="utf-8")
    safety_path = tmp_path / "safety.json"
    safety_path.write_text("{}", encoding="utf-8")
    untracked = tmp_path / "operator-notes.txt"
    untracked.write_text("first", encoding="utf-8")
    runner = _git_runner_with(b"?? operator-notes.txt\0", b"tracked patch")

    first = build_run_manifest(
        args=_args(config_path, safety_path),
        config=None,
        project_root=tmp_path,
        environ={},
        git_runner=runner,
    )["git"]
    untracked.write_text("second", encoding="utf-8")
    second = build_run_manifest(
        args=_args(config_path, safety_path),
        config=None,
        project_root=tmp_path,
        environ={},
        git_runner=runner,
    )["git"]

    assert first["dirty"] is True
    assert first["untracked_file_count"] == 1
    assert first["dirty_digest"] != second["dirty_digest"]


def test_manifest_records_selected_shelf_and_body_profile_effects(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("vision: {}\n", encoding="utf-8")
    safety_path = tmp_path / "safety.json"
    profiles = {
        "profiles": {
            "shelf_template": {
                "verified_for_execution": False,
                "grasp_frame": {"opening_normal_base": [0, 1, 0]},
                "tool_mount_calibration": {"verified": False},
            },
            "side_table_delivery": {
                "verified_for_execution": False,
                "side_table_delivery": {
                    "shelf_ready_verified": False,
                    "rotation_sweep": {"positive": {"verified": False}},
                },
            },
        }
    }
    safety_path.write_text(json.dumps(profiles), encoding="utf-8")

    first = build_run_manifest(
        args=_args(config_path, safety_path),
        config=None,
        project_root=tmp_path,
        environ={},
        git_runner=_git_runner_with(b"", b""),
    )
    source = first["profiles"]["effective"]["source"]
    delivery = first["profiles"]["effective"]["delivery"]
    assert source["state"] == "present"
    assert source["config"]["grasp_frame"]["opening_normal_base"] == [0, 1, 0]
    assert source["config"]["tool_mount_calibration"]["verified"] is False
    assert delivery["state"] == "present"
    assert delivery["config"]["side_table_delivery"]["shelf_ready_verified"] is False
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(first), encoding="utf-8")
    assert manifest_profile_expectations(
        manifest_path, args=_args(config_path, safety_path), project_root=tmp_path
    ) == {
        "source": source["sha256"],
        "delivery": delivery["sha256"],
    }

    profiles["profiles"]["shelf_template"]["grasp_frame"][
        "opening_normal_base"
    ] = [1, 0, 0]
    safety_path.write_text(json.dumps(profiles), encoding="utf-8")
    # The manifest verifier catches a changed file before initialization, and
    # the loader independently compares the exact parsed profile it uses.
    with pytest.raises(SafetyAbort, match="safety_config 已改变"):
        manifest_profile_expectations(
            manifest_path,
            args=_args(config_path, safety_path),
            project_root=tmp_path,
        )
    with pytest.raises(SafetyAbort, match="与 run manifest 记录不一致"):
        load_safety_profile(
            safety_path,
            "shelf_template",
            require_verified=False,
            expected_profile_sha256=source["sha256"],
        )
    second = build_run_manifest(
        args=_args(config_path, safety_path),
        config=None,
        project_root=tmp_path,
        environ={},
        git_runner=_git_runner_with(b"", b""),
    )
    assert (
        first["profiles"]["effective"]["source"]["sha256"]
        != second["profiles"]["effective"]["source"]["sha256"]
    )


def test_manifest_freeze_allows_a_non_dispense_source_profile_load(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("vision: {}\n", encoding="utf-8")
    safety_path = (
        Path(__file__).parents[2] / "shelf_dispenser" / "safety_profiles.json"
    )
    args = _args(config_path, safety_path)
    args.safety_profile = "table_demo"
    args.delivery_safety_profile = None
    args.dispense = False
    manifest = build_run_manifest(
        args=args,
        config=None,
        project_root=tmp_path,
        environ={},
        git_runner=_git_runner_with(b"", b""),
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    expected = manifest_profile_expectations(
        manifest_path, args=args, project_root=tmp_path
    )
    assert expected["source"]
    assert expected["delivery"] is None
    profile = load_safety_profile(
        safety_path,
        "table_demo",
        require_verified=False,
        expected_profile_sha256=expected["source"],
    )
    assert profile.name == "table_demo"


def test_demo_rejects_a_profile_changed_after_manifest_write(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("vision: {}\n", encoding="utf-8")
    source_profiles = (
        Path(__file__).parents[2] / "shelf_dispenser" / "safety_profiles.json"
    )
    safety_path = tmp_path / "safety.json"
    payload = json.loads(source_profiles.read_text(encoding="utf-8"))
    safety_path.write_text(json.dumps(payload), encoding="utf-8")
    args = _args(config_path, safety_path)
    args.safety_profile = "table_demo"
    args.delivery_safety_profile = None
    args.dispense = False
    args.execute = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_run_manifest(
        run_dir,
        args=args,
        config=None,
        project_root=tmp_path,
        environ={},
        git_runner=_git_runner_with(b"", b""),
    )

    payload["profiles"]["table_demo"]["description"] = "mutated after manifest"
    safety_path.write_text(json.dumps(payload), encoding="utf-8")
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.args = args
    demo.params = DemoParams()
    demo.run_dir = run_dir
    demo.project_root = tmp_path
    demo.safety = None
    demo.source_safety = None
    demo.delivery_safety = None

    with pytest.raises(SafetyAbort, match="safety_config 已改变"):
        demo._load_safety_profiles()


def test_complete_launcher_source_provenance_is_authoritative(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("vision: {}\n", encoding="utf-8")
    safety_path = tmp_path / "safety.json"
    safety_path.write_text("{}", encoding="utf-8")
    launcher_sha = "a" * 40
    manifest = build_run_manifest(
        args=_args(config_path, safety_path),
        config=None,
        project_root=tmp_path,
        environ={
            "BOTTLE_GRASP_SOURCE_GIT_SHA": launcher_sha,
            "BOTTLE_GRASP_SOURCE_DIRTY": "1",
            "BOTTLE_GRASP_SOURCE_DIRTY_DIGEST": "b" * 64,
            "BOTTLE_GRASP_SOURCE_DIRTY_DIGEST_ALGORITHM": "source-v1",
        },
        # A conflicting robot-side checkout proves the source fields won.
        git_runner=_git_runner_with(b" M stale.py\0", b"robot patch"),
    )

    assert manifest["git"] == {
        "commit_sha": launcher_sha,
        "dirty": True,
        "dirty_digest": "b" * 64,
        "dirty_digest_algorithm": "source-v1",
        "untracked_file_count": None,
        "state": "available",
        "provenance": "launcher_source",
    }


def test_incomplete_launcher_source_provenance_keeps_local_git_fallback(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("vision: {}\n", encoding="utf-8")
    safety_path = tmp_path / "safety.json"
    safety_path.write_text("{}", encoding="utf-8")
    manifest = build_run_manifest(
        args=_args(config_path, safety_path),
        config=None,
        project_root=tmp_path,
        environ={"BOTTLE_GRASP_SOURCE_GIT_SHA": "a" * 40},
        git_runner=_git_runner_with(b"", b""),
    )

    assert manifest["git"]["commit_sha"] == "1" * 40
    assert "provenance" not in manifest["git"]


def test_write_manifest_is_json_evidence_and_needs_no_hardware(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("vision: {}\n", encoding="utf-8")
    safety_path = tmp_path / "safety.json"
    safety_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "evidence"
    run_dir.mkdir()

    destination = write_run_manifest(
        run_dir,
        args=_args(config_path, safety_path),
        config=None,
        project_root=tmp_path,
        environ={},
        git_runner=_git_runner_with(b"", b""),
        created_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    assert destination == run_dir / "run_manifest.json"
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["created_at_utc"] == "2026-07-23T00:00:00+00:00"
    assert payload["execution"]["trajectory_mode"] == "continuous"
