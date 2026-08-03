#!/usr/bin/env python3
"""Small offline check for the no-motion Gate C reviewer."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile

import yaml

PKG = pathlib.Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "gate_c_review", PKG / "scripts" / "gate_c_review.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def _write(path: pathlib.Path, value, *, yaml_file: bool = False) -> pathlib.Path:
    path.write_text(
        yaml.safe_dump(value) if yaml_file else json.dumps(value),
        encoding="utf-8",
    )
    return path


def test_fixture_blocks_and_live_scenario_reaches_human_review():
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        result = {
            "plan_only": True,
            "solved": True,
            "selected_arm": "right_arm",
            "execution_eligible": True,
            "scenario_id": "demo",
            "scene_version": "live-1",
            "fixture_source": True,
            "start_state": {
                "all_zero": False,
                "selected_arm": "right_arm",
                "selected_arm_complete": True,
                "joint_state_stamp_ns": 1_700_000_000_000_000_000,
                "joint_state_age_s_at_planning": 0.05,
                "joints": {"r_joint1": 0.2},
            },
            "tool_version": "tool-1",
            "calibration_version": "cal-1",
        }
        bridge = {
            "read_only": True,
            "publishing": True,
            "lift_motion_ready": True,
            "read_failures": 0,
            "skew_violations": 0,
            "average_rate_hz": 20.0,
        }
        arms = {
            "arms": [{
                "arm_id": "right_arm",
                "execution_eligible": True,
                "tool_version": "tool-1",
                "calibration_version": "cal-1",
            }]
        }
        scenario = {
            "scenario_id": "demo",
            "scene_version": "live-1",
            "fixture_source": True,
        }
        paths = (
            _write(root / "result.json", result),
            _write(root / "bridge.json", bridge),
            _write(root / "arms.yaml", arms, yaml_file=True),
            _write(root / "scenario.yaml", scenario, yaml_file=True),
        )

        report = MODULE.review(*paths)
        assert report["verdict"] == "BLOCKED"
        assert report["blockers"] == ["LIVE_RELOCALIZATION_REQUIRED"]

        result["fixture_source"] = False
        scenario["fixture_source"] = False
        paths = (
            _write(paths[0], result),
            paths[1],
            paths[2],
            _write(paths[3], scenario, yaml_file=True),
        )
        report = MODULE.review(*paths)
        assert report["verdict"] == "READY_FOR_HUMAN_REVIEW"
        assert report["blockers"] == []


if __name__ == "__main__":
    test_fixture_blocks_and_live_scenario_reaches_human_review()
    print("all Gate C review checks passed")
