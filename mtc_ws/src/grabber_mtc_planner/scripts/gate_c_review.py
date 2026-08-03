#!/usr/bin/env python3
"""Fail-closed, no-motion review of one MTC plan-only result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(result_path: Path, bridge_path: Path, arms_path: Path, scenario_path: Path) -> dict:
    result = _read_json(result_path)
    bridge = _read_json(bridge_path)
    arms_doc = _read_yaml(arms_path)
    scenario = _read_yaml(scenario_path)
    arm_items = arms_doc.get("arms")
    if not isinstance(arm_items, list):
        raise ValueError(f"{arms_path} must contain an arms list")
    arms = {
        item.get("arm_id"): item
        for item in arm_items
        if isinstance(item, dict) and item.get("arm_id")
    }
    selected = result.get("selected_arm")
    arm = arms.get(selected)
    blockers: list[str] = []

    def require(condition: bool, reason: str) -> None:
        if not condition:
            blockers.append(reason)

    require(result.get("plan_only") is True, "RESULT_IS_NOT_PLAN_ONLY")
    require(result.get("solved") is True, "NO_COMPLETE_SOLUTION")
    require(isinstance(selected, str) and bool(selected), "NO_SELECTED_ARM")
    require(result.get("execution_eligible") is True, "RESULT_ARM_NOT_EXECUTION_ELIGIBLE")
    require(arm is not None, "SELECTED_ARM_MISSING_FROM_CONFIG")
    require(result.get("scenario_id") == scenario.get("scenario_id"), "SCENARIO_ID_MISMATCH")
    require(result.get("scene_version") == scenario.get("scene_version"), "SCENE_VERSION_MISMATCH")
    require(result.get("fixture_source") == scenario.get("fixture_source"), "FIXTURE_FLAG_MISMATCH")
    require(scenario.get("fixture_source") is False, "LIVE_RELOCALIZATION_REQUIRED")

    start = result.get("start_state")
    age_s = start.get("joint_state_age_s_at_planning") if isinstance(start, dict) else None
    stamp_ns = start.get("joint_state_stamp_ns") if isinstance(start, dict) else None
    require(
        isinstance(start, dict)
        and start.get("selected_arm") == selected
        and start.get("selected_arm_complete") is True
        and isinstance(age_s, (int, float))
        and not isinstance(age_s, bool)
        and math.isfinite(float(age_s))
        and 0.0 <= float(age_s) <= 0.5
        and isinstance(stamp_ns, int)
        and not isinstance(stamp_ns, bool)
        and stamp_ns > 0,
        "FRESH_SELECTED_ARM_START_STATE_REQUIRED",
    )
    joints = start.get("joints") if isinstance(start, dict) else None
    require(
        isinstance(joints, dict)
        and bool(joints)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in joints.values()
        ),
        "LIVE_START_JOINTS_INVALID",
    )

    if arm is not None:
        require(arm.get("execution_eligible") is True, "CONFIG_ARM_NOT_EXECUTION_ELIGIBLE")
        require(bool(arm.get("tool_version")), "TOOL_VERSION_MISSING")
        require(bool(arm.get("calibration_version")), "CALIBRATION_VERSION_MISSING")
        require(result.get("tool_version") == arm.get("tool_version"), "TOOL_VERSION_MISMATCH")
        require(
            result.get("calibration_version") == arm.get("calibration_version"),
            "CALIBRATION_VERSION_MISMATCH",
        )

    require(bridge.get("read_only") is True, "STATE_BRIDGE_NOT_READ_ONLY")
    require(bridge.get("publishing") is True, "STATE_BRIDGE_NOT_PUBLISHING")
    require(bridge.get("lift_motion_ready") is True, "LIFT_NOT_MOTION_READY")
    read_failures = bridge.get("read_failures")
    skew_violations = bridge.get("skew_violations")
    rate = bridge.get("average_rate_hz")
    require(
        isinstance(read_failures, int) and not isinstance(read_failures, bool) and read_failures == 0,
        "STATE_BRIDGE_READ_FAILURES",
    )
    require(
        isinstance(skew_violations, int) and not isinstance(skew_violations, bool) and skew_violations == 0,
        "STATE_BRIDGE_SKEW_VIOLATIONS",
    )
    require(
        isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and math.isfinite(float(rate))
        and float(rate) >= 10.0,
        "STATE_BRIDGE_RATE_TOO_LOW",
    )

    return {
        "schema_version": 1,
        "mode": "review_only_no_motion",
        "verdict": "BLOCKED" if blockers else "READY_FOR_HUMAN_REVIEW",
        "execution_ready": False,
        "blockers": blockers,
        "always_required_before_motion": [
            "FRESH_START_STATE_RECHECK",
            "CURRENT_SCENE_COLLISION_RECHECK",
            "EXPLICIT_HUMAN_APPROVAL",
            "HARDWARE_ESTOP_WITNESS",
        ],
        "selected_arm": selected,
        "scenario_id": result.get("scenario_id"),
        "artifacts": {
            "result": {"path": str(result_path), "sha256": _sha256(result_path)},
            "bridge_status": {"path": str(bridge_path), "sha256": _sha256(bridge_path)},
            "arms": {"path": str(arms_path), "sha256": _sha256(arms_path)},
            "scenario": {"path": str(scenario_path), "sha256": _sha256(scenario_path)},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--bridge-status", required=True, type=Path)
    parser.add_argument("--arms", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = review(args.result, args.bridge_status, args.arms, args.scenario)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["verdict"] == "READY_FOR_HUMAN_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
