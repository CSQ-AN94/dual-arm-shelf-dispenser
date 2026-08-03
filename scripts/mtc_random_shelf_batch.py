#!/usr/bin/env python3
"""Generate and optionally plan synthetic shelf pick-only MTC cases.

The generated scenarios are always marked as simulation fixtures.  They can
exercise the real MoveIt/MTC collision and export path, but can never satisfy
the hardware execution bundle.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.safety import load_safety_profile
from localization_to_mtc_scenario import build_scenario

TEMPLATE = (
    ROOT
    / "mtc_ws/src/grabber_mtc_planner/scenarios/shelf_transfer_fixture.yaml"
)
SAFETY_PROFILES = ROOT / "shelf_dispenser/safety_profiles.json"
ARMS = ("right_arm", "left_arm")
BOTTLE_RADIUS_M = 0.033
BOTTLE_CENTER_X_BAND_M = (-0.25, 0.25)
# The padded shelf-bottom collision box ends at y=-0.5878 m.  Keeping each
# centre at least one radius inward makes the whole bottle footprint supported.
BOTTLE_CENTER_Y_BAND_M = (-0.66, -0.625)
# Measured shelf-board top (-0.193 m) plus half the 0.21 m bottle height.
BOTTLE_CENTER_Z_M = -0.088
MIN_LATERAL_SEPARATION_M = 0.11


def _profile_point(profile, moveit_point: np.ndarray) -> list[float]:
    inverse = np.linalg.inv(profile.T_moveit_from_profile)
    return (inverse @ np.r_[moveit_point, 1.0])[:3].tolist()


def _sample_bottles(seed: int, count: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        bottles = rng.uniform(
            [BOTTLE_CENTER_X_BAND_M[0], BOTTLE_CENTER_Y_BAND_M[0]],
            [BOTTLE_CENTER_X_BAND_M[1], BOTTLE_CENTER_Y_BAND_M[1]],
            size=(count, 2),
        )
        if np.min(np.diff(np.sort(bottles[:, 0]))) >= MIN_LATERAL_SEPARATION_M:
            return bottles
    raise SafetyAbort("无法生成互不遮挡的随机瓶位")


def generate_case(
    seed: int, arm_id: str, *, local_motion_planner: str
) -> tuple[dict, dict]:
    if arm_id not in ARMS:
        raise SafetyAbort(f"不支持的规划臂: {arm_id}")
    if local_motion_planner not in {"cartesian", "pilz_lin"}:
        raise SafetyAbort("局部规划器只允许 cartesian/pilz_lin")
    bottles_xy = _sample_bottles(seed)
    target_index = int(np.random.default_rng(seed + 10_000).integers(len(bottles_xy)))
    centers = np.column_stack(
        (bottles_xy, np.full(len(bottles_xy), BOTTLE_CENTER_Z_M))
    )
    target_center = centers[target_index]
    # RGB-D locks the visible near surface. The scenario converter moves one
    # radius inward along -Y to recover the cylinder centre.
    target_surface = target_center + np.asarray([0.0, BOTTLE_RADIUS_M, 0.0])

    profile = load_safety_profile(
        SAFETY_PROFILES, "shelf_template", require_verified=False
    )
    target_profile = _profile_point(profile, target_surface)
    obstacle_profile = [
        _profile_point(profile, center + np.asarray([0.0, 0.0, dz]))
        for index, center in enumerate(centers)
        if index != target_index
        for dz in (-0.075, -0.025, 0.025, 0.075)
    ]
    stamp = datetime.now(timezone.utc).isoformat()
    localization = {
        "point_base": target_profile,
        "depth_m": 0.9,
        "depth_mad_m": 0.001,
        "position_spread_m": 0.002,
        "confidence": 0.9,
        "frame_count": DemoParams().samples,
        "captured_at_utc": stamp,
    }
    scene = {
        "captured_at_utc": stamp,
        "safety_profile": "shelf_template",
        "frame": profile.frame,
        "target_point_base": target_profile,
        "image_height_px": 480,
        "observed_row_limit_px": 480,
        "voxel_size_m": DemoParams().scene_voxel_m,
        "scene_voxels": [target_profile, *obstacle_profile],
        "non_target_scene_voxels": obstacle_profile,
        "target_occupancy_voxels": [target_profile],
        "collision_boxes": profile.moveit_collision_boxes(),
    }
    with TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        localization_path = temporary_path / "synthetic_localization.json"
        scene_path = temporary_path / "synthetic_scene.json"
        localization_path.write_text(json.dumps(localization), encoding="utf-8")
        scene_path.write_text(json.dumps(scene), encoding="utf-8")
        scenario = build_scenario(
            localization_path,
            TEMPLATE,
            SAFETY_PROFILES,
            "shelf_template",
            max_age_s=3600.0,
            allow_stale=False,
            scene_path=scene_path,
            pick_only=True,
            planning_arm_id=arm_id,
        )

    scenario_id = f"synthetic_shelf_pick_v2_{seed:04d}_{arm_id}"
    scenario.update(
        {
            "scenario_id": scenario_id,
            "scene_version": f"synthetic_shelf_v2@seed:{seed}",
            "fixture_source": True,
            "simulation_source": True,
            "freshness_max_age_s": 3600.0,
            "local_motion_planner": local_motion_planner,
            "simulation_obstacle_bottles": [
                {"xyz": center.tolist()}
                for index, center in enumerate(centers)
                if index != target_index
            ],
            "simulation_provenance": {
                "seed": seed,
                "target_index": target_index,
                "bottle_centers_moveit_m": centers.tolist(),
                "x_band_m": list(BOTTLE_CENTER_X_BAND_M),
                "y_band_m": list(BOTTLE_CENTER_Y_BAND_M),
                "minimum_lateral_separation_m": MIN_LATERAL_SEPARATION_M,
                "hardware_execution_forbidden": True,
            },
        }
    )
    return scenario, scenario["simulation_provenance"]


def _selected_failure(result: dict, arm_id: str) -> str | None:
    candidate_id = result.get("selected_grasp_candidate")
    branch_id = f"{arm_id}__{candidate_id}" if candidate_id else None
    failures = result.get("earliest_failure_stage_by_arm") or {}
    if branch_id in failures:
        return failures[branch_id]
    return next(
        (
            value
            for key, value in failures.items()
            if key.startswith(f"{arm_id}__") and value
        ),
        None,
    )


def plan_case(scenario_path: Path, result_path: Path, timeout_s: float) -> dict:
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not (
        isinstance(scenario, dict)
        and scenario.get("simulation_source") is True
        and scenario.get("fixture_source") is True
        and scenario.get("mode") == "pick_only"
    ):
        raise SafetyAbort("批跑器只接受 simulation_source fixture pick-only 场景")
    if result_path.exists():
        raise SafetyAbort(f"拒绝复用旧 MTC 结果路径: {result_path}")
    command = [
        "ros2",
        "launch",
        "grabber_mtc_planner",
        "plan_shelf_transfer_experimental.launch.py",
        f"scenario:={scenario_path}",
        f"out:={result_path}",
        "hold_seconds:=0",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    result_path.with_suffix(".log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode not in (0, 1) or not result_path.exists():
        raise SafetyAbort(
            f"MTC 未生成结果（exit={completed.returncode}），见 {result_path.with_suffix('.log')}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("scenario_id") != scenario.get("scenario_id")
        or result.get("fixture_source") is not True
        or result.get("plan_only") is not True
        or result.get("execution_eligible") is not False
    ):
        raise SafetyAbort("MTC 合成结果丢失 fixture/plan-only 安全标记")
    start = result.get("start_state") or {}
    stamp_ns = start.get("joint_state_stamp_ns")
    age_s = start.get("joint_state_age_s_at_planning")
    if not (
        start.get("selected_arm") == scenario.get("planning_arm_id")
        and start.get("selected_arm_complete") is True
        and isinstance(stamp_ns, int)
        and not isinstance(stamp_ns, bool)
        and stamp_ns > 0
        and isinstance(age_s, (int, float))
        and not isinstance(age_s, bool)
        and 0.0 <= age_s <= 0.5
    ):
        raise SafetyAbort("MTC 合成批跑缺少所选臂新鲜实时起点证据")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--local-motion-planner",
        choices=("cartesian", "pilz_lin"),
        default="cartesian",
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--launch-timeout-s", type=float, default=90.0)
    cli = parser.parse_args()
    if cli.count < 1 or cli.launch_timeout_s <= 0:
        parser.error("--count 和 --launch-timeout-s 必须为正数")
    if cli.plan and shutil.which("ros2") is None:
        parser.error("--plan 需要 ROS 2/MoveIt 环境")

    cli.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for offset in range(cli.count):
        seed = cli.seed + offset
        for arm_id in cli.arms:
            scenario, provenance = generate_case(
                seed, arm_id, local_motion_planner=cli.local_motion_planner
            )
            stem = scenario["scenario_id"]
            scenario_path = cli.output_dir / f"{stem}.yaml"
            result_path = cli.output_dir / f"{stem}.result.json"
            scenario_path.write_text(
                yaml.safe_dump(scenario, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            case = {
                "scenario_id": stem,
                "arm_id": arm_id,
                "seed": seed,
                "scenario": str(scenario_path),
                "target_xyz_m": provenance["bottle_centers_moveit_m"][
                    provenance["target_index"]
                ],
                "planned": False,
            }
            if cli.plan:
                try:
                    result = plan_case(
                        scenario_path, result_path, cli.launch_timeout_s
                    )
                    case.update(
                        {
                            "planned": True,
                            "solved": result.get("solved") is True,
                            "selected_candidate": result.get(
                                "selected_grasp_candidate"
                            ),
                            "failure_stage": _selected_failure(result, arm_id),
                            "planning_wall_time_s": result.get(
                                "planning_wall_time"
                            ),
                            "result": str(result_path),
                        }
                    )
                except (SafetyAbort, subprocess.TimeoutExpired) as exc:
                    case["error"] = str(exc)
            cases.append(case)

    by_arm = {}
    for arm_id in cli.arms:
        arm_cases = [case for case in cases if case["arm_id"] == arm_id]
        by_arm[arm_id] = {
            "generated": len(arm_cases),
            "planned": sum(case["planned"] for case in arm_cases),
            "solved": sum(case.get("solved") is True for case in arm_cases),
            "failure_stages": dict(
                Counter(
                    case["failure_stage"]
                    for case in arm_cases
                    if case.get("failure_stage")
                )
            ),
        }
    summary = {
        "schema_version": "grabber.mtc_synthetic_batch.v1",
        "simulation_only": True,
        "hardware_execution_forbidden": True,
        "local_motion_planner": cli.local_motion_planner,
        "infrastructure_errors": sum("error" in case for case in cases),
        "by_arm": by_arm,
        "cases": cases,
    }
    summary_path = cli.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 2 if summary["infrastructure_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
