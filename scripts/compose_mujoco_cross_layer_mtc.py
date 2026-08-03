#!/usr/bin/env python3
"""Compose real MTC pick/place exports and a stationary-arm lift for MuJoCo."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.mtc_pick_contract import (
    validate_full_transfer_trajectory,
    validate_pick_trajectory,
    validate_place_trajectory,
)
from scripts.mujoco_full_workflow import _validate_platform_arm_interlock


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SafetyAbort(f"{path} 顶层必须是对象")
    return payload


def _yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SafetyAbort(f"{path} 顶层必须是对象")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_cross_layer_replay(
    *,
    scene: dict,
    manifest: dict,
    pick_scenario: dict,
    pick: dict,
    place_scenario: dict,
    place: dict,
    left_joints_deg: list[float],
    lift_duration_s: float,
    lift_sample_s: float,
) -> tuple[dict, dict]:
    validate_pick_trajectory(pick)
    validate_place_trajectory(place)
    scene_id = scene.get("scenario_id")
    if (
        not isinstance(scene_id, str)
        or not scene_id
        or manifest.get("scenario_id") != scene_id
        or scene.get("simulation_scene_only") is not True
    ):
        raise SafetyAbort("随机场景与 manifest 身份不一致")
    if pick_scenario.get("scenario_id") != pick.get("scenario_id"):
        raise SafetyAbort("pick scenario/trajectory 的 scenario_id 不一致")
    if place_scenario.get("scenario_id") != place.get("scenario_id"):
        raise SafetyAbort("place scenario/trajectory 的 scenario_id 不一致")
    for label, scenario in (
        ("pick", pick_scenario),
        ("place", place_scenario),
    ):
        if scenario.get("simulation_scene_id") != scene_id:
            raise SafetyAbort(f"{label} MTC 场景未绑定随机仿真场景")
        if scenario.get("simulation_source") is not True:
            raise SafetyAbort(f"{label} MTC 场景没有 simulation_source 标记")
    if pick.get("joint_names") != place.get("joint_names"):
        raise SafetyAbort("pick/place 右臂关节顺序不一致")
    pick_end = np.asarray(pick["points"][-1]["positions_deg"], dtype=float)
    place_start = np.asarray(place["points"][0]["positions_deg"], dtype=float)
    if float(np.max(np.abs(pick_end - place_start))) > 0.1:
        raise SafetyAbort("place MTC 起点不是 pick 抓后收拢终点")
    left = np.asarray(left_joints_deg, dtype=float)
    if left.shape != (7,) or not np.all(np.isfinite(left)):
        raise SafetyAbort("仿真左臂静止姿态必须是七个有限关节角")
    if (
        not math.isfinite(lift_duration_s)
        or lift_duration_s <= 0.0
        or not math.isfinite(lift_sample_s)
        or lift_sample_s <= 0.0
        or lift_sample_s > lift_duration_s
    ):
        raise SafetyAbort("升降时长/采样周期无效")

    coordinates = manifest.get("coordinate_contract")
    if not isinstance(coordinates, dict):
        raise SafetyAbort("manifest 缺少跨层坐标契约")
    source_lift = float(coordinates["visualization_reference_lift_mm"])
    target_lift = float(coordinates["place_planning_lift_mm"])
    frame_shift = float(coordinates["place_frame_z_shift_m"])
    if (
        not np.all(np.isfinite([source_lift, target_lift, frame_shift]))
        or abs(source_lift - 647.0) > 1e-6
        or abs(target_lift - 250.0) > 1e-6
        or abs(frame_shift - 0.397) > 1e-6
    ):
        raise SafetyAbort("跨层平台坐标契约必须是 647->250 mm")
    expected_target = np.asarray(
        coordinates["target_place_xyz_at_250mm"], dtype=float
    )
    planned_target = np.asarray(
        place_scenario["target_place_pose"]["xyz"], dtype=float
    )
    scene_target = np.asarray(scene["target_place_pose"]["xyz"], dtype=float)
    planned_quat = np.asarray(
        place_scenario["target_place_pose"]["quat_xyzw"], dtype=float
    )
    if (
        expected_target.shape != (3,)
        or planned_target.shape != (3,)
        or scene_target.shape != (3,)
        or planned_quat.shape != (4,)
        or not np.all(np.isfinite(expected_target))
        or not np.all(np.isfinite(planned_target))
        or not np.all(np.isfinite(scene_target))
        or not np.all(np.isfinite(planned_quat))
        or abs(float(np.linalg.norm(planned_quat)) - 1.0) > 1e-6
        or float(np.max(np.abs(expected_target - planned_target))) > 1e-6
        or float(
            np.max(
                np.abs(
                    scene_target + np.asarray([0.0, 0.0, frame_shift])
                    - planned_target
                )
            )
        )
        > 1e-6
    ):
        raise SafetyAbort("place MTC 目标没有使用 250 mm 重标定坐标")

    points: list[dict] = []
    for point in pick["points"]:
        copied = deepcopy(point)
        copied["platform_height_mm"] = source_lift
        points.append(copied)
    pick_end_index = len(points) - 1
    lift_steps = int(math.ceil(lift_duration_s / lift_sample_s))
    lift_start_time = float(points[-1]["time_from_start_s"])
    for step in range(1, lift_steps + 1):
        alpha = step / lift_steps
        points.append(
            {
                "time_from_start_s": lift_start_time
                + alpha * lift_duration_s,
                "positions_deg": pick_end.tolist(),
                "velocities_deg_s": [0.0] * 7,
                "accelerations_deg_s2": [0.0] * 7,
                "platform_height_mm": source_lift
                + alpha * (target_lift - source_lift),
            }
        )
    lift_end_index = len(points) - 1
    place_index_map = {0: lift_end_index}
    place_start_time = float(place["points"][0]["time_from_start_s"])
    for old_index, point in enumerate(place["points"][1:], start=1):
        copied = deepcopy(point)
        copied["time_from_start_s"] = (
            float(points[lift_end_index]["time_from_start_s"])
            + float(point["time_from_start_s"])
            - place_start_time
        )
        copied["platform_height_mm"] = target_lift
        points.append(copied)
        place_index_map[old_index] = len(points) - 1

    pick_bounds = {
        item["name"]: item for item in pick["phase_boundaries"]
    }
    place_bounds = {
        item["name"]: item for item in place["phase_boundaries"]
    }

    def mapped(name: str, endpoint: str) -> int:
        return place_index_map[int(place_bounds[name][endpoint])]

    attach_index = int(pick_bounds["attach"]["start_index"])
    release_index = mapped("release", "start_index")
    trajectory = {
        "schema_version": "grabber.mtc_full_transfer.v1",
        "plan_only": True,
        "execution_supported": False,
        "execution_block_reason": "PLAN_ONLY_FULL_TRANSFER",
        "simulation_replay_only": True,
        "mode": "full_transfer",
        "scenario_id": scene_id,
        "arm_id": "right_arm",
        "grasp_candidate_id": pick["grasp_candidate_id"],
        "joint_units": "degrees",
        "joint_names": list(pick["joint_names"]),
        "left_joints_deg": left.tolist(),
        "cross_layer_transport": True,
        "points": points,
        "phase_boundaries": [
            deepcopy(pick_bounds["pregrasp"]),
            deepcopy(pick_bounds["approach"]),
            deepcopy(pick_bounds["attach"]),
            {
                "name": "source_retreat",
                "start_index": int(pick_bounds["retreat"]["start_index"]),
                "end_index": pick_end_index,
            },
            {
                "name": "platform_lower",
                "start_index": pick_end_index,
                "end_index": lift_end_index,
            },
            {
                "name": "transport",
                "start_index": lift_end_index,
                "end_index": mapped("transport", "end_index"),
            },
            {
                "name": "place",
                "start_index": mapped("approach", "start_index"),
                "end_index": mapped("approach", "end_index"),
            },
            {
                "name": "release",
                "start_index": release_index,
                "end_index": release_index,
            },
            {
                "name": "target_retreat",
                "start_index": mapped("retreat", "start_index"),
                "end_index": mapped("retreat", "end_index"),
            },
        ],
        "gripper_events": [
            {"name": "open_before_motion", "point_index": 0},
            {"name": "close_at_attach", "point_index": attach_index},
            {"name": "open_at_release", "point_index": release_index},
        ],
        "platform_lift_phase": {
            "start_index": pick_end_index,
            "end_index": lift_end_index,
            "source_height_mm": source_lift,
            "target_height_mm": target_lift,
            "right_arm_stationary": True,
            "left_arm_stationary": True,
        },
    }
    validate_full_transfer_trajectory(trajectory)
    _validate_platform_arm_interlock(trajectory)
    replay_scene = deepcopy(scene)
    replay_scene["mode"] = "full_transfer_replay"
    replay_scene["simulation_scene_only"] = False
    # Replay geometry stays in the 647 mm visualization frame; only the
    # orientation comes from the 250 mm place plan.
    replay_scene["target_place_pose"]["quat_xyzw"] = planned_quat.tolist()
    replay_scene["simulation_mtc_sources"] = {
        "pick_scenario_id": pick["scenario_id"],
        "place_scenario_id": place["scenario_id"],
    }
    return replay_scene, trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pick-scenario", type=Path, required=True)
    parser.add_argument("--pick-trajectory", type=Path, required=True)
    parser.add_argument("--place-scenario", type=Path, required=True)
    parser.add_argument("--place-trajectory", type=Path, required=True)
    parser.add_argument("--left-joints-deg", type=float, nargs=7, required=True)
    parser.add_argument("--lift-duration-s", type=float, default=3.97)
    parser.add_argument("--lift-sample-s", type=float, default=0.05)
    parser.add_argument("--out-scene", type=Path, required=True)
    parser.add_argument("--out-trajectory", type=Path, required=True)
    cli = parser.parse_args()
    paths = {
        "scene": cli.scene,
        "manifest": cli.manifest,
        "pick_scenario": cli.pick_scenario,
        "pick_trajectory": cli.pick_trajectory,
        "place_scenario": cli.place_scenario,
        "place_trajectory": cli.place_trajectory,
    }
    replay_scene, trajectory = compose_cross_layer_replay(
        scene=_yaml(cli.scene),
        manifest=_json(cli.manifest),
        pick_scenario=_yaml(cli.pick_scenario),
        pick=_json(cli.pick_trajectory),
        place_scenario=_yaml(cli.place_scenario),
        place=_json(cli.place_trajectory),
        left_joints_deg=cli.left_joints_deg,
        lift_duration_s=cli.lift_duration_s,
        lift_sample_s=cli.lift_sample_s,
    )
    trajectory["source_sha256"] = {
        label: _sha256(path) for label, path in paths.items()
    }
    cli.out_scene.write_text(
        yaml.safe_dump(replay_scene, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    cli.out_trajectory.write_text(
        json.dumps(trajectory, indent=2) + "\n", encoding="utf-8"
    )
    print(f"MuJoCo replay scene: {cli.out_scene}")
    print(f"MuJoCo replay trajectory: {cli.out_trajectory}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, SafetyAbort) as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
