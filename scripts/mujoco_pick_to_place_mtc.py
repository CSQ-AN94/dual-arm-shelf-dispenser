#!/usr/bin/env python3
"""Build an exact simulated place-only MTC scene from a real pick export."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bottle_grasp.core import SafetyAbort
from bottle_grasp.mtc_pick_contract import validate_pick_trajectory
from scripts.mujoco_full_workflow import (
    PLATFORM_ORIGIN,
    _set_initial_state,
    build_full_model_xml,
)
from scripts.mujoco_grabber_sim import _site_transform


def _pose_matrix(pose: dict) -> np.ndarray:
    transform = np.eye(4)
    xyz = np.asarray(pose.get("xyz"), dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise SafetyAbort("位姿 xyz 必须是三个有限数")
    if "quat_xyzw" in pose:
        transform[:3, :3] = Rotation.from_quat(pose["quat_xyzw"]).as_matrix()
    else:
        transform[:3, :3] = Rotation.from_euler(
            "xyz", pose.get("rpy_deg", [0.0, 0.0, 0.0]), degrees=True
        ).as_matrix()
    transform[:3, 3] = xyz
    return transform


def _pose_dict(transform: np.ndarray) -> dict:
    return {
        "xyz": transform[:3, 3].tolist(),
        "quat_xyzw": Rotation.from_matrix(transform[:3, :3]).as_quat().tolist(),
    }


def _shift_pose_z(item: dict, shift_m: float) -> dict:
    shifted = deepcopy(item)
    shifted["pose"]["xyz"][2] = float(shifted["pose"]["xyz"][2]) + shift_m
    return shifted


def build_place_scenario(
    *, scene: dict, manifest: dict, pick_scenario: dict, pick: dict
) -> dict:
    validate_pick_trajectory(pick)
    scene_id = scene.get("scenario_id")
    if (
        not isinstance(scene_id, str)
        or not scene_id
        or scene.get("simulation_scene_only") is not True
        or manifest.get("scenario_id") != scene_id
    ):
        raise SafetyAbort("随机场景与 manifest 身份不一致")
    if (
        pick_scenario.get("mode") != "pick_only"
        or pick_scenario.get("simulation_source") is not True
        or pick_scenario.get("fixture_source") is not True
        or pick_scenario.get("simulation_scene_id") != scene_id
        or pick_scenario.get("scenario_id") != pick.get("scenario_id")
    ):
        raise SafetyAbort("pick MTC 产物没有绑定该随机仿真场景")
    for key, label in (
        ("bottle", "瓶体几何"),
        ("shelf_boxes", "货架几何"),
        ("obstacle_voxels", "障碍几何"),
        ("simulation_obstacle_bottles", "随机瓶位"),
    ):
        if pick_scenario.get(key) != scene.get(key):
            raise SafetyAbort(f"pick MTC 的{label}与随机场景不一致")

    coordinates = manifest.get("coordinate_contract")
    if not isinstance(coordinates, dict):
        raise SafetyAbort("manifest 缺少跨层坐标契约")
    source_lift_mm = float(coordinates["visualization_reference_lift_mm"])
    target_lift_mm = float(coordinates["place_planning_lift_mm"])
    shift_m = float(coordinates["place_frame_z_shift_m"])
    if (
        abs(source_lift_mm - 647.0) > 1e-9
        or abs(target_lift_mm - 250.0) > 1e-9
        or abs(shift_m - (source_lift_mm - target_lift_mm) / 1000.0) > 1e-9
    ):
        raise SafetyAbort("跨层坐标契约必须严格为 647→250 mm")

    candidate = next(
        (
            item
            for item in pick_scenario.get("source_grasp_candidates", [])
            if item.get("id") == pick.get("grasp_candidate_id")
        ),
        None,
    )
    if candidate is None:
        raise SafetyAbort("pick 轨迹抓姿不在随机场景中")

    end_joints_deg = np.asarray(pick["points"][-1]["positions_deg"], dtype=float)
    model = mujoco.MjModel.from_xml_string(build_full_model_xml(scene))
    data = mujoco.MjData(model)
    _set_initial_state(
        model,
        data,
        list(pick["joint_names"]),
        np.radians(end_joints_deg),
        source_lift_mm / 1000.0,
    )
    end_tcp = _site_transform(model, data, "r_tcp").copy()
    end_tcp[:3, 3] -= PLATFORM_ORIGIN
    source_tcp = _pose_matrix(candidate["pose"])
    source_bottle = _pose_matrix(scene["bottle"]["pose"])
    held_bottle = end_tcp @ np.linalg.inv(source_tcp) @ source_bottle

    target_xyz = np.asarray(coordinates["target_place_xyz_at_250mm"], dtype=float)
    if target_xyz.shape != (3,) or not np.all(np.isfinite(target_xyz)):
        raise SafetyAbort("250 mm 放置目标必须是三个有限数")
    shifted_boxes = [_shift_pose_z(item, shift_m) for item in scene["shelf_boxes"]]
    support = next(
        item for item in shifted_boxes if item["id"] == "second_shelf_board"
    )
    support_top_z = float(support["pose"]["xyz"][2]) + float(support["size"][2]) / 2.0
    bottle_half_height_m = float(scene["bottle"]["height_m"]) / 2.0
    if abs(float(target_xyz[2]) - bottle_half_height_m - support_top_z) > 1e-6:
        raise SafetyAbort("放置目标没有让瓶底精确落在第二层支撑面")

    place = deepcopy(scene)
    place.update(
        {
            "scenario_id": f"{scene_id}_place",
            "simulation_scene_id": scene_id,
            "simulation_source": True,
            "simulation_scene_only": False,
            "fixture_source": True,
            "mode": "place_only",
            "planning_arm_id": "right_arm",
            "source_layer_id": "simulated_held_bottle_after_pick",
            "target_layer_id": "simulated_second_shelf_empty_patch",
            "lift_state_id": "simulated_platform_joint_250mm",
            "source_support_surface_id": "",
            "target_support_surface_id": "second_shelf_board",
            "scene_version": f"{scene.get('scene_version', 'digital_twin')}@250mm",
            "scene_captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "freshness_max_age_s": 3600.0,
            "source_grasp_pose": _pose_dict(end_tcp),
            "source_grasp_candidates": [
                {"id": "held_bottle", "pose": _pose_dict(end_tcp)}
            ],
            "target_place_pose": {
                "xyz": target_xyz.tolist(),
                "quat_xyzw": list(candidate["pose"]["quat_xyzw"]),
            },
            "target_insert_direction": [0.0, 0.0, -1.0],
            "target_retreat_direction": [0.0, 0.0, 1.0],
            "target_preplace_offset_m": 0.085,
            "target_contact_distance_m": 0.010,
            # The vendor r_hand collision mesh is 117 mm tall in this
            # orientation; 120 mm still overlaps the 210 mm bottle.
            "target_retreat_distance_m": 0.200,
            "post_place_home_joints_deg": end_joints_deg.tolist(),
            "local_motion_planner": "pilz_lin",
            "shelf_boxes": shifted_boxes,
            "obstacle_voxels": [
                [float(x), float(y), float(z) + shift_m]
                for x, y, z in scene["obstacle_voxels"]
            ],
            "simulation_obstacle_bottles": [
                {
                    **deepcopy(item),
                    "xyz": [
                        float(item["xyz"][0]),
                        float(item["xyz"][1]),
                        float(item["xyz"][2]) + shift_m,
                    ],
                }
                for item in scene["simulation_obstacle_bottles"]
            ],
            "placement_provenance": {
                "pick_scenario_id": pick["scenario_id"],
                "pick_grasp_candidate_id": pick["grasp_candidate_id"],
                "pick_end_joints_deg": end_joints_deg.tolist(),
                "platform_height_mm": target_lift_mm,
                "place_frame_z_shift_m": shift_m,
            },
        }
    )
    place["bottle"]["pose"] = _pose_dict(held_bottle)
    return place


def build_place_fixture_state(*, result: dict, pick: dict, place: dict) -> dict:
    if (
        result.get("plan_only") is not True
        or result.get("solved") is not True
        or result.get("mode") != "pick_only"
        or result.get("fixture_source") is not True
        or result.get("selected_arm") != "right_arm"
        or result.get("scenario_id") != pick.get("scenario_id")
    ):
        raise SafetyAbort("place fixture 只接受已求解的 simulation pick result")
    joints = (result.get("start_state") or {}).get("joints")
    if not isinstance(joints, dict):
        raise SafetyAbort("pick result 缺少完整起点关节")
    right_start_deg = np.degrees(
        np.asarray([joints[f"r_joint{i}"] for i in range(1, 8)], dtype=float)
    )
    planned_start_deg = np.asarray(pick["points"][0]["positions_deg"], dtype=float)
    if float(np.max(np.abs(right_start_deg - planned_start_deg))) > 0.1:
        raise SafetyAbort("pick result 起点与轨迹不连续")
    platform_start_mm = 1000.0 * float(joints["platform_joint"])
    if abs(platform_start_mm - 647.0) > 0.5:
        raise SafetyAbort("pick fixture 起点不是 647 mm")
    return {
        "schema_version": "grabber.mtc_fixture_joint_state.v1",
        "simulation_only": True,
        "hardware_connections": 0,
        "source_pick_scenario_id": pick["scenario_id"],
        "place_scenario_id": place["scenario_id"],
        "platform_height_mm": 250.0,
        "head_joints_rad": [
            float(joints.get("head_joint1", 0.0)),
            float(joints.get("head_joint2", 0.0)),
        ],
        "left_joints_deg": np.degrees(
            np.asarray([joints[f"l_joint{i}"] for i in range(1, 8)], dtype=float)
        ).tolist(),
        "right_joints_deg": list(pick["points"][-1]["positions_deg"]),
    }


def _load(path: Path) -> dict:
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.suffix == ".json"
        else yaml.safe_load(path.read_text(encoding="utf-8"))
    )
    if not isinstance(payload, dict):
        raise SafetyAbort(f"{path} 顶层必须是对象")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pick-scenario", type=Path, required=True)
    parser.add_argument("--pick-result", type=Path, required=True)
    parser.add_argument("--pick-trajectory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-fixture-state", type=Path, required=True)
    args = parser.parse_args()
    pick = _load(args.pick_trajectory)
    place = build_place_scenario(
        scene=_load(args.scene),
        manifest=_load(args.manifest),
        pick_scenario=_load(args.pick_scenario),
        pick=pick,
    )
    place["placement_provenance"]["source_sha256"] = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in (
            ("scene", args.scene),
            ("manifest", args.manifest),
            ("pick_scenario", args.pick_scenario),
            ("pick_result", args.pick_result),
            ("pick_trajectory", args.pick_trajectory),
        )
    }
    args.out.write_text(
        yaml.safe_dump(place, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    fixture = build_place_fixture_state(
        result=_load(args.pick_result), pick=pick, place=place
    )
    args.out_fixture_state.write_text(
        json.dumps(fixture, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "place_scenario": str(args.out),
                "fixture_state": str(args.out_fixture_state),
                "scenario_id": place["scenario_id"],
                "simulation_scene_id": place["simulation_scene_id"],
                "platform_height_mm": 250,
                "start_right_joints_deg": place["post_place_home_joints_deg"],
                "target_place_xyz": place["target_place_pose"]["xyz"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, StopIteration, SafetyAbort) as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
