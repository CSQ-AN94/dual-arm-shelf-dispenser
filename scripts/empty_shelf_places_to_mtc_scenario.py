#!/usr/bin/env python3
"""Convert a fresh empty-shelf observation into an MTC place-only scenario."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.core import SafetyAbort, pose_matrix
from shelf_dispenser.safety import FenceBox, load_safety_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-index", type=int, default=0)
    # An arm_clear_of_view map is taken before the pick, so it cannot know
    # where the bottle ends up being held.  That has to come from the pick
    # itself, which is a different question from "where is the shelf empty".
    parser.add_argument("--pick-execution-record", type=Path)
    parser.add_argument("--target-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    cli = parser.parse_args(argv)

    payload = json.loads(cli.observation.read_text(encoding="utf-8"))
    captured = datetime.fromisoformat(payload["captured_at_utc"])
    if captured.tzinfo is None:
        raise SafetyAbort("第二层空位观测时间缺少时区")
    age_s = (datetime.now(timezone.utc) - captured).total_seconds()
    if age_s < 0.0 or age_s > 900.0:
        raise SafetyAbort(f"第二层空位观测不新鲜: age={age_s:.1f}s")
    candidates = payload["observation"]["candidates"]
    if not 0 <= cli.candidate_index < len(candidates):
        raise SafetyAbort("空位候选索引越界")
    candidate = candidates[cli.candidate_index]
    surface_z = float(payload["observation"]["table_height_m"])
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )

    bottom = next(
        box for box in profile.keepout_boxes if box.id == "shelf_bottom"
    )
    shift_z = surface_z - bottom.maximum[2]
    shifted = tuple(
        replace(
            box,
            minimum=(
                box.minimum[0],
                box.minimum[1],
                box.minimum[2] + shift_z,
            ),
            maximum=(
                box.maximum[0],
                box.maximum[1],
                box.maximum[2] + shift_z,
            ),
        )
        for box in profile.keepout_boxes
    )
    shifted_profile = replace(profile, keepout_boxes=shifted)
    collision_boxes = shifted_profile.moveit_collision_boxes()
    exact_bottom = next(box for box in shifted if box.id == "shelf_bottom")
    exact_item = exact_bottom.moveit_box()
    exact_item["center"] = shifted_profile.point_to_moveit(
        exact_item["center"]
    ).tolist()
    collision_boxes = [
        exact_item if item["id"] == exact_item["id"] else item
        for item in collision_boxes
    ]

    # An empty-handed map was taken before the pick, with the arm parked clear
    # of the head camera's view, so there is nothing to subtract and -- more
    # importantly -- no occlusion shadow in which "empty" and "not visible"
    # would be indistinguishable.  A held map still needs the arm and bottle
    # removed, and still carries that shadow.
    regime = payload.get("occlusion_regime", "held_arm_subtracted")
    if regime not in ("arm_clear_of_view", "held_arm_subtracted"):
        raise SafetyAbort(f"捕获产物 occlusion_regime 无效: {regime}")
    voxels = np.asarray(payload["scene_voxels"], dtype=float)
    if regime == "arm_clear_of_view":
        if cli.pick_execution_record is None:
            raise SafetyAbort(
                "arm_clear_of_view 空位图拍摄于抓取之前，必须用 "
                "--pick-execution-record 提供当前持瓶位姿"
            )
        record = json.loads(
            cli.pick_execution_record.read_text(encoding="utf-8")
        )
        if (
            record.get("schema_version") != "grabber.mtc_execution.v1"
            or record.get("mode") != "pick"
            or not isinstance(record.get("completion"), dict)
        ):
            raise SafetyAbort("pick 执行证据格式无效")
        picked_at = datetime.fromisoformat(record["completed_at_utc"])
        if picked_at.tzinfo is None:
            raise SafetyAbort("pick 执行证据时间缺少时区")
        pick_age_s = (datetime.now(timezone.utc) - picked_at).total_seconds()
        if pick_age_s < 0.0 or pick_age_s > 900.0:
            raise SafetyAbort(f"pick 执行证据不新鲜: age={pick_age_s:.1f}s")
        completion = record["completion"]
        held_pose_input = completion.get("final_tcp_base_xyz_rpy_rad")
        held_joints_input = completion.get("final_right_joints_deg")
        keep = np.ones(len(voxels), dtype=bool)
    else:
        held_pose_input = payload["held_tcp_base_xyz_rpy_rad"]
        held_joints_input = payload.get("held_right_joints_deg")

    held_pose = np.asarray(held_pose_input, dtype=float)
    if held_pose.shape != (6,) or not np.all(np.isfinite(held_pose)):
        raise SafetyAbort("held TCP 必须是有限的 xyz 米 + rpy 弧度")
    held_right_joints = np.asarray(held_joints_input, dtype=float)
    if (
        held_right_joints.shape != (7,)
        or not np.all(np.isfinite(held_right_joints))
    ):
        raise SafetyAbort("缺少持瓶时的七个右臂关节")
    held_tcp = pose_matrix(held_pose)
    held_center = held_tcp[:3, 3].copy()
    # Side grasp is 40% down from the 21 cm bottle top.
    held_center[2] -= 0.021
    if regime == "held_arm_subtracted":
        link7_to_flange, flange_to_tcp = (
            profile.tool_mount_calibration.require_transforms()
        )
        link7 = held_tcp @ np.linalg.inv(link7_to_flange @ flange_to_tcp)
        relative = voxels - held_center
        held = (
            np.linalg.norm(relative[:, :2], axis=1)
            <= 0.033 + float(payload["voxel_size_m"]) / math.sqrt(2.0)
        ) & (
            np.abs(relative[:, 2])
            <= 0.105 + float(payload["voxel_size_m"]) / 2.0
        )
        segment = held_tcp[:3, 3] - link7[:3, 3]
        segment_length_sq = float(segment @ segment)
        along = np.clip(
            ((voxels - link7[:3, 3]) @ segment) / segment_length_sq,
            0.0,
            1.0,
        )
        nearest_tool = link7[:3, 3] + along[:, None] * segment
        robot_tool = (
            np.linalg.norm(voxels - nearest_tool, axis=1)
            <= 0.055
            + math.sqrt(3.0) * float(payload["voxel_size_m"]) / 2.0
        )
        keep = ~(held | robot_tool)
    obstacle_voxels = shifted_profile.moveit_obstacles_outside_fences(
        voxels[keep].tolist(), collision_boxes
    )

    target_tcp = held_tcp.copy()
    target_tcp[:3, :3] = (
        Rotation.from_euler("z", cli.target_yaw_offset_deg, degrees=True)
        .as_matrix()
        @ target_tcp[:3, :3]
    )
    target_tcp[0, 3], target_tcp[1, 3] = candidate["xy_base"]
    target_tcp[2, 3] = surface_z + 0.126 + 0.005
    target_moveit = profile.T_moveit_from_profile @ target_tcp
    held_tcp_moveit = profile.T_moveit_from_profile @ held_tcp
    held_center_moveit = profile.point_to_moveit(held_center)

    def pose(transform):
        return {
            "xyz": transform[:3, 3].tolist(),
            "quat_xyzw": Rotation.from_matrix(
                transform[:3, :3]
            ).as_quat().tolist(),
        }

    scenario = {
        "scenario_id": f"second_layer_place_{captured:%Y%m%dT%H%M%SZ}",
        "mode": "place_only",
        "frame_id": profile.moveit_frame,
        "planning_arm_id": "right_arm",
        "source_layer_id": "held_bottle",
        "target_layer_id": "second_layer_empty_patch",
        "lift_state_id": f"lift_{payload['lift_height_mm']}_mm",
        "source_support_surface_id": "",
        "target_support_surface_id": "fence_shelf_bottom",
        "scene_version": "fixed_head_empty_shelf_v1",
        "fixture_source": False,
        "start_state_source": "current_state",
        "source_grasp_pose": pose(held_tcp_moveit),
        "source_grasp_candidates": [
            {"id": "held_bottle", "pose": pose(held_tcp_moveit)}
        ],
        "target_place_pose": pose(target_moveit),
        "source_approach_direction": [0.0, -1.0, 0.0],
        "source_lift_direction": [0.0, 0.0, 1.0],
        "source_retreat_direction": [0.0, 1.0, 0.0],
        "target_insert_direction": [0.0, 0.0, -1.0],
        "target_retreat_direction": [0.0, 0.0, 1.0],
        "source_pregrasp_offset_m": 0.085,
        "source_contact_distance_m": 0.020,
        "source_lift_distance_m": 0.050,
        "source_retreat_distance_m": 0.150,
        "target_preplace_offset_m": 0.085,
        "target_contact_distance_m": 0.010,
        "target_retreat_distance_m": 0.120,
        "cartesian_min_fraction": 1.0,
        "post_place_home_joints_deg": list(profile.grasp_start_right_joints_deg),
        "bottle": {
            "id": "bottle",
            "radius_m": 0.033,
            "height_m": 0.21,
            "pose": {
                "xyz": held_center_moveit.tolist(),
                "rpy_deg": [0.0, 0.0, 0.0],
            },
        },
        "spawn_scene_objects": True,
        "shelf_boxes": [
            {
                "id": item["id"],
                "size": item["size"],
                "pose": {
                    "xyz": item["center"],
                    "rpy_deg": [0.0, 0.0, 0.0],
                },
            }
            for item in collision_boxes
        ],
        "dynamic_obstacle_id": "head_rgbd_non_target",
        "obstacle_voxel_size_m": payload["voxel_size_m"],
        "obstacle_voxels": obstacle_voxels,
        "scene_captured_at_utc": payload["captured_at_utc"],
        "freshness_max_age_s": 900.0,
        "planner_id": "RRTConnectkConfigDefault",
        "planning_timeout_s": 5.0,
        "max_ik_solutions": 8,
        "max_solutions": 10,
        "placement_provenance": {
            "candidate": candidate,
            "support_source": payload.get("support_source"),
            "target_yaw_offset_deg": cli.target_yaw_offset_deg,
            "surface_z_base": surface_z,
            # An empty-handed map removed nothing because nothing was in the
            # way; that is the point of taking it before the pick.
            "occlusion_regime": regime,
            "held_voxels_removed": (
                0 if regime == "arm_clear_of_view"
                else int(np.count_nonzero(held))
            ),
            "robot_tool_voxels_removed": (
                0 if regime == "arm_clear_of_view"
                else int(np.count_nonzero(robot_tool))
            ),
            "non_target_voxel_count": len(obstacle_voxels),
            "held_right_joints_deg": held_right_joints.tolist(),
        },
    }
    cli.output.write_text(
        yaml.safe_dump(scenario, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"place-only MTC 场景已写入 {cli.output}")
    print(json.dumps(scenario["placement_provenance"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SafetyAbort, OSError, ValueError, KeyError) as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
