#!/usr/bin/env python3
"""Convert a fresh empty-shelf observation into an MTC place-only scenario."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
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
from shelf_dispenser.relative_place import (
    build_relative_place_route,
    product_geometry,
)
from shelf_dispenser.safety import load_safety_profile
from scripts.apply_demonstrated_grasp_to_scenario import select_row_candidate

# How much daylight the bottle's base keeps over the support while travelling.
# The operator's three placements cleared it by 51, 63 and 93 mm; the low end
# of that is plenty and costs the least reach.
TRANSIT_CLEARANCE_M = 0.040


def _transit_raise_m(
    collision_boxes,
    held_tcp_z: float,
    bottle_center_in_tcp,
    bottle_height_m: float,
) -> float:
    """How far to lift before moving in, so the bottle's base clears the panel.

    Every term is measured this round: the support box comes from the profile
    the fence adapted to this capture, the held TCP from the lift record, and
    the bottle's own dimensions from its scenario entry.  Swap in a taller
    bottle and this grows by itself.
    """
    support = next(
        (
            item
            for item in collision_boxes
            if str(item["id"]) == "fence_shelf_bottom"
        ),
        None,
    )
    if support is None:
        raise SafetyAbort("放置场景缺少 fence_shelf_bottom，无法算出过渡抬升高度")
    support_top_z = float(support["center"][2]) + float(support["size"][2]) / 2.0
    # The bottle stands upright in the gripper, so its base sits half its height
    # below its centre, and the centre is the recorded offset from the TCP.
    base_below_tcp = bottle_height_m / 2.0 - float(bottle_center_in_tcp[2])
    needed_tcp_z = support_top_z + base_below_tcp + TRANSIT_CLEARANCE_M
    return max(0.0, needed_tcp_z - float(held_tcp_z))


RELEASE_CLEARANCE_M = 0.005
UPPER_SHELF_SURFACE_GROUND_M = 0.995
LOWER_SHELF_SURFACE_GROUND_M = 0.625
UPPER_SHELF_LIFT_MM = 647
LOWER_SHELF_SURFACE_TOLERANCE_M = 0.025
# How far a perceived empty patch may sit behind the taught depth before the
# pull-back stops being a correction and starts being a guess.  The 2026-08-08
# failure was 55 mm; the lower shelf's floor is only 460 mm deep and the taught
# points sit 82 mm behind its lip, so a candidate more than 80 mm past them is
# not the same slot seen imprecisely, it is a different surface.  Refuse rather
# than drag the bottle 150 mm forward and drop it off the front edge.
MAX_DEPTH_PULLBACK_M = 0.080


def select_place_demonstration(
    directory: Path,
    *,
    layer_id: str,
    arm_id: str,
    target_tcp_x_m: float,
    max_x_error_m: float,
    target_slot: str | None = None,
) -> dict:
    """Select the successful teaching whose full route will be retargeted."""
    prefix = {"upper": "A", "lower": "B"}.get(layer_id)
    expected_lift = {"upper": 647, "lower": 250}.get(layer_id)
    if prefix is None or not math.isfinite(float(target_tcp_x_m)):
        raise SafetyAbort("放置示教层或目标 X 无效")
    if target_slot is not None and (
        target_slot not in {f"{prefix}{index}" for index in range(1, 5)}
    ):
        raise SafetyAbort(f"目标槽位 {target_slot!r} 不属于 {layer_id}")
    if not 0.005 <= float(max_x_error_m) <= 0.15:
        raise SafetyAbort("放置示教最大 X 匹配误差必须在 5..150 mm")

    matches = []
    pattern = (
        f"place_{target_slot}_*_*.json"
        if target_slot is not None
        else f"place_{prefix}[1-4]_*_*.json"
    )
    for path in sorted(Path(directory).glob(pattern)):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("arm_id") != arm_id:
            continue
        if (
            record.get("schema_version")
            != "grabber.demonstrated_joint_trajectory.v1"
            or record.get("pose_frame") != "platform_base_link"
            or abs(int(record.get("lift_height_mm", -1)) - expected_lift) > 5
            or record.get("usable_for_planning_constraints") is not True
            or record.get("fence_violations")
            or not record.get("samples")
        ):
            raise SafetyAbort(f"放置示教不可用于规划: {path.name}")
        final = record["samples"][-1]
        pose = final.get("tcp_pose_moveit") or {}
        xyz = np.asarray(pose.get("xyz"), dtype=float)
        quaternion = np.asarray(pose.get("quat_xyzw"), dtype=float)
        joints = np.asarray(final.get("joints_deg"), dtype=float)
        if (
            xyz.shape != (3,)
            or quaternion.shape != (4,)
            or joints.shape != (7,)
            or not np.all(np.isfinite(xyz))
            or not np.all(np.isfinite(quaternion))
            or not np.all(np.isfinite(joints))
            or float(np.linalg.norm(quaternion)) < 1e-9
        ):
            raise SafetyAbort(f"放置示教终点无效: {path.name}")
        matches.append(
            {
                "path": path.name,
                "slot_id": str(record.get("label", "")).split("_")[1],
                "label": str(record.get("label", path.stem)),
                "tcp_pose_moveit": {
                    "xyz": xyz.tolist(),
                    "quat_xyzw": (quaternion / np.linalg.norm(quaternion)).tolist(),
                },
                "reference_joints_deg": joints.tolist(),
                "x_error_m": abs(float(xyz[0]) - float(target_tcp_x_m)),
            }
        )
    if not matches:
        raise SafetyAbort(f"没有 {layer_id}/{arm_id} 的逐点放置示教")
    selected = min(matches, key=lambda item: item["x_error_m"])
    if selected["x_error_m"] > float(max_x_error_m):
        raise SafetyAbort(
            "目标超出逐点放置示教覆盖范围: "
            f"nearest_error={selected['x_error_m'] * 1000:.1f} mm > "
            f"{float(max_x_error_m) * 1000:.1f} mm"
        )
    return selected


def expected_lower_surface_z(profile, lift_height_mm: int) -> float:
    upper_surface = next(
        box.maximum[2]
        for box in profile.keepout_boxes
        if box.id == "shelf_bottom"
    )
    shelf_delta = LOWER_SHELF_SURFACE_GROUND_M - UPPER_SHELF_SURFACE_GROUND_M
    lift_delta = (int(lift_height_mm) - UPPER_SHELF_LIFT_MM) / 1000.0
    return float(upper_surface + shelf_delta - lift_delta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument(
        "--product-code",
        required=True,
        choices=("P01", "P02", "P03", "P04", "P05", "P06"),
        help="本轮已知商品编号；尺寸从商品目录读取，不允许使用通用瓶子常量",
    )
    parser.add_argument(
        "--target-slot",
        choices=("B3", "B4"),
        help=(
            "固定右臂下层槽位；B1/B2 的左臂执行仍受工具标定安全闸限制。"
            "不给时按实时候选 X 选择最近的右臂逐点示教"
        ),
    )
    parser.add_argument(
        "--row-templates",
        type=Path,
        default=ROOT / "outputs" / "row_templates.json",
    )
    parser.add_argument("--max-template-x-error-m", type=float, default=0.04)
    parser.add_argument(
        "--place-demonstrations",
        type=Path,
        default=ROOT / "outputs" / "demonstrated_trajectories",
    )
    parser.add_argument("--max-place-demo-x-error-m", type=float, default=0.08)
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
    geometry = product_geometry(cli.product_code)
    captured_product_code = payload.get("product_code")
    if (
        captured_product_code is not None
        and captured_product_code != geometry["product_code"]
    ):
        raise SafetyAbort(
            "空位采集与放置规划商品编号不一致: "
            f"capture={captured_product_code}, plan={geometry['product_code']}"
        )
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
    expected_surface_z = expected_lower_surface_z(
        profile, int(payload["lift_height_mm"])
    )
    surface_z_error = surface_z - expected_surface_z
    if abs(surface_z_error) > LOWER_SHELF_SURFACE_TOLERANCE_M:
        raise SafetyAbort(
            "下层底板高度与现场测量不一致: "
            f"视觉 z={surface_z:.4f}m, 物理约束 z={expected_surface_z:.4f}m, "
            f"误差={surface_z_error * 1000:.1f}mm > "
            f"{LOWER_SHELF_SURFACE_TOLERANCE_M * 1000:.0f}mm；"
            "拒绝把瓶盖、背板横带或其他平面当成底板"
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
        bottle_center_in_tcp_input = completion.get(
            "bottle_center_in_tcp_m"
        )
        keep = np.ones(len(voxels), dtype=bool)
    else:
        held_pose_input = payload["held_tcp_base_xyz_rpy_rad"]
        held_joints_input = payload.get("held_right_joints_deg")
        bottle_center_in_tcp_input = payload.get(
            "bottle_center_in_tcp_m"
        )

    held_pose = np.asarray(held_pose_input, dtype=float)
    if held_pose.shape != (6,) or not np.all(np.isfinite(held_pose)):
        raise SafetyAbort("held TCP 必须是有限的 xyz 米 + rpy 弧度")
    held_right_joints = np.asarray(held_joints_input, dtype=float)
    if (
        held_right_joints.shape != (7,)
        or not np.all(np.isfinite(held_right_joints))
    ):
        raise SafetyAbort("缺少持瓶时的七个右臂关节")
    bottle_center_in_tcp = np.asarray(
        bottle_center_in_tcp_input, dtype=float
    )
    if (
        bottle_center_in_tcp.shape != (3,)
        or not np.all(np.isfinite(bottle_center_in_tcp))
        or float(np.linalg.norm(bottle_center_in_tcp)) > 0.25
    ):
        raise SafetyAbort("缺少抓取时记录的瓶心相对 TCP 偏移")
    held_tcp = pose_matrix(held_pose)
    held_center = (
        held_tcp[:3, 3]
        + held_tcp[:3, :3] @ bottle_center_in_tcp
    )
    if regime == "held_arm_subtracted":
        link7_to_flange, flange_to_tcp = (
            profile.tool_mount_calibration.require_transforms()
        )
        link7 = held_tcp @ np.linalg.inv(link7_to_flange @ flange_to_tcp)
        relative = voxels - held_center
        held = (
            np.linalg.norm(relative[:, :2], axis=1)
            <= geometry["radius_m"]
            + float(payload["voxel_size_m"]) / math.sqrt(2.0)
        ) & (
            np.abs(relative[:, 2])
            <= geometry["height_m"] / 2.0
            + float(payload["voxel_size_m"]) / 2.0
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

    target_bottle_center = np.asarray(
        [
            *candidate["xy_base"],
            surface_z + geometry["height_m"] / 2.0 + RELEASE_CLEARANCE_M,
        ],
        dtype=float,
    )
    target_bottle_center_moveit = profile.point_to_moveit(
        target_bottle_center
    )
    template_bytes = cli.row_templates.read_bytes()
    templates = json.loads(template_bytes)
    selected_template = select_row_candidate(
        templates,
        layer_id="lower",
        arm_id="right_arm",
        target_tcp_x_m=float(target_bottle_center_moveit[0]),
        max_x_error_m=cli.max_template_x_error_m,
        operation="place",
        pose_semantics="grasp_curve_place_initial_guess_only",
    )
    selected_demo = select_place_demonstration(
        cli.place_demonstrations,
        layer_id="lower",
        arm_id="right_arm",
        target_tcp_x_m=float(target_bottle_center_moveit[0]),
        max_x_error_m=cli.max_place_demo_x_error_m,
        target_slot=cli.target_slot,
    )
    template_rotation_moveit = Rotation.from_quat(
        selected_demo["tcp_pose_moveit"]["quat_xyzw"]
    ).as_matrix()
    target_tcp = held_tcp.copy()
    target_tcp[:3, :3] = (
        Rotation.from_euler("z", cli.target_yaw_offset_deg, degrees=True)
        .as_matrix()
        @ profile.T_moveit_from_profile[:3, :3].T
        @ template_rotation_moveit
    )
    target_tcp[:3, 3] = (
        target_bottle_center
        - target_tcp[:3, :3] @ bottle_center_in_tcp
    )
    target_moveit = profile.T_moveit_from_profile @ target_tcp
    # Perception says WHERE ALONG THE ROW the slot is free.  It does not get to
    # say how deep.  The empty-patch centroid is biased toward the back of the
    # shelf -- the lip and the neighbouring bottles occlude the front, so the
    # visible free surface starts behind the reachable part of it -- and the ROI
    # widening of 2026-08-07 (140 mm -> 350 mm of depth, needed to get any
    # candidates at all) removed the only thing that had been bounding it.
    #
    # 2026-08-08 it asked for a place 55 mm behind the taught depth at the same
    # row position, which is 30 mm past where the right arm can reach with this
    # orientation at all.  Nothing downstream noticed: the template match is
    # validated on x only, so `orientation_template_x_error_m` read 1.6 mm while
    # the endpoint sat outside the workspace.  target_approach then truncated at
    # fraction 0.655 and MoveIt reported only "failed to move full distance",
    # which reads like a collision and is not one: an unreachable goal has no
    # colliding pair to find, so looking for one finds nothing.
    #
    # The taught point is a demonstrated depth: a human drove the arm there, so
    # it is reachable by construction.  Never place beyond it.  Deeper is the
    # only unsafe direction -- pulling the bottle toward the shelf mouth keeps
    # it on the same support and strictly inside the reachable set.
    template_depth_m = float(
        selected_template["candidate"]["tcp_pose_moveit"]["xyz"][1]
    )
    requested_depth_m = float(target_moveit[1, 3])
    depth_pullback_m = max(0.0, template_depth_m - requested_depth_m)
    if depth_pullback_m > MAX_DEPTH_PULLBACK_M:
        raise SafetyAbort(
            "空位候选比已示教深度深 "
            f"{depth_pullback_m * 1000:.0f} mm，超过 "
            f"{MAX_DEPTH_PULLBACK_M * 1000:.0f} mm 上限；"
            "感知与行模板对不上，拒绝硬拉回"
        )
    target_moveit[1, 3] = requested_depth_m + depth_pullback_m
    held_tcp_moveit = profile.T_moveit_from_profile @ held_tcp
    held_center_moveit = profile.point_to_moveit(held_center)
    required_raise_m = _transit_raise_m(
        collision_boxes,
        held_tcp[2, 3],
        bottle_center_in_tcp,
        geometry["height_m"],
    )
    demonstration = json.loads(
        (cli.place_demonstrations / selected_demo["path"]).read_text(
            encoding="utf-8"
        )
    )
    route = build_relative_place_route(
        demonstration,
        current_tcp_moveit_xyz=held_tcp_moveit[:3, 3],
        target_tcp_moveit_xyz=target_moveit[:3, 3],
        target_tcp_moveit_quat_xyzw=Rotation.from_matrix(
            target_moveit[:3, :3]
        ).as_quat(),
        contact_distance_m=0.010,
        minimum_transit_raise_m=required_raise_m,
    )

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
        "target_preplace_pose": route["preplace_pose"],
        "source_approach_direction": [0.0, -1.0, 0.0],
        "source_lift_direction": [0.0, 0.0, 1.0],
        "source_retreat_direction": [0.0, 1.0, 0.0],
        # This slot's taught terminal path replaces the old shared 45° policy.
        "target_insert_direction": route["insert_direction"],
        "target_contact_direction": route["contact_direction"],
        "target_retreat_direction": route["retreat_direction"],
        "source_pregrasp_offset_m": 0.085,
        "source_contact_distance_m": 0.020,
        "source_lift_distance_m": 0.050,
        "source_retreat_distance_m": 0.150,
        "target_preplace_offset_m": route["preplace_offset_m"],
        # Phase one of the operator's own description: "先有一个原地抬升，把瓶子
        # 底部抬到货架底板之上，然后再运动，最后放下."
        #
        # It is not a preference.  fence_shelf_bottom is not the 2 cm panel, it
        # is a 507 mm slab of keepout whose top is the measured panel top, and
        # the held bottle hangs ~108 mm below the TCP.  So the bottle is inside
        # that slab whenever the TCP is below (panel top + 108 mm), which is
        # exactly where it sits in the carry pose.  A joint-space straight line
        # rises and advances at the same time, so it enters the slab's footprint
        # before it has risen out of it -- MoveIt puts the first violation at 5%
        # of the path, contact pair fence_shelf_bottom <-> held_bottle.  Raising
        # first, straight up and outside the shelf footprint, is what makes the
        # rest of the motion representable at all.
        #
        # Computed, never a constant: change the bottle and every term moves.
        "target_transit_raise_m": route["transit_raise_m"],
        "target_contact_distance_m": route["contact_distance_m"],
        "target_retreat_distance_m": route["preplace_offset_m"],
        "cartesian_min_fraction": 1.0,
        "target_preplace_reference_joints_deg": route[
            "preplace_reference_joints_deg"
        ],
        "target_place_reference_joints_deg": route[
            "place_reference_joints_deg"
        ],
        "post_place_home_joints_deg": list(profile.grasp_start_right_joints_deg),
        "product_code": geometry["product_code"],
        "bottle": {
            "id": "bottle",
            "radius_m": geometry["radius_m"],
            "height_m": geometry["height_m"],
            "graspable_cylinder_height_m": geometry["graspable_height_m"],
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
            "expected_surface_z_base": expected_surface_z,
            "surface_z_error_m": surface_z_error,
            "upper_shelf_surface_ground_m": UPPER_SHELF_SURFACE_GROUND_M,
            "lower_shelf_surface_ground_m": LOWER_SHELF_SURFACE_GROUND_M,
            "orientation_template_point_id": selected_template["point_id"],
            "orientation_template_x_error_m": selected_template["x_error_m"],
            "template_depth_m": template_depth_m,
            "requested_depth_m": requested_depth_m,
            "depth_pullback_m": depth_pullback_m,
            "place_demonstration": selected_demo["path"],
            "target_slot": route["slot_id"],
            "place_demonstration_label": selected_demo["label"],
            "place_demonstration_x_error_m": selected_demo["x_error_m"],
            "place_demonstration_usage": "relative_route",
            "relative_route": route,
            "row_templates_sha256": hashlib.sha256(
                template_bytes
            ).hexdigest(),
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
            "bottle_center_in_tcp_m": bottle_center_in_tcp.tolist(),
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
