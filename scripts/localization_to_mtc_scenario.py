#!/usr/bin/env python3
"""Convert one fresh bottle localization into a plan-only MTC scenario."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bottle_grasp.core import DemoParams, SafetyAbort
from bottle_grasp.grasp_orientation import installed_rmg24_tcp_rotation
from bottle_grasp.safety import load_safety_profile

# Leading part of the shelf insertion that stays fully collision checked; only
# the remainder may relax bottle<->hand contact.
COLLISION_FORBIDDEN_INSERTION_M = 0.013


def _finite_number(data: dict, key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SafetyAbort(f"定位字段 {key} 必须是数字")
    value = float(value)
    if not math.isfinite(value):
        raise SafetyAbort(f"定位字段 {key} 必须是有限数字")
    return value


def _vector3(value, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SafetyAbort(f"{label} 必须是 3 个有限数字") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise SafetyAbort(f"{label} 必须是 3 个有限数字")
    return vector


def _support_surface_id(point: np.ndarray, boxes: list[dict]) -> str:
    candidates = []
    for box in boxes:
        center = np.asarray(box["center"], dtype=float)
        size = np.asarray(box["size"], dtype=float)
        if (
            abs(point[0] - center[0]) <= size[0] / 2
            and abs(point[1] - center[1]) <= size[1] / 2
            and center[2] + size[2] / 2 <= point[2]
        ):
            candidates.append((center[2] + size[2] / 2, str(box["id"])))
    if not candidates:
        raise SafetyAbort("定位瓶子下方没有可识别的静态支撑面")
    return max(candidates)[1]


def _capture_timestamp(
    payload: dict,
    path: Path,
    *,
    label: str,
    max_age_s: float,
    allow_stale: bool,
) -> tuple[float, str, float]:
    captured_at = payload.get("captured_at_utc")
    freshness_source = "captured_at_utc"
    if captured_at is None:
        if not allow_stale:
            raise SafetyAbort(
                f"{label}缺少 captured_at_utc；重新采集，"
                "或仅做历史回放时显式传 --allow-stale"
            )
        captured_timestamp = path.stat().st_mtime
        freshness_source = "legacy_file_mtime"
    elif not isinstance(captured_at, str):
        raise SafetyAbort(f"{label} captured_at_utc 必须是带时区的 ISO 时间")
    else:
        try:
            captured = datetime.fromisoformat(captured_at)
        except ValueError as exc:
            raise SafetyAbort(f"{label} captured_at_utc 不是有效 ISO 时间") from exc
        if captured.tzinfo is None:
            raise SafetyAbort(f"{label} captured_at_utc 必须带时区")
        captured_timestamp = captured.timestamp()
    age_s = time.time() - captured_timestamp
    if age_s < -60:
        raise SafetyAbort(f"{label}时间在未来，拒绝使用")
    if age_s > max_age_s and not allow_stale:
        raise SafetyAbort(
            f"{label}已过期（{age_s:.1f}s > {max_age_s:.1f}s）；"
            "重新采集，或仅做历史回放时显式传 --allow-stale"
        )
    return captured_timestamp, freshness_source, max(0.0, age_s)


def build_scenario(
    localization_path: Path,
    template_path: Path,
    safety_profiles_path: Path,
    profile_name: str,
    *,
    max_age_s: float,
    allow_stale: bool,
    scene_path: Path | None = None,
    pick_only: bool = False,
    planning_arm_id: str = "right_arm",
) -> dict:
    raw_bytes = localization_path.read_bytes()
    localization = json.loads(raw_bytes)
    if not isinstance(localization, dict):
        raise SafetyAbort("定位 JSON 顶层必须是对象")

    params = DemoParams()
    freshness_limit = min(max_age_s, params.scene_max_age_s) if pick_only else max_age_s
    if pick_only and allow_stale:
        raise SafetyAbort("pick-only 直接抓取禁止 --allow-stale")
    captured_timestamp, freshness_source, age_s = _capture_timestamp(
        localization,
        localization_path,
        label="定位文件",
        max_age_s=freshness_limit,
        allow_stale=allow_stale,
    )
    point_base = _vector3(localization.get("point_base"), "定位字段 point_base")
    confidence = _finite_number(localization, "confidence")
    depth_m = _finite_number(localization, "depth_m")
    depth_mad_m = _finite_number(localization, "depth_mad_m")
    spread_m = _finite_number(localization, "position_spread_m")
    frame_count = localization.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise SafetyAbort("定位字段 frame_count 必须是整数")
    if confidence < params.confidence:
        raise SafetyAbort(
            f"定位置信度不足（{confidence:.3f} < {params.confidence:.3f}）"
        )
    # Each camera entry keeps its own range semantics. pick-only is localized by
    # the fixed head camera, whose usable range is far longer than the wrist
    # camera's close-in 0.65 m; reusing the wrist gate rejects shelf targets the
    # head sees perfectly well.
    min_depth_m = params.head_min_depth_m if pick_only else params.min_depth_m
    max_depth_m = params.head_max_depth_m if pick_only else params.max_depth_m
    if not min_depth_m <= depth_m <= max_depth_m:
        raise SafetyAbort(
            f"定位深度超界: {depth_m:.4f}m 不在"
            f"{'固定头部' if pick_only else '腕部'}相机量程 "
            f"{min_depth_m:.2f}..{max_depth_m:.2f}m 内"
        )
    if depth_mad_m > params.max_depth_mad_m:
        raise SafetyAbort(f"定位深度 MAD 过大: {depth_mad_m:.4f}m")
    if spread_m > params.max_position_spread_m:
        raise SafetyAbort(f"定位位置散布过大: {spread_m:.4f}m")
    if frame_count < params.samples:
        raise SafetyAbort(
            f"定位共识帧不足（{frame_count} < {params.samples}）"
        )

    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise SafetyAbort("MTC 模板顶层必须是对象")
    approach = _vector3(
        template.get("source_approach_direction"),
        "MTC 模板 source_approach_direction",
    )
    approach_norm = float(np.linalg.norm(approach))
    if approach_norm < 1e-9:
        raise SafetyAbort("MTC 模板 source_approach_direction 不能为零")
    approach /= approach_norm

    profile = load_safety_profile(
        safety_profiles_path,
        profile_name,
        require_verified=False,
    )
    if pick_only and profile_name != "shelf_template":
        raise SafetyAbort("pick-only 直接抓取第一版只允许 shelf_template")
    if pick_only and planning_arm_id not in {"right_arm", "left_arm"}:
        raise SafetyAbort("pick-only planning_arm_id 只允许 right_arm/left_arm")
    if pick_only and scene_path is None:
        raise SafetyAbort("pick-only 直接抓取必须提供本轮固定头部 --scene-json")
    rotation = profile.T_moveit_from_profile[:3, :3]
    approach_profile = rotation.T @ approach
    pick_radius_m = None
    if pick_only:
        radius = (template.get("bottle") or {}).get("radius_m")
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not math.isfinite(float(radius))
            or float(radius) <= 0
        ):
            raise SafetyAbort("pick-only 模板 bottle.radius_m 必须是正有限数")
        pick_radius_m = float(radius)
    # The fixed head sees the near surface, and grasp_stop_short_m is the one
    # working distance in this project: it was tuned on the real right arm
    # until grasps held, so it already absorbs whatever the true link7->finger
    # offset is.  pick-only briefly used surface + radius instead, i.e. the
    # bottle axis, which assumes the modelled TCP *is* the finger centre --
    # an assumption nothing here has ever measured.  That drove every grasp
    # 63 mm too deep.  Both paths use the validated distance.
    tcp_profile = point_base - approach_profile * params.grasp_stop_short_m
    profile.assert_tcp_point(tcp_profile, label="MTC 定位抓取 TCP")

    surface_moveit = profile.point_to_moveit(point_base)
    tcp_moveit = profile.point_to_moveit(tcp_profile)
    digest = hashlib.sha256(raw_bytes).hexdigest()
    stamp = datetime.fromtimestamp(
        captured_timestamp, tz=timezone.utc
    ).isoformat()

    scenario = dict(template)
    scenario["scenario_id"] = f"localized_bottle_{digest[:12]}"
    scenario["source_layer_id"] = "localized_bottle_surface"
    scenario["target_layer_id"] = "localized_placeback"
    scenario["scene_version"] = f"localization@sha256:{digest[:12]}"
    scenario["fixture_source"] = not pick_only
    scenario["start_state_source"] = "current_state"
    scenario["spawn_scene_objects"] = True
    if pick_only:
        scenario["mode"] = "pick_only"
        scenario["planning_arm_id"] = planning_arm_id
        scenario["local_motion_planner"] = "pilz_lin"
        scenario["target_captured_at_utc"] = stamp
        scenario["freshness_max_age_s"] = freshness_limit
        # Give the planner the posture a person used for this grasp.  Without
        # it, IK and RRTConnect pick among equivalent arm configurations at
        # random: eight replays of one scene returned the demonstrated posture
        # four times, a posture 160-210 deg away twice, nothing twice.
        if (
            planning_arm_id == "right_arm"
            and profile.demonstrated_grasp_right_joints_deg is not None
        ):
            scenario["source_grasp_reference_joints_deg"] = list(
                profile.demonstrated_grasp_right_joints_deg
            )
        # The offset used to be extended by grasp_stop_short_m + radius to keep
        # the standoff in place while the grasp itself moved to the bottle
        # axis.  With the grasp back at the validated distance the template
        # value already puts the standoff at the same physical point.
        workspace_corners = [
            [
                profile.tcp_workspace.minimum[axis]
                if side == 0
                else profile.tcp_workspace.maximum[axis]
                for axis, side in enumerate(bits)
            ]
            for bits in itertools.product((0, 1), repeat=3)
        ]
        workspace_moveit = np.asarray(
            profile.points_to_moveit(workspace_corners), dtype=float
        )
        workspace_min = workspace_moveit.min(axis=0) + profile.clearance_m
        workspace_max = workspace_moveit.max(axis=0) - profile.clearance_m
        if np.any(workspace_max <= workspace_min):
            raise SafetyAbort("pick-only TCP 工作区扣除安全余量后为空")
        scenario["tcp_path_workspace"] = {
            "id": "tcp_path_workspace",
            "size": (workspace_max - workspace_min).tolist(),
            "pose": {
                "xyz": ((workspace_min + workspace_max) / 2).tolist(),
                "rpy_deg": [0.0, 0.0, 0.0],
            },
        }
        # The installed MoveIt hand mesh is more conservative than the RMG24.
        # Keep target contact forbidden through current->pregrasp and the first
        # 13 mm insertion, then allow it only for the final bounded segment.
        # Derive it from the offset so the two can never drift apart again.
        scenario["source_contact_distance_m"] = (
            float(scenario["source_pregrasp_offset_m"])
            - COLLISION_FORBIDDEN_INSERTION_M
        )

    for key in ("source_grasp_pose", "target_place_pose"):
        if not isinstance(scenario.get(key), dict):
            raise SafetyAbort(f"MTC 模板 {key} 必须是 pose 对象")
        pose = dict(scenario[key])
        pose["xyz"] = tcp_moveit.tolist()
        scenario[key] = pose
    if pick_only:
        if profile.grasp_frame is None:
            raise SafetyAbort("pick-only shelf_template 缺少抓取轴语义")
        base_rotation = Rotation.from_matrix(
            rotation @ installed_rmg24_tcp_rotation(profile.grasp_frame)
        )
        source_pose = {
            **scenario["source_grasp_pose"],
            "quat_xyzw": base_rotation.as_quat().tolist(),
        }
        scenario["source_grasp_pose"] = source_pose
        # The one authored orientation is the one the operator demonstrated on
        # the real shelf; installed_rmg24_tcp_rotation() reproduces it to
        # within their own wrist tilt.  Pitch about the finger axis is a small
        # tilt of that same grasp and stays.  Roll about the approach axis is
        # not offered: it is only equivalent to roll 0 if the finger centre
        # sits exactly on the tool axis, and the link7->TCP transform carrying
        # that claim is nominal, never measured.  A lateral offset d there
        # would move the real grasp by 2d when rolled.
        scenario["source_grasp_candidates"] = [
            {
                "id": f"horizontal_fingers_roll_0{pitch_suffix}",
                "pose": {
                    **source_pose,
                    # Right multiplication keeps this offset in TCP-local Y.
                    "quat_xyzw": (
                        base_rotation
                        * Rotation.from_euler("y", pitch_deg, degrees=True)
                    ).as_quat().tolist(),
                },
            }
            for pitch_deg, pitch_suffix in (
                (0.0, ""),
                (3.0, "_pitch_plus_3"),
                (-3.0, "_pitch_minus_3"),
            )
        ]

    bottle = dict(scenario.get("bottle", {}))
    bottle_pose = dict(bottle.get("pose", {}))
    bottle_center = surface_moveit
    if pick_only:
        # Fixed-head RGB-D locks the visible near surface, not the cylinder
        # centre. The centre is one radius farther along the approach axis.
        bottle_center = surface_moveit + approach * pick_radius_m
    bottle_pose["xyz"] = bottle_center.tolist()
    bottle_pose.setdefault("rpy_deg", [0.0, 0.0, 0.0])
    bottle["pose"] = bottle_pose
    scenario["bottle"] = bottle
    scene_digest = None
    scene_age_s = None
    if pick_only:
        assert scene_path is not None
        scene_bytes = scene_path.read_bytes()
        live_scene = json.loads(scene_bytes)
        if not isinstance(live_scene, dict):
            raise SafetyAbort("固定头部场景 JSON 顶层必须是对象")
        scene_timestamp, _, scene_age_s = _capture_timestamp(
            live_scene,
            scene_path,
            label="固定头部场景",
            max_age_s=freshness_limit,
            allow_stale=False,
        )
        scenario["scene_captured_at_utc"] = datetime.fromtimestamp(
            scene_timestamp, tz=timezone.utc
        ).isoformat()
        if live_scene.get("safety_profile") != "shelf_template":
            raise SafetyAbort("固定头部场景不是 shelf_template")
        if live_scene.get("frame") != profile.frame:
            raise SafetyAbort(
                "固定头部场景坐标系与 shelf_template 不一致"
            )
        image_height_px = live_scene.get("image_height_px")
        observed_row_limit_px = live_scene.get("observed_row_limit_px")
        if (
            isinstance(image_height_px, bool)
            or not isinstance(image_height_px, int)
            or image_height_px <= 0
            or isinstance(observed_row_limit_px, bool)
            or not isinstance(observed_row_limit_px, int)
            or observed_row_limit_px != image_height_px
        ):
            raise SafetyAbort(
                "固定头部直抓场景必须处理完整深度帧；底部裁剪会把未知区域伪装为空闲"
            )
        scene_target = _vector3(
            live_scene.get("target_point_base"),
            "固定头部场景 target_point_base",
        )
        if float(np.linalg.norm(scene_target - point_base)) > 0.01:
            raise SafetyAbort("固定头部场景与定位目标不匹配（差值超过 1 cm）")
        collision_boxes = live_scene.get("collision_boxes")
        if not isinstance(collision_boxes, list):
            raise SafetyAbort("固定头部场景缺少 collision_boxes")
        required_shelf_ids = {
            "fence_shelf_bottom",
            "fence_shelf_top",
            "fence_shelf_back",
        }
        actual_ids = {str(item.get("id", "")) for item in collision_boxes}
        missing = sorted(required_shelf_ids - actual_ids)
        if missing:
            raise SafetyAbort(
                "固定头部场景缺少真实货架底/顶/背几何: " + ", ".join(missing)
            )
        for item in collision_boxes:
            center = _vector3(item.get("center"), "固定头部 collision box center")
            size = _vector3(item.get("size"), "固定头部 collision box size")
            if np.any(size <= 0) or not str(item.get("id", "")):
                raise SafetyAbort("固定头部 collision box 必须有 id 和正尺寸")
        scene_voxels = np.asarray(live_scene.get("scene_voxels"), dtype=float)
        target_voxels = np.asarray(
            live_scene.get("target_occupancy_voxels"), dtype=float
        )
        raw_non_target_voxels = live_scene.get("non_target_scene_voxels")
        if not isinstance(raw_non_target_voxels, list):
            raise SafetyAbort(
                "固定头部场景缺少原始深度点级过滤后的 "
                "non_target_scene_voxels"
            )
        non_target_profile = np.asarray(raw_non_target_voxels, dtype=float)
        if non_target_profile.size == 0:
            non_target_profile = np.empty((0, 3), dtype=float)
        if (
            scene_voxels.ndim != 2
            or scene_voxels.shape[1] != 3
            or not np.all(np.isfinite(scene_voxels))
        ):
            raise SafetyAbort("固定头部 scene_voxels 必须是有限 Nx3 数组")
        if (
            target_voxels.ndim != 2
            or target_voxels.shape[1] != 3
            or len(target_voxels) == 0
            or not np.all(np.isfinite(target_voxels))
        ):
            raise SafetyAbort("固定头部场景没有可安全剔除的目标占据体素")
        if (
            non_target_profile.ndim != 2
            or non_target_profile.shape[1] != 3
            or not np.all(np.isfinite(non_target_profile))
        ):
            raise SafetyAbort(
                "固定头部 non_target_scene_voxels 必须是有限 Nx3 数组"
            )
        voxel_size = float(live_scene.get("voxel_size_m", params.scene_voxel_m))
        if (
            not math.isfinite(voxel_size)
            or voxel_size <= 0
            or voxel_size > params.scene_voxel_m
        ):
            raise SafetyAbort("固定头部场景体素不得粗于当前安全参数")
        scene_keys = {
            tuple(item)
            for item in np.floor(scene_voxels / voxel_size).astype(np.int64)
        }
        target_keys = {
            tuple(item)
            for item in np.floor(target_voxels / voxel_size).astype(np.int64)
        }
        non_target_keys = {
            tuple(item)
            for item in np.floor(non_target_profile / voxel_size).astype(np.int64)
        }
        if scene_keys != target_keys | non_target_keys:
            raise SafetyAbort(
                "固定头部场景的目标/非目标点级分类与完整场景不一致"
            )
        non_target_moveit = profile.moveit_obstacles_outside_fences(
            non_target_profile.tolist(),
            collision_boxes,
        )
        # A voxel inside the target's own collision cylinder is the target,
        # whatever the per-pixel classifier decided.  Leaving it in the
        # obstacle cloud makes the bottle collide with itself the moment it is
        # attached: on 2026-08-03 two voxels 10.5 mm from the bottle axis --
        # its own cap -- failed attach_bottle with "head_rgbd_non_target
        # colliding with bottle" on five consecutive live captures, while the
        # 2026-08-02 scenes that planned had none inside the cylinder at all.
        # An obstacle genuinely overlapping the target would mean the target
        # model is wrong, and no grasp of it is well posed either way.
        if pick_only and non_target_moveit:
            centre = np.asarray(bottle_center, dtype=float)
            half_height = float(bottle["height_m"]) / 2.0 + voxel_size / 2.0
            radius = pick_radius_m + voxel_size / 2.0
            kept = [
                voxel
                for voxel in non_target_moveit
                if not (
                    float(
                        np.linalg.norm(
                            np.asarray(voxel, dtype=float)[:2] - centre[:2]
                        )
                    )
                    <= radius
                    and abs(float(voxel[2]) - float(centre[2])) <= half_height
                )
            ]
            scenario["target_cylinder_voxels_removed"] = len(
                non_target_moveit
            ) - len(kept)
            non_target_moveit = kept
        scenario["dynamic_obstacle_id"] = "head_rgbd_non_target"
        scenario["obstacle_voxel_size_m"] = voxel_size
        scenario["obstacle_voxels"] = non_target_moveit
        scene_digest = hashlib.sha256(scene_bytes).hexdigest()
    else:
        collision_boxes = profile.moveit_collision_boxes()
    scenario["shelf_boxes"] = [
        {
            "id": box["id"],
            "size": box["size"],
            "pose": {
                "xyz": box["center"],
                "rpy_deg": [0.0, 0.0, 0.0],
            },
        }
        for box in collision_boxes
    ]
    support_surface_id = _support_surface_id(surface_moveit, collision_boxes)
    scenario["source_support_surface_id"] = support_surface_id
    scenario["target_support_surface_id"] = support_surface_id
    scenario["localization_provenance"] = {
        "source_file": localization_path.name,
        "sha256": digest,
        "captured_at_utc": stamp,
        "freshness_source": freshness_source,
        "age_s_at_generation": age_s,
        "profile": profile_name,
        "confidence": confidence,
        "frame_count": frame_count,
        "depth_mad_m": depth_mad_m,
        "position_spread_m": spread_m,
        "execution_caveat": (
            "Plan-only export; real motion requires execute_mtc_trajectory.py "
            "and all live gates."
            if pick_only
            else (
                "Bottle localization plus static keepouts only; dynamic RGB-D "
                "obstacles are not represented. Plan-only fixture."
            )
        ),
    }
    if pick_only:
        scenario["scene_provenance"] = {
            "source_file": scene_path.name,
            "sha256": scene_digest,
            "age_s_at_generation": scene_age_s,
            "non_target_voxel_count": len(scenario["obstacle_voxels"]),
            "target_associated_voxel_count": len(target_voxels),
            "target_filter_stage": "raw_depth_before_voxelization",
            "image_height_px": image_height_px,
            "observed_row_limit_px": observed_row_limit_px,
        }
    return scenario


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("localization_json", type=Path)
    parser.add_argument("output_yaml", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--safety-profiles", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--max-age-s", type=float, default=300.0)
    parser.add_argument("--allow-stale", action="store_true")
    parser.add_argument(
        "--scene-json",
        type=Path,
        help="本轮固定头部 head_scene.json（pick-only 必填）",
    )
    parser.add_argument(
        "--pick-only",
        action="store_true",
        help="生成固定头部直抓场景，MTC 在 source retreat 后停止",
    )
    parser.add_argument(
        "--planning-arm",
        choices=("right_arm", "left_arm"),
        default="right_arm",
        help="pick-only 规划臂；左臂当前只允许 plan-only",
    )
    args = parser.parse_args()
    if not math.isfinite(args.max_age_s) or args.max_age_s <= 0:
        parser.error("--max-age-s 必须是正有限数字")

    try:
        scenario = build_scenario(
            args.localization_json,
            args.template,
            args.safety_profiles,
            args.profile,
            max_age_s=args.max_age_s,
            allow_stale=args.allow_stale,
            scene_path=args.scene_json,
            pick_only=args.pick_only,
            planning_arm_id=args.planning_arm,
        )
        args.output_yaml.write_text(
            yaml.safe_dump(
                scenario,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, yaml.YAMLError, SafetyAbort) as exc:
        parser.exit(2, f"拒绝生成 MTC 场景: {exc}\n")

    if args.pick_only:
        print(
            f"已生成 MTC {args.planning_arm} pick-only 场景: {args.output_yaml}\n"
            "包含本轮固定头部非目标障碍；仍为 plan-only，当前没有安全执行桥。"
        )
    else:
        print(
            f"已生成仅规划场景: {args.output_yaml}\n"
            "fixture_source=true；仍需动态 RGB-D 场景和人工审核，禁止据此直接运动。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
