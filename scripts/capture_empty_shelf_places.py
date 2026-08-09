#!/usr/bin/env python3
"""Capture fixed-head RGB-D and rank empty patches on one shelf layer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser import head_lock
from shelf_dispenser.core import SafetyAbort, pose_matrix
from shelf_dispenser.delivery_table import observe_output_table
from shelf_dispenser.live_arm import open_live_arm
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.mobile_body import LiftSocketAdapter
from shelf_dispenser.relative_place import product_geometry
from shelf_dispenser.safety import load_safety_profile
from shelf_dispenser.scene import (
    head_scene_points,
    union_scene_voxels,
    voxelize_scene_points,
)
from utils.config import load_config

# Same band normalize_to_grasp_start accepts for the taught height.
LIFT_TOLERANCE_MM = 5


def _demo_args(cli) -> SimpleNamespace:
    return SimpleNamespace(
        task_mode=None,
        execute=False,
        plan_only=False,
        config=cli.config,
        safety_config=cli.safety_config,
        safety_profile="shelf_template",
        stop_after_observation=False,
        confirm_before_grasp=False,
        place_back=False,
        return_home=False,
        resume_at_wrist=False,
        finish_from_current=False,
        target_product=None,
        host="127.0.0.1",
        port=cli.port,
        output_dir=str(cli.output.parent),
        observe_seconds=0.0,
    )


def _candidate_config(cli, demo) -> SimpleNamespace:
    return SimpleNamespace(
        table_roi_min=tuple(cli.roi_min),
        table_roi_max=tuple(cli.roi_max),
        table_height_bin_m=0.01,
        table_inlier_band_m=0.012,
        table_min_inliers=80,
        table_frame_agreement_m=0.012,
        table_edge_margin_m=0.08,
        table_support_radius_m=0.07,
        table_min_patch_points=4,
        place_clearance_radius_m=cli.clearance_radius_m,
        place_grid_m=0.04,
        obstacle_min_height_m=0.025,
        # Candidate clearance is an XY projection, so including geometry above
        # the bottle turns the shelf ceiling into a fake bottle obstacle.  That
        # pushed the 2026-08-08 target 65--82 mm behind every successful taught
        # placement.  The full 3-D scene still goes to MoveIt afterwards.
        obstacle_max_height_m=float(demo.params.held_bottle_height_m),
        max_place_candidates=8,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument("--expected-lift-mm", type=int, required=True)
    parser.add_argument(
        "--product-code",
        required=True,
        choices=("P01", "P02", "P03", "P04", "P05", "P06"),
    )
    parser.add_argument("--roi-min", type=float, nargs=3, required=True)
    parser.add_argument("--roi-max", type=float, nargs=3, required=True)
    parser.add_argument("--clearance-radius-m", type=float, default=0.10)
    parser.add_argument(
        "--camera",
        choices=("head", "right_wrist", "left_wrist"),
        default="head",
    )
    parser.add_argument(
        "--operator-confirms-shelf-obstacles-complete",
        action="store_true",
        help=(
            "Use the verified shelf box as support where the held arm occludes "
            "depth; requires the operator to confirm every shelf obstacle is visible"
        ),
    )
    held = parser.add_mutually_exclusive_group(required=True)
    held.add_argument(
        "--no-held-object",
        action="store_true",
        help=(
            "Capture before the pick, with nothing in the gripper.  The arm is "
            "then clear of the head camera's view of the shelf, so no arm "
            "subtraction happens and no occlusion shadow is left behind -- "
            "which is the only way 'empty' and 'not visible' stay "
            "distinguishable"
        ),
    )
    held.add_argument(
        "--held-tcp-base-rad",
        type=float,
        nargs=6,
        metavar=("X_M", "Y_M", "Z_M", "ROLL_RAD", "PITCH_RAD", "YAW_RAD"),
    )
    held.add_argument("--lift-execution-record", type=Path)
    held.add_argument("--pick-execution-record", type=Path)
    parser.add_argument("--held-right-joints-deg", type=float, nargs=7)
    parser.add_argument(
        "--held-bottle-center-in-tcp-m",
        type=float,
        nargs=3,
        help="手工 held TCP 模式下，瓶心在 TCP 局部坐标中的三维偏移",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8884)
    cli = parser.parse_args(argv)
    geometry = product_geometry(cli.product_code)
    if not 0.08 <= cli.clearance_radius_m <= 0.30:
        raise SafetyAbort("放置净空半径必须在 0.08..0.30 m")

    held_right_joints = cli.held_right_joints_deg
    held_pose_input = cli.held_tcp_base_rad
    bottle_center_in_tcp_input = cli.held_bottle_center_in_tcp_m
    if cli.lift_execution_record:
        lift_record = json.loads(
            cli.lift_execution_record.read_text(encoding="utf-8")
        )
        if (
            lift_record.get("schema_version")
            != "grabber.mtc_lift_execution.v1"
            or not isinstance(lift_record.get("completion"), dict)
        ):
            raise SafetyAbort("升降执行证据格式无效")
        captured = datetime.fromisoformat(lift_record["completed_at_utc"])
        if captured.tzinfo is None:
            raise SafetyAbort("升降执行证据时间缺少时区")
        age_s = (datetime.now(timezone.utc) - captured).total_seconds()
        if age_s < 0.0 or age_s > 900.0:
            raise SafetyAbort(f"升降执行证据不新鲜: age={age_s:.1f}s")
        completion = lift_record["completion"]
        # The column stops where it stops.  2026-08-07: a completed descent
        # reported 251 mm against an expected 250 and this refused the whole
        # place stage over one millimetre, after the pick, the tuck and the
        # descent had all succeeded.  Every other lift comparison in this repo
        # already carries a tolerance (normalize_to_grasp_start's
        # LIFT_TOLERANCE_MM, mobile_body's own 3 mm no-op band); this was the
        # only exact one.
        reached_mm = int(completion.get("target_height_mm", -1))
        if abs(reached_mm - cli.expected_lift_mm) > LIFT_TOLERANCE_MM:
            raise SafetyAbort(
                "升降执行证据高度与空位采集高度不一致: "
                f"实到 {reached_mm} mm, 期望 {cli.expected_lift_mm} mm, "
                f"容差 {LIFT_TOLERANCE_MM} mm"
            )
        held_pose_input = completion.get("held_tcp_base_xyz_rpy_rad")
        held_right_joints = completion.get("right_joints_deg")
        bottle_center_in_tcp_input = completion.get(
            "bottle_center_in_tcp_m"
        )
    elif cli.pick_execution_record:
        pick_record = json.loads(
            cli.pick_execution_record.read_text(encoding="utf-8")
        )
        if (
            pick_record.get("schema_version") != "grabber.mtc_execution.v1"
            or pick_record.get("mode") != "pick"
            or not isinstance(pick_record.get("completion"), dict)
        ):
            raise SafetyAbort("pick 执行证据格式无效")
        captured = datetime.fromisoformat(pick_record["completed_at_utc"])
        if captured.tzinfo is None:
            raise SafetyAbort("pick 执行证据时间缺少时区")
        age_s = (datetime.now(timezone.utc) - captured).total_seconds()
        completion = pick_record["completion"]
        if age_s < 0.0 or age_s > 900.0:
            raise SafetyAbort(f"pick 执行证据不新鲜: age={age_s:.1f}s")
        if int(completion.get("lift_start_mm", -1)) != cli.expected_lift_mm:
            raise SafetyAbort("pick 执行高度与空位采集高度不一致")
        held_pose_input = completion.get("final_tcp_base_xyz_rpy_rad")
        held_right_joints = completion.get("final_right_joints_deg")
        bottle_center_in_tcp_input = completion.get(
            "bottle_center_in_tcp_m"
        )
    if not cli.no_held_object:
        if held_right_joints is None:
            raise SafetyAbort(
                "手工 held TCP 模式必须同时提供 --held-right-joints-deg"
            )
        held_right_joints = np.asarray(held_right_joints, dtype=float)
        if (
            held_right_joints.shape != (7,)
            or not np.all(np.isfinite(held_right_joints))
        ):
            raise SafetyAbort("held 右臂关节必须是七个有限数")
        bottle_center_in_tcp = np.asarray(
            bottle_center_in_tcp_input, dtype=float
        )
        if (
            bottle_center_in_tcp.shape != (3,)
            or not np.all(np.isfinite(bottle_center_in_tcp))
            or float(np.linalg.norm(bottle_center_in_tcp)) > 0.25
        ):
            raise SafetyAbort("缺少抓取时记录的瓶心相对 TCP 偏移")

    lift = LiftSocketAdapter().state()
    if lift.mode != 0 or abs(lift.height_mm - cli.expected_lift_mm) > 5:
        raise SafetyAbort(
            "升降未在静止目标高度: "
            f"actual={lift.height_mm} mm mode={lift.mode}, "
            f"expected={cli.expected_lift_mm} mm"
        )
    demo = RunOrchestrator(_demo_args(cli), load_config(cli.config))
    # Model the shelf at the same resolution the pick does.  DemoParams
    # defaults to 65 mm cells; capture_mtc_direct_pick_scene overrides that to
    # 25 mm and this script never did, so the same shelf was carried at 25 mm
    # for the pick and 65 mm for the place.  A 65 mm cube can present a corner
    # 46 mm proud of the surface it came from, against 18 mm at 25 mm cells,
    # and 2026-08-07 the place kept failing at target_preplace_ik with
    # "eef in collision: head_rgbd_non_target - r_hand" -- the gripper meeting
    # that overhang at a hover point the operator's own hand passes through.
    demo.params = replace(
        demo.params,
        scene_voxel_m=0.025,
        scene_max_voxels=3000,
        held_bottle_height_m=geometry["height_m"],
    )
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    try:
        if cli.camera != "head":
            demo.model_asset_contract = demo._verify_detector_assets()
        demo._start_camera(cli.camera)
        if cli.camera == "head":
            angle = head_lock.read_current_angle_direct()
            if angle is None:
                angle = head_lock.read_current_angle()
            if not head_lock.is_at_reference(angle):
                raise SafetyAbort(
                    f"固定头部不在标定基准角: current={angle}, "
                    f"expected={head_lock.HEAD_REFERENCE}"
                )
            T_base_camera = demo.T_base_head_camera
        else:
            angle = None
            arm_id = (
                "right_arm" if cli.camera == "right_wrist" else "left_arm"
            )
            robot, _view = open_live_arm(
                demo.cfg,
                demo.params,
                profile,
                arm_id,
                take_control=False,
            )
            try:
                if cli.camera == "right_wrist":
                    T_base_camera = (
                        robot.current_flange()
                        @ demo.cfg.calibration.wrist_extrinsic("right")
                    )
                else:
                    T_base_camera = (
                        demo.cfg.calibration.T_base_right_to_base_left
                        @ robot.current_flange()
                        @ demo.cfg.calibration.wrist_extrinsic("left")
                    )
            finally:
                robot.close()
        K, _ = demo.camera.get_camera_intrinsics()
        if K is None:
            raise SafetyAbort(f"{cli.camera} 相机内参不可用")
        depths = demo._collect_fresh_depth_frames(
            demo.params.scene_samples, label="第二层放置场景"
        )
        point_frames = [
            head_scene_points(
                depth,
                K,
                T_base_camera,
                demo.params,
                min_depth_m=demo.params.head_min_depth_m,
                max_depth_m=demo.params.head_max_depth_m,
                bottom_crop=demo.params.scene_image_bottom_crop,
            )
            for depth in depths
        ]
        if cli.no_held_object:
            # Nothing is held and the arm is parked clear of the shelf, so the
            # depth frames are the shelf itself.  Subtracting an arm that is
            # not there would only delete real bottles, and there is no
            # occlusion shadow to reconstruct -- which is the whole point of
            # capturing before the pick rather than during it.
            held_pose = None
            held_tcp = None
        else:
            link7_to_flange, flange_to_tcp = (
                profile.tool_mount_calibration.require_transforms()
            )
            held_pose = np.asarray(held_pose_input, dtype=float)
            if held_pose.shape != (6,) or not np.all(np.isfinite(held_pose)):
                raise SafetyAbort("held TCP 必须是有限的 xyz 米 + rpy 弧度")
            held_tcp = pose_matrix(held_pose)
            link7 = held_tcp @ np.linalg.inv(link7_to_flange @ flange_to_tcp)
            held_center = (
                held_tcp[:3, 3]
                + held_tcp[:3, :3] @ bottle_center_in_tcp
            )
            segment = held_tcp[:3, 3] - link7[:3, 3]
            segment_length_sq = float(segment @ segment)

        def without_held(points):
            relative = points - held_center
            held = (
                np.linalg.norm(relative[:, :2], axis=1) <= 0.05
            ) & (np.abs(relative[:, 2]) <= 0.12)
            along = np.clip(
                ((points - link7[:3, 3]) @ segment) / segment_length_sq,
                0.0,
                1.0,
            )
            nearest_tool = link7[:3, 3] + along[:, None] * segment
            robot_tool = (
                np.linalg.norm(points - nearest_tool, axis=1) <= 0.075
            )
            return points[~(held | robot_tool)]

        if not cli.no_held_object:
            point_frames = [without_held(points) for points in point_frames]
        # Keep the points the voxels came from.  Without them the only record
        # of the scene is a 25 mm grid, and the one question that decides
        # whether docs/rgbd_voxel_inflation.md is an algorithm bug or a camera
        # bug -- does the arm clear the RAW measurements, or only the voxels? --
        # cannot be asked after the fact.  A few hundred kB per capture.
        raw_points_path = cli.output.with_suffix(".raw_points.npy")
        np.save(
            raw_points_path,
            np.vstack(point_frames).astype(np.float32),
        )
        logging.info(
            "原始深度点已保存 %s: %d 点（体素化之前）",
            raw_points_path,
            sum(len(points) for points in point_frames),
        )
        center = (np.asarray(cli.roi_min) + np.asarray(cli.roi_max)) / 2.0
        voxels = union_scene_voxels(
            [
                voxelize_scene_points(points, demo.params, center_base=center)
                for points in point_frames
            ],
            demo.params,
        )
        config = _candidate_config(cli, demo)
        observation = observe_output_table(
            point_frames, config, require_candidates=False
        )
        if cli.operator_confirms_shelf_obstacles_complete:
            shelf_bottom = next(
                box for box in profile.keepout_boxes if box.id == "shelf_bottom"
            )
            roi_min = np.asarray(cli.roi_min, dtype=float)
            roi_max = np.asarray(cli.roi_max, dtype=float)
            if (
                roi_min[0] < shelf_bottom.minimum[0]
                or roi_min[1] < shelf_bottom.minimum[1]
                or roi_max[0] > shelf_bottom.maximum[0]
                or roi_max[1] > shelf_bottom.maximum[1]
            ):
                raise SafetyAbort("遮挡补全 ROI 超出已验证货架支撑面")
            support_x = np.arange(roi_min[0], roi_max[0] + 1e-9, 0.03)
            support_y = np.arange(roi_min[1], roi_max[1] + 1e-9, 0.03)
            xx, yy = np.meshgrid(support_x, support_y)
            verified_support = np.column_stack(
                (
                    xx.ravel(),
                    yy.ravel(),
                    np.full(xx.size, observation.table_height_m),
                )
            )
            scene_voxels = np.asarray(voxels, dtype=float)
            observation = observe_output_table(
                [
                    np.vstack((scene_voxels, verified_support))
                    for points in point_frames
                ],
                config,
                require_candidates=False,
            )
        # Drop the support surface's own cells.  The panel is already a keepout
        # box the place stage relaxes against (allow_final_support_contact names
        # fence_shelf_bottom), but the same panel also arrives as RGB-D voxels
        # under a different id, and nothing relaxes against those.  A 25 mm cell
        # centred on the measured surface stands 12.5 mm proud of it, so a
        # bottle resting where it belongs sinks into the duplicate and the plan
        # is refused: 2026-08-07 measured exactly 12 such cells, all at
        # z = -0.204 and 16 to 42 mm from the bottle axis, inside the target
        # slot the detector had just called empty and well supported.
        #
        # Cut one cell above the fitted plane, which is the tallest a surface
        # cell can be.  Anything genuinely standing there -- a neighbouring
        # bottle is 217 mm, nine cells -- keeps everything above that.
        surface_cut = observation.table_height_m + demo.params.scene_voxel_m
        kept_voxels = [v for v in voxels if float(v[2]) > surface_cut]
        logging.info(
            "支撑面体素扣除 支撑面 %.3f m，切到 %.3f m：删 %d 个，保留 %d 个",
            observation.table_height_m,
            surface_cut,
            len(voxels) - len(kept_voxels),
            len(kept_voxels),
        )
        voxels = kept_voxels
        payload = {
            "schema_version": "grabber.empty_shelf_places.v1",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "frame": "right_controller_base",
            "lift_height_mm": lift.height_mm,
            "head_angle": angle,
            "capture_camera": cli.camera,
            "product_code": geometry["product_code"],
            "held_tcp_base_xyz_rpy_rad": (
                None if held_pose is None else held_pose.tolist()
            ),
            "held_right_joints_deg": (
                None if cli.no_held_object else held_right_joints.tolist()
            ),
            "bottle_center_in_tcp_m": (
                None if cli.no_held_object else bottle_center_in_tcp.tolist()
            ),
            # Which of the two occlusion regimes produced this map.  A map
            # taken empty-handed has no arm shadow in it at all; one taken
            # while holding a bottle has a region the camera could not see and
            # that "empty" cannot be claimed for.
            "occlusion_regime": (
                "arm_clear_of_view" if cli.no_held_object
                else "held_arm_subtracted"
            ),
            "support_source": (
                "verified_shelf_geometry_operator_obstacle_confirmation"
                if cli.operator_confirms_shelf_obstacles_complete
                else "visible_rgbd"
            ),
            "roi_min": cli.roi_min,
            "roi_max": cli.roi_max,
            "observation": asdict(observation),
            "voxel_size_m": demo.params.scene_voxel_m,
            "scene_voxels": voxels,
        }
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    finally:
        demo.close()
    print(f"第二层空位候选已写入 {cli.output}")
    print(json.dumps(payload["observation"]["candidates"], ensure_ascii=False))
    if not observation.candidates:
        raise SafetyAbort(
            "实时点云中没有满足支撑与 "
            f"{cli.clearance_radius_m * 100:.0f} cm 净空的放置区域"
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
