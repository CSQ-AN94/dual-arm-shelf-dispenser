#!/usr/bin/env python3
"""Place an already-held bottle on the side table without moving the chassis.

Commissioning entry.  The body is wherever the operator parked it: this reads
the chassis pose and lift, refuses to command either, and runs only the arm.

The place point is computed, not replayed.  A taught trajectory may be handed
in with ``--demonstration``; it contributes three references to the search --
seed configuration, wrist orientation and which patch to try first -- and a
contact-measured support height the fitted plane is checked against.  It
never contributes a waypoint.  See ``free_table_place.DemonstratedPlace``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import signal
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_pick_place_task import build_parser, validate_args
from shelf_dispenser import console
from shelf_dispenser.core import (
    SafetyAbort,
    interpolate_poses,
    matrix_pose,
    pose_matrix,
)
from shelf_dispenser.free_table_place import (
    build_free_table_place_targets,
    demonstrated_place_from_trajectory,
)
from shelf_dispenser.mobile_body import BodySnapshot, wrap_angle_rad
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.relative_place import product_geometry
from shelf_dispenser.run_manifest import write_run_manifest
from shelf_dispenser.safe_planner import PlanTarget
from utils.config import load_config


# What this gate actually protects: it bounds how long ago anyone last saw
# this bottle enter this grip.  It is not the only thing standing there --
# the resume revalidates the live gripper feedback against the pick's own
# empty-close baseline and the live tuck pose against the profile, every run,
# and those refuse on their own.  The operator may widen or remove it (see
# --pick-record-max-age-s); the default stays conservative so that a later
# session does not inherit an unlimited window it never agreed to.
PICK_RESUME_MAX_AGE_S = 7200.0
# One bottle on an otherwise clear table supports far fewer plane points per
# patch than a shelf row does; the profile default assumes the latter.
SINGLE_BOTTLE_MIN_PATCH_POINTS = 30
NOMINAL_RELEASE_GAP_M = 0.005


def _gripper_released(demo) -> bool:
    """Did the hand actually open, whatever the run then went on to decide?

    An abort *after* release is not an unreleased bottle.  On 2026-08-11 a
    completed pick was recorded as a failure because the banner never
    printed, and two more runs were spent on it; the same trap sits here,
    because every abort path used to hardcode False.  Ask the hand.
    """
    try:
        state = demo.robot.gripper_state()
        return int(state["pos"][0]) > int(demo.params.gripper_open_position) // 2
    except Exception:
        # Never let bookkeeping mask the real failure being reported.
        return False


def _record_age_s(path: Path) -> float:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    completed = datetime.fromisoformat(
        str(record["completed_at_utc"]).replace("Z", "+00:00")
    )
    if completed.tzinfo is None:
        raise SafetyAbort("抓取证据时间缺少时区")
    return (datetime.now(timezone.utc) - completed).total_seconds()


def read_body_only(coordinator):
    """Read pose/lift without requiring or acquiring automatic chassis mode."""
    output = coordinator.chassis._run([coordinator.chassis.pose_query_path], 8.0)
    chassis = coordinator.chassis._parse_pose(
        output, mode="read_only", state="operator_locked"
    )
    if abs(chassis.linear_mps) > 0.005 or abs(chassis.angular_radps) > 0.01:
        raise SafetyAbort("placement-only 检测到底盘未静止")
    return BodySnapshot(chassis=chassis, lift=coordinator.lift.state())


class StationaryBody:
    """Allow the existing place tail to consume a verified no-motion body pose."""

    def __init__(self, coordinator, snapshot):
        self.coordinator = coordinator
        self.snapshot = snapshot

    def position_for_delivery(self, config, *, start=None):
        if start is not self.snapshot:
            raise SafetyAbort("placement-only 必须使用本轮现场快照")
        current = read_body_only(self.coordinator)
        translation = math.hypot(
            current.chassis.x_m - start.chassis.x_m,
            current.chassis.y_m - start.chassis.y_m,
        )
        yaw_error = abs(
            wrap_angle_rad(current.chassis.yaw_rad - start.chassis.yaw_rad)
        )
        lift_error = abs(
            current.lift.height_mm - int(config.target_lift_height_mm)
        )
        if translation > 0.02 or yaw_error > math.radians(1.0):
            raise SafetyAbort(
                "placement-only 启动后底盘发生移动: "
                f"translation={translation * 1000:.1f}mm, "
                f"yaw={math.degrees(yaw_error):.2f}deg"
            )
        if lift_error > int(config.target_lift_tolerance_mm):
            raise SafetyAbort(
                "placement-only 升降不在桌面放置高度: "
                f"actual={current.lift.height_mm}mm, "
                f"target={config.target_lift_height_mm}mm"
            )
        return current

    def close(self):
        # Placement-only must never acquire chassis control, including cleanup.
        pass


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--demonstration",
        type=Path,
        help=(
            "一条示教放置轨迹，只用来给搜索提供种子位形、腕部姿态基准和"
            "接触实测的落瓶高度；不会被回放"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "跑到选出落点为止就停：验证夹持与收拢位、采集桌面、枚举端点并"
            "解 IK，然后打印结果退出。全程不下发任何运动指令"
        ),
    )
    parser.add_argument(
        "--pick-record-max-age-s",
        type=float,
        default=PICK_RESUME_MAX_AGE_S,
        help=(
            "抓取证据的时效上限，秒；给 inf 取消。放宽是操作者的决定，"
            "会原样写进 placement_only_execution.json。"
            "实时夹持反馈与收拢位复核不受影响，每次照常执行"
        ),
    )
    parser.add_argument(
        "--bottle-bottom-below-tcp-m",
        type=float,
        default=None,
        help=(
            "操作者实测的 TCP→瓶底距离，米。抓取点高度是那次抓取的属性、"
            "不是商品的属性，商品表的 height/2 只是假设夹在瓶身中部。"
            "给了它就取代该假设；头部相机看不见瓶底时，下降改走开环，"
            "但每一步仍受单步/累计上限约束，证据文件会写明未经观测"
        ),
    )
    parser.add_argument("--target-product", default="P01")
    parser.add_argument("--shelf-layer-lift-mm", default="647")
    parser.add_argument("--commissioning-speed", default="10")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser/safety_profiles.json"),
    )
    parser.add_argument("--safety-profile", default="shelf_template")
    parser.add_argument(
        "--delivery-safety-profile", default="side_table_template"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_cli().parse_args(argv)
    args = validate_args(
        build_parser().parse_args(
            [
                "--execute",
                "--task-mode",
                "from-held",
                "--dispense",
                "--config",
                cli.config,
                "--safety-config",
                cli.safety_config,
                "--safety-profile",
                cli.safety_profile,
                "--delivery-safety-profile",
                cli.delivery_safety_profile,
                "--target-product",
                cli.target_product,
                "--shelf-layer-lift-mm",
                str(cli.shelf_layer_lift_mm),
                "--mtc-pick-execution-record",
                str(cli.pick_record),
                "--commissioning-speed",
                str(cli.commissioning_speed),
                "--output-dir",
                str(cli.output_dir),
            ]
        )
    )
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[])
    logging.getLogger("bottle_demo").setLevel(logging.DEBUG)
    demo = RunOrchestrator(args, load_config(args.config))
    write_run_manifest(
        demo.run_dir,
        args=demo.args,
        config=demo.cfg,
        project_root=demo.project_root,
        params=demo.params,
    )
    demo.timeline = console.install(latest_log=cli.output_dir / "latest.log")
    (demo.run_dir / "placement_only_execution.json").write_text(
        json.dumps(
            {
                "chassis_motion": "forbidden",
                "automatic_body_return": False,
                "pick_record": str(cli.pick_record),
                "demonstration": (
                    str(cli.demonstration) if cli.demonstration else None
                ),
                "demonstration_role": (
                    "search prior only: IK seed, wrist orientation reference, "
                    "patch ordering, and a contact-measured support height "
                    "the fitted plane is checked against; never replayed"
                ),
                "release_policy": (
                    "only after table observation, refreshed scene, wrist "
                    "landing clearance, and bottle-bottom visual servo"
                ),
                "pick_resume_age_limit_s": float(cli.pick_record_max_age_s),
                "pick_resume_age_limit_default_s": PICK_RESUME_MAX_AGE_S,
                "pick_resume_age_limit_widened_by": (
                    None
                    if float(cli.pick_record_max_age_s)
                    == PICK_RESUME_MAX_AGE_S
                    else "operator, this run only, on the command line"
                ),
                "single_bottle_min_patch_points": SINGLE_BOTTLE_MIN_PATCH_POINTS,
                "pick_resume_age_policy": (
                    "extended only around live gripper+tuck revalidation after "
                    "commissioning delay; scene_max_age_s restored immediately"
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def request_stop(*_):
        demo.stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    result = {"status": "fault", "bottle_released": False}
    try:
        demo.initialize()
        demo._preflight()
        demo._validate_side_table_profile_pair()
        coordinator = demo._ensure_mobile_body()
        # Read pose/lift only.  coordinator.preflight() initializes the chassis
        # session and sends zero twists, which placement-only deliberately avoids.
        snapshot = read_body_only(coordinator)
        config = demo._side_table_config()
        demo._resolved_side_table_config = replace(
            config,
            table_min_patch_points=SINGLE_BOTTLE_MIN_PATCH_POINTS,
        )
        config = demo._side_table_config()
        if abs(snapshot.lift.height_mm - int(config.target_lift_height_mm)) > int(
            config.target_lift_tolerance_mm
        ):
            raise SafetyAbort(
                "请先把升降停在桌面放置高度: "
                f"actual={snapshot.lift.height_mm}mm, "
                f"target={config.target_lift_height_mm}mm"
            )
        if cli.bottle_bottom_below_tcp_m is not None:
            offset = float(cli.bottle_bottom_below_tcp_m)
            geometry = product_geometry(cli.target_product)
            if not 0.03 <= offset <= float(geometry["height_m"]):
                raise SafetyAbort(
                    "TCP→瓶底距离必须在 30mm 到整瓶高之间: "
                    f"{offset * 1000:.0f}mm, 瓶高 {geometry['height_m'] * 1000:.0f}mm"
                )
            demo._operator_bottle_bottom_below_tcp_m = offset
            demo.stage(
                "操作者给定抓取高度",
                f"TCP→瓶底 {offset * 1000:.0f} mm，取代商品表的 "
                f"{geometry['height_m'] * 500:.0f} mm 中部假设",
            )
        demo.shelf_ready_body_snapshot = snapshot
        demo._delivery_head_extrinsic = demo.T_base_head_camera.copy()
        demo.mobile_body = StationaryBody(coordinator, snapshot)
        demo.stage(
            "placement-only 底盘锁定",
            "使用人工摆放后的实时静止 pose；禁止自动升降、旋转和返程",
        )
        resume_max_age_s = float(cli.pick_record_max_age_s)
        if resume_max_age_s <= 0.0:
            raise SafetyAbort("--pick-record-max-age-s 必须为正")
        if resume_max_age_s != PICK_RESUME_MAX_AGE_S:
            logging.getLogger("bottle_demo").warning(
                "抓取证据时效上限由操作者设为 %.0fs（默认 %.0fs）："
                "实时夹持反馈与收拢位复核仍然逐次执行",
                resume_max_age_s,
                PICK_RESUME_MAX_AGE_S,
            )
        if cli.dry_run:
            # The freshness gate protects a *motion*: it bounds how long ago
            # anyone last saw the bottle enter this grip.  A dry run commands
            # nothing, and it is the cheapest way to find out whether the
            # geometry solves before paying for a fresh pick.  The live
            # gripper and tuck checks inside the resume still run and still
            # decide; only the clock is relaxed, and only here.
            resume_max_age_s = max(
                resume_max_age_s, _record_age_s(cli.pick_record) + 60.0
            )
        normal_params = demo.params
        try:
            demo.params = replace(
                demo.params, scene_max_age_s=resume_max_age_s
            )
            lifted = demo._resume_mtc_pick_for_delivery()
        finally:
            demo.params = normal_params
        pick_record = json.loads(cli.pick_record.read_text(encoding="utf-8"))
        completion = pick_record["completion"]
        tcp_from_bottle = np.eye(4)
        tcp_from_bottle[:3, :3] = pose_matrix(
            completion["final_tcp_base_xyz_rpy_rad"]
        )[:3, :3].T
        tcp_from_bottle[:3, 3] = np.asarray(
            completion["bottle_center_in_tcp_m"], dtype=float
        )
        geometry = product_geometry(cli.target_product)

        demonstration = None
        if cli.demonstration is not None:
            demonstration = demonstrated_place_from_trajectory(
                json.loads(cli.demonstration.read_text(encoding="utf-8")),
                tcp_from_joints=demo.robot.tcp_from_joints,
                tcp_from_bottle=tcp_from_bottle,
                bottle_height_m=float(geometry["height_m"]),
            )
            # Rank live empty patches by distance from the taught place point
            # rather than from the centre of the authored ROI, so the patches
            # that survive max_place_candidates are the reachable ones.
            demo._preferred_place_xy = np.asarray(
                demonstration.bottle_center[:2], dtype=float
            )
            # And offer the taught point itself as a candidate.  Its support
            # was measured by setting a bottle on it, which the head camera
            # cannot match and, at this range, often cannot even see.  Live
            # clearance is still checked against it like any other patch.
            demo._vouched_place_candidates = (
                (
                    tuple(map(float, demonstration.bottle_center[:2])),
                    float(demonstration.support_z),
                ),
            )
            demo.stage(
                "示教放置先验",
                "示教终点落瓶 "
                f"xy=({demonstration.bottle_center[0]:.3f}, "
                f"{demonstration.bottle_center[1]:.3f})、"
                f"支撑高度 {demonstration.support_z:.4f} m；"
                "用作候选排序中心、IK 种子和腕部姿态基准，不回放任何路点",
            )

        def plan_free_table_place(observation):
            current_tcp = demo.robot.current_tcp()
            current_joints = demo.robot.joints_deg()
            ik_failures = 0

            def solve_free_ik(flange, seed_joints_deg):
                nonlocal ik_failures
                try:
                    return demo.robot.solve_flange_ik_candidates(
                        flange,
                        demo.params,
                        seed_joints_deg=list(map(float, seed_joints_deg)),
                    )
                except SafetyAbort as exc:
                    ik_failures += 1
                    if ik_failures <= 12:
                        print(
                            f"FREE_IK_REJECT flange={matrix_pose(flange)} "
                            f"reason={exc}",
                            flush=True,
                        )
                    raise

            free_targets = build_free_table_place_targets(
                candidates=observation.candidates,
                current_tcp=current_tcp,
                current_joints_deg=current_joints,
                tcp_from_bottle=tcp_from_bottle,
                flange_from_tcp=demo.T_flange_tcp,
                bottle_height_m=float(geometry["height_m"]),
                bottle_radius_m=float(geometry["radius_m"]),
                release_gap_m=NOMINAL_RELEASE_GAP_M,
                preplace_clearance_m=float(config.preplace_clearance_m),
                table_xy_min=config.table_roi_min[:2],
                table_xy_max=config.table_roi_max[:2],
                solve_flange_ik_candidates=solve_free_ik,
                demonstration=demonstration,
            )
            targets = []
            final_by_label = {}
            for index, target in enumerate(free_targets):
                demo.safety.assert_tcp_point(
                    target.place_tcp[:3, 3], label=target.label + " 放置点"
                )
                demo.safety.assert_tcp_point(
                    target.preplace_tcp[:3, 3], label=target.label + " 预放置点"
                )
                final_by_label[target.label] = target.place_tcp
                targets.append(
                    PlanTarget(
                        label=target.label,
                        flange=target.flange,
                        goal_joints=target.goal_joints,
                        score=float(index),
                        goal_constraint="joints",
                    )
                )

            print(
                f"FREE_PLACE empty_patches={len(observation.candidates)} "
                f"reachable_targets={len(targets)}",
                flush=True,
            )

            def validate_lowering(target, trajectory):
                endpoint = trajectory.get("points_deg", [])[-1]
                start_tcp = demo.robot.tcp_from_joints(endpoint)
                standoff_tcp = final_by_label[target.label].copy()
                standoff_tcp[2, 3] += float(
                    demo.params.place_servo_standoff_m
                )
                poses = interpolate_poses(
                    matrix_pose(start_tcp),
                    matrix_pose(standoff_tcp),
                    demo.params.segment_m,
                )
                joints = demo.robot.plan_ik(
                    poses,
                    demo.params,
                    seed_joints_deg=endpoint,
                )
                demo._validate_local_joint_path(
                    name="free_output_lowering_precheck",
                    joints=joints,
                    target_base=None,
                    start_joints_deg=endpoint,
                    scene_points_override=demo._output_contact_scene_voxels(
                        observation
                    ),
                )

            verified = demo._verified_plan_targets(
                "moveit_free_output_table",
                targets,
                continuation_validator=validate_lowering,
            )
            return (
                verified.trajectory,
                final_by_label[verified.target.label],
            )

        demo._plan_to_output_table = plan_free_table_place
        if cli.dry_run:
            # Everything the run uses to *decide* where to place, and nothing
            # it uses to move: gripper and tuck revalidation above, one table
            # capture here, then the endpoint enumeration and the controller's
            # own IK.  No trajectory is planned and none is sent.
            observation = demo._capture_output_table_scene()
            targets = build_free_table_place_targets(
                candidates=observation.candidates,
                current_tcp=demo.robot.current_tcp(),
                current_joints_deg=demo.robot.joints_deg(),
                tcp_from_bottle=tcp_from_bottle,
                flange_from_tcp=demo.T_flange_tcp,
                bottle_height_m=float(geometry["height_m"]),
                bottle_radius_m=float(geometry["radius_m"]),
                release_gap_m=NOMINAL_RELEASE_GAP_M,
                preplace_clearance_m=float(config.preplace_clearance_m),
                table_xy_min=config.table_roi_min[:2],
                table_xy_max=config.table_roi_max[:2],
                solve_flange_ik_candidates=lambda flange, seed: (
                    demo.robot.solve_flange_ik_candidates(
                        flange,
                        demo.params,
                        seed_joints_deg=list(map(float, seed)),
                    )
                ),
                demonstration=demonstration,
            )
            result = {
                "status": "dry_run",
                "bottle_released": False,
                "motion_commanded": False,
                "table_height_m": float(observation.table_height_m),
                "table_height_spread_m": float(observation.height_spread_m),
                "empty_patches": len(observation.candidates),
                "reachable_targets": len(targets),
                "selected": [
                    {
                        "label": target.label,
                        "bottle_xy": list(map(float, target.candidate.xy_base)),
                        "place_tcp_xyz": target.place_tcp[:3, 3].tolist(),
                        "goal_joints_deg": list(target.goal_joints),
                    }
                    for target in targets[:5]
                ],
                "demonstrated_support_z_m": (
                    demonstration.support_z if demonstration else None
                ),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return 0
        demo._dispense_to_side_table(lifted, start=snapshot)
        result = {"status": "success", "bottle_released": True}
        return 0
    except SafetyAbort as exc:
        result = {
            "status": "safe_abort",
            "bottle_released": _gripper_released(demo),
            "error": str(exc),
        }
        logging.getLogger("bottle_demo").error("安全中止: %s", exc)
        return 2
    except Exception as exc:
        result = {
            "status": "fault",
            "bottle_released": _gripper_released(demo),
            "error": f"{type(exc).__name__}: {exc}",
        }
        logging.getLogger("bottle_demo").exception("placement-only 未处理异常")
        return 1
    finally:
        (demo.run_dir / "placement_only_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        demo.close()
        if demo.timeline is not None:
            summary = demo.timeline.render()
            if summary:
                print(summary)


if __name__ == "__main__":
    raise SystemExit(main())
