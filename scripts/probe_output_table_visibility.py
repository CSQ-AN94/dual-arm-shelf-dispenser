#!/usr/bin/env python3
"""Read-only: where does the head camera actually see the output table?

Placement kept selecting patches at y=0.76..0.84 while the only endpoint ever
proven reachable was y=0.60.  Two candidate explanations, both cheap to test
from one capture:

  1. ``scene_image_bottom_crop`` is a *row limit*, not a crop amount, and the
     405 in DemoParams was tuned for the shelf.  Rows below it are discarded,
     which at the head's 24.9 deg downward optical axis lands on the tabletop
     around y=0.55.
  2. ``_fit_one_height`` now seeds from the *highest* occupied bin, so on a
     table whose far end reads higher than its near end the +/-12 mm inlier
     band keeps the far strip and drops the near strip.

This entry never connects an arm, never moves the lift and never touches the
chassis.  It opens the head camera, takes the same frames the delivery capture
would take, and prints where the table-height points are.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser import head_lock
from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.delivery_table import observe_output_table
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.safety import load_safety_profile
from shelf_dispenser.scene import head_scene_points
from run_pick_place_task import build_parser as build_task_parser
from utils.config import load_config


def _row_profile(points: np.ndarray, table_z: float, band_m: float) -> list[dict]:
    """Point counts per 5 cm of y, restricted to a slab about ``table_z``."""
    near_table = points[np.abs(points[:, 2] - table_z) <= band_m]
    if not len(near_table):
        return []
    edges = np.arange(0.30, 1.05, 0.05)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = near_table[
            (near_table[:, 1] >= low) & (near_table[:, 1] < high)
        ]
        if not len(selected):
            continue
        rows.append(
            {
                "y_range": [round(float(low), 3), round(float(high), 3)],
                "points": int(len(selected)),
                "z_median": round(float(np.median(selected[:, 2])), 4),
                "x_range": [
                    round(float(np.min(selected[:, 0])), 3),
                    round(float(np.max(selected[:, 0])), 3),
                ],
            }
        )
    return rows


def _xy_map(points: np.ndarray, table_z: float, band_m: float) -> list[str]:
    """ASCII occupancy of the table slab: rows are y, columns are x."""
    slab = points[np.abs(points[:, 2] - table_z) <= band_m]
    x_edges = np.arange(-0.50, 0.51, 0.05)
    y_edges = np.arange(0.40, 0.96, 0.05)
    header = "  y\\x  " + "".join(f"{value:+.2f} "[:6] for value in x_edges[:-1])
    rows = [header]
    for y_low, y_high in zip(y_edges[:-1], y_edges[1:]):
        band = slab[(slab[:, 1] >= y_low) & (slab[:, 1] < y_high)]
        cells = []
        for x_low, x_high in zip(x_edges[:-1], x_edges[1:]):
            count = int(
                np.count_nonzero((band[:, 0] >= x_low) & (band[:, 0] < x_high))
            )
            cells.append("     ." if count == 0 else f"{count:>6d}")
        rows.append(f"{y_low:+.2f} " + "".join(cells))
    return rows


def _xy_height_map(points: np.ndarray, table_z: float, band_m: float) -> list[str]:
    """Median height of the table slab per cell, in millimetres about ``table_z``.

    Counts say where the surface is; this says what shape it is.  A tilt reads
    as a monotone gradient and is a plane; anything else is not, and a plane
    fit would be the wrong model however well it converges.
    """
    slab = points[np.abs(points[:, 2] - table_z) <= band_m]
    x_edges = np.arange(-0.50, 0.51, 0.05)
    y_edges = np.arange(0.40, 0.96, 0.05)
    rows = ["  y\\x  " + "".join(f"{value:+.2f} "[:6] for value in x_edges[:-1])]
    for y_low, y_high in zip(y_edges[:-1], y_edges[1:]):
        band = slab[(slab[:, 1] >= y_low) & (slab[:, 1] < y_high)]
        cells = []
        for x_low, x_high in zip(x_edges[:-1], x_edges[1:]):
            cell = band[(band[:, 0] >= x_low) & (band[:, 0] < x_high)]
            if len(cell) < 4:
                cells.append("     .")
            else:
                offset_mm = (float(np.median(cell[:, 2])) - table_z) * 1000.0
                cells.append(f"{offset_mm:>+6.0f}")
        rows.append(f"{y_low:+.2f} " + "".join(cells))
    return rows


def _z_histogram(points: np.ndarray, bin_m: float = 0.01, top: int = 8) -> list[dict]:
    """Where the horizontal surfaces are, whether or not the fit succeeded.

    The fit refusing to seed tells you it found no bin with enough support; it
    does not tell you what *is* there.  This does, so a failed fit still
    produces a measurement instead of only an error string.
    """
    if not len(points):
        return []
    keys = np.floor(points[:, 2] / bin_m).astype(np.int64)
    unique, counts = np.unique(keys, return_counts=True)
    order = np.argsort(counts)[::-1][:top]
    bands = []
    for index in sorted(order, key=lambda i: -counts[i]):
        selected = points[keys == unique[index]]
        bands.append(
            {
                "z_low": round(float(unique[index] * bin_m), 3),
                "points": int(counts[index]),
                "y_range": [
                    round(float(np.min(selected[:, 1])), 3),
                    round(float(np.max(selected[:, 1])), 3),
                ],
                "x_range": [
                    round(float(np.min(selected[:, 0])), 3),
                    round(float(np.max(selected[:, 0])), 3),
                ],
                "rows_by_y": _row_profile(
                    points, float(np.median(selected[:, 2])), 0.015
                ),
            }
        )
    return bands


def _observe(frames, config, label: str, preferred_xy=None) -> dict:
    try:
        observation = observe_output_table(
            frames,
            config,
            require_candidates=False,
            preferred_xy=preferred_xy,
        )
    except SafetyAbort as exc:
        return {"label": label, "error": str(exc)}
    candidates = observation.candidates
    return {
        "label": label,
        "table_height_m": round(observation.table_height_m, 4),
        "height_spread_m": round(observation.height_spread_m, 4),
        "frame_inliers": list(observation.frame_inliers),
        "candidate_count": len(candidates),
        "candidate_y_range": (
            [
                round(min(item.xy_base[1] for item in candidates), 3),
                round(max(item.xy_base[1] for item in candidates), 3),
            ]
            if candidates
            else None
        ),
        "candidates": [
            {
                "xy": [round(item.xy_base[0], 3), round(item.xy_base[1], 3)],
                "support": item.support_points,
                "nearest_obstacle_m": round(item.nearest_obstacle_m, 3),
            }
            for item in candidates[:40]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument(
        "--delivery-safety-profile", default="side_table_template"
    )
    parser.add_argument("--port", type=int, default=8881)
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs/output_table_probe")
    )
    parser.add_argument(
        "--inlier-band-m",
        type=float,
        default=0.030,
        help="额外再用一个更宽的平面内点带复跑一次拟合",
    )
    parser.add_argument(
        "--min-patch-points",
        type=int,
        default=None,
        help=(
            "覆盖 table_min_patch_points；placement-only 入口用 30，"
            "profile 的 60 是按货架行来的"
        ),
    )
    parser.add_argument(
        "--preferred-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="候选排序中心，通常是示教终点的落瓶 xy；不给则用 ROI 中心",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="深度图采样步长；桌面拟合的门槛是点数，默认 6 会把它饿死",
    )
    parser.add_argument(
        "--table-z",
        type=float,
        default=-0.310,
        help="画桌面占据图用的参考桌高（2026-08-13 探针实测的最密 z 带）",
    )
    cli = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    Path(cli.output_dir).mkdir(parents=True, exist_ok=True)
    demo_args = build_task_parser().parse_args(
        [
            "--config",
            cli.config,
            "--safety-config",
            cli.safety_config,
            "--safety-profile",
            "shelf_template",
            "--port",
            str(cli.port),
            "--output-dir",
            cli.output_dir,
        ]
    )
    # Same camera-only construction as capture_mtc_direct_pick_scene: no arm
    # session, no gripper command, no body motion.
    demo_args.plan_only = False
    demo = RunOrchestrator(demo_args, load_config(cli.config))
    delivery = load_safety_profile(
        cli.safety_config,
        cli.delivery_safety_profile,
        require_verified=False,
    )
    config = delivery.side_table_delivery
    if cli.min_patch_points is not None:
        config = replace(config, table_min_patch_points=int(cli.min_patch_points))

    report: dict = {"probe": "output_table_visibility"}
    try:
        demo.initialize()
        current_head = head_lock.read_current_angle()
        if current_head is None:
            current_head = head_lock.read_current_angle_direct()
        if not head_lock.is_at_reference(current_head):
            raise SafetyAbort(
                "固定头部不在已标定基准角，只读探针拒绝自动转动: "
                f"current={current_head}, expected={head_lock.HEAD_REFERENCE}"
            )
        report["head_angle"] = current_head
        if demo.camera_name != "head":
            demo._start_camera("head")
        K, _ = demo.camera.get_camera_intrinsics()
        if K is None:
            raise SafetyAbort("头部相机内参不可用")
        depth_frames = demo._collect_fresh_depth_frames(
            demo.params.scene_samples, label="输出桌面可见性探针"
        )
        report["depth_shape"] = list(np.asarray(depth_frames[0]).shape)
        report["scene_image_bottom_crop"] = int(
            demo.params.scene_image_bottom_crop
        )
        report["table_roi"] = {
            "min": list(config.table_roi_min),
            "max": list(config.table_roi_max),
        }

        windows = {
            "cropped_405": int(demo.params.scene_image_bottom_crop),
            "full_frame": int(np.asarray(depth_frames[0]).shape[0]),
        }
        report["windows"] = {}
        for name, bottom_crop in windows.items():
            frames = [
                np.asarray(
                    head_scene_points(
                        depth,
                        K,
                        demo.T_base_head_camera,
                        demo.params,
                        min_depth_m=demo.params.head_min_depth_m,
                        max_depth_m=demo.params.head_max_depth_m,
                        bottom_crop=bottom_crop,
                        stride=cli.stride,
                    ),
                    dtype=float,
                )
                for depth in depth_frames
            ]
            stacked = np.vstack(frames)
            roi = stacked[
                np.all(stacked >= np.asarray(config.table_roi_min), axis=1)
                & np.all(stacked <= np.asarray(config.table_roi_max), axis=1)
            ]
            entry = {
                "bottom_crop_rows": bottom_crop,
                "points_total": int(len(stacked)),
                "points_in_roi": int(len(roi)),
            }
            fitted = _observe(
                frames, config, "profile_band", preferred_xy=cli.preferred_xy
            )
            entry["fit_profile_band"] = fitted
            # Same frames, wider inlier band: separates "the near strip is not
            # in the cloud" from "the near strip is in the cloud but reads a
            # couple of centimetres lower than the far strip".
            entry["fit_wide_band"] = _observe(
                frames,
                replace(config, table_inlier_band_m=float(cli.inlier_band_m)),
                f"band_{cli.inlier_band_m:.3f}m",
            )
            # Unconditional: a failed fit must still say where the surfaces
            # are, otherwise the probe reports only the same error the
            # production run already reported.
            entry["z_bands_in_roi"] = _z_histogram(roi)
            # Occupancy of the table slab on a coarse xy grid.  A table that
            # ends at y=0.70 and a table whose near half sits in the held
            # bottle's shadow produce the same "no candidates below 0.70";
            # only the x distribution of the surviving near points separates
            # them, because the arm shadows one x band and not the others.
            entry["table_slab_xy_map"] = _xy_map(roi, float(cli.table_z), 0.020)
            entry["table_slab_height_map_mm"] = _xy_height_map(
                roi, float(cli.table_z), 0.030
            )
            reference_z = fitted.get("table_height_m")
            if reference_z is not None:
                entry["rows_about_fitted_plane"] = _row_profile(
                    stacked, float(reference_z), float(cli.inlier_band_m)
                )
            report["windows"][name] = entry
    finally:
        demo.close()

    output = Path(cli.output_dir) / "output_table_visibility.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"\n已写入 {output}；本次未连接机械臂、未移动升降或底盘。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
