"""Pure multi-frame output-table fitting and collision-aware place selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .core import SafetyAbort


@dataclass(frozen=True)
class PlaceCandidate:
    xy_base: tuple[float, float]
    table_height_m: float
    score: float
    support_points: int
    nearest_obstacle_m: float


@dataclass(frozen=True)
class OutputTableObservation:
    table_height_m: float
    height_spread_m: float
    frame_inliers: tuple[int, ...]
    candidates: tuple[PlaceCandidate, ...]

    @property
    def best(self) -> PlaceCandidate:
        if not self.candidates:
            raise SafetyAbort("桌面观测没有可用放置候选")
        return self.candidates[0]

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _validate_roi(minimum: Sequence[float], maximum: Sequence[float]):
    lower = np.asarray(minimum, dtype=float)
    upper = np.asarray(maximum, dtype=float)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(upper <= lower)
    ):
        raise SafetyAbort("输出桌面 ROI 必须是有效的三维 min/max")
    return lower, upper


def _fit_one_height(
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    bin_m: float,
    inlier_band_m: float,
    min_inliers: int,
) -> tuple[float, np.ndarray]:
    inside = points[
        np.all(points >= lower, axis=1) & np.all(points <= upper, axis=1)
    ]
    if len(inside) < min_inliers:
        raise SafetyAbort(
            f"输出桌面 ROI 点不足: {len(inside)} < {min_inliers}"
        )
    keys = np.floor(inside[:, 2] / bin_m).astype(np.int64)
    unique, counts = np.unique(keys, return_counts=True)
    mode = unique[int(np.argmax(counts))]
    seed = float(np.median(inside[keys == mode, 2]))
    inliers = inside[np.abs(inside[:, 2] - seed) <= inlier_band_m]
    if len(inliers) < min_inliers:
        raise SafetyAbort(
            f"输出桌面平面内点不足: {len(inliers)} < {min_inliers}"
        )
    return float(np.median(inliers[:, 2])), inliers


def observe_output_table(
    point_frames_base: Sequence[np.ndarray],
    config,
    *,
    require_candidates: bool = True,
) -> OutputTableObservation:
    """Fit the live table in every frame and rank genuinely free patches.

    The ROI is a measured safety envelope, not a fixed place coordinate.  The
    actual table height and XY release point are recomputed from fresh point
    clouds after lift and chassis rotation.
    """
    if not point_frames_base:
        raise SafetyAbort("输出桌面拟合没有点云帧")
    lower, upper = _validate_roi(config.table_roi_min, config.table_roi_max)
    frames: list[np.ndarray] = []
    heights: list[float] = []
    plane_frames: list[np.ndarray] = []
    for index, raw in enumerate(point_frames_base, 1):
        points = np.asarray(raw, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise SafetyAbort(f"第 {index} 帧输出桌面点云不是有限 Nx3")
        height, inliers = _fit_one_height(
            points,
            lower,
            upper,
            bin_m=float(config.table_height_bin_m),
            inlier_band_m=float(config.table_inlier_band_m),
            min_inliers=int(config.table_min_inliers),
        )
        frames.append(points)
        heights.append(height)
        plane_frames.append(inliers)
    spread = float(np.ptp(heights))
    if spread > float(config.table_frame_agreement_m):
        raise SafetyAbort(
            "输出桌面多帧高度不一致: "
            f"spread={spread * 1000:.1f} mm"
        )
    table_height = float(np.median(heights))

    edge = float(config.table_edge_margin_m)
    radius = float(config.place_clearance_radius_m)
    grid = float(config.place_grid_m)
    x_values = np.arange(lower[0] + edge, upper[0] - edge + 1e-9, grid)
    y_values = np.arange(lower[1] + edge, upper[1] - edge + 1e-9, grid)
    if len(x_values) == 0 or len(y_values) == 0:
        raise SafetyAbort("输出桌面 ROI 扣除边缘余量后没有候选区域")
    roi_center = (lower[:2] + upper[:2]) / 2.0
    all_points = np.vstack(frames)
    obstacle_band = all_points[
        (all_points[:, 2] > table_height + float(config.obstacle_min_height_m))
        & (all_points[:, 2] < table_height + float(config.obstacle_max_height_m))
    ]
    candidates: list[PlaceCandidate] = []
    for x in x_values:
        for y in y_values:
            xy = np.asarray([x, y], dtype=float)
            per_frame_support = [
                int(
                    np.count_nonzero(
                        np.linalg.norm(plane[:, :2] - xy, axis=1)
                        <= float(config.table_support_radius_m)
                    )
                )
                for plane in plane_frames
            ]
            support = min(per_frame_support)
            if support < int(config.table_min_patch_points):
                continue
            if len(obstacle_band):
                nearest = float(
                    np.min(np.linalg.norm(obstacle_band[:, :2] - xy, axis=1))
                )
            else:
                nearest = math.inf
            if nearest < radius:
                continue
            centre_cost = float(np.linalg.norm(xy - roi_center))
            clearance_reward = min(nearest, 0.5) if math.isfinite(nearest) else 0.5
            score = centre_cost - 0.25 * clearance_reward
            candidates.append(
                PlaceCandidate(
                    xy_base=(float(x), float(y)),
                    table_height_m=table_height,
                    score=score,
                    support_points=support,
                    nearest_obstacle_m=nearest,
                )
            )
    if require_candidates and not candidates:
        raise SafetyAbort("实时桌面点云中找不到同时满足支撑与净空的放置区域")
    candidates.sort(key=lambda item: (item.score, -item.support_points))
    return OutputTableObservation(
        table_height_m=table_height,
        height_spread_m=spread,
        frame_inliers=tuple(len(item) for item in plane_frames),
        candidates=tuple(candidates[: int(config.max_place_candidates)]),
    )


def placement_still_valid(
    *,
    planned_table_height_m: float,
    planned_xy_base: Sequence[float],
    refreshed: OutputTableObservation,
    height_tolerance_m: float,
    xy_tolerance_m: float,
) -> None:
    """Fail closed unless the *selected* patch survives a fresh observation."""
    planned_xy = np.asarray(planned_xy_base, dtype=float)
    if planned_xy.shape != (2,) or not np.all(np.isfinite(planned_xy)):
        raise SafetyAbort("已规划放置点必须是有限的二维坐标")
    height_error = abs(
        float(planned_table_height_m) - refreshed.table_height_m
    )
    if refreshed.candidates:
        xy_error = min(
            float(
                np.linalg.norm(
                    planned_xy - np.asarray(candidate.xy_base, dtype=float)
                )
            )
            for candidate in refreshed.candidates
        )
    else:
        xy_error = math.inf
    if height_error > height_tolerance_m or xy_error > xy_tolerance_m:
        raise SafetyAbort(
            "规划后刷新桌面发生变化，原选定落点已无法确认: "
            f"height={height_error * 1000:.1f} mm, "
            f"xy={xy_error * 1000:.1f} mm"
        )


def observations_agree(
    planned: OutputTableObservation,
    refreshed: OutputTableObservation,
    *,
    height_tolerance_m: float,
    xy_tolerance_m: float,
) -> None:
    """Backward-compatible comparison using the originally best patch."""
    placement_still_valid(
        planned_table_height_m=planned.table_height_m,
        planned_xy_base=planned.best.xy_base,
        refreshed=refreshed,
        height_tolerance_m=height_tolerance_m,
        xy_tolerance_m=xy_tolerance_m,
    )
