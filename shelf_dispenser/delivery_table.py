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
    # Bounding box of the points actually fitted as tabletop.  The ROI is a
    # search envelope an operator authored; this is where the table was seen.
    surface_xy_min: tuple[float, float] = (0.0, 0.0)
    surface_xy_max: tuple[float, float] = (0.0, 0.0)

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


def _plane_z(plane: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Height of the fitted surface at one or many xy points."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    return xy @ plane[:2] + plane[2]


def _fit_one_plane(
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    bin_m: float,
    inlier_band_m: float,
    min_inliers: int,
    min_extent_m: float,
    refinements: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the tabletop as a tilted plane and return ``(coefficients, inliers)``.

    A constant height cannot represent this surface.  On 2026-08-13 the live
    tabletop read 22 mm across the visible area -- rising about 11 mm per
    200 mm in y, falling about 10 mm per 650 mm in x -- against an inlier band
    of 12 mm, so a single-z fit could hold the near strip or the far strip but
    never both, and the strip it dropped was silently reported as "no table
    here".  A plane fitted to the same points predicted the taught, contact
    measured support height to 2.5 mm where the constant fit was 22 mm out.

    Seeding still runs highest bin first, because a broad commissioning ROI
    usually contains more floor than table.  That alone is not enough: a dense
    enough sweep lifts small *high* objects past the point count too, and the
    arm carrying the bottle is the highest thing in this ROI.  A tabletop is
    also wide, so the accepted surface must span ``min_extent_m`` in both axes
    -- which an arm does not.
    """
    inside = points[
        np.all(points >= lower, axis=1) & np.all(points <= upper, axis=1)
    ]
    if len(inside) < min_inliers:
        raise SafetyAbort(
            f"输出桌面 ROI 点不足: {len(inside)} < {min_inliers}"
        )
    keys = np.floor(inside[:, 2] / bin_m).astype(np.int64)
    unique, counts = np.unique(keys, return_counts=True)
    best_extent = 0.0
    for key in unique[np.argsort(unique)[::-1]]:
        seed = float(np.median(inside[keys == key, 2]))
        inliers = inside[np.abs(inside[:, 2] - seed) <= inlier_band_m]
        plane = np.array([0.0, 0.0, seed], dtype=float)
        for _ in range(refinements):
            if len(inliers) < min_inliers:
                break
            design = np.column_stack(
                (inliers[:, 0], inliers[:, 1], np.ones(len(inliers)))
            )
            plane = np.linalg.lstsq(design, inliers[:, 2], rcond=None)[0]
            residual = inside[:, 2] - _plane_z(plane, inside[:, :2])
            inliers = inside[np.abs(residual) <= inlier_band_m]
        if len(inliers) < min_inliers:
            continue
        extent = np.ptp(inliers[:, :2], axis=0)
        best_extent = max(best_extent, float(np.min(extent)))
        if float(np.min(extent)) < min_extent_m:
            continue
        return plane, inliers
    raise SafetyAbort(
        "输出桌面平面拟合失败: "
        f"max_bin={int(np.max(counts))}, min_inliers={min_inliers}, "
        f"best_extent={best_extent * 1000:.0f}mm < {min_extent_m * 1000:.0f}mm"
    )


def observe_output_table(
    point_frames_base: Sequence[np.ndarray],
    config,
    *,
    require_candidates: bool = True,
    preferred_xy: Sequence[float] | None = None,
    vouched_candidates: Sequence[tuple[Sequence[float], float]] | None = None,
    # Smallest span, in both axes, that an accepted surface must cover.  Not
    # a measured constant: it separates a tabletop from the arm in front of
    # it, and both are far from 250 mm.  The side table is about 680x400.
    min_surface_extent_m: float = 0.25,
) -> OutputTableObservation:
    """Fit the live table in every frame and rank genuinely free patches.

    The ROI is a measured safety envelope, not a fixed place coordinate.  The
    actual table height and XY release point are recomputed from fresh point
    clouds after lift and chassis rotation.

    ``preferred_xy`` is where the caller would rather put the bottle -- in
    practice a point some human has already set one down on.  Without it the
    ranking falls back to the centre of the ROI, which is an authored safety
    envelope and says nothing about what the arm can reach: on 2026-08-13
    that ranking, followed by the ``max_place_candidates`` truncation below,
    handed the planner twenty endpoints that were every one of them past full
    extension while reachable patches sat unreported further down the list.
    """
    if not point_frames_base:
        raise SafetyAbort("输出桌面拟合没有点云帧")
    lower, upper = _validate_roi(config.table_roi_min, config.table_roi_max)
    frames: list[np.ndarray] = []
    planes: list[np.ndarray] = []
    plane_frames: list[np.ndarray] = []
    for index, raw in enumerate(point_frames_base, 1):
        points = np.asarray(raw, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise SafetyAbort(f"第 {index} 帧输出桌面点云不是有限 Nx3")
        plane, inliers = _fit_one_plane(
            points,
            lower,
            upper,
            bin_m=float(config.table_height_bin_m),
            inlier_band_m=float(config.table_inlier_band_m),
            min_inliers=int(config.table_min_inliers),
            min_extent_m=float(min_surface_extent_m),
        )
        frames.append(points)
        planes.append(plane)
        plane_frames.append(inliers)
    # Frames must agree where the bottle is actually going, not at some
    # arbitrary origin: on a tilted surface those are different numbers.
    reference_xy = (
        (lower[:2] + upper[:2]) / 2.0
        if preferred_xy is None
        else np.asarray(preferred_xy, dtype=float)
    )
    if reference_xy.shape != (2,) or not np.all(np.isfinite(reference_xy)):
        raise SafetyAbort("首选放置点必须是有限的二维坐标")
    heights = [
        float(_plane_z(plane, reference_xy)[0]) for plane in planes
    ]
    spread = float(np.ptp(heights))
    if spread > float(config.table_frame_agreement_m):
        raise SafetyAbort(
            "输出桌面多帧高度不一致: "
            f"spread={spread * 1000:.1f} mm"
        )
    plane = np.median(np.vstack(planes), axis=0)
    # One scalar for the callers that still need a single number -- the
    # dynamic fence lid and the refresh comparison.  It is the surface *at
    # the reference point*, not a property of the table.
    table_height = float(_plane_z(plane, reference_xy)[0])

    edge = float(config.table_edge_margin_m)
    radius = float(config.place_clearance_radius_m)
    grid = float(config.place_grid_m)
    x_values = np.arange(lower[0] + edge, upper[0] - edge + 1e-9, grid)
    y_values = np.arange(lower[1] + edge, upper[1] - edge + 1e-9, grid)
    if len(x_values) == 0 or len(y_values) == 0:
        raise SafetyAbort("输出桌面 ROI 扣除边缘余量后没有候选区域")
    all_points = np.vstack(frames)
    # Height above the *surface* underneath each point.  Measured against one
    # scalar, a 22 mm tilt reads as 22 mm of phantom obstacle at one end of
    # the table and hides real ones at the other.
    above_surface = all_points[:, 2] - _plane_z(plane, all_points[:, :2])
    obstacle_band = all_points[
        (above_surface > float(config.obstacle_min_height_m))
        & (above_surface < float(config.obstacle_max_height_m))
    ]

    def clearance(xy: np.ndarray) -> float:
        if not len(obstacle_band):
            return math.inf
        return float(np.min(np.linalg.norm(obstacle_band[:, :2] - xy, axis=1)))

    def scored(xy: np.ndarray, height: float, support: int, nearest: float):
        reward = min(nearest, 0.5) if math.isfinite(nearest) else 0.5
        return PlaceCandidate(
            xy_base=(float(xy[0]), float(xy[1])),
            table_height_m=float(height),
            score=float(np.linalg.norm(xy - reference_xy)) - 0.25 * reward,
            support_points=int(support),
            nearest_obstacle_m=nearest,
        )

    candidates: list[PlaceCandidate] = []
    for x in x_values:
        for y in y_values:
            xy = np.asarray([x, y], dtype=float)
            support = min(
                int(
                    np.count_nonzero(
                        np.linalg.norm(inliers[:, :2] - xy, axis=1)
                        <= float(config.table_support_radius_m)
                    )
                )
                for inliers in plane_frames
            )
            if support < int(config.table_min_patch_points):
                continue
            nearest = clearance(xy)
            if nearest < radius:
                continue
            # Each patch gets the surface height under *itself*.
            candidates.append(
                scored(xy, _plane_z(plane, xy)[0], support, nearest)
            )

    # Patches whose support is vouched for by something other than this
    # camera -- in practice a taught point where a bottle was physically set
    # down.  Contact beats a depth image, and the camera cannot always see
    # the near table at all.  The clearance check is still live and still
    # binding: the demonstration proves there was a surface, never that the
    # surface is empty right now.
    for extra_xy, extra_height in vouched_candidates or ():
        xy = np.asarray(extra_xy, dtype=float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise SafetyAbort("外部担保的放置点必须是有限的二维坐标")
        if not math.isfinite(float(extra_height)):
            raise SafetyAbort("外部担保的放置点高度必须有限")
        if np.any(xy < lower[:2] + edge) or np.any(xy > upper[:2] - edge):
            raise SafetyAbort(
                f"外部担保的放置点 {xy.tolist()} 超出 ROI 扣除边距后的范围"
            )
        nearest = clearance(xy)
        if nearest < radius:
            continue
        candidates.append(scored(xy, float(extra_height), 0, nearest))

    if require_candidates and not candidates:
        raise SafetyAbort("实时桌面点云中找不到同时满足支撑与净空的放置区域")
    candidates.sort(key=lambda item: (item.score, -item.support_points))
    # Where the tabletop was actually seen, across every frame.  Callers build
    # the tabletop collision box from this rather than from the ROI: the ROI
    # is an authored search envelope reaching back to the robot, and on
    # 2026-08-13 a box built from it was 1.00x0.70 m with its near edge at
    # y=0.30 for a table measured at 0.70x0.32 starting near y=0.55 -- a
    # quarter of a metre of tabletop that is not there, standing exactly where
    # the arm has to pass.
    surface = np.vstack([item[:, :2] for item in plane_frames])
    return OutputTableObservation(
        table_height_m=table_height,
        height_spread_m=spread,
        frame_inliers=tuple(len(item) for item in plane_frames),
        candidates=tuple(candidates[: int(config.max_place_candidates)]),
        surface_xy_min=tuple(map(float, np.min(surface, axis=0))),
        surface_xy_max=tuple(map(float, np.max(surface, axis=0))),
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
