"""Per-run table-surface fitting so the electronic fence tracks the real table.

The MoveIt layer already sees the actual table every run through the dynamic
RGB-D voxels. The independent fence layer, however, uses hand-measured static
boxes: whenever the chassis parks at a different distance or the table height
changes, the static `table_top` keepout either blocks free space above a
table that is lower/farther than configured, or under-protects one that is
higher/closer. This module measures the real surface from the same head
depth frame and adapts the fence within a bounded envelope; anything outside
the envelope is a hard abort, never a silent guess.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import DemoParams, SafetyAbort
from .safety import FenceBox, SafetyProfile

TABLE_KEEPOUT_ID = "table_top"

_BIN_M = 0.01
_PLANE_THICKNESS_M = 0.015


@dataclass(frozen=True)
class TableFit:
    """Robustly measured table surface in the right-arm base frame."""

    height_m: float
    top_m: float
    x_range: tuple[float, float]
    y_front: float
    inliers: int


def fit_table_top(
    points: np.ndarray,
    target_base: np.ndarray,
    params: DemoParams,
) -> TableFit | None:
    """Find the dominant horizontal surface just below the bottle target.

    The search band [min_below, max_below] under the target keeps the fit
    anchored to the surface the bottle stands on: the floor and lower shelf
    boards sit far below the band, and points at the very target height
    (the bottle body) sit above it. Returns None when there is no plane with
    enough support — the caller decides whether that is fatal.
    """
    points = np.asarray(points, dtype=float)
    target = np.asarray(target_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or target.shape != (3,):
        raise SafetyAbort("桌面拟合的输入点云或目标坐标形状无效")
    z = points[:, 2]
    horizontal = np.linalg.norm(points[:, :2] - target[:2], axis=1)
    band = (
        (z <= target[2] - params.table_fit_min_below_m)
        & (z >= target[2] - params.table_fit_max_below_m)
        & (horizontal <= params.table_fit_horizontal_radius_m)
    )
    candidates = points[band]
    if len(candidates) < params.table_fit_min_inliers:
        return None
    bins = np.round(candidates[:, 2] / _BIN_M).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    mode_z = float(values[np.argmax(counts)]) * _BIN_M
    selected = np.abs(candidates[:, 2] - mode_z) <= _PLANE_THICKNESS_M
    inliers = candidates[selected]
    if len(inliers) < params.table_fit_min_inliers:
        return None
    height = float(np.median(inliers[:, 2]))
    # 顶面取 90 分位再抬 1cm：既盖住表面起伏，又不被个别飞点抬高。
    top = float(np.percentile(inliers[:, 2], 90)) + 0.01
    x_lo, x_hi = np.percentile(inliers[:, 0], [5, 95])
    y_front = float(np.percentile(inliers[:, 1], 5))
    return TableFit(
        height_m=height,
        top_m=top,
        x_range=(float(x_lo), float(x_hi)),
        y_front=y_front,
        inliers=int(len(inliers)),
    )


def combine_table_fits(
    fits: Sequence[TableFit | None],
    params: DemoParams,
) -> TableFit:
    """Require the sampled frames to agree, then combine conservatively.

    A table surface does not move between consecutive frames, so a spread
    larger than ``table_fit_agreement_m`` is evidence that something else
    changed during capture — a hand crossing the view, an exposure switch,
    a burst of IR interference.  That is exactly the case where trusting a
    single frame would silently shift the fence, so it aborts instead.

    Height (which gates the ±tolerance check against the configured
    profile) uses the median: robust to one outlier frame.  Every field
    that sizes the *keepout box* takes the most protective value seen, so
    multi-frame sampling can only grow the protected volume.
    """
    if not fits:
        raise SafetyAbort("桌面拟合收到空的帧列表")
    if any(fit is None for fit in fits):
        missing = sum(1 for fit in fits if fit is None)
        raise SafetyAbort(
            f"{len(fits)} 帧中有 {missing} 帧找不到桌面平面，"
            "采集期间视野不稳定——检查是否有人/物体经过头部相机视野后重跑"
        )
    measured = [fit for fit in fits if fit is not None]
    heights = np.asarray([fit.height_m for fit in measured], dtype=float)
    spread = float(np.max(heights) - np.min(heights))
    if spread > params.table_fit_agreement_m:
        raise SafetyAbort(
            f"{len(measured)} 帧实测桌面高度不一致，跨度 {spread * 1000:.1f} mm "
            f"超过上限 {params.table_fit_agreement_m * 1000:.0f} mm："
            "采集期间场景在动，拒绝用不稳定的测量调整电子围栏"
        )
    return TableFit(
        height_m=float(np.median(heights)),
        # Highest surface, furthest-forward edge and widest extent seen:
        # the keepout box only ever grows across frames.
        top_m=max(fit.top_m for fit in measured),
        x_range=(
            min(fit.x_range[0] for fit in measured),
            max(fit.x_range[1] for fit in measured),
        ),
        y_front=min(fit.y_front for fit in measured),
        # Report the weakest supporting evidence, not the most flattering.
        inliers=min(fit.inliers for fit in measured),
    )


def adapt_profile_to_table(
    profile: SafetyProfile,
    fit: TableFit | None,
    params: DemoParams,
) -> SafetyProfile:
    """Return a profile whose table keepout and table-attached zone floors
    follow the measured surface.

    Adaptation rules (all bounded, all fail-closed):
    - the measured height must be within table_fit_height_tolerance_m of the
      configured keepout top, otherwise the physical setup changed too much
      and the run is refused;
    - the keepout top follows the measurement in both directions (higher →
      more protective; lower → stops blocking the free space just above the
      real surface);
    - horizontally the box only ever grows (union with the measured extent):
      the camera cannot see the true front edge below the image crop, so
      shrinking based on "no points seen" would be un-measured trust;
    - allowed zones whose floor was authored just above the configured table
      top keep their authored clearance relative to the *measured* top.
    """
    table = next(
        (box for box in profile.keepout_boxes if box.id == TABLE_KEEPOUT_ID),
        None,
    )
    if table is None:
        return profile
    if fit is None:
        raise SafetyAbort(
            "头部点云中找不到有足够支撑的桌面平面，无法核对 table_top 围栏"
            "与真实桌子是否一致——检查深度画面/目标定位，或按 runbook 走"
            "静态测量流程"
        )
    old_top = table.maximum[2]
    height_error = abs(fit.height_m - old_top)
    if height_error > params.table_fit_height_tolerance_m:
        raise SafetyAbort(
            f"实测桌面高度 z={fit.height_m:.3f} 与围栏配置顶面 z={old_top:.3f} "
            f"相差 {height_error * 100:.1f}cm，超出自适应容差 "
            f"{params.table_fit_height_tolerance_m * 100:.0f}cm——桌子或底盘"
            "位置变化过大，请按 runbook 重新测量并更新 safety_profiles.json"
        )
    margin = params.table_fit_edge_margin_m
    new_table = FenceBox(
        id=table.id,
        minimum=(
            min(table.minimum[0], fit.x_range[0] - margin),
            min(table.minimum[1], fit.y_front - margin),
            table.minimum[2],
        ),
        maximum=(
            max(table.maximum[0], fit.x_range[1] + margin),
            table.maximum[1],
            fit.top_m,
        ),
    )
    zones = []
    for zone in profile.allowed_tcp_zones:
        floor = zone.minimum[2]
        attached = (
            old_top <= floor <= old_top + params.table_fit_zone_attach_band_m
        )
        if not attached:
            zones.append(zone)
            continue
        new_floor = fit.top_m + (floor - old_top)
        if new_floor >= zone.maximum[2] - 0.05:
            raise SafetyAbort(
                f"允许区 {zone.id} 底面随桌面自适应后过高，区域退化"
            )
        if not profile.tcp_workspace.contains(
            (zone.minimum[0], zone.minimum[1], new_floor), 0.0
        ):
            raise SafetyAbort(
                f"允许区 {zone.id} 底面随桌面自适应后越出总工作空间"
            )
        zones.append(
            FenceBox(
                id=zone.id,
                minimum=(zone.minimum[0], zone.minimum[1], new_floor),
                maximum=zone.maximum,
            )
        )
    keepouts = tuple(
        new_table if box.id == TABLE_KEEPOUT_ID else box
        for box in profile.keepout_boxes
    )
    return dataclasses.replace(
        profile,
        keepout_boxes=keepouts,
        allowed_tcp_zones=tuple(zones),
    )
