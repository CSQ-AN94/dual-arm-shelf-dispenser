"""Per-run shelf-panel fitting so the electronic fence tracks the real shelf.

Generalizes ``table_model.py``'s "single horizontal plane below the target"
fit to any of the five shelf-compartment faces (bottom/top/back/left/right
panel). Each face is a plane normal to one axis, searched on the keepout
side of the target and adapted with the same fail-closed, only-grow-in-plane
rules that ``table_model.py`` already validated on real hardware for the
``table_top`` case.

``table_model.py`` itself is intentionally left untouched: ``table_demo``
keeps running on its existing, real-machine-verified code path. This module
is additive and only activates for profiles whose keepout boxes use one of
the ids in ``FACE_SPECS``.

Unlike the table (whose height was independently measured on real hardware
before this module existed), no shelf face has ever been measured on real
hardware. The default tolerances below are provisional engineering
judgement, not empirical fits — they must be revisited once the first
on-site shelf measurement exists (see ``scripts/measure_shelf_geometry.py``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import DemoParams, SafetyAbort
from .safety import FenceBox, SafetyProfile

_BIN_M = 0.01
_PLANE_THICKNESS_M = 0.015

# The head camera looks roughly along +y in the right-arm base frame. Whatever
# face is being fitted, the near (smaller-y) edge is the only side the camera
# can reliably see; the far edge is routinely cropped or occluded. So — same
# reasoning as table_model.fit_table_top's y_front handling — this axis's
# near bound is the only one ever adapted; its far bound stays exactly as
# configured.
_DEPTH_AXIS = 1


@dataclass(frozen=True)
class FaceSpec:
    """How to search for and adapt one named shelf panel.

    ``axis``: the coordinate axis (0=x, 1=y, 2=z) normal to this panel.
    ``free_space_sign``: +1 if the reachable/free volume lies on the
    increasing side of ``axis`` relative to the panel (e.g. the space above a
    shelf bottom, or to the right of a left panel), else -1.
    ``min_gap_m``/``max_gap_m``: per-face override of
    ``DemoParams.shelf_fit_min_gap_m``/``shelf_fit_max_gap_m``. ``None``
    means "use the shared default". A product standing in its slot is much
    taller than it is wide/deep, so it confounds the search band far more
    for the *top* face (the object's own cap sits well within the default
    near-gap) than for bottom/back/left/right (the object barely extends
    past its own footprint in those directions). 2026-07-20 real-hardware
    finding: with the shared default (0.02m), ``shelf_top`` locked onto the
    bottle's own cap instead of the panel above it.
    """

    axis: int
    free_space_sign: float
    min_gap_m: float | None = None
    max_gap_m: float | None = None

    @property
    def tracked_bound(self) -> str:
        """Which keepout-box bound on this axis faces the free-space side."""
        return "max" if self.free_space_sign > 0 else "min"


# A shelf compartment has (at most) these five rigid faces. This is a fixed,
# small registry, not a generic n-face system — a real shelf bin only ever
# has these surfaces, so there is nothing to gain from a more open design.
FACE_SPECS: dict[str, FaceSpec] = {
    "shelf_bottom": FaceSpec(axis=2, free_space_sign=1.0),
    "shelf_top": FaceSpec(axis=2, free_space_sign=-1.0, min_gap_m=0.20),
    "shelf_back": FaceSpec(axis=1, free_space_sign=-1.0),
    "shelf_left_panel": FaceSpec(axis=0, free_space_sign=1.0),
    "shelf_right_panel": FaceSpec(axis=0, free_space_sign=-1.0),
}


@dataclass(frozen=True)
class FaceFit:
    """Robustly measured shelf panel plane in the right-arm base frame."""

    face: str
    plane_m: float
    in_plane_ranges: dict[int, tuple[float, float]]
    inliers: int


def fit_shelf_face(
    points: np.ndarray,
    target_base: np.ndarray,
    face: str,
    params: DemoParams,
    *,
    min_gap_m: float | None = None,
    max_gap_m: float | None = None,
) -> FaceFit | None:
    """Find the dominant plane for one shelf panel, searched near the target.

    Mirrors ``table_model.fit_table_top``'s band+mode+percentile algorithm;
    the search band and in-plane extent axes come from ``FACE_SPECS`` instead
    of being hardcoded to "horizontal plane below the target". Returns
    ``None`` when there is no plane with enough support — the caller decides
    whether that is fatal.

    Gap precedence (narrowest scope wins): explicit ``min_gap_m``/
    ``max_gap_m`` arguments > ``FACE_SPECS[face]``'s per-face override >
    ``params.shelf_fit_min_gap_m``/``max_gap_m`` shared default.
    """
    if face not in FACE_SPECS:
        raise SafetyAbort(f"未知的货架面标识: {face}")
    spec = FACE_SPECS[face]
    effective_min_gap = (
        min_gap_m
        if min_gap_m is not None
        else (
            spec.min_gap_m
            if spec.min_gap_m is not None
            else params.shelf_fit_min_gap_m
        )
    )
    effective_max_gap = (
        max_gap_m
        if max_gap_m is not None
        else (
            spec.max_gap_m
            if spec.max_gap_m is not None
            else params.shelf_fit_max_gap_m
        )
    )
    points = np.asarray(points, dtype=float)
    target = np.asarray(target_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or target.shape != (3,):
        raise SafetyAbort("货架面拟合的输入点云或目标坐标形状无效")
    axis = spec.axis
    in_plane_axes = [a for a in (0, 1, 2) if a != axis]
    coord = points[:, axis]
    # The search band sits on the keepout side of the target, i.e. the
    # opposite side from free_space_sign.
    if spec.free_space_sign > 0:
        band = (
            (coord <= target[axis] - effective_min_gap)
            & (coord >= target[axis] - effective_max_gap)
        )
    else:
        band = (
            (coord >= target[axis] + effective_min_gap)
            & (coord <= target[axis] + effective_max_gap)
        )
    lateral = np.linalg.norm(
        points[:, in_plane_axes] - target[in_plane_axes], axis=1
    )
    band &= lateral <= params.shelf_fit_lateral_radius_m
    candidates = points[band]
    if len(candidates) < params.shelf_fit_min_inliers:
        return None
    bins = np.round(candidates[:, axis] / _BIN_M).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    mode_coord = float(values[np.argmax(counts)]) * _BIN_M
    selected = np.abs(candidates[:, axis] - mode_coord) <= _PLANE_THICKNESS_M
    inliers = candidates[selected]
    if len(inliers) < params.shelf_fit_min_inliers:
        return None
    if spec.free_space_sign > 0:
        plane_m = (
            float(np.percentile(inliers[:, axis], 90))
            + params.shelf_fit_conservative_margin_m
        )
    else:
        plane_m = (
            float(np.percentile(inliers[:, axis], 10))
            - params.shelf_fit_conservative_margin_m
        )
    in_plane_ranges: dict[int, tuple[float, float]] = {}
    for in_axis in in_plane_axes:
        lo, hi = np.percentile(inliers[:, in_axis], [5, 95])
        in_plane_ranges[in_axis] = (float(lo), float(hi))
    return FaceFit(
        face=face,
        plane_m=plane_m,
        in_plane_ranges=in_plane_ranges,
        inliers=int(len(inliers)),
    )


def combine_shelf_fits(
    fits: Sequence[FaceFit | None],
    params: DemoParams,
) -> FaceFit:
    """Require the sampled frames to agree on one face, then combine
    conservatively — same three rules as ``table_model.combine_table_fits``:
    median for the tracked bound, only-grows for in-plane extents, hard abort
    on missing planes or cross-frame disagreement.
    """
    if not fits:
        raise SafetyAbort("货架面拟合收到空的帧列表")
    if any(fit is None for fit in fits):
        missing = sum(1 for fit in fits if fit is None)
        raise SafetyAbort(
            f"{len(fits)} 帧中有 {missing} 帧找不到货架面平面，"
            "采集期间视野不稳定——检查是否有人/物体经过头部相机视野后重跑"
        )
    measured = [fit for fit in fits if fit is not None]
    faces = {fit.face for fit in measured}
    if len(faces) != 1:
        raise SafetyAbort("combine_shelf_fits 收到了不同货架面的拟合结果混在一起")
    face = next(iter(faces))
    planes = np.asarray([fit.plane_m for fit in measured], dtype=float)
    spread = float(np.max(planes) - np.min(planes))
    if spread > params.table_fit_agreement_m:
        raise SafetyAbort(
            f"{len(measured)} 帧实测 {face} 面位置不一致，跨度 "
            f"{spread * 1000:.1f}mm 超过上限 "
            f"{params.table_fit_agreement_m * 1000:.0f}mm："
            "采集期间场景在动，拒绝用不稳定的测量调整电子围栏"
        )
    combined_ranges: dict[int, tuple[float, float]] = {}
    for axis in measured[0].in_plane_ranges:
        combined_ranges[axis] = (
            min(fit.in_plane_ranges[axis][0] for fit in measured),
            max(fit.in_plane_ranges[axis][1] for fit in measured),
        )
    return FaceFit(
        face=face,
        plane_m=float(np.median(planes)),
        in_plane_ranges=combined_ranges,
        inliers=min(fit.inliers for fit in measured),
    )


def adapt_profile_to_shelf(
    profile: SafetyProfile,
    fits_by_face: dict[str, FaceFit | None],
    params: DemoParams,
) -> SafetyProfile:
    """Return a profile whose shelf-panel keepout boxes follow this run's
    measured panels.

    Every keepout box whose id is a recognized shelf face (``FACE_SPECS``) is
    updated from ``fits_by_face[box.id]``; a missing measurement for a face
    the profile actually has is fatal, mirroring
    ``adapt_profile_to_table``'s "expects a table but found none" rule.
    Boxes with unrecognized ids (including ``table_top``) pass through
    untouched.
    """
    recognized = [box for box in profile.keepout_boxes if box.id in FACE_SPECS]
    if not recognized:
        return profile
    updated_by_id: dict[str, FenceBox] = {}
    for box in recognized:
        spec = FACE_SPECS[box.id]
        fit = fits_by_face.get(box.id)
        if fit is None:
            raise SafetyAbort(
                f"头部点云中找不到有足够支撑的 {box.id} 平面，无法核对该"
                "货架面围栏与真实货架是否一致——检查深度画面/目标定位，或"
                "按现场测量流程重新测量"
            )
        axis = spec.axis
        old_bound = (
            box.maximum[axis] if spec.tracked_bound == "max" else box.minimum[axis]
        )
        bound_error = abs(fit.plane_m - old_bound)
        if bound_error > params.shelf_fit_bound_tolerance_m:
            raise SafetyAbort(
                f"实测 {box.id} 面位置 {fit.plane_m:.3f} 与围栏配置 "
                f"{old_bound:.3f} 相差 {bound_error * 100:.1f}cm，超出自适应"
                f"容差 {params.shelf_fit_bound_tolerance_m * 100:.0f}cm——货架"
                "或机械臂位置变化过大，请重新测量并更新 safety_profiles.json"
            )
        new_minimum = list(box.minimum)
        new_maximum = list(box.maximum)
        if spec.tracked_bound == "max":
            new_maximum[axis] = fit.plane_m
        else:
            new_minimum[axis] = fit.plane_m
        margin = params.shelf_fit_edge_margin_m
        for in_axis, (lo, hi) in fit.in_plane_ranges.items():
            if in_axis == _DEPTH_AXIS:
                new_minimum[in_axis] = min(new_minimum[in_axis], lo - margin)
            else:
                new_minimum[in_axis] = min(new_minimum[in_axis], lo - margin)
                new_maximum[in_axis] = max(new_maximum[in_axis], hi + margin)
        updated_by_id[box.id] = FenceBox(
            id=box.id,
            minimum=tuple(new_minimum),
            maximum=tuple(new_maximum),
        )
    keepouts = tuple(
        updated_by_id.get(box.id, box) for box in profile.keepout_boxes
    )
    return dataclasses.replace(profile, keepout_boxes=keepouts)
