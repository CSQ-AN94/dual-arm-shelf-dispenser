"""On-site, read-only shelf-panel measurement (no arm motion).

Companion to `scripts/measure_shelf_geometry.py`: reuses the same head
RGB-D + calibration pipeline `_build_head_scene` already relies on
(`RunOrchestrator._collect_fresh_depth_frames`, `T_base_head_camera`,
`scene.head_scene_points`) and the fitting algorithm in `shelf_model.py`,
but as a one-time manual survey instead of a per-run runtime adaptation.

This module never writes `safety_profiles.json`. It only produces a
human-reviewable draft — measuring, adapting a live electronic fence, and
marking a profile `verified_for_execution` are three separate steps on
purpose (see `shelf_dispenser/SAFETY_PROFILES.md`), and a script should not
collapse "I measured something" into "this is now safe to run".
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np

from .core import SafetyAbort
from .scene import head_scene_points
from .shelf_model import FACE_SPECS, FaceFit, combine_shelf_fits, fit_shelf_face


def suggest_box(
    face: str,
    fit: FaceFit,
    *,
    far_offset_m: float = 0.5,
    margin_m: float = 0.03,
) -> dict:
    """A draft keepout box from a single face measurement.

    Unlike `shelf_model.adapt_profile_to_shelf` (which only ever adjusts an
    *existing* configured box), this synthesizes a whole box from scratch —
    there is nothing to adapt yet on a first survey. The tracked axis's far
    bound is an arbitrary generous offset, not a measurement; the operator
    is expected to review and correct every field before it goes anywhere
    near `safety_profiles.json`.
    """
    if face not in FACE_SPECS:
        raise SafetyAbort(f"未知的货架面标识: {face}")
    spec = FACE_SPECS[face]
    axis = spec.axis
    minimum = [0.0, 0.0, 0.0]
    maximum = [0.0, 0.0, 0.0]
    if spec.tracked_bound == "max":
        maximum[axis] = fit.plane_m
        minimum[axis] = fit.plane_m - far_offset_m
    else:
        minimum[axis] = fit.plane_m
        maximum[axis] = fit.plane_m + far_offset_m
    for in_axis, (lo, hi) in fit.in_plane_ranges.items():
        minimum[in_axis] = lo - margin_m
        maximum[in_axis] = hi + margin_m
    return {
        "id": face,
        "min": [round(float(v), 4) for v in minimum],
        "max": [round(float(v), 4) for v in maximum],
    }


def run_shelf_survey(
    demo,
    faces: Sequence[str],
    target_base: Sequence[float],
    *,
    frame_samples: int | None = None,
    min_gap_m: float | None = None,
    max_gap_m: float | None = None,
) -> dict[str, dict]:
    """Measure each requested shelf face near `target_base`.

    `demo` must already have its head camera streaming (see
    `scripts/measure_shelf_geometry.py`'s `_start_head_camera_only`, which
    starts the camera without ever creating a RobotSession — this survey has
    no business touching the arm or teleop). Reuses
    `RunOrchestrator._collect_fresh_depth_frames` for the same multi-frame
    consensus discipline every other adaptive-fence measurement in this
    project uses; a single noisy frame is not enough to draft safety
    geometry from.
    """
    unknown = [face for face in faces if face not in FACE_SPECS]
    if unknown:
        raise SafetyAbort(f"未知的货架面标识: {unknown}")
    K, _ = demo.camera.get_camera_intrinsics()
    if K is None:
        raise SafetyAbort("头部相机内参不可用")
    target = np.asarray(target_base, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise SafetyAbort("现场测量的目标坐标无效，必须是 3 个有限数")
    samples = frame_samples or demo.params.scene_samples
    depth_frames = demo._collect_fresh_depth_frames(
        samples, label="货架现场测量"
    )
    results: dict[str, dict] = {}
    for face in faces:
        per_frame_fits = [
            fit_shelf_face(
                head_scene_points(
                    depth,
                    K,
                    demo.T_base_head_camera,
                    demo.params,
                    min_depth_m=demo.params.head_min_depth_m,
                    max_depth_m=demo.params.head_max_depth_m,
                    bottom_crop=demo.params.scene_image_bottom_crop,
                ),
                target,
                face,
                demo.params,
                min_gap_m=min_gap_m,
                max_gap_m=max_gap_m,
            )
            for depth in depth_frames
        ]
        combined = combine_shelf_fits(per_frame_fits, demo.params)
        results[face] = {
            "fit": asdict(combined),
            "suggested_box": suggest_box(face, combined),
        }
    return results
