"""Read-only shelf-panel measurement helper (no hardware).

Covers suggest_box() at the pure-function level (all five faces) and
run_shelf_survey()'s orchestration (multi-frame collection -> per-face
fitting -> draft box) with a synthetic constant-depth plane. The
fitting/combination math itself is already covered thoroughly at the
point-cloud level in test_shelf_model.py; this file only needs to prove the
survey wiring is correct, not re-derive that math.
"""

import threading

import numpy as np
import pytest

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.shelf_model import FACE_SPECS, FaceFit
from shelf_dispenser.shelf_survey import run_shelf_survey, suggest_box


def _fit(face, *, plane_m, ranges=None, inliers=100):
    return FaceFit(
        face=face,
        plane_m=plane_m,
        in_plane_ranges=ranges
        or {a: (-0.3, 0.3) for a in (0, 1, 2) if a != FACE_SPECS[face].axis},
        inliers=inliers,
    )


@pytest.mark.parametrize("face", sorted(FACE_SPECS))
def test_suggest_box_tracked_bound_matches_plane(face):
    fit = _fit(face, plane_m=0.5 if FACE_SPECS[face].tracked_bound == "max" else -0.5)
    box = suggest_box(face, fit)
    axis = FACE_SPECS[face].axis
    if FACE_SPECS[face].tracked_bound == "max":
        assert box["max"][axis] == pytest.approx(fit.plane_m)
        assert box["min"][axis] < fit.plane_m
    else:
        assert box["min"][axis] == pytest.approx(fit.plane_m)
        assert box["max"][axis] > fit.plane_m
    assert box["id"] == face


def test_suggest_box_in_plane_axes_use_measured_range_plus_margin():
    fit = _fit("shelf_back", plane_m=0.75, ranges={0: (-0.2, 0.3), 2: (-0.1, 0.4)})
    box = suggest_box("shelf_back", fit, margin_m=0.03)
    assert box["min"][0] == pytest.approx(-0.23)
    assert box["max"][0] == pytest.approx(0.33)
    assert box["min"][2] == pytest.approx(-0.13)
    assert box["max"][2] == pytest.approx(0.43)


def test_suggest_box_rejects_unknown_face():
    fit = _fit("shelf_bottom", plane_m=-0.2)
    with pytest.raises(SafetyAbort, match="未知的货架面"):
        suggest_box("shelf_ceiling_fan", fit)


class _FakeCamera:
    """Serves a fixed number of distinct constant-depth frames + intrinsics."""

    def __init__(self, depth, intrinsics, frame_count=3):
        self._depth = depth
        self._intrinsics = intrinsics
        self._frame_count = frame_count
        self._tick = 0.0
        self._served = 0

    def get_camera_intrinsics(self):
        return self._intrinsics, None

    def get_frame_timestamp(self):
        self._tick += 1.0
        return self._tick

    def get_latest_frames(self):
        self._served += 1
        return None, self._depth


class _FakeCalibration:
    T_base_right_to_camera_head = np.eye(4).tolist()


class _FakeConfig:
    calibration = _FakeCalibration()


def _fake_demo(depth, K, *, frame_samples=3):
    import shelf_dispenser.orchestrator as demo_module

    demo = demo_module.RunOrchestrator.__new__(demo_module.RunOrchestrator)
    demo.params = DemoParams(scene_samples=frame_samples)
    demo.stop_event = threading.Event()
    demo.camera = _FakeCamera(depth, K)
    demo.cfg = _FakeConfig()  # T_base_head_camera reads self.cfg.calibration...
    return demo


def _plane_depth_image(plane_value, *, size=100):
    return np.full((size, size), plane_value, dtype=float)


def _identity_intrinsics(size=100, focal=1000.0):
    center = size / 2.0
    return np.array(
        [[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]]
    )


def test_run_shelf_survey_fits_a_clean_plane_end_to_end():
    # shelf_bottom: free_space_sign=+1, band searched below target[2].
    target = np.array([0.0, 0.0, 1.0])
    plane_value = 0.80  # within [target_z - 0.35, target_z - 0.02]
    demo = _fake_demo(
        _plane_depth_image(plane_value), _identity_intrinsics()
    )
    results = run_shelf_survey(demo, ["shelf_bottom"], target)
    fit = results["shelf_bottom"]["fit"]
    assert fit["plane_m"] == pytest.approx(plane_value + 0.01, abs=1e-6)
    box = results["shelf_bottom"]["suggested_box"]
    assert box["id"] == "shelf_bottom"
    assert box["max"][2] == pytest.approx(plane_value + 0.01, abs=1e-6)


def test_run_shelf_survey_rejects_unknown_face_before_touching_camera():
    demo = _fake_demo(_plane_depth_image(0.8), _identity_intrinsics())
    with pytest.raises(SafetyAbort, match="未知的货架面"):
        run_shelf_survey(demo, ["shelf_bottom", "shelf_ceiling_fan"], [0, 0, 1.0])
    # No frames should have been consumed from the camera.
    assert demo.camera._served == 0


def test_run_shelf_survey_rejects_invalid_target_shape():
    demo = _fake_demo(_plane_depth_image(0.8), _identity_intrinsics())
    with pytest.raises(SafetyAbort, match="目标坐标无效"):
        run_shelf_survey(demo, ["shelf_bottom"], [0.0, 0.0])


def test_run_shelf_survey_uses_the_configured_frame_count():
    target = np.array([0.0, 0.0, 1.0])
    demo = _fake_demo(_plane_depth_image(0.80), _identity_intrinsics(), frame_samples=5)
    run_shelf_survey(demo, ["shelf_bottom"], target)
    assert demo.camera._served == 5
