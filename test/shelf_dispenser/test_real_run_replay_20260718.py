"""Regression replays distilled from the 2026-07-18 real-robot failures.

These are not generic mocked-success tests.  The box, intrinsics and measured
surface depth below come from ``outputs/shelf_dispenser/20260718_203207``.  The
saved wrist image contains a bottle cut off by the bottom image border; using
that view-dependent surface depth as the object centre displaced the grasp
target 47.3 mm from the fixed-head observation.
"""

from types import SimpleNamespace
import threading

import numpy as np
import pytest

from shelf_dispenser.core import (
    BottleDetectionLost,
    CameraFrameUnavailable,
    DemoParams,
    Detection,
    Localization,
    SafetyAbort,
)
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.lift_evidence import LiftEvidenceKind, LiftVisualEvidence


def test_bottom_clipped_wrist_box_keeps_locked_target_depth(monkeypatch):
    """A partial bottle view may refine image-plane position, not overwrite
    the locked object's depth with the visible near surface.
    """
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.params = DemoParams(samples=3)
    demo.stop_event = threading.Event()
    demo.camera_name = "right_wrist"
    demo.state = SimpleNamespace(update=lambda **_: None)
    demo._save_localization = lambda *_: None
    demo.stage = lambda *_: None

    intrinsics = np.array(
        [[602.6, 0.0, 327.4], [0.0, 602.3, 248.8], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    detection = Detection((344, 165, 467, 479), 0.837, "bottle")
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 0.50, dtype=np.float32)
    # Reproduce the stable 0.315 m near-surface patch from the saved run.
    depth[250:440, 380:430] = 0.315

    class Camera:
        timestamp = 0.0

        def get_camera_intrinsics(self):
            return intrinsics, None

        def get_frame_timestamp(self):
            self.timestamp += 1.0
            return self.timestamp

        def get_latest_frames(self):
            return color, depth

    class Detector:
        @staticmethod
        def detect(_color, predicate=None, target_classes=None):
            if predicate is None or predicate(detection):
                return detection
            return None

    demo.camera = Camera()
    demo.wrist_detector = Detector()
    demo.detector = None
    monkeypatch.setattr("shelf_dispenser.orchestrator.time.sleep", lambda _: None)

    # Keep the recorded 0.36 m locked depth, but reproduce the harder camera
    # boundary the task must support: the physical grasp-height projection is
    # 60 px below the image while the recorded upper bottle box still touches
    # the bottom edge.  This drives the full localization path, not only the
    # association helper.
    locked_v = 540.0
    locked = np.array(
        [
            0.0,
            (locked_v - intrinsics[1, 2]) * 0.36 / intrinsics[1, 1],
            0.36,
        ]
    )
    result = demo.localize(
        "预抓取复检",
        lambda: np.eye(4),
        demo.params,
        depth_prior_base=locked,
    )

    assert result.depth_m == 0.36
    assert result.point_camera[2] == 0.36
    assert result.pixel[1] == locked_v


def test_independent_head_measurement_never_synthesizes_prior_depth(monkeypatch):
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.params = DemoParams(samples=3)
    demo.stop_event = threading.Event()
    demo.camera_name = "head"
    demo.state = SimpleNamespace(update=lambda **_: None)
    demo._save_localization = lambda *_: None
    demo.stage = lambda *_: None
    intrinsics = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    detection = Detection((280, 100, 360, 400), 0.9, "bottle")

    class Camera:
        @staticmethod
        def get_camera_intrinsics():
            return intrinsics, None

        @staticmethod
        def get_frame_timestamp():
            return 1.0

        @staticmethod
        def get_latest_frames():
            return (
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.full((480, 640), np.nan, dtype=float),
            )

    class Detector:
        @staticmethod
        def detect(_color, predicate=None, target_classes=None):
            return detection if predicate is None or predicate(detection) else None

    demo.camera = Camera()
    demo.detector = Detector()
    demo.wrist_detector = None
    clock = iter((0.0, 1.0, 11.0))
    monkeypatch.setattr("shelf_dispenser.orchestrator.time.time", lambda: next(clock))

    with pytest.raises(SafetyAbort, match="检测/深度稳定帧不足: 0/3"):
        demo.localize(
            "独立头部三维确认",
            lambda: np.eye(4),
            demo.params,
            depth_prior_base=np.array([0.0, 0.0, 0.5]),
            allow_depth_prior_fallback=False,
        )


def test_post_lift_recorded_wrist_outage_never_uses_head_fallback():
    """The old 2026-07-21 fixed-head success path is now fail-closed."""
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.params = DemoParams()
    demo.stage = lambda *_: None
    demo._post_lift_visual_evidence = lambda _target: LiftVisualEvidence(
        LiftEvidenceKind.CAMERA_UNAVAILABLE,
        "recorded wrist outage",
        0,
        0,
    )
    demo._measure_target_from_head_3d = lambda *_args: pytest.fail(
        "fixed-head airborne-bottle gate must stay deleted"
    )
    locked = Localization(
        point_camera=[0.0, 0.0, 0.5],
        point_base=[0.0, 0.0, 0.5],
        pixel=[320.0, 240.0],
        depth_m=0.5,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[280, 100, 360, 400],
        confidence=0.9,
        frame_count=7,
    )

    with pytest.raises(CameraFrameUnavailable, match="不能当作遮挡"):
        demo._confirm_lifted_target(
            locked,
            prelift_tcp=np.eye(4),
            postlift_tcp=np.eye(4),
        )
