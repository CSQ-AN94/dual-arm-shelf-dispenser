"""Regression tests for transient YOLO loss during the pregrasp transit.

The real 2026-07-15 logs localised the bottle 7/7 frames, validated the point
cloud corridor, executed the first movel, then aborted before segment two only
because the translated wrist view no longer passed the close-bottle shape
filter.  The locked target and geometric safety gates remain valid; RGB-D
stream loss must still abort.
"""

from types import SimpleNamespace
import time

import numpy as np
import pytest

from shelf_dispenser.core import (
    BottleDetectionLost,
    CameraFrameUnavailable,
    Detection,
    DemoParams,
    Localization,
    SafetyAbort,
)
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.target_guard import GuardResult


def _localization() -> Localization:
    return Localization(
        point_camera=[0.0, 0.0, 0.3],
        point_base=[0.02, 0.59, -0.135],
        pixel=[304.0, 180.0],
        depth_m=0.3,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[268, 52, 340, 294],
        confidence=0.12,
        frame_count=7,
    )


def _demo_with_two_segment_path():
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.params = DemoParams()
    poses = [
        [0.22, 0.59, -0.135, 0.0, 0.0, 0.0],
        [0.105, 0.59, -0.135, 0.0, 0.0, 0.0],
    ]
    demo.candidate_path = lambda target: (poses[-1], poses[-1], poses)
    demo.safety = SimpleNamespace(assert_tcp_point=lambda *args, **kwargs: None)
    demo.collision_gate = lambda *args, **kwargs: None
    demo.stage = lambda *args, **kwargs: None

    class FakeRobot:
        def __init__(self):
            self.moves = []

        def plan_ik(self, path, params, *, allow_first_jump=False):
            assert not allow_first_jump
            return [[0.0] * 7 for _ in path]

        def move_linear(self, pose, speed):
            self.moves.append(pose)

        @staticmethod
        def current_tcp():
            tcp = np.eye(4)
            tcp[:3, 3] = [0.32, 0.59, -0.135]
            return tcp

    demo.robot = FakeRobot()
    return demo, poses


def test_transient_detection_loss_after_first_segment_does_not_abort_locked_target():
    demo, poses = _demo_with_two_segment_path()
    head_checks = []
    outcomes = iter(
        [None, BottleDetectionLost("移动过程中符合形状的 bottle 检测丢失")]
    )

    def visibility_check(target_base=None):
        outcome = next(outcomes)
        if outcome:
            raise outcome

    demo.ensure_bottle_visible = visibility_check
    demo._confirm_locked_target_from_head = lambda target: head_checks.append(
        list(target)
    )

    demo._approach_pregrasp(_localization())

    assert demo.robot.moves == poses
    assert head_checks == [[0.02, 0.59, -0.135]]


def test_rgbd_stream_loss_still_aborts_before_next_segment():
    demo, poses = _demo_with_two_segment_path()
    outcomes = iter([None, CameraFrameUnavailable("RGB-D 画面中断")])

    def visibility_check(target_base=None):
        outcome = next(outcomes)
        if outcome:
            raise outcome

    demo.ensure_bottle_visible = visibility_check

    try:
        demo._approach_pregrasp(_localization())
    except SafetyAbort as exc:
        assert "画面中断" in str(exc)
    else:
        raise AssertionError("RGB-D stream loss must remain fatal")

    assert demo.robot.moves == poses[:1]


def test_visible_raw_bottle_is_associated_to_locked_projection_without_shape_gate():
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.params = DemoParams()
    demo.camera_name = "right_wrist"
    demo.cfg = SimpleNamespace(
        calibration=SimpleNamespace(
            T_end_right_to_camera_rightwrist=np.eye(4)
        )
    )

    class FakeCamera:
        @staticmethod
        def get_frame_timestamp():
            return time.time()

        @staticmethod
        def get_latest_frames():
            return np.zeros((480, 640, 3), dtype=np.uint8), None

        @staticmethod
        def get_camera_intrinsics():
            return (
                np.array(
                    [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
                ),
                None,
            )

    candidate = Detection((280, 220, 380, 270), 0.08, "bottle")
    assert not RunOrchestrator._plausible_close_bottle(candidate, (480, 640, 3))

    class FakeDetector:
        @staticmethod
        def detect(color, predicate=None, target_classes=None):
            return candidate if predicate is None or predicate(candidate) else None

    demo.camera = FakeCamera()
    demo.wrist_detector = FakeDetector()
    demo.detector = None
    demo.robot = SimpleNamespace(current_flange=lambda: np.eye(4))

    demo.ensure_bottle_visible(target_base=np.array([0.0, 0.0, 1.0]))


def test_pregrasp_path_that_moves_farther_from_locked_bottle_is_rejected():
    demo, _ = _demo_with_two_segment_path()
    divergent = [
        [0.37, 0.59, -0.135, 0.0, 0.0, 0.0],
        [0.42, 0.59, -0.135, 0.0, 0.0, 0.0],
    ]
    demo.candidate_path = lambda target: (
        divergent[-1],
        divergent[-1],
        divergent,
    )

    with pytest.raises(SafetyAbort, match="没有朝锁定目标收敛"):
        demo._approach_pregrasp(_localization())

    assert demo.robot.moves == []


def test_pregrasp_confirmation_uses_head_when_partial_wrist_view_is_lost():
    """2026-07-18 replay: after two approach segments the partial wrist
    detection became 0/3, while the fixed head still confirmed the same
    locked target.  Presence confirmation must use that independent adapter
    instead of rerunning full wrist 3-D localization.
    """
    demo = RunOrchestrator.__new__(RunOrchestrator)
    target = _localization()
    head_checks = []
    stages = []
    demo.ensure_bottle_visible = lambda target_base=None: (_ for _ in ()).throw(
        BottleDetectionLost("partial wrist view lost")
    )
    demo._confirm_locked_target_from_head = lambda point: head_checks.append(
        np.asarray(point).copy()
    )
    demo.stage = lambda name, message="": stages.append((name, message))

    result = demo._confirm_target_at_pregrasp(target)

    assert result.source == "head"
    assert result.detection is None
    np.testing.assert_allclose(head_checks, [target.point_base])
    assert any(name == "预抓取目标确认" for name, _ in stages)


def test_pregrasp_confirmation_returns_the_current_wrist_box():
    demo = RunOrchestrator.__new__(RunOrchestrator)
    target = _localization()
    current = Detection((310, 190, 470, 479), 0.88, "bottle")
    demo.ensure_bottle_visible = lambda target_base=None: current
    demo._confirm_locked_target_from_head = lambda _: pytest.fail(
        "current associated wrist detection should be sufficient"
    )
    demo.stage = lambda *_: None

    result = demo._confirm_target_at_pregrasp(target)

    assert result.source == "wrist"
    assert result.detection is current


def _demo_capturing_final_corridor(tmp_path, confirmation):
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.params = DemoParams()
    demo.run_dir = tmp_path
    demo.stage = lambda *_: None
    demo._approach_pregrasp = lambda _: None
    demo._confirm_target_at_pregrasp = lambda _: confirmation
    pose = [0.0] * 6
    demo.candidate_path = lambda _: (pose, pose, [])
    captured = []
    demo.collision_gate = lambda target_box, target: captured.append(
        target_box
    )
    demo.ensure_bottle_visible = lambda target_base=None: None
    demo.planned_local_legs = []
    demo._plan_local_leg = lambda *args, **kwargs: (
        demo.planned_local_legs.append((args, kwargs)) or []
    )
    demo._confirm_lifted_target = lambda target, **_kwargs: target

    class Robot:
        @staticmethod
        def calibrate_empty_close(_params):
            return 0

        @staticmethod
        def current_tcp():
            return np.eye(4)

        @staticmethod
        def plan_ik(_path, _params):
            return None

        @staticmethod
        def move_linear(_pose, _speed):
            return None

        @staticmethod
        def close_gripper(_params):
            return {"pos": [500]}

    demo.robot = Robot()
    return demo, captured


def test_final_corridor_uses_the_current_pregrasp_wrist_box(tmp_path):
    old = _localization()
    current = Detection((310, 190, 470, 479), 0.88, "bottle")
    demo, captured = _demo_capturing_final_corridor(
        tmp_path, GuardResult("wrist", current)
    )

    demo._grasp_and_lift(old)

    assert captured == [current.box]


def test_final_corridor_has_no_wrist_mask_after_head_only_confirmation(tmp_path):
    old = _localization()
    demo, captured = _demo_capturing_final_corridor(
        tmp_path, GuardResult("head")
    )

    demo._grasp_and_lift(old)

    assert captured == [None]


def test_runtime_lift_disables_all_authored_orientation_roll_fallbacks(tmp_path):
    demo, _captured = _demo_capturing_final_corridor(
        tmp_path, GuardResult("head")
    )

    demo._grasp_and_lift(_localization())

    lift_args, lift_kwargs = demo.planned_local_legs[-1]
    assert lift_args[0] == "抬升"
    assert lift_kwargs["roll_degrees"] == (0,)
