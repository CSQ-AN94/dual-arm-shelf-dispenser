"""Typed post-lift evidence and conservative held-object contracts."""

from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from shelf_dispenser.core import (
    CameraFrameUnavailable,
    DemoParams,
    Detection,
    Localization,
    SafetyAbort,
)
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.lift_evidence import LiftEvidenceKind, LiftVisualEvidence
from shelf_dispenser.scene import conservative_scene_union
from shelf_dispenser.target_guard import PostLiftTargetAssociation


def _locked():
    return Localization(
        point_camera=[0.0, 0.0, 0.5],
        point_base=[0.0, 0.0, 0.5],
        pixel=[50.0, 50.0],
        depth_m=0.5,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[40, 20, 60, 80],
        confidence=0.9,
        frame_count=3,
        class_name="bottle",
    )


def _visual_demo(frames, detections):
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.params = DemoParams()
    demo.camera_name = "right_wrist"
    demo.stop_event = threading.Event()
    demo.stage = lambda *_args: None
    demo.cfg = SimpleNamespace(
        calibration=SimpleNamespace(
            T_end_right_to_camera_rightwrist=np.eye(4)
        )
    )
    demo.robot = SimpleNamespace(current_flange=lambda: np.eye(4))
    K = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )

    class Camera:
        index = 0
        timestamp = time.time()

        @staticmethod
        def get_camera_intrinsics():
            return K, None

        def get_frame_timestamp(self):
            self.timestamp += 0.01
            return self.timestamp

        def get_latest_frames(self):
            value = frames[self.index]
            self.index += 1
            return value

    class Detector:
        index = 0

        def detect(self, _color, predicate=None, target_classes=None):
            detection = detections[min(self.index, len(detections) - 1)]
            # One frame invokes the associated query and possibly the
            # unrestricted "another bottle" query.
            if predicate is None:
                self.index += 1
                return detection
            accepted = (
                detection
                if detection is not None and predicate(detection)
                else None
            )
            if accepted is not None:
                self.index += 1
            return accepted

    demo.camera = Camera()
    demo.wrist_detector = Detector()
    return demo


def _rgbd(depth):
    return (
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.full((100, 100), depth, dtype=float),
    )


def test_post_lift_visual_confirmation_uses_fresh_measured_depth():
    detection = Detection((40, 20, 60, 80), 0.9, "bottle")
    demo = _visual_demo([_rgbd(0.55)] * 3, [detection] * 3)

    evidence = demo._post_lift_visual_evidence(_locked())

    assert evidence.kind is LiftEvidenceKind.VISUAL_CONFIRMED
    assert evidence.measurement is not None
    assert evidence.measurement.point_base[2] == pytest.approx(0.55)


def test_fresh_visual_negative_wins_over_gripper_fallback():
    detection = Detection((40, 20, 60, 80), 0.9, "bottle")
    demo = _visual_demo([_rgbd(0.50)] * 3, [detection] * 3)

    evidence = demo._post_lift_visual_evidence(_locked())

    assert evidence.kind is LiftEvidenceKind.VISUAL_NEGATIVE


def test_wrong_horizontal_bottle_is_visual_negative():
    wrong = Detection((72, 20, 92, 80), 0.9, "bottle")
    demo = _visual_demo([_rgbd(0.55)] * 3, [wrong] * 3)

    evidence = demo._post_lift_visual_evidence(_locked())

    assert evidence.kind is LiftEvidenceKind.VISUAL_NEGATIVE
    assert "另一瓶" in evidence.reason


def test_wrong_product_class_is_a_visual_negative_not_occlusion_fusion():
    """Use a class-aware detector; the old mock ignored this production seam."""
    wrong_product = Detection((72, 20, 92, 80), 0.9, "water")
    demo = _visual_demo([_rgbd(0.30)] * 3, [wrong_product] * 3)
    demo.params.target_product_classes = ("coke_bottle",)
    calls = []

    class ClassAwareDetector:
        def detect(self, _color, predicate=None, target_classes=None):
            calls.append(target_classes)
            if (
                target_classes is not None
                and wrong_product.class_name not in target_classes
            ):
                return None
            if predicate is not None and not predicate(wrong_product):
                return None
            return wrong_product

    demo.wrist_detector = ClassAwareDetector()
    evidence = demo._post_lift_visual_evidence(_locked())

    assert evidence.kind is LiftEvidenceKind.VISUAL_NEGATIVE
    assert "另一瓶" in evidence.reason
    assert calls[0] == {"coke_bottle"}
    assert calls[1] is None


def test_post_lift_association_accepts_bottom_truncation_without_vertical_veto():
    association = PostLiftTargetAssociation(
        projected_u=50.0,
        projected_v=30.0,
        expected_depth_m=0.55,
        expected_width_px=20.0,
        image_size=(100, 100),
        class_name="bottle",
    )
    bottom_truncated = Detection((40, 75, 60, 99), 0.9, "bottle")

    assert association.accepts(bottom_truncated)


def test_bottom_truncation_cannot_cast_a_vertical_negative_vote():
    truncated = Detection((40, 20, 60, 99), 0.9, "bottle")
    demo = _visual_demo([_rgbd(0.50)] * 3, [truncated] * 3)

    evidence = demo._post_lift_visual_evidence(_locked())

    assert evidence.kind is LiftEvidenceKind.INSUFFICIENT_DEPTH


def test_all_nan_depth_enters_explicit_zero_of_three_branch():
    demo = _visual_demo(
        [
            (
                np.zeros((100, 100, 3), dtype=np.uint8),
                np.full((100, 100), np.nan),
            )
        ]
        * 3,
        [None] * 3,
    )

    evidence = demo._post_lift_visual_evidence(_locked())

    assert evidence.kind is LiftEvidenceKind.INSUFFICIENT_DEPTH
    assert "关联=0/3" in evidence.reason


def test_three_fresh_nearer_depth_frames_are_typed_occlusion():
    demo = _visual_demo([_rgbd(0.30)] * 3, [None] * 3)

    evidence = demo._post_lift_visual_evidence(_locked())

    assert evidence.kind is LiftEvidenceKind.OCCLUDED_WITH_FRESH_FRAME
    assert evidence.fresh_frames == 3


def test_sparse_foreground_patch_is_not_complete_occlusion():
    depth = np.full((100, 100), np.nan, dtype=float)
    depth[38:41, 38:44] = 0.30
    demo = _visual_demo(
        [(np.zeros((100, 100, 3), dtype=np.uint8), depth)] * 3,
        [None] * 3,
    )

    evidence = demo._post_lift_visual_evidence(_locked())

    assert evidence.kind is LiftEvidenceKind.INSUFFICIENT_DEPTH


def test_camera_buffer_loss_is_not_occlusion():
    demo = _visual_demo([(None, None)] * 3, [None] * 3)

    evidence = demo._post_lift_visual_evidence(_locked())

    assert evidence.kind is LiftEvidenceKind.CAMERA_UNAVAILABLE


@pytest.mark.parametrize(
    ("frames", "expected_exception"),
    [
        (
            [
                (
                    np.zeros((100, 100, 3), dtype=np.uint8),
                    np.full((100, 100), np.nan),
                )
            ]
            * 3,
            SafetyAbort,
        ),
        ([(None, None)] * 3, CameraFrameUnavailable),
    ],
)
def test_zero_of_three_or_camera_loss_cannot_be_promoted_to_held(
    frames, expected_exception
):
    demo = _visual_demo(frames, [None] * 3)
    before = np.eye(4)
    after = np.eye(4)
    after[2, 3] = demo.params.lift_m

    with pytest.raises(expected_exception):
        demo._confirm_lifted_target(
            _locked(), prelift_tcp=before, postlift_tcp=after
        )


def _occlusion_confirmation_demo(states):
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.params = DemoParams()
    demo.stage = lambda *_args: None
    demo._post_lift_visual_evidence = lambda _locked_target: LiftVisualEvidence(
        LiftEvidenceKind.OCCLUDED_WITH_FRESH_FRAME,
        "3/3 fresh",
        3,
        0,
    )

    class Robot:
        empty_close_pos = 0

        def __init__(self):
            self.states = iter(states)

        def gripper_state(self):
            return next(self.states)

    demo.robot = Robot()
    return demo


def _held_state(position=50):
    return {"dof_state": [3], "pos": [position]}


def test_occlusion_fusion_requires_all_three_gripper_rereads():
    demo = _occlusion_confirmation_demo(
        [_held_state(), _held_state(), {"dof_state": [2], "pos": [50]}]
    )
    before = np.eye(4)
    after = np.eye(4)
    after[2, 3] = 0.05

    with pytest.raises(SafetyAbort, match="第 3/3 帧夹持证据丢失"):
        demo._confirm_lifted_target(
            _locked(), prelift_tcp=before, postlift_tcp=after
        )


def test_occlusion_fusion_rejects_tcp_that_did_not_rise():
    demo = _occlusion_confirmation_demo([_held_state()] * 3)

    with pytest.raises(SafetyAbort, match="实际 TCP 抬升不成立"):
        demo._confirm_lifted_target(
            _locked(), prelift_tcp=np.eye(4), postlift_tcp=np.eye(4)
        )


def test_occlusion_fusion_accepts_stable_grip_and_measured_five_cm_rise():
    demo = _occlusion_confirmation_demo(
        [_held_state(50), _held_state(51), _held_state(50)]
    )
    before = np.eye(4)
    after = np.eye(4)
    after[2, 3] = 0.05

    measured = demo._confirm_lifted_target(
        _locked(), prelift_tcp=before, postlift_tcp=after
    )

    assert measured is None
    assert demo.last_lift_confirmation_camera == "occlusion_fusion"


def test_scene_union_requires_same_base_pose_and_fails_over_budget():
    params = DemoParams(scene_max_voxels=2)
    pose = np.eye(4)

    union = conservative_scene_union(
        [[0.0, 0.0, 0.0]],
        [[0.1, 0.0, 0.0]],
        params,
        before_frame="right_controller_base",
        after_frame="right_controller_base",
        before_base_pose=pose,
        after_base_pose=pose,
    )
    assert len(union) == 2

    moved = pose.copy()
    moved[0, 3] = 0.1
    with pytest.raises(SafetyAbort, match="跨底盘姿态"):
        conservative_scene_union(
            [],
            [],
            params,
            before_frame="right_controller_base",
            after_frame="right_controller_base",
            before_base_pose=pose,
            after_base_pose=moved,
        )
    with pytest.raises(SafetyAbort, match="超出安全场景预算"):
        conservative_scene_union(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.2, 0.0, 0.0]],
            params,
            before_frame="right_controller_base",
            after_frame="right_controller_base",
            before_base_pose=pose,
            after_base_pose=pose,
        )


def test_shelf_fallback_guard_covers_bottle_above_and_below_grasp(tmp_path):
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.params = DemoParams(grasp_height_fraction=0.40)
    demo.delivery_safety = None
    demo.run_dir = tmp_path
    demo.robot = SimpleNamespace(current_tcp=lambda: np.eye(4))

    demo._set_held_bottle_guard(_locked())

    guard = demo.held_object_guard
    T_base_link7 = np.linalg.inv(demo.T_link7_tcp)
    center_base = (
        T_base_link7 @ np.r_[np.asarray(guard["center"], dtype=float), 1.0]
    )[:3]
    guarded_height = float(guard["size"][2])
    guarded_bottom = center_base[2] - guarded_height / 2.0
    guarded_top = center_base[2] + guarded_height / 2.0
    assert guarded_bottom <= -(0.60 * demo.params.held_bottle_height_m)
    assert guarded_top >= 0.40 * demo.params.held_bottle_height_m


def test_explicit_detach_clears_guard_only_after_live_readback(tmp_path):
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.held_object_guard = {"size": [1, 1, 1]}
    demo.args = SimpleNamespace(task_mode="from-observation")
    demo.safety = SimpleNamespace(moveit_frame="platform_base_link")
    demo.stage = lambda *_args: None
    calls = []
    demo.planner = SimpleNamespace(
        detach_attached_object=lambda **kwargs: calls.append(kwargs)
    )

    demo._detach_held_bottle_guard()

    assert demo.held_object_guard is None
    assert calls[0]["object_id"] == "held_bottle_guard"


def test_detach_failure_preserves_guard_unknown_state():
    demo = RunOrchestrator.__new__(RunOrchestrator)
    guard = {"size": [1, 1, 1]}
    demo.held_object_guard = guard
    demo.args = SimpleNamespace(task_mode="from-observation")
    demo.safety = SimpleNamespace(moveit_frame="platform_base_link")
    demo.stage = lambda *_args: None

    def fail(**_kwargs):
        raise SafetyAbort("still attached")

    demo.planner = SimpleNamespace(detach_attached_object=fail)

    with pytest.raises(SafetyAbort, match="still attached"):
        demo._detach_held_bottle_guard()
    assert demo.held_object_guard is guard
