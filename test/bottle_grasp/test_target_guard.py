import numpy as np
import pytest

from bottle_grasp.core import (
    BottleDetectionLost,
    CameraFrameUnavailable,
    Detection,
    DemoParams,
    Localization,
    SafetyAbort,
)
from bottle_grasp.demo import BottleDemo
from bottle_grasp.lift_evidence import LiftEvidenceKind, LiftVisualEvidence
from bottle_grasp.target_guard import LockedTargetGuard, ProjectedTargetAssociation


def _intrinsics() -> np.ndarray:
    return np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def test_bottom_truncated_box_accepts_a_bounded_offscreen_locked_grasp_point():
    detection = Detection((280, 240, 360, 479), 0.9, "bottle")
    association = ProjectedTargetAssociation.from_view(
        # Projects to (320, 540): the locked grasp height is 60 px below the
        # image while the upper part of the bottle remains visibly bottom-cut.
        target_base=np.array([0.0, 0.18, 0.36]),
        T_base_camera=np.eye(4),
        intrinsics=_intrinsics(),
        image_shape=(480, 640, 3),
    )
    far_outside = ProjectedTargetAssociation.from_view(
        target_base=np.array([0.0, 0.396, 0.36]),
        T_base_camera=np.eye(4),
        intrinsics=_intrinsics(),
        image_shape=(480, 640, 3),
    )

    assert association.pixel == pytest.approx((320.0, 540.0))
    assert not association.in_image
    assert association.accepts(detection)
    assert not far_outside.accepts(detection)


def test_side_truncated_box_does_not_pull_locked_horizontal_point_inward():
    detection = Detection((0, 150, 80, 330), 0.9, "bottle")
    locked = np.array([-0.21, 0.0, 0.36])  # projects to u=-30
    association = ProjectedTargetAssociation.from_view(
        target_base=locked,
        T_base_camera=np.eye(4),
        intrinsics=_intrinsics(),
        image_shape=(480, 640, 3),
    )

    point_camera, point_base, pixel, depth = association.refine_locked_depth(
        detection=detection,
        target_base=locked,
        T_base_camera=np.eye(4),
        intrinsics=_intrinsics(),
    )

    assert pixel == pytest.approx((-30.0, 240.0))
    np.testing.assert_allclose(point_camera, locked)
    np.testing.assert_allclose(point_base, locked)
    assert depth == pytest.approx(0.36)


def test_guard_uses_head_adapter_when_wrist_adapter_loses_detection():
    target = np.array([0.02, 0.59, -0.135])
    head_calls = []
    guard = LockedTargetGuard(
        wrist_check=lambda _: (_ for _ in ()).throw(
            BottleDetectionLost("wrist detector miss")
        ),
        head_confirm=lambda point: head_calls.append(point.copy()),
    )

    result = guard.verify(target)

    assert result.source == "head"
    assert result.detection is None
    assert len(head_calls) == 1
    np.testing.assert_allclose(head_calls[0], target)


def test_guard_returns_the_current_associated_wrist_detection():
    target = np.array([0.02, 0.59, -0.135])
    detection = Detection((250, 80, 390, 479), 0.8, "bottle")
    guard = LockedTargetGuard(
        wrist_check=lambda _: detection,
        head_confirm=lambda _: pytest.fail("visible wrist target must not use head"),
    )

    result = guard.verify(target)

    assert result.source == "wrist"
    assert result.detection is detection


def test_guard_never_converts_rgbd_stream_failure_into_head_fallback():
    guard = LockedTargetGuard(
        wrist_check=lambda _: (_ for _ in ()).throw(
            CameraFrameUnavailable("RGB-D 画面中断")
        ),
        head_confirm=lambda _: pytest.fail("stream failure must stop immediately"),
    )

    with pytest.raises(CameraFrameUnavailable, match="画面中断"):
        guard.verify(np.zeros(3))


def _demo_with_head_adapter(head_point):
    demo = BottleDemo.__new__(BottleDemo)
    demo.params = DemoParams()
    demo.detector = object()
    demo.cfg = type(
        "Cfg",
        (),
        {
            "calibration": type(
                "Calibration",
                (),
                {"T_base_right_to_camera_head": np.eye(4)},
            )()
        },
    )()
    calls = []
    demo._start_camera = lambda name: calls.append(("camera", name))
    demo.stage = lambda name, message="": calls.append(("stage", name, message))

    def localize(label, transform_provider, params, depth_prior_base=None):
        calls.append(("localize", label, params.samples, list(depth_prior_base)))
        return Localization(
            point_camera=list(head_point),
            point_base=list(head_point),
            pixel=[320, 240],
            depth_m=0.5,
            depth_mad_m=0.001,
            position_spread_m=0.002,
            box=[280, 100, 360, 400],
            confidence=0.8,
            frame_count=params.samples,
        )

    demo.localize = localize
    return demo, calls


def test_head_adapter_confirms_without_rewriting_locked_target():
    target = np.array([0.02, 0.59, -0.135])
    demo, calls = _demo_with_head_adapter(target + np.array([0.01, 0.0, 0.0]))

    demo._confirm_locked_target_from_head(target)

    assert calls[0] == ("camera", "head")
    assert calls[1][0:3] == ("localize", "头部补充确认", 3)
    assert calls[2] == ("camera", "right_wrist")
    assert any(call[0:2] == ("stage", "头部补充确认通过") for call in calls)


def test_head_adapter_restores_wrist_then_aborts_if_target_moved():
    target = np.array([0.02, 0.59, -0.135])
    demo, calls = _demo_with_head_adapter(target + np.array([0.10, 0.0, 0.0]))

    with pytest.raises(SafetyAbort, match="当前路径作废"):
        demo._confirm_locked_target_from_head(target)

    assert ("camera", "right_wrist") in calls


def test_release_confirmation_uses_wrist_and_does_not_call_occluded_head():
    target = np.array([0.02, 0.59, -0.135])
    demo = BottleDemo.__new__(BottleDemo)
    demo.params = DemoParams()
    calls = []
    locked = Localization(
        point_camera=target.tolist(),
        point_base=target.tolist(),
        pixel=[320, 240],
        depth_m=0.5,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[280, 100, 360, 400],
        confidence=0.8,
        frame_count=7,
    )
    demo.stage = lambda name, message="": calls.append(
        ("stage", name, message)
    )
    demo._measure_target_from_wrist_3d = lambda label, expected: (
        calls.append(("wrist", label, np.asarray(expected).copy())),
        locked,
    )[1]
    demo._measure_target_from_head_3d = lambda *_args: pytest.fail(
        "visible wrist target must not use the occluded head camera"
    )

    assert demo._confirm_released_target(locked) == "right_wrist"
    np.testing.assert_allclose(calls[0][2], target)
    assert any(call[0:2] == ("stage", "放回视觉确认") for call in calls)


def _demo_with_independent_head_measurement(measured_point):
    demo = BottleDemo.__new__(BottleDemo)
    demo.params = DemoParams()
    demo.detector = object()
    demo.camera_name = "right_wrist"
    demo.cfg = type(
        "Cfg",
        (),
        {
            "calibration": type(
                "Calibration",
                (),
                {"T_base_right_to_camera_head": np.eye(4)},
            )()
        },
    )()
    calls = []

    def start_camera(name):
        demo.camera_name = name
        calls.append(("camera", name))

    def localize(
        label,
        transform_provider,
        params,
        depth_prior_base=None,
        allow_depth_prior_fallback=True,
        required_consensus_frames=None,
    ):
        calls.append(
            (
                "localize",
                label,
                np.asarray(depth_prior_base).copy(),
                allow_depth_prior_fallback,
                required_consensus_frames,
            )
        )
        return Localization(
            point_camera=list(measured_point),
            point_base=list(measured_point),
            pixel=[320, 240],
            depth_m=0.5,
            depth_mad_m=0.001,
            position_spread_m=0.002,
            box=[280, 100, 360, 400],
            confidence=0.8,
            frame_count=params.samples,
        )

    demo._start_camera = start_camera
    demo.localize = localize
    demo.stage = lambda name, message="": calls.append(("stage", name, message))
    return demo, calls


def test_lift_confirmation_never_falls_back_to_fixed_head():
    locked = Localization(
        point_camera=[0.0, 0.0, 0.5],
        point_base=[0.0, 0.0, 0.5],
        pixel=[320, 240],
        depth_m=0.5,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[280, 100, 360, 400],
        confidence=0.8,
        frame_count=7,
    )
    demo, _calls = _demo_with_independent_head_measurement([0.0, 0.0, 0.55])
    demo._post_lift_visual_evidence = lambda _target: LiftVisualEvidence(
        LiftEvidenceKind.CAMERA_UNAVAILABLE,
        "recorded wrist outage",
        0,
        0,
    )
    demo._measure_target_from_head_3d = lambda *_args: pytest.fail(
        "post-lift confirmation must not query the fixed head"
    )

    with pytest.raises(CameraFrameUnavailable, match="不能当作遮挡"):
        demo._confirm_lifted_target(
            locked,
            prelift_tcp=np.eye(4),
            postlift_tcp=np.eye(4),
        )


def test_release_confirmation_uses_fresh_3d_measurement_at_locked_point():
    locked = Localization(
        point_camera=[0.0, 0.0, 0.5],
        point_base=[0.0, 0.0, 0.5],
        pixel=[320, 240],
        depth_m=0.5,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[280, 100, 360, 400],
        confidence=0.8,
        frame_count=7,
    )
    demo, calls = _demo_with_independent_head_measurement([0.0, 0.0, 0.5])

    assert demo._confirm_released_target(locked) == "right_wrist"

    localize_call = next(call for call in calls if call[0] == "localize")
    np.testing.assert_allclose(localize_call[2], locked.point_base)
    assert localize_call[3] is False


def test_release_confirmation_accepts_20260720_returned_bottle_trace():
    """Replay the successful grasp/place that aborted before return-home.

    Wrist lock and post-release localization can sample different
    semantic pixels on the bottle. The horizontal association is good, but
    the sampled height moved 34.4 mm and made the raw 3-D norm 41.6 mm.
    """
    locked = Localization(
        point_camera=[0.016410, -0.001285, 0.299033],
        point_base=[0.010814, 0.622928, -0.149620],
        pixel=[360.5, 246.230275985172],
        depth_m=0.299033,
        depth_mad_m=0.0,
        position_spread_m=0.0004567,
        box=[326, 0, 397, 378],
        confidence=0.8,
        frame_count=7,
    )
    lifted = Localization(
        point_camera=[0.117794, -0.007661, 0.557000],
        point_base=[-0.002952, 0.621046, -0.030264],
        pixel=[549.0, 239.26],
        depth_m=0.557000,
        depth_mad_m=0.003,
        position_spread_m=0.002803,
        box=[514, 133, 587, 294],
        confidence=0.8,
        frame_count=3,
    )
    measured_point = [-0.000404, 0.602440, -0.115207]
    demo, _calls = _demo_with_independent_head_measurement(measured_point)

    assert demo._confirm_released_target(locked, lifted) == "right_wrist"


def test_release_confirmation_falls_back_to_head_when_wrist_is_unavailable():
    target = np.array([0.02, 0.59, -0.135])
    demo, calls = _demo_with_independent_head_measurement(target)
    locked = Localization(
        point_camera=target.tolist(),
        point_base=target.tolist(),
        pixel=[320, 240],
        depth_m=0.5,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[280, 100, 360, 400],
        confidence=0.8,
        frame_count=7,
    )
    demo._measure_target_from_wrist_3d = lambda *_args: (
        (_ for _ in ()).throw(BottleDetectionLost("腕部看不到已放回水瓶"))
    )

    assert demo._confirm_released_target(locked) == "head"
    assert any(call[0:2] == ("stage", "放回腕部确认不可用") for call in calls)


@pytest.mark.parametrize(
    ("measured_point", "message"),
    (
        ([0.060, 0.000, -0.010], "水平"),
        ([0.000, 0.000, 0.070], "下降"),
    ),
)
def test_directional_release_confirmation_still_rejects_wrong_release(
    measured_point,
    message,
):
    def localization(point):
        return Localization(
            point_camera=list(point),
            point_base=list(point),
            pixel=[320, 240],
            depth_m=0.5,
            depth_mad_m=0.001,
            position_spread_m=0.002,
            box=[280, 100, 360, 400],
            confidence=0.8,
            frame_count=3,
        )

    locked = localization([0.0, 0.0, 0.0])
    lifted = localization([0.0, 0.0, 0.08])
    demo, _calls = _demo_with_independent_head_measurement(measured_point)

    with pytest.raises(SafetyAbort, match=message):
        demo._confirm_released_target(locked, lifted)
