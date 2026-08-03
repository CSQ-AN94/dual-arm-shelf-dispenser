"""Multi-frame collection for the obstacle scene / table fit (no hardware).

2026-07-19: target localization already required 7-frame consensus before
trusting a position; the environment point cloud (obstacle voxels + table
height) trusted a single get_latest_frames() call. That asymmetry meant a
transient bad frame (motion blur, exposure switch, a hand crossing the
head-camera view) could silently misshape the electronic fence for the
whole run without ever being caught. These tests cover the sampling loop
that closes that gap; the combination logic itself (union for occupancy,
median+agreement for table height) is covered in test_algorithms.py and
test_table_model.py.
"""

import threading

import numpy as np
import pytest

import bottle_grasp.demo as demo_module
from bottle_grasp.core import DemoParams, SafetyAbort
from bottle_grasp.demo import BottleDemo


def _demo_with_camera(camera):
    demo = BottleDemo.__new__(BottleDemo)
    demo.camera = camera
    demo.stop_event = threading.Event()
    return demo


class _TickingCamera:
    """Advances its timestamp by one tick per get_frame_timestamp() call."""

    def __init__(self, depths):
        self._depths = list(depths)
        self._tick = 0.0
        self._served = -1

    def get_frame_timestamp(self):
        self._tick += 1.0
        return self._tick

    def get_latest_frames(self):
        self._served += 1
        if self._served >= len(self._depths):
            return None, self._depths[-1]
        return None, self._depths[self._served]


class _StaleCamera:
    """Timestamp freezes after the very first read — a dead/frozen stream.

    The first poll is legitimately "new" relative to the collector's
    initial last_timestamp=0.0, so it must count. Every poll after that
    reports the same timestamp and must not.
    """

    def get_frame_timestamp(self):
        return 1.0

    def get_latest_frames(self):
        return None, np.zeros((4, 4))


def test_collects_exactly_the_requested_number_of_distinct_frames():
    depths = [np.full((2, 2), value) for value in (1.0, 2.0, 3.0, 4.0)]
    demo = _demo_with_camera(_TickingCamera(depths))
    frames = demo._collect_fresh_depth_frames(3, label="测试")
    assert len(frames) == 3
    # Distinct frames, in the order the camera produced them.
    assert [float(frame[0, 0]) for frame in frames] == [1.0, 2.0, 3.0]


def test_rejects_a_frame_count_below_one():
    demo = _demo_with_camera(_TickingCamera([np.zeros((2, 2))]))
    with pytest.raises(SafetyAbort, match="至少为 1"):
        demo._collect_fresh_depth_frames(0, label="测试")


def test_frozen_camera_stream_times_out_short_of_the_requested_count(
    monkeypatch,
):
    """A camera stream that freezes after its first frame must not be read
    as repeated agreement — the collector must stop accepting it as 'fresh'
    once the timestamp stops advancing, and time out short rather than loop
    forever or silently return duplicates. Fake the clock: the real deadline
    is several real seconds, and this test only needs to prove the branch
    fires, not wait it out."""
    fake_now = [0.0]
    monkeypatch.setattr(demo_module.time, "time", lambda: fake_now[0])

    def fake_sleep(_seconds):
        fake_now[0] += 1.0  # jumps straight past the deadline

    monkeypatch.setattr(demo_module.time, "sleep", fake_sleep)

    demo = _demo_with_camera(_StaleCamera())
    demo.params = DemoParams()
    with pytest.raises(SafetyAbort, match=r"只取到 1/2 个新鲜深度帧"):
        demo._collect_fresh_depth_frames(2, label="测试")


def test_stop_event_aborts_collection_immediately():
    demo = _demo_with_camera(_TickingCamera([np.zeros((2, 2))] * 5))
    demo.stop_event.set()
    with pytest.raises(SafetyAbort, match="用户停止"):
        demo._collect_fresh_depth_frames(3, label="测试")


def test_local_scene_removes_voxels_intersecting_the_whole_locked_bottle():
    """The RGB-D lock is on the camera-facing surface, not bottle centre.

    A coarse voxel whose centre lies behind or beside that surface point can
    still intersect the physical bottle.  Such cells are expected contact,
    while a separate cell beyond the bottle footprint and the shelf below it
    must remain obstacles.
    """
    demo = BottleDemo.__new__(BottleDemo)
    demo.params = DemoParams(scene_voxel_m=0.065)

    class Calibration:
        T_base_right_to_camera_head = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.10],
                [0.0, 0.0, 1.0, 0.30],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    demo.cfg = type("Config", (), {"calibration": Calibration()})()
    target = np.array([0.0, 0.60, 0.0])
    target_front = [0.0, 0.60, 0.0]
    target_rear_voxel = [0.0, 0.665, 0.0]
    target_side_voxel = [0.065, 0.665, 0.0]
    separate_obstacle = [0.13, 0.665, 0.0]
    shelf_below = [0.0, 0.665, -0.13]
    demo.scene_voxels = [
        target_front,
        target_rear_voxel,
        target_side_voxel,
        separate_obstacle,
        shelf_below,
    ]

    remaining = demo._scene_without_locked_target(target)

    np.testing.assert_allclose(remaining, [separate_obstacle, shelf_below])


def test_local_scene_also_removes_target_silhouette_voxel_recorded_at_capture():
    """Replay the 2026-07-21 residual voxel 4 mm outside cylinder padding."""
    demo = BottleDemo.__new__(BottleDemo)
    demo.params = DemoParams(scene_voxel_m=0.065)

    class Calibration:
        T_base_right_to_camera_head = np.eye(4)

    demo.cfg = type("Config", (), {"calibration": Calibration()})()
    target = np.array([0.15902154, 0.69046435, -0.08527139])
    residual_bottle_voxel = [0.0975, 0.6825, -0.0975]
    real_neighbour = [0.0325, 0.6825, -0.0975]
    demo.scene_voxels = [residual_bottle_voxel, real_neighbour]
    demo.target_occupancy_voxels = [residual_bottle_voxel]

    remaining = demo._scene_without_locked_target(target)

    np.testing.assert_allclose(remaining, [real_neighbour])


def test_adapt_fence_skips_multi_frame_fitting_when_profile_has_no_table():
    """A profile without a table keepout has nothing for the measurement to
    adjust. It must not call fit_table_top at all — frame disagreement on a
    measurement that would be discarded anyway must not abort a run it
    cannot affect."""
    demo = BottleDemo.__new__(BottleDemo)

    class NoTableSafety:
        keepout_boxes = ()

    demo.safety = NoTableSafety()

    def _must_not_be_called(*_a, **_k):
        raise AssertionError("fit_table_top must not run without a table keepout")

    original = demo_module.fit_table_top
    demo_module.fit_table_top = _must_not_be_called
    try:
        result = demo._adapt_fence_to_measured_table(
            [np.zeros((2, 2)), np.zeros((2, 2))],
            np.eye(3),
            _localization(),
        )
    finally:
        demo_module.fit_table_top = original
    assert result is None


def test_adapt_fence_fits_every_frame_then_combines_them():
    """Wiring check: each collected frame gets its own fit_table_top call
    (not just the first/last), and the per-frame results are handed to
    combine_table_fits rather than averaged ad hoc inline."""
    import bottle_grasp.table_model as table_model

    from bottle_grasp.table_model import TABLE_KEEPOUT_ID

    demo = BottleDemo.__new__(BottleDemo)
    demo.params = DemoParams()

    class FakeCalibration:
        T_base_right_to_camera_head = np.eye(4)

    class FakeConfig:
        calibration = FakeCalibration()

    demo.cfg = FakeConfig()

    class TableSafety:
        keepout_boxes = (type("Box", (), {"id": TABLE_KEEPOUT_ID})(),)

    demo.safety = TableSafety()

    seen_depths = []

    def fake_fit(depth, target, params):
        seen_depths.append(depth)
        return object()  # identity is enough to prove pass-through

    combine_calls = []

    def fake_combine(fits, params):
        combine_calls.append(fits)
        raise SafetyAbort("stop before touching the real fence adaptation")

    monkey_targets = [
        (demo_module, "fit_table_top", fake_fit),
        (demo_module, "head_scene_points", lambda *a, **k: np.zeros((1, 3))),
        (demo_module, "combine_table_fits", fake_combine),
    ]
    originals = [(mod, name, getattr(mod, name)) for mod, name, _ in monkey_targets]
    for mod, name, value in monkey_targets:
        setattr(mod, name, value)
    try:
        depths = [np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), 2.0)]
        with pytest.raises(SafetyAbort, match="stop before touching"):
            demo._adapt_fence_to_measured_table(
                depths, np.eye(3), _localization()
            )
    finally:
        for mod, name, original in originals:
            setattr(mod, name, original)

    assert len(seen_depths) == 3
    assert len(combine_calls[0]) == 3


def _localization():
    from bottle_grasp.core import Localization

    return Localization(
        [0, 0, 0.5], [0, 0.6, -0.1], [320, 240], 0.5, 0.001, 0.002,
        [0, 0, 10, 10], 0.9, 7,
    )


def test_none_depth_frames_are_skipped_not_counted():
    class FlakyCamera:
        def __init__(self):
            self._tick = 0.0
            self._n = 0

        def get_frame_timestamp(self):
            self._tick += 1.0
            return self._tick

        def get_latest_frames(self):
            self._n += 1
            # Every other frame arrives with no depth payload.
            if self._n % 2 == 0:
                return None, None
            return None, np.full((2, 2), float(self._n))

    demo = _demo_with_camera(FlakyCamera())
    frames = demo._collect_fresh_depth_frames(2, label="测试")
    assert len(frames) == 2
