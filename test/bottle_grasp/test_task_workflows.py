from __future__ import annotations

import logging
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from bottle_grasp.core import DemoParams, Localization, SafetyAbort
from bottle_grasp.demo import BottleDemo
import bottle_grasp.demo as demo_module
from bottle_grasp.mobile_body import (
    ChassisState,
    LiftState,
    MobileBodyCoordinator,
)
from bottle_grasp.task import (
    BottlePickPlaceTask,
    DeliverMode,
    ObjectState,
    RunStatus,
    StartMode,
    TaskPhase,
)


def _localization(point):
    return Localization(
        point_camera=list(point),
        point_base=list(point),
        pixel=[320.0, 240.0],
        depth_m=0.36,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[280, 100, 360, 479],
        confidence=0.9,
        frame_count=7,
    )


class _ShelfReadyChassis:
    """Inert chassis adapter used to exercise the real body admission gate."""

    def __init__(self, *, yaw_deg):
        self._state = ChassisState(
            x_m=1.0,
            y_m=2.0,
            yaw_rad=math.radians(yaw_deg),
            linear_mps=0.0,
            angular_radps=0.0,
            control_mode="kAuto",
            robot_state="kIdle",
            captured_monotonic=1.0,
        )
        self.stops = 0
        self.prepared = 0

    def state(self):
        return self._state

    def prepare_for_motion(self):
        self.prepared += 1
        return True

    def stop(self):
        self.stops += 1


class _ShelfReadyLift:
    @staticmethod
    def state():
        return LiftState(
            height_mm=716,
            enabled=True,
            error_flag=0,
            mode=0,
            captured_monotonic=1.0,
        )


def _shelf_ready_config():
    return SimpleNamespace(
        transport_pose_verified=True,
        shelf_ready_verified=True,
        lift_transition_verified=True,
        table_roi_verified=True,
        workspace_verified=True,
        keepouts_verified=True,
        bottle_tcp_verified=True,
        shelf_ready=SimpleNamespace(
            x_m=1.0,
            y_m=2.0,
            yaw_deg=0.0,
            lift_height_mm=716,
            xy_tolerance_m=0.02,
            yaw_tolerance_deg=2.0,
            lift_tolerance_mm=5,
        ),
        source_lift_height_mm=716,
        target_lift_height_mm=900,
        target_lift_tolerance_mm=5,
        body_lift_speed=15,
        body_rotation_yaw_deg=-90.0,
        max_angular_speed_radps=0.12,
        rotation_tolerance_deg=2.0,
        max_base_translation_m=0.035,
        rotation_timeout_s=25.0,
        rotation_sweep=SimpleNamespace(
            positive_clearance_m=0.08,
            negative_clearance_m=0.08,
            positive_verified=True,
            negative_verified=True,
        ),
    )


def _wire_real_shelf_ready_gate(demo, tmp_path, *, chassis_yaw_deg):
    """Use BottleDemo's real pre-arm body admission with inert adapters."""
    chassis = _ShelfReadyChassis(yaw_deg=chassis_yaw_deg)
    demo.mobile_body = MobileBodyCoordinator(
        chassis=chassis,
        lift=_ShelfReadyLift(),
        stop_event=demo.stop_event,
        evidence_dir=tmp_path,
    )
    demo.delivery_safety = SimpleNamespace(
        side_table_delivery=_shelf_ready_config()
    )
    demo.shelf_ready_body_snapshot = None
    demo._load_safety_profiles = lambda: None
    demo._validate_side_table_profile_pair = lambda: None
    demo._ensure_mobile_body = lambda: demo.mobile_body

    def capture():
        demo.calls.append(("capture_shelf_ready",))
        snapshot = BottleDemo._capture_shelf_ready_for_dispense(demo)
        demo.shelf_ready_snapshot = snapshot
        return snapshot

    demo._capture_shelf_ready_for_dispense = capture
    return chassis


class FakeDemo:
    def __init__(self, tmp_path, mode):
        self.args = SimpleNamespace(task_mode=mode.value)
        self.params = DemoParams()
        self.safety = SimpleNamespace(
            observation_staging_joints_deg=None,
        )
        self.stop_event = threading.Event()
        self.run_dir = tmp_path
        self.calls = []
        self.head = _localization([0.12, 0.52, -0.12])
        self.wrist = _localization([0.13, 0.51, -0.12])
        self.lifted = _localization([0.13, 0.51, -0.07])
        self.left_robot = SimpleNamespace(joints_deg=lambda: [0.0] * 7)
        self.start_home_moved = False

    def stage(self, name, message=""):
        self.calls.append(("stage", name))

    def initialize(self):
        self.calls.append(("initialize",))

    def _preflight(self):
        self.calls.append(("preflight",))

    def _preflight_side_table_delivery(self, *, start=None):
        assert start is self.shelf_ready_snapshot
        self.calls.append(("preflight_delivery",))

    def _capture_shelf_ready_for_dispense(self):
        self.shelf_ready_snapshot = object()
        self.calls.append(("capture_shelf_ready",))
        return self.shelf_ready_snapshot

    def _fresh_head_target(self):
        self.calls.append(("fresh_head",))
        return self.head

    def _build_head_scene(self, target):
        assert target is self.head
        self.calls.append(("scene",))

    def _plan_observation(self, target):
        np.testing.assert_allclose(target, self.head.point_base)
        self.calls.append(("plan_observation",))
        return {"points_deg": [[1.0] * 7]}

    def _normalize_start_home(self):
        self.calls.append(("normalize_start_home",))
        return self.start_home_moved

    def _plan_observation_staging(self):
        self.calls.append(("plan_observation_staging",))
        return {"points_deg": [[2.0] * 7]}

    def _refresh_and_revalidate_plan(self, *, name, plan, locked_target):
        assert plan["points_deg"]
        assert locked_target is self.head
        if name == "moveit_observation_staging":
            self.calls.append(("refresh_validate_staging",))
        else:
            assert name == "moveit_observation"
            self.calls.append(("refresh_validate_observation",))

    def _execute_plan(self, name, plan):
        if name == "抬高展开到观察准备位":
            self.calls.append(("execute_staging", name))
        else:
            self.calls.append(("execute_observation", name))

    def _fresh_wrist_target(self, head_target):
        assert head_target is self.head
        self.calls.append(("fresh_wrist",))
        return self.wrist

    def _verify_wrist_observation_start(self, wrist_target):
        assert wrist_target is self.wrist
        self.calls.append(("verify_wrist_start",))

    def _verify_wrist_pregrasp_start(self, wrist_target):
        assert wrist_target is self.wrist
        self.calls.append(("verify_pregrasp_start",))

    def _grasp_and_lift(self, wrist_target):
        assert wrist_target is self.wrist
        self.calls.append(("grasp_lift",))
        return self.lifted

    def _grasp_and_lift_from_pregrasp(self, wrist_target):
        assert wrist_target is self.wrist
        self.calls.append(("grasp_lift_from_pregrasp",))
        return self.lifted

    def _place_back(self, wrist_target, lifted_target):
        assert wrist_target is self.wrist
        assert lifted_target is self.lifted
        self.calls.append(("place_release_retreat",))

    def _dispense_to_side_table(self, lifted_target, *, start=None):
        assert lifted_target is self.lifted
        assert start is self.shelf_ready_snapshot
        self.calls.append(("dispense_side_table",))

    def _capture_output_table_scene(self, *, require_place_candidate=True):
        self.calls.append(
            ("refresh_output_scene", require_place_candidate)
        )

    def _refresh_head_scene_for_global_motion(self, wrist_target):
        assert wrist_target is self.wrist
        self.calls.append(("refresh_return_scene",))

    def _return_home(self):
        self.calls.append(("return_home",))

    def _right_arm_at_delivery_home(self):
        self.calls.append(("right_arm_home_check",))
        return True

    def _return_body_to_shelf_ready(self, *, start, authorization):
        assert start is self.shelf_ready_snapshot
        assert authorization.release_verified is True
        assert authorization.object_state == "empty"
        assert authorization.right_arm_compact_or_home is True
        assert authorization.left_arm_stable is True
        self.calls.append(("return_body",))


def _hardware_calls(demo):
    return [call[0] for call in demo.calls if call[0] != "stage"]


def test_from_observation_is_fresh_and_uses_one_complete_shared_tail(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_OBSERVATION)

    result = BottlePickPlaceTask(demo).run(StartMode.FROM_OBSERVATION)

    assert _hardware_calls(demo) == [
        "initialize",
        "preflight",
        "fresh_head",
        "scene",
        "fresh_wrist",
        "verify_wrist_start",
        "grasp_lift",
        "place_release_retreat",
        "refresh_return_scene",
        "return_home",
    ]
    assert result.status == RunStatus.DONE.value
    assert result.phase == TaskPhase.DONE.value
    assert result.object_state == ObjectState.EMPTY.value
    assert (tmp_path / "task_result.json").is_file()


def test_from_pregrasp_verifies_current_hover_and_skips_completed_transit(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_PREGRASP)

    result = BottlePickPlaceTask(demo).run(StartMode.FROM_PREGRASP)

    assert _hardware_calls(demo) == [
        "initialize",
        "preflight",
        "fresh_head",
        "scene",
        "fresh_wrist",
        "verify_pregrasp_start",
        "grasp_lift_from_pregrasp",
        "place_release_retreat",
        "refresh_return_scene",
        "return_home",
    ]
    assert result.status == RunStatus.DONE.value
    assert result.object_state == ObjectState.EMPTY.value


def test_dispense_uses_body_and_live_side_table_flow_before_home(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_OBSERVATION)
    demo.args.dispense = True

    result = BottlePickPlaceTask(demo).run(
        StartMode.FROM_OBSERVATION, DeliverMode.DISPENSE
    )

    assert _hardware_calls(demo) == [
        "capture_shelf_ready",
        "initialize",
        "preflight",
        "preflight_delivery",
        "fresh_head",
        "scene",
        "fresh_wrist",
        "verify_wrist_start",
        "grasp_lift",
        "dispense_side_table",
        "refresh_output_scene",
        "return_home",
        "right_arm_home_check",
        "return_body",
    ]
    assert result.status == RunStatus.DONE.value
    assert result.object_state == ObjectState.EMPTY.value


def test_delivery_mode_must_match_the_declared_dispense_flag_before_hardware(
    tmp_path,
):
    demo = FakeDemo(tmp_path, StartMode.FROM_OBSERVATION)
    demo.args.dispense = True

    with pytest.raises(SafetyAbort, match="DeliverMode 不一致"):
        BottlePickPlaceTask(demo).run(
            StartMode.FROM_OBSERVATION, DeliverMode.PLACE_BACK
        )

    assert _hardware_calls(demo) == []


def test_unknown_delivery_mode_is_rejected_before_hardware(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_OBSERVATION)

    with pytest.raises(SafetyAbort, match="未知送货模式"):
        BottlePickPlaceTask(demo).run(
            StartMode.FROM_OBSERVATION, "sideways"
        )

    assert _hardware_calls(demo) == []



def _pregrasp_verification_demo(distance_m):
    demo = BottleDemo.__new__(BottleDemo)
    demo.params = DemoParams()
    demo.stage = lambda *_args: None
    tcp = np.eye(4)
    tcp[0, 3] = distance_m

    class Robot:
        @staticmethod
        def assert_arm_healthy():
            pass

        @staticmethod
        def current_tcp():
            return tcp

    class Safety:
        @staticmethod
        def assert_tcp_point(_point, *, label):
            assert label

    planned = []
    demo.robot = Robot()
    demo.safety = Safety()
    demo.candidate_path = lambda point: planned.append(list(point))
    return demo, planned


def test_pregrasp_entry_accepts_only_the_measured_hover_distance():
    demo, planned = _pregrasp_verification_demo(0.085)
    target = _localization([0.0, 0.0, 0.0])

    demo._verify_wrist_pregrasp_start(target)

    assert planned == [[0.0, 0.0, 0.0]]
    np.testing.assert_allclose(demo.local_contact_target_base, [0.0, 0.0, 0.0])


def test_pregrasp_entry_rejects_a_pose_that_was_moved_after_abort():
    demo, planned = _pregrasp_verification_demo(0.13)

    with pytest.raises(SafetyAbort, match="不在预抓取悬停位"):
        demo._verify_wrist_pregrasp_start(_localization([0.0, 0.0, 0.0]))

    assert planned == []


def test_from_start_adds_only_transfer_prefix_and_home_suffix(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_START)

    result = BottlePickPlaceTask(demo).run(StartMode.FROM_START)

    assert _hardware_calls(demo) == [
        "initialize",
        "preflight",
        "fresh_head",
        "scene",
        "normalize_start_home",
        "plan_observation",
        "refresh_validate_observation",
        "execute_observation",
        "fresh_wrist",
        "verify_wrist_start",
        "grasp_lift",
        "place_release_retreat",
        "refresh_return_scene",
        "return_home",
    ]
    assert result.status == RunStatus.DONE.value
    assert result.object_state == ObjectState.EMPTY.value


def test_from_natural_hang_uses_staging_then_reacquires_scene(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_START)
    demo.safety.observation_staging_joints_deg = tuple([10.0] * 7)

    result = BottlePickPlaceTask(demo).run(StartMode.FROM_START)

    assert _hardware_calls(demo) == [
        "initialize",
        "preflight",
        "fresh_head",
        "scene",
        "normalize_start_home",
        "plan_observation_staging",
        "refresh_validate_staging",
        "execute_staging",
        # The arm occupied a different part of the camera scene after the
        # departure motion, so observation planning must use a fresh lock.
        "fresh_head",
        "scene",
        "plan_observation",
        "refresh_validate_observation",
        "execute_observation",
        "fresh_wrist",
        "verify_wrist_start",
        "grasp_lift",
        "place_release_retreat",
        "refresh_return_scene",
        "return_home",
    ]
    assert result.status == RunStatus.DONE.value


def test_from_non_home_reacquires_target_and_scene_before_observation(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_START)
    demo.start_home_moved = True

    result = BottlePickPlaceTask(demo).run(StartMode.FROM_START)

    assert _hardware_calls(demo)[:8] == [
        "initialize",
        "preflight",
        "fresh_head",
        "scene",
        "normalize_start_home",
        "fresh_head",
        "scene",
        "plan_observation",
    ]
    assert result.status == RunStatus.DONE.value


def test_from_start_can_stop_after_real_observation_without_grasp(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_START)
    demo.args.stop_after_observation = True

    result = BottlePickPlaceTask(demo).run(StartMode.FROM_START)

    assert _hardware_calls(demo) == [
        "initialize",
        "preflight",
        "fresh_head",
        "scene",
        "normalize_start_home",
        "plan_observation",
        "refresh_validate_observation",
        "execute_observation",
        "fresh_wrist",
        "verify_wrist_start",
    ]
    assert result.status == RunStatus.DONE.value
    assert result.object_state == ObjectState.EMPTY.value


def test_stop_after_with_body_guard_keeps_the_gripper_untouched(tmp_path):
    """Exercise actual SHELF_READY before the real gripper preflight branch."""
    demo = FakeDemo(tmp_path, StartMode.FROM_OBSERVATION)
    demo.args.execute = True
    demo.args.dispense = True
    demo.args.stop_after_observation = True
    chassis = _wire_real_shelf_ready_gate(
        demo, tmp_path, chassis_yaw_deg=0.0
    )
    gripper_calls = []

    class Robot:
        @staticmethod
        def recover_transient_joint_frame_loss():
            return []

        @staticmethod
        def assert_arm_healthy():
            return {"controller": {"return_code": 0, "system_error": 0}}

        @staticmethod
        def current_tcp():
            return np.eye(4)

        @staticmethod
        def controller_fence_status():
            return {"state": {"enable_state": False}, "current": (0, {}), "saved": {"len": 0}}

        @staticmethod
        def gripper_state():
            return {"enable_state": True, "pos": [902]}

        @staticmethod
        def close_empty_gripper(_params):
            gripper_calls.append("close_empty")

    demo.robot = Robot()
    demo._is_read_only_vision_check = lambda: False
    demo._preflight = lambda: BottleDemo._preflight(demo)

    result = BottlePickPlaceTask(demo).run(
        StartMode.FROM_OBSERVATION, DeliverMode.DISPENSE
    )

    calls = _hardware_calls(demo)
    assert calls[:4] == [
        "capture_shelf_ready",
        "initialize",
        "preflight_delivery",
        "fresh_head",
    ]
    assert gripper_calls == []
    assert "grasp_lift" not in calls
    assert "dispense_side_table" not in calls
    assert "return_home" not in calls
    assert "return_body" not in calls
    assert chassis.prepared == 1
    assert result.status == RunStatus.DONE.value
    assert result.object_state == ObjectState.EMPTY.value


def test_shelf_ready_92_degree_guard_aborts_before_any_arm_or_gripper_path(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_OBSERVATION)
    demo.args.execute = True
    demo.args.dispense = True
    chassis = _wire_real_shelf_ready_gate(
        demo, tmp_path, chassis_yaw_deg=92.0
    )

    with pytest.raises(SafetyAbort, match="SHELF_READY yaw 超出.*容差"):
        BottlePickPlaceTask(demo).run(
            StartMode.FROM_OBSERVATION, DeliverMode.DISPENSE
        )

    assert _hardware_calls(demo) == ["capture_shelf_ready"]
    assert chassis.stops == 3


def test_held_or_unknown_failure_never_homes_or_returns_the_body(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_OBSERVATION)

    def fail_after_lift(_lifted, *, start):
        assert start is demo.shelf_ready_snapshot
        demo.calls.append(("dispense_started",))
        raise SafetyAbort("release evidence unavailable")

    demo._dispense_to_side_table = fail_after_lift
    task = BottlePickPlaceTask(demo)

    with pytest.raises(SafetyAbort, match="release evidence unavailable"):
        task.run(StartMode.FROM_OBSERVATION, DeliverMode.DISPENSE)

    calls = _hardware_calls(demo)
    assert "grasp_lift" in calls
    assert "return_home" not in calls
    assert "right_arm_home_check" not in calls
    assert "return_body" not in calls
    assert task.object_state is ObjectState.UNKNOWN
    assert task.phase is TaskPhase.ABORTED


def test_done_is_impossible_until_body_and_lift_restore_succeeds(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_OBSERVATION)

    def fail_restore(*, start, authorization):
        assert start is demo.shelf_ready_snapshot
        assert authorization.object_state == "empty"
        demo.calls.append(("return_body",))
        raise SafetyAbort("body/lift restore failed")

    demo._return_body_to_shelf_ready = fail_restore
    task = BottlePickPlaceTask(demo)

    with pytest.raises(SafetyAbort, match="body/lift restore failed"):
        task.run(StartMode.FROM_OBSERVATION, DeliverMode.DISPENSE)

    assert task.status is RunStatus.SAFE_ABORT
    assert task.phase is TaskPhase.ABORTED
    journal = (tmp_path / "task_journal.jsonl").read_text(encoding="utf-8")
    assert '"phase": "shelf_restored"' not in journal
    assert '"phase": "done"' not in journal


def test_stop_after_rejects_pregrasp_entry_before_any_hardware_touch(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_PREGRASP)
    demo.args.stop_after_observation = True

    with pytest.raises(SafetyAbort, match="已越过观察位"):
        BottlePickPlaceTask(demo).run(StartMode.FROM_PREGRASP)

    assert demo.calls == []


def test_confirm_before_grasp_is_a_task_fsm_gate_before_any_gripper_path(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_OBSERVATION)
    demo.args.confirm_before_grasp = True
    task = BottlePickPlaceTask(demo)

    def confirm():
        # The confirmation is a legal FSM state while the gripper is still
        # empty; a later failure cannot make this evidence claim HELD.
        assert task.phase is TaskPhase.CONFIRM_BEFORE_GRASP
        assert task.object_state is ObjectState.EMPTY
        demo.calls.append(("confirm_before_grasp",))

    demo._wait_for_grasp_confirmation = confirm
    result = task.run(StartMode.FROM_OBSERVATION)

    calls = _hardware_calls(demo)
    assert calls.index("confirm_before_grasp") < calls.index("grasp_lift")
    assert result.status == RunStatus.DONE.value
    assert result.object_state == ObjectState.EMPTY.value
    journal = (tmp_path / "task_journal.jsonl").read_text(encoding="utf-8")
    assert '"phase": "confirm_before_grasp"' in journal


def test_failure_during_grasp_is_recorded_as_unknown_not_success(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_OBSERVATION)

    def fail(_target):
        raise SafetyAbort("captured hardware failure")

    demo._grasp_and_lift = fail
    task = BottlePickPlaceTask(demo)

    with pytest.raises(SafetyAbort, match="captured hardware failure"):
        task.run(StartMode.FROM_OBSERVATION)

    assert task.status is RunStatus.SAFE_ABORT
    assert task.phase is TaskPhase.ABORTED
    assert task.object_state is ObjectState.UNKNOWN
    assert '"status": "safe_abort"' in (
        tmp_path / "task_result.json"
    ).read_text(encoding="utf-8")


def test_failed_independent_lift_confirmation_cannot_place_or_reach_done(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_OBSERVATION)

    def reject_false_grasp(_target):
        raise SafetyAbort("抬升三维确认失败：瓶子仍在原桌面点")

    demo._grasp_and_lift = reject_false_grasp
    task = BottlePickPlaceTask(demo)

    with pytest.raises(SafetyAbort, match="抬升三维确认失败"):
        task.run(StartMode.FROM_OBSERVATION)

    assert task.status is RunStatus.SAFE_ABORT
    assert task.object_state is ObjectState.UNKNOWN
    assert "place_release_retreat" not in _hardware_calls(demo)
    assert task.phase is TaskPhase.ABORTED


def test_task_rejects_parser_and_runtime_mode_disagreement(tmp_path):
    demo = FakeDemo(tmp_path, StartMode.FROM_START)

    with pytest.raises(SafetyAbort, match="任务模式在解析后发生变化"):
        BottlePickPlaceTask(demo).run(StartMode.FROM_OBSERVATION)

    assert demo.calls == []


def test_two_runs_started_at_the_same_timestamp_get_distinct_evidence_dirs(
    monkeypatch, tmp_path
):
    class FrozenDateTime:
        @staticmethod
        def now():
            return SimpleNamespace(
                strftime=lambda _format: "20260718_203207"
            )

    monkeypatch.setattr(demo_module, "datetime", FrozenDateTime)
    args = SimpleNamespace(
        config=str(tmp_path / "config.yaml"),
        output_dir=str(tmp_path / "outputs"),
    )
    first = BottleDemo(args, None)
    second = BottleDemo(args, None)
    try:
        assert first.run_dir != second.run_dir
    finally:
        for demo in (first, second):
            logging.getLogger().removeHandler(demo.run_log_handler)
            demo.run_log_handler.close()
