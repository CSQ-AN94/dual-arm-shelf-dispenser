from types import SimpleNamespace
import json
import math
import threading

import numpy as np
import pytest
import bottle_grasp.mobile_body as mobile_body_module

from bottle_grasp.core import SafetyAbort
from bottle_grasp.mobile_body import (
    BodySnapshot,
    ChassisState,
    LiftState,
    MobileBodyCoordinator,
    ReturnAuthorization,
    WooshChassisAdapter,
    wrap_angle_rad,
)


def _chassis(
    x=1.0,
    y=2.0,
    yaw=0.2,
    *,
    linear=0.0,
    angular=0.0,
    control_mode="kAuto",
    robot_state="kIdle",
):
    return ChassisState(
        x_m=x,
        y_m=y,
        yaw_rad=yaw,
        linear_mps=linear,
        angular_radps=angular,
        control_mode=control_mode,
        robot_state=robot_state,
        captured_monotonic=1.0,
    )


def _lift(height=716, *, enabled=True, error_flag=0):
    return LiftState(
        height_mm=height,
        enabled=enabled,
        error_flag=error_flag,
        mode=0,
        captured_monotonic=1.0,
    )


def test_world_platform_transform_applies_lift_once_without_camera_extrinsic():
    snapshot = BodySnapshot(chassis=_chassis(), lift=_lift(900))
    transform = snapshot.world_from_platform(reference_lift_height_mm=700)

    assert transform[2, 3] == pytest.approx(0.2)
    assert transform[0, 3] == pytest.approx(1.0)
    assert transform[1, 3] == pytest.approx(2.0)
    # There is intentionally no camera transform input to this interface.
    np.testing.assert_allclose(
        transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12
    )


def test_woosh_pose_parser_reads_live_pose_and_twist():
    state = WooshChassisAdapter._parse_pose(
        "twist linear=0.000 angular=0.000\n"
        "pose x=1.25 y=-0.4 theta=3.10 mileage=0 robot_id=10001\n",
        mode="kAuto",
        state="kIdle",
    )
    assert state.x_m == pytest.approx(1.25)
    assert state.y_m == pytest.approx(-0.4)
    assert state.yaw_rad == pytest.approx(3.10)


def test_woosh_motion_adapter_forwards_shared_stop_as_sigint(monkeypatch):
    event = threading.Event()
    event.set()

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, value):
            assert value == mobile_body_module.signal.SIGINT
            self.returncode = 130

        def communicate(self, timeout):
            return "", "stopped"

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr(
        mobile_body_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    adapter = WooshChassisAdapter(stop_event=event)

    with pytest.raises(SafetyAbort, match="用户停止"):
        adapter._run_motion(["unused"], 1.0)


def test_woosh_motion_timeout_is_not_reinterpreted_as_a_success(monkeypatch):
    signals = []

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, value):
            signals.append(value)
            self.returncode = 0

        def communicate(self, timeout):
            return "", ""

        def terminate(self):
            self.returncode = -15

    monotonic_values = iter((10.0, 11.1))
    monkeypatch.setattr(
        mobile_body_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        mobile_body_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    adapter = WooshChassisAdapter()

    with pytest.raises(SafetyAbort, match="旋转超时"):
        adapter._run_motion(["unused"], 1.0)

    assert signals == [mobile_body_module.signal.SIGINT]


def test_woosh_preflight_initializes_uninit_then_requires_fresh_idle(monkeypatch):
    adapter = WooshChassisAdapter()
    calls = []
    diagnostics = iter(
        [
            "[Mode] ok=true msg=ok\nctrl: kAuto\n[RobotState] ok=true msg=ok\nstate: kUninit\n",
            "[Mode] ok=true msg=ok\nctrl: kAuto\n[RobotState] ok=true msg=ok\nstate: kIdle\n",
        ]
    )

    def fake_run(command, _timeout):
        calls.append(command)
        if command == [adapter.diagnostic_path]:
            return next(diagnostics)
        if command == [adapter.init_helper_path]:
            return "init-record-true ok=true"
        if command == [adapter.pose_query_path]:
            return "twist linear=0 angular=0\npose x=1 y=2 theta=0.3\n"
        raise AssertionError(command)

    monkeypatch.setattr(adapter, "assert_ready", lambda: None)
    monkeypatch.setattr(adapter, "_run", fake_run)

    assert adapter.prepare_for_motion() is True
    assert [adapter.init_helper_path] in calls


def test_woosh_preflight_reacquires_clean_session_when_already_idle(monkeypatch):
    adapter = WooshChassisAdapter()
    calls = []
    diagnostics = iter(
        [
            "[Mode] ok=true msg=ok\nctrl: kAuto\n[RobotState] ok=true msg=ok\nstate: kIdle\n",
            "[Mode] ok=true msg=ok\nctrl: kAuto\n[RobotState] ok=true msg=ok\nstate: kIdle\n",
        ]
    )

    def fake_run(command, _timeout):
        calls.append(command)
        if command == [adapter.diagnostic_path]:
            return next(diagnostics)
        if command == [adapter.init_helper_path]:
            return "init-record-true ok=true"
        if command == [adapter.pose_query_path]:
            return "twist linear=0 angular=0\npose x=1 y=2 theta=0.3\n"
        raise AssertionError(command)

    monkeypatch.setattr(adapter, "assert_ready", lambda: None)
    monkeypatch.setattr(adapter, "_run", fake_run)

    assert adapter.prepare_for_motion() is True
    assert calls.count([adapter.init_helper_path]) == 1


def test_woosh_preflight_rejects_fault_without_initializing(monkeypatch):
    adapter = WooshChassisAdapter()
    calls = []

    def fake_run(command, _timeout):
        calls.append(command)
        return "[Mode] ok=true msg=ok\nctrl: kAuto\n[RobotState] ok=true msg=ok\nstate: kFault\n"

    monkeypatch.setattr(adapter, "assert_ready", lambda: None)
    monkeypatch.setattr(adapter, "_run", fake_run)

    with pytest.raises(SafetyAbort, match="既不是 kUninit 也不是 kIdle"):
        adapter.prepare_for_motion()
    assert [adapter.init_helper_path] not in calls


class FakeLift:
    def __init__(self, *, fail_target=None):
        self.current = _lift()
        self.moves = []
        self.fail_target = fail_target

    def state(self):
        return self.current

    def move_to(self, height_mm, *, speed):
        self.moves.append((height_mm, speed))
        if height_mm != self.fail_target:
            self.current = _lift(height_mm)
        return self.current


class FakeChassis:
    def __init__(self, translation=0.0, *, yaw=0.2, rotate_error=None):
        self.current = _chassis(yaw=yaw)
        self.translation = translation
        self.rotate_error = rotate_error
        self.rotations = []
        self.stops = 0
        self.prepares = 0

    def state(self):
        return self.current

    def prepare_for_motion(self):
        self.prepares += 1
        return True

    def rotate_relative(self, yaw_rad, **kwargs):
        self.rotations.append((yaw_rad, kwargs))
        if self.rotate_error is not None:
            raise self.rotate_error
        self.current = _chassis(
            x=self.current.x_m + self.translation,
            y=self.current.y_m,
            yaw=self.current.yaw_rad + yaw_rad,
        )
        return self.current

    def stop(self):
        self.stops += 1


def _config(*, yaw=0.2, turn_deg=-90.0, target_lift=900):
    return SimpleNamespace(
        transport_pose_verified=True,
        shelf_ready=SimpleNamespace(
            x_m=1.0,
            y_m=2.0,
            yaw_deg=math.degrees(yaw),
            lift_height_mm=716,
            xy_tolerance_m=0.02,
            yaw_tolerance_deg=2.0,
            lift_tolerance_mm=5,
        ),
        shelf_ready_verified=True,
        source_lift_height_mm=716,
        target_lift_height_mm=target_lift,
        target_lift_tolerance_mm=5,
        lift_transition_verified=True,
        body_lift_speed=15,
        body_rotation_yaw_deg=turn_deg,
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
        table_roi_verified=True,
        workspace_verified=True,
        keepouts_verified=True,
        bottle_tcp_verified=True,
    )


def _coordinator(tmp_path, chassis=None, lift=None):
    return MobileBodyCoordinator(
        chassis=chassis or FakeChassis(),
        lift=lift or FakeLift(),
        stop_event=threading.Event(),
        evidence_dir=tmp_path,
    )


def _authorized(**overrides):
    data = {
        "release_verified": True,
        "object_state": "empty",
        "right_arm_compact_or_home": True,
        "left_arm_stable": True,
    }
    data.update(overrides)
    return ReturnAuthorization(**data)


def test_coordinator_lifts_then_commands_rotation_only_and_verifies_state(tmp_path):
    chassis, lift = FakeChassis(), FakeLift()
    coordinator = _coordinator(tmp_path, chassis, lift)
    start = coordinator.capture_shelf_ready(_config())

    result = coordinator.position_for_delivery(_config(), start=start)

    assert lift.moves == [(900, 15)]
    assert chassis.rotations[0][0] == pytest.approx(-math.pi / 2)
    assert result.lift.height_mm == 900
    assert (tmp_path / "delivery_body_state.json").is_file()
    shelf_evidence = json.loads(
        (tmp_path / "shelf_ready_body_snapshot.json").read_text()
    )
    assert shelf_evidence["profile_contract"]["shelf_ready"]["yaw_deg"] == pytest.approx(
        math.degrees(0.2)
    )


def test_coordinator_rejects_translation_during_nominal_in_place_turn(tmp_path):
    chassis = FakeChassis(translation=0.06)
    coordinator = _coordinator(tmp_path, chassis, FakeLift())
    config = _config()
    start = coordinator.capture_shelf_ready(config)

    with pytest.raises(SafetyAbort, match="过大平移"):
        coordinator.position_for_delivery(config, start=start)
    assert chassis.stops >= 3


def test_outbound_turn_uses_the_stricter_shelf_restore_translation_limit(tmp_path):
    chassis = FakeChassis(translation=0.03)
    coordinator = _coordinator(tmp_path, chassis, FakeLift())
    config = _config()
    start = coordinator.capture_shelf_ready(config)

    with pytest.raises(SafetyAbort, match="过大平移"):
        coordinator.position_for_delivery(config, start=start)

    assert chassis.rotations[0][1]["max_translation_m"] == pytest.approx(0.02)


def test_shelf_ready_guard_rejects_92_degree_yaw_before_lift_or_turn(tmp_path):
    chassis = FakeChassis(yaw=math.radians(92.0))
    lift = FakeLift()
    coordinator = _coordinator(tmp_path, chassis, lift)

    with pytest.raises(SafetyAbort, match="SHELF_READY yaw"):
        coordinator.capture_shelf_ready(_config(yaw=0.0))

    assert lift.moves == []
    assert chassis.rotations == []
    assert chassis.stops >= 3


def test_unverified_delivery_contract_is_rejected_before_body_preflight(tmp_path):
    chassis = FakeChassis()
    coordinator = _coordinator(tmp_path, chassis, FakeLift())
    config = _config()
    config.shelf_ready_verified = False

    with pytest.raises(SafetyAbort, match="shelf_ready_verified"):
        coordinator.capture_shelf_ready(config)

    assert chassis.prepares == 0
    assert chassis.rotations == []


def test_position_without_a_captured_shelf_snapshot_fails_closed(tmp_path):
    chassis = FakeChassis()
    coordinator = _coordinator(tmp_path, chassis, FakeLift())

    with pytest.raises(SafetyAbort, match="capture_shelf_ready"):
        coordinator.position_for_delivery(_config())

    assert chassis.rotations == []


def test_forged_or_cross_coordinator_snapshot_cannot_bypass_preflight(tmp_path):
    chassis, lift = FakeChassis(), FakeLift()
    coordinator = _coordinator(tmp_path, chassis, lift)
    config = _config()
    captured = coordinator.capture_shelf_ready(config)
    forged = BodySnapshot(chassis=captured.chassis, lift=captured.lift)

    with pytest.raises(SafetyAbort, match="本 coordinator.*原始 BodySnapshot"):
        coordinator.position_for_delivery(config, start=forged)

    # No lift or turn may be sent when a caller manufactures a value-equal
    # snapshot instead of using the token returned by the real admission.
    assert lift.moves == []
    assert chassis.rotations == []


def test_return_rejects_a_value_equal_snapshot_not_issued_by_this_coordinator(
    tmp_path,
):
    chassis, lift = FakeChassis(), FakeLift()
    coordinator = _coordinator(tmp_path, chassis, lift)
    config = _config()
    captured = coordinator.capture_shelf_ready(config)
    coordinator.position_for_delivery(config, start=captured)
    forged = BodySnapshot(chassis=captured.chassis, lift=captured.lift)
    moves_before = list(lift.moves)
    turns_before = list(chassis.rotations)

    with pytest.raises(SafetyAbort, match="本 coordinator.*原始 BodySnapshot"):
        coordinator.return_to_shelf_ready(
            config, start=forged, authorization=_authorized()
        )

    assert lift.moves == moves_before
    assert chassis.rotations == turns_before


def test_return_requires_a_successful_outbound_body_transition(tmp_path):
    chassis, lift = FakeChassis(), FakeLift()
    coordinator = _coordinator(tmp_path, chassis, lift)
    config = _config()
    start = coordinator.capture_shelf_ready(config)

    with pytest.raises(SafetyAbort, match="成功完成的送桌 body 操作"):
        coordinator.return_to_shelf_ready(
            config, start=start, authorization=_authorized()
        )

    assert lift.moves == []
    assert chassis.rotations == []


def test_shelf_ready_and_return_handle_yaw_wraparound(tmp_path):
    initial_yaw = math.radians(179.0)
    chassis = FakeChassis(yaw=initial_yaw)
    coordinator = _coordinator(tmp_path, chassis, FakeLift())
    config = _config(yaw=initial_yaw, turn_deg=90.0)
    start = coordinator.capture_shelf_ready(config)

    side = coordinator.position_for_delivery(config, start=start)
    restored = coordinator.return_to_shelf_ready(
        config, start=start, authorization=_authorized()
    )

    assert wrap_angle_rad(side.chassis.yaw_rad - math.radians(-91.0)) == pytest.approx(0.0)
    assert wrap_angle_rad(restored.chassis.yaw_rad - initial_yaw) == pytest.approx(0.0)
    assert restored.lift.height_mm == 716


def test_return_requires_verified_release_empty_and_stable_arms(tmp_path):
    blocked = (
        _authorized(release_verified=False),
        _authorized(object_state="held"),
        _authorized(object_state="unknown"),
        _authorized(right_arm_compact_or_home=False),
        _authorized(left_arm_stable=False),
    )
    for index, authorization in enumerate(blocked):
        # A failed return clears the coordinator's capture token by design.
        # Give every authorization predicate an independent real outbound
        # transition so this test cannot pass merely on a stale-token reject.
        chassis = FakeChassis()
        coordinator = _coordinator(
            tmp_path / str(index), chassis, FakeLift()
        )
        config = _config()
        start = coordinator.capture_shelf_ready(config)
        coordinator.position_for_delivery(config, start=start)
        turns_before = len(chassis.rotations)

        with pytest.raises(SafetyAbort):
            coordinator.return_to_shelf_ready(
                config, start=start, authorization=authorization
            )

        assert len(chassis.rotations) == turns_before
        assert chassis.stops >= 3


def test_return_refuses_unrestorable_translation_before_commanding_reverse_turn(tmp_path):
    chassis = FakeChassis()
    coordinator = _coordinator(tmp_path, chassis, FakeLift())
    config = _config()
    start = coordinator.capture_shelf_ready(config)
    coordinator.position_for_delivery(config, start=start)
    chassis.current = _chassis(
        # This is below the general 35 mm sweep cap, but already outside the
        # 20 mm SHELF_READY acceptance band and cannot be corrected by a
        # rotation-only adapter.
        x=1.03,
        y=2.0,
        yaw=chassis.current.yaw_rad,
    )
    turns_before = len(chassis.rotations)

    with pytest.raises(SafetyAbort, match="产生过大平移"):
        coordinator.return_to_shelf_ready(
            config, start=start, authorization=_authorized()
        )

    assert len(chassis.rotations) == turns_before


def test_timeout_or_rotation_error_emits_repeated_zero_stops(tmp_path):
    chassis = FakeChassis(rotate_error=SafetyAbort("controller timeout"))
    coordinator = _coordinator(tmp_path, chassis, FakeLift())
    config = _config()
    start = coordinator.capture_shelf_ready(config)

    with pytest.raises(SafetyAbort, match="controller timeout"):
        coordinator.position_for_delivery(config, start=start)

    assert chassis.stops >= 3


def test_stop_request_prevents_body_motion_and_emits_zero_stops(tmp_path):
    chassis = FakeChassis()
    event = threading.Event()
    event.set()
    coordinator = MobileBodyCoordinator(
        chassis=chassis,
        lift=FakeLift(),
        stop_event=event,
        evidence_dir=tmp_path,
    )

    with pytest.raises(SafetyAbort, match="用户停止"):
        coordinator.capture_shelf_ready(_config())

    assert chassis.rotations == []
    assert chassis.stops >= 3


def test_controller_state_error_and_close_are_fail_closed_and_idempotent(tmp_path):
    chassis = FakeChassis()
    chassis.current = _chassis(robot_state="kFault")
    coordinator = _coordinator(tmp_path, chassis, FakeLift())

    with pytest.raises(SafetyAbort, match="kIdle"):
        coordinator.capture_shelf_ready(_config())
    stops_after_error = chassis.stops
    coordinator.close()
    coordinator.close()

    assert stops_after_error >= 3
    assert chassis.stops == stops_after_error + 6


def test_two_body_round_trips_do_not_accumulate_pose_or_lift_error(tmp_path):
    chassis = FakeChassis(yaw=0.0)
    lift = FakeLift()
    coordinator = _coordinator(tmp_path, chassis, lift)
    config = _config(yaw=0.0, turn_deg=-90.0)

    for _ in range(2):
        start = coordinator.capture_shelf_ready(config)
        coordinator.position_for_delivery(config, start=start)
        restored = coordinator.return_to_shelf_ready(
            config, start=start, authorization=_authorized()
        )
        assert restored.chassis.x_m == pytest.approx(1.0)
        assert restored.chassis.y_m == pytest.approx(2.0)
        assert wrap_angle_rad(restored.chassis.yaw_rad) == pytest.approx(0.0)
        assert restored.lift.height_mm == 716

    assert [round(math.degrees(item[0])) for item in chassis.rotations] == [
        -90,
        90,
        -90,
        90,
    ]


def test_lift_not_restored_prevents_a_verified_return_snapshot(tmp_path):
    chassis = FakeChassis()
    lift = FakeLift(fail_target=716)
    coordinator = _coordinator(tmp_path, chassis, lift)
    config = _config()
    start = coordinator.capture_shelf_ready(config)
    coordinator.position_for_delivery(config, start=start)

    with pytest.raises(SafetyAbort, match="返程升降 未到位"):
        coordinator.return_to_shelf_ready(
            config, start=start, authorization=_authorized()
        )

    assert not (tmp_path / "return_body_state.json").exists()
