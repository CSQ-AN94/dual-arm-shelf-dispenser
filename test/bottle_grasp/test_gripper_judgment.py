"""close_gripper grasp/empty judgment against the measured baseline.

2026-07-15 real-hardware finding: a genuinely held narrow metal bottle closed
at pos=402, only +8 over the empty baseline 394, and the old static threshold
(394 + margin 35) misreported it as an empty grasp. The judgment now uses a
per-run measured baseline (calibrate_empty_close) and a small margin.
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bottle_grasp.core import DemoParams, SafetyAbort
from bottle_grasp.robot import RobotSession


def _session(feedback: dict, baseline: int | None):
    session = RobotSession.__new__(RobotSession)
    session.take_control = True
    if baseline is not None:
        session.empty_close_pos = baseline
    session._command_gripper_position = lambda *a, **k: feedback
    return session


def _feedback(pos: int, state: int = 3, current: int = 120) -> dict:
    return {"dof_state": [state], "pos": [pos], "current": [current]}


class _FakeArm:
    def rm_set_hand_force(self, force):
        return 0

    def rm_set_hand_speed(self, speed):
        return 0

    def rm_set_hand_follow_pos(self, command, blocking):
        return 0


def test_gripper_state_retries_transient_rm_plus_startup(monkeypatch):
    session = RobotSession.__new__(RobotSession)
    states = iter(
        [
            (1, {}),
            (1, {}),
            (
                0,
                {
                    "sys_state": 0,
                    "dof_state": [2],
                    "dof_err": [0],
                    "pos": [0],
                },
            ),
        ]
    )
    session.arm = type(
        "TransientArm",
        (),
        {"rm_get_rm_plus_state_info": lambda _self: next(states)},
    )()
    monkeypatch.setattr("bottle_grasp.robot.time.sleep", lambda _: None)

    assert session.gripper_state()["dof_state"] == [2]


def test_open_waits_through_state_6_while_gripper_is_still_moving(monkeypatch):
    """Replay the 2026-07-17 real feedback that aborted reopening at pos=728."""
    session = RobotSession.__new__(RobotSession)
    session.arm = _FakeArm()
    session.stop_event = threading.Event()
    states = iter(
        [
            {"dof_state": [3], "pos": [0], "speed": [0]},
            {"dof_state": [6], "pos": [728], "speed": [74]},
            {"dof_state": [2], "pos": [900], "speed": [0]},
            {"dof_state": [2], "pos": [901], "speed": [0]},
            {"dof_state": [2], "pos": [902], "speed": [0]},
        ]
    )
    session.gripper_state = lambda: next(states)
    monkeypatch.setattr("bottle_grasp.robot.time.sleep", lambda _: None)

    result = session._command_gripper_position(
        900,
        speed=100,
        force=None,
    )

    assert result["dof_state"] == [2]
    assert result["pos"] == [902]


def test_persistent_stopped_state_6_is_still_a_fault(monkeypatch):
    session = RobotSession.__new__(RobotSession)
    session.arm = _FakeArm()
    session.stop_event = threading.Event()
    states = [
        {"dof_state": [3], "pos": [0], "speed": [0]},
        {"dof_state": [6], "pos": [728], "speed": [0]},
        {"dof_state": [6], "pos": [728], "speed": [0]},
        {"dof_state": [6], "pos": [728], "speed": [0]},
    ]
    reads = 0

    def read_state():
        nonlocal reads
        state = states[min(reads, len(states) - 1)]
        reads += 1
        return state

    session.gripper_state = read_state
    monkeypatch.setattr("bottle_grasp.robot.time.sleep", lambda _: None)

    with pytest.raises(SafetyAbort, match="连续 3 帧"):
        session._command_gripper_position(900, speed=100, force=None)

    assert reads == 4


def _calibration_session(feedbacks: list[dict]):
    session = RobotSession.__new__(RobotSession)
    session.take_control = True
    remaining = iter(feedbacks)
    session._command_gripper_position = lambda *a, **k: next(remaining)
    return session


def test_empty_close_calibration_accepts_zero_measured_baseline():
    # 2026-07-17 real hardware: open feedback was 902 and an unobstructed
    # empty close reached 0. A stale static reference of 394 must not arbitrate
    # a per-run dynamic calibration.
    session = _calibration_session(
        [_feedback(902, state=2), _feedback(0), _feedback(901, state=2)]
    )

    assert session.calibrate_empty_close(DemoParams()) == 0
    assert session.empty_close_pos == 0


def test_empty_close_calibration_rejects_when_gripper_barely_moves():
    session = _calibration_session(
        [_feedback(902, state=2), _feedback(850), _feedback(901, state=2)]
    )

    with pytest.raises(SafetyAbort, match="闭合行程不足"):
        session.calibrate_empty_close(DemoParams())


def test_narrow_object_over_measured_baseline_is_a_grasp():
    # The 2026-07-15 real case: baseline 394, closed at 402 while holding.
    session = _session(_feedback(402), baseline=394)
    state = session.close_gripper(DemoParams())
    assert int(state["pos"][0]) == 402


def test_close_at_baseline_is_still_an_empty_grasp():
    session = _session(_feedback(395), baseline=394)
    with pytest.raises(SafetyAbort, match="空夹"):
        session.close_gripper(DemoParams())


def test_static_fallback_baseline_used_without_calibration():
    session = _session(_feedback(399), baseline=None)
    with pytest.raises(SafetyAbort, match="空夹基线=394"):
        session.close_gripper(DemoParams())


def test_no_internal_force_reached_aborts():
    session = _session(_feedback(500, state=2), baseline=394)
    with pytest.raises(SafetyAbort, match="夹持力"):
        session.close_gripper(DemoParams())


def test_empty_close_after_retreat_does_not_apply_object_grasp_judgment():
    session = _session(_feedback(0, state=3), baseline=0)

    state = session.close_empty_gripper(DemoParams())

    assert int(state["pos"][0]) == 0
