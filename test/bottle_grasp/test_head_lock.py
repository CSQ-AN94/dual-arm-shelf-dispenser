"""Head-servo reference-angle enforcement (no hardware).

2026-07-17: the head camera's hand-eye calibration (T_base_right_to_camera_head
in config.yaml) is only valid when the head servo sits at HEAD_REFERENCE
(pitch lowest, yaw centered). The demo must force it back there before doing
anything else, every run, regardless of why it might have drifted.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bottle_grasp.demo as demo_module
from bottle_grasp import head_lock
from bottle_grasp.core import SafetyAbort
from bottle_grasp.head_lock import (
    HEAD_REFERENCE,
    TOLERANCE,
    _decode_direct_angle_response,
    is_at_reference,
)


def test_is_at_reference_within_tolerance():
    assert is_at_reference(dict(HEAD_REFERENCE))
    drifted = {
        "angle1": HEAD_REFERENCE["angle1"] + TOLERANCE,
        "angle2": HEAD_REFERENCE["angle2"] - TOLERANCE,
    }
    assert is_at_reference(drifted)


def test_is_at_reference_rejects_drift_and_missing_reading():
    drifted = {
        "angle1": HEAD_REFERENCE["angle1"] + TOLERANCE + 1,
        "angle2": HEAD_REFERENCE["angle2"],
    }
    assert not is_at_reference(drifted)
    assert not is_at_reference(None)


def test_direct_angle_response_is_strictly_decoded():
    response = bytes.fromhex("55 55 09 15 02 01 8e 01 02 08 02")
    assert _decode_direct_angle_response(response) == {
        "angle1": 398,
        "angle2": 520,
    }
    assert _decode_direct_angle_response(response[:-1]) is None
    assert _decode_direct_angle_response(
        bytes.fromhex("55 55 08 15 02 01 8e 01 02 08 02")
    ) is None
    assert _decode_direct_angle_response(
        bytes.fromhex("55 55 09 15 02 03 8e 01 04 08 02")
    ) is None


def test_direct_angle_read_never_opens_a_busy_serial(monkeypatch):
    monkeypatch.setattr(head_lock, "_serial_owner_pids", lambda _path: {9876})
    monkeypatch.setattr(
        head_lock.os,
        "open",
        lambda *_args: pytest.fail("busy serial must not be opened"),
    )
    assert head_lock.read_current_angle_direct() is None


def test_direct_angle_read_ignores_missing_fallback_device(monkeypatch):
    response = bytes.fromhex("55 55 09 15 02 01 8e 01 02 08 02")
    monkeypatch.setattr(
        head_lock,
        "DIRECT_SERIAL_PATHS",
        ("/dev/rmUSB3", "/dev/ttyUSB0"),
    )
    monkeypatch.setattr(
        head_lock.Path,
        "exists",
        lambda path: str(path) == "/dev/rmUSB3",
    )
    monkeypatch.setattr(
        head_lock,
        "_serial_owner_pids",
        lambda path: set() if path == "/dev/rmUSB3" else None,
    )
    monkeypatch.setattr(head_lock.os, "open", lambda *_args: 42)
    monkeypatch.setattr(head_lock.fcntl, "ioctl", lambda *_args: None)
    monkeypatch.setattr(
        head_lock.termios,
        "tcgetattr",
        lambda _port: [0, 0, 0, 0, 0, 0, [0] * 32],
    )
    monkeypatch.setattr(head_lock.termios, "tcsetattr", lambda *_args: None)
    monkeypatch.setattr(head_lock.termios, "tcflush", lambda *_args: None)
    monkeypatch.setattr(head_lock.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(head_lock.os, "write", lambda *_args: len(response))
    monkeypatch.setattr(
        head_lock.select,
        "select",
        lambda *_args: ([42], [], []),
    )
    monkeypatch.setattr(head_lock.os, "read", lambda *_args: response)
    monkeypatch.setattr(head_lock.os, "close", lambda _port: None)

    assert head_lock.read_current_angle_direct() == {
        "angle1": 398,
        "angle2": 520,
    }


def _make_demo(*, execute):
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    demo.args = type("Args", (), {"execute": execute})()
    calls = []
    demo.stage = lambda name, msg="": calls.append((name, msg))
    return demo, calls


def test_already_at_reference_skips_restore(monkeypatch):
    demo, calls = _make_demo(execute=True)
    monkeypatch.setattr(
        demo_module.head_lock, "read_current_angle", lambda: dict(HEAD_REFERENCE)
    )
    called = []
    monkeypatch.setattr(
        demo_module.head_lock,
        "restore_reference",
        lambda: called.append("restore") or {"ok": True, "angle": {}, "steps": 0},
    )
    demo._ensure_head_reference()
    assert not called
    assert any("确认" in name for name, _ in calls)


def test_drifted_without_execute_warns_but_does_not_move(monkeypatch):
    demo, calls = _make_demo(execute=False)
    monkeypatch.setattr(
        demo_module.head_lock,
        "read_current_angle",
        lambda: {"angle1": 0, "angle2": 0},
    )
    called = []
    monkeypatch.setattr(
        demo_module.head_lock,
        "restore_reference",
        lambda: called.append("restore") or {"ok": True, "angle": {}, "steps": 0},
    )
    demo._ensure_head_reference()
    assert not called


def test_drifted_with_execute_restores(monkeypatch):
    demo, calls = _make_demo(execute=True)
    monkeypatch.setattr(
        demo_module.head_lock,
        "read_current_angle",
        lambda: {"angle1": 0, "angle2": 0},
    )
    monkeypatch.setattr(
        demo_module.head_lock,
        "restore_reference",
        lambda: {"ok": True, "angle": dict(HEAD_REFERENCE), "steps": 4},
    )
    demo._ensure_head_reference()
    assert any("已校正" in name for name, _ in calls)


def test_restore_failure_aborts(monkeypatch):
    demo, calls = _make_demo(execute=True)
    monkeypatch.setattr(
        demo_module.head_lock,
        "read_current_angle",
        lambda: {"angle1": 0, "angle2": 0},
    )
    monkeypatch.setattr(
        demo_module.head_lock,
        "restore_reference",
        lambda: {"ok": False, "angle": None, "reason": "没收到角度广播"},
    )
    with pytest.raises(SafetyAbort, match="头部无法回到标定基准角度"):
        demo._ensure_head_reference()


def test_resume_motion_restores_head_for_fallback_camera(monkeypatch):
    demo, calls = _make_demo(execute=True)
    demo.args.resume_at_wrist = True
    demo.args.stop_after_observation = False
    monkeypatch.setattr(
        demo_module.head_lock,
        "read_current_angle",
        lambda: {"angle1": 0, "angle2": 0},
    )
    restored = []
    monkeypatch.setattr(
        demo_module.head_lock,
        "restore_reference",
        lambda: restored.append(True)
        or {"ok": True, "angle": dict(HEAD_REFERENCE), "steps": 4},
    )

    demo._ensure_head_reference()

    assert restored == [True]


def test_read_only_resume_check_does_not_move_head(monkeypatch):
    demo, calls = _make_demo(execute=False)
    demo.args.resume_at_wrist = True
    demo.args.stop_after_observation = True
    monkeypatch.setattr(
        demo_module.head_lock,
        "read_current_angle",
        lambda: pytest.fail("read-only wrist check does not use the head camera"),
    )

    demo._ensure_head_reference()
