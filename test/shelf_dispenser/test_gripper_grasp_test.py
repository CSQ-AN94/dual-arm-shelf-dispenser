from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from shelf_dispenser.core import SafetyAbort

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gripper_grasp_test.py"
SPEC = importlib.util.spec_from_file_location("gripper_grasp_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeRobot:
    def __init__(self, grasp=None):
        self.calls = []
        self._grasp = grasp or {"pos": [420], "dof_state": [3], "current": [55]}

    def recover_transient_joint_frame_loss(self):
        self.calls.append("recover")

    def assert_arm_healthy(self):
        self.calls.append("healthy")

    def joints_deg(self):
        return [0.0] * 7

    def tcp_from_joints(self, joints):
        return MODULE.np.eye(4)

    def calibrate_empty_close(self, *args):
        self.calls.append(("calibrate", args))
        return 100

    def close_gripper(self, *args):
        self.calls.append(("close", args))
        if isinstance(self._grasp, SafetyAbort):
            raise self._grasp
        return self._grasp

    def open_gripper(self, *args):
        self.calls.append(("open", args))
        return {"pos": [900]}

    def close_empty_gripper(self, *args):
        self.calls.append(("retract", args))
        return {"pos": [0]}

    def close(self):
        self.calls.append("closed")


class FakeView:
    def assert_tcp_point(self, point, *, label):
        pass


def _run(monkeypatch, argv, answers, robot):
    monkeypatch.setattr(MODULE, "load_config", lambda _: object())
    monkeypatch.setattr(MODULE, "load_safety_profile", lambda *a, **k: object())
    monkeypatch.setattr(
        MODULE, "open_live_arm", lambda *a, **k: (robot, FakeView())
    )
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    return MODULE.main(argv)


def test_left_arm_grasp_cycle_never_serializes_demo_params(monkeypatch, capsys):
    """ArmProxy crosses JSON; every gripper call must go over with no args."""
    robot = FakeRobot()

    assert _run(monkeypatch, ["--arm", "left_arm"], ["y", "y", "y"], robot) == 0

    assert [name for name, _ in (c for c in robot.calls if isinstance(c, tuple))] == [
        "calibrate",
        "close",
        "open",
        "retract",
    ]
    assert all(args == () for _, args in (c for c in robot.calls if isinstance(c, tuple)))
    assert "recover" not in robot.calls  # right-arm-only recovery
    assert "抓取判定通过" in capsys.readouterr().out


def test_right_arm_passes_params_and_recovers_joint_frame(monkeypatch):
    robot = FakeRobot()

    assert _run(monkeypatch, ["--arm", "right_arm"], ["y", "y", "y"], robot) == 0

    assert robot.calls[0] == "recover"
    assert all(
        args != () for _, args in (c for c in robot.calls if isinstance(c, tuple))
    )


def test_a_refused_grasp_reports_and_leaves_the_gripper_alone(monkeypatch, capsys):
    robot = FakeRobot(grasp=SafetyAbort("夹爪闭合位置等同空夹"))

    assert _run(monkeypatch, ["--arm", "left_arm"], ["y", "y"], robot) == 2

    assert not any(
        isinstance(call, tuple) and call[0] in ("open", "retract")
        for call in robot.calls
    )
    assert "抓取判定未通过" in capsys.readouterr().out


def test_an_unconfirmed_empty_hand_sends_no_gripper_command(monkeypatch):
    robot = FakeRobot()

    assert _run(monkeypatch, ["--arm", "left_arm"], ["n"], robot) == 1

    assert not any(isinstance(call, tuple) for call in robot.calls)
    assert robot.calls[-1] == "closed"  # the session is still released


def test_keep_holding_skips_the_release(monkeypatch):
    robot = FakeRobot()

    assert _run(
        monkeypatch, ["--arm", "left_arm", "--keep-holding"], ["y", "y"], robot
    ) == 0

    assert not any(
        isinstance(call, tuple) and call[0] in ("open", "retract")
        for call in robot.calls
    )
