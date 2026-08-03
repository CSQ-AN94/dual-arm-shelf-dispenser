"""_wait_for_grasp_confirmation: interruptible operator checkpoint (no hardware).

2026-07-18: added --confirm-before-grasp so a `watch` run can pause after
reaching the observation position (bottle already detected) and continue
straight into the same grasp/cycle code path on operator confirmation,
without restarting the process (camera/YOLO/MoveIt init otherwise costs
~30-40s) or falling back to a different, disconnected code path
(run_bottle_grasp_resume.sh) the way the 2026-07-18 real-run session did.

The wait must not swallow Ctrl+C/STOP: bottle_grasp_demo.py's signal handler
only sets stop_event and returns (it does not raise), so a plain blocking
input() would ignore a stop request until Enter is also pressed. The
implementation polls stop_event instead of blocking on stdin directly.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bottle_grasp.demo as demo_module
from bottle_grasp.core import Localization, SafetyAbort


class _FakeStdin:
    def __init__(self, readline_fn):
        self.readline = readline_fn


def _make_demo(calls):
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    demo.stage = lambda name, msg="": calls.append((name, msg))
    demo.stop_event = threading.Event()
    return demo


def test_confirmation_returns_once_operator_presses_enter(monkeypatch):
    calls = []
    demo = _make_demo(calls)
    monkeypatch.setattr(demo_module.sys, "stdin", _FakeStdin(lambda: "\n"))

    demo._wait_for_grasp_confirmation()

    assert any(name == "人工确认通过" for name, _ in calls)
    assert any(name == "等待人工确认" for name, _ in calls)


def test_confirmation_aborts_immediately_on_pending_stop(monkeypatch):
    calls = []
    demo = _make_demo(calls)

    def _never_returns():
        time.sleep(10)
        return "\n"

    monkeypatch.setattr(demo_module.sys, "stdin", _FakeStdin(_never_returns))
    demo.stop_event.set()

    with pytest.raises(SafetyAbort, match="确认抓取前收到停止请求"):
        demo._wait_for_grasp_confirmation()
    assert not any(name == "人工确认通过" for name, _ in calls)


@pytest.mark.parametrize("line", ("", "continue\n"))
def test_confirmation_fails_closed_on_eof_or_nonempty_input(monkeypatch, line):
    calls = []
    demo = _make_demo(calls)
    monkeypatch.setattr(demo_module.sys, "stdin", _FakeStdin(lambda: line))

    with pytest.raises(SafetyAbort, match="拒绝"):
        demo._wait_for_grasp_confirmation()

    assert not any(name == "人工确认通过" for name, _ in calls)


def test_confirmation_aborts_on_stop_arriving_mid_wait(monkeypatch):
    calls = []
    demo = _make_demo(calls)

    def _never_returns():
        time.sleep(10)
        return "\n"

    monkeypatch.setattr(demo_module.sys, "stdin", _FakeStdin(_never_returns))
    threading.Timer(0.05, demo.stop_event.set).start()

    with pytest.raises(SafetyAbort, match="确认抓取前收到停止请求"):
        demo._wait_for_grasp_confirmation()


def _make_execute_demo(calls, *, confirm_before_grasp, stop_after_observation=False):
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)

    class Args:
        pass

    demo.args = Args()
    demo.args.execute = True
    demo.args.plan_only = False
    demo.args.resume_at_wrist = False
    demo.args.stop_after_observation = stop_after_observation
    demo.args.confirm_before_grasp = confirm_before_grasp
    demo.args.observe_seconds = 0
    demo.params = demo_module.DemoParams()

    class FakeCalibration:
        T_base_right_to_camera_head = np.eye(4)
        T_end_right_to_camera_rightwrist = np.eye(4)

    class FakeConfig:
        calibration = FakeCalibration()

    demo.cfg = FakeConfig()

    class FakeSafety:
        def assert_tcp_point(self, point, label=""):
            pass

    class FakeRobot:
        @staticmethod
        def current_flange():
            return np.eye(4)

    demo.safety = FakeSafety()
    demo.robot = FakeRobot()
    demo.stage = lambda name, msg="": calls.append(("stage", name))
    demo.initialize = lambda: calls.append(("initialize",))
    demo._preflight = lambda: calls.append(("preflight",))
    demo._build_head_scene = lambda target: calls.append(("build_head_scene",))
    demo._plan_observation = lambda target: (
        calls.append(("plan_observation",)),
        {"points_deg": [[0.0] * 7]},
    )[1]
    demo._execute_plan = lambda name, plan: calls.append(("execute_plan", name))
    demo._start_camera = lambda name: calls.append(("start_camera", name))
    demo._wait_for_grasp_confirmation = lambda: calls.append(("confirm",))
    demo._finish_grasp_from_wrist = lambda target: calls.append(("finish_grasp",))
    loc = Localization(
        [0, 0, 0.5], [0, 0.6, -0.05], [320, 240], 0.5, 0.001, 0.002,
        [0, 0, 10, 10], 0.9, 7,
    )
    demo.localize = lambda *a, **k: (calls.append(("localize",)), loc)[1]
    return demo


def test_confirm_before_grasp_pauses_between_observation_and_grasp():
    calls = []
    demo = _make_execute_demo(calls, confirm_before_grasp=True)

    demo.run()

    assert calls.index(("confirm",)) < calls.index(("finish_grasp",))
    assert ("start_camera", "right_wrist") in calls


def test_cycle_without_confirm_flag_goes_straight_to_grasp():
    calls = []
    demo = _make_execute_demo(calls, confirm_before_grasp=False)

    demo.run()

    assert ("confirm",) not in calls
    assert ("finish_grasp",) in calls


def test_stop_after_observation_wins_over_confirm_flag():
    """两者互斥由 CLI 层拒绝；run() 本身仍按 stop_after_observation 优先短路。"""
    calls = []
    demo = _make_execute_demo(
        calls, confirm_before_grasp=True, stop_after_observation=True
    )

    demo.run()

    assert ("confirm",) not in calls
    assert ("finish_grasp",) not in calls
