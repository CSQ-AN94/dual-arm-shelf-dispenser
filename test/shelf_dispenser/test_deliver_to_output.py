"""_deliver_to_output orchestration (no hardware).

Real dispensing (as opposed to table_demo's place-back-in-place cycle):
after grasp+lift, carry the bottle to a configured output point instead of
returning it to the pick location. Mirrors test_return_home.py's approach —
same _plan_flange/_execute_plan mechanism, same fail-closed-on-missing-
target contract — plus the release-confirmation branch, which must not
pretend to have visual evidence the output point was never configured to
provide.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import shelf_dispenser.orchestrator as demo_module
from shelf_dispenser.core import SafetyAbort


def _make_demo(calls, **safety_overrides):
    demo = demo_module.RunOrchestrator.__new__(demo_module.RunOrchestrator)
    demo.params = demo_module.DemoParams()

    class FakeSafety:
        name = "fake"
        output_joints_deg = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        output_visible_to_head_camera = False
        output_point_base = None

        @staticmethod
        def assert_tcp_point(_point, *, label):
            calls.append(("fence", label))

    safety = FakeSafety()
    for key, value in safety_overrides.items():
        setattr(safety, key, value)
    demo.safety = safety
    demo.stage = lambda name, msg="": calls.append(("stage", name))

    class FakeRobot:
        def controller_flange_from_joints(self, joints):
            calls.append(("fk", tuple(joints)))
            return np.eye(4)

        def current_tcp(self):
            return np.eye(4)

        def move_linear(self, pose, speed):
            calls.append(("move", tuple(pose)))

        def open_gripper(self, params):
            calls.append(("open",))

        def close_empty_gripper(self, params):
            calls.append(("close_empty",))

    demo.robot = FakeRobot()
    demo._plan_flange = lambda name, flange, goal_joints=None: (
        calls.append(("plan", name, tuple(goal_joints))),
        {"points_deg": [[0.0] * 7]},
    )[1]
    demo._execute_plan = lambda name, plan: calls.append(("execute", name))
    demo._plan_local_leg = lambda name, build_fn, params: build_fn()
    demo._confirm_point_released = lambda point, **kwargs: calls.append(
        ("confirm", tuple(point), kwargs.get("label"))
    )
    return demo


def test_deliver_to_output_aborts_without_configured_target():
    calls = []
    demo = _make_demo(calls, output_joints_deg=None)
    with pytest.raises(SafetyAbort, match="output_joints_deg"):
        demo._deliver_to_output()


def test_deliver_to_output_plans_and_executes_to_profile_target():
    calls = []
    demo = _make_demo(calls)
    demo._deliver_to_output()
    assert (
        "plan",
        "moveit_deliver_output",
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
    ) in calls
    assert any(c[0] == "execute" for c in calls)
    assert ("open",) in calls
    assert ("close_empty",) in calls


def test_deliver_to_output_skips_vision_confirm_when_not_visible():
    calls = []
    demo = _make_demo(calls, output_visible_to_head_camera=False)
    demo._deliver_to_output()
    assert not any(c[0] == "confirm" for c in calls)
    assert any(
        c[0] == "stage" and c[1] == "出货口释放（仅夹爪反馈）" for c in calls
    )


def test_deliver_to_output_confirms_when_visible_and_point_configured():
    calls = []
    demo = _make_demo(
        calls,
        output_visible_to_head_camera=True,
        output_point_base=(0.1, 0.6, 0.05),
    )
    demo._deliver_to_output()
    confirm_calls = [c for c in calls if c[0] == "confirm"]
    assert confirm_calls == [("confirm", (0.1, 0.6, 0.05), "出货口三维确认")]


def test_deliver_to_output_aborts_if_visible_but_point_missing():
    calls = []
    demo = _make_demo(
        calls, output_visible_to_head_camera=True, output_point_base=None
    )
    with pytest.raises(SafetyAbort, match="output_point_base"):
        demo._deliver_to_output()


def test_deliver_to_output_closes_gripper_only_after_retreat():
    """Fingers stay open until the post-release retreat leg completes,
    mirroring _place_back's ordering contract."""
    calls = []
    demo = _make_demo(calls)
    demo._deliver_to_output()
    open_index = calls.index(("open",))
    retreat_stage = calls.index(("stage", "退开"))
    close_index = calls.index(("close_empty",))
    completed_stage = calls.index(("stage", "送货完成"))
    assert open_index < retreat_stage < close_index < completed_stage
