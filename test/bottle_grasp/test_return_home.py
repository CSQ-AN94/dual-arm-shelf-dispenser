"""_return_home / --finish-from-current orchestration (no hardware).

2026-07-17: the removed guided corridor used to also carry the robot back to
its hang pose. This restores a "go home" leg using the same trusted mechanism
already used for the observation leg (MoveIt plan + electronic-fence dense
check inside _plan_flange), driven by a single home_joints_deg target stored
in the safety profile instead of a recorded corridor.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bottle_grasp.demo as demo_module
from bottle_grasp.core import DemoParams, Localization, SafetyAbort
from bottle_grasp.safety import load_safety_profile


def test_table_demo_profile_has_home_joints_deg():
    profile = load_safety_profile(
        Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json",
        "table_demo",
        require_verified=False,
    )
    assert profile.home_joints_deg is not None
    assert len(profile.home_joints_deg) == 7


def test_shelf_template_has_taught_right_home_posture():
    # 2026-07-20 现场重新示教右臂的高净空停靠位。左臂每轮任务
    # 采集实时快照，不用这份 profile 里的静态目标。
    profile = load_safety_profile(
        Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json",
        "shelf_template",
        require_verified=False,
    )
    assert profile.home_joints_deg is not None
    assert len(profile.home_joints_deg) == 7
    assert profile.home_joints_deg == pytest.approx(
        [7.665, 113.884, -7.937, 33.977, -82.214, -83.986, -13.099],
        abs=1e-3,
    )


def _make_demo(calls):
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    demo.params = demo_module.DemoParams()

    class FakeSafety:
        name = "fake"
        home_joints_deg = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)

    demo.safety = FakeSafety()
    demo.stage = lambda name, msg="": calls.append(("stage", name))

    class FakeRobot:
        def controller_flange_from_joints(self, joints):
            calls.append(("fk", tuple(joints)))
            return np.eye(4)

        def joints_deg(self):
            return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    demo.robot = FakeRobot()
    demo._plan_flange = lambda name, flange, goal_joints=None: (
        calls.append(("plan", name, tuple(goal_joints))),
        {"points_deg": [[0.0] * 7]},
    )[1]
    demo._execute_plan = lambda name, plan: calls.append(("execute", name))
    return demo


def test_return_home_plans_and_executes_to_profile_target():
    calls = []
    demo = _make_demo(calls)
    demo._return_home()
    assert ("plan", "moveit_return_home", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)) in calls
    assert any(c[0] == "execute" for c in calls)


def test_return_home_aborts_without_configured_target():
    calls = []
    demo = _make_demo(calls)
    demo.safety.home_joints_deg = None
    with pytest.raises(SafetyAbort, match="home_joints_deg"):
        demo._return_home()


def test_start_home_check_skips_motion_when_already_at_taught_home():
    calls = []
    demo = _make_demo(calls)
    demo._return_home = lambda: calls.append(("unexpected_return_home",))

    assert demo._normalize_start_home() is False
    assert ("unexpected_return_home",) not in calls
    assert ("stage", "起点 home 检查") in calls


def test_start_home_check_uses_safe_return_when_current_pose_is_not_home():
    calls = []
    demo = _make_demo(calls)
    demo.robot.joints_deg = lambda: [0.0] * 7
    demo._return_home = lambda: calls.append(("safe_return_home",))

    assert demo._normalize_start_home() is True
    assert ("stage", "起点 home 归位") in calls
    assert calls.count(("safe_return_home",)) == 1


def test_return_scene_refresh_uses_post_action_majority_consensus():
    """One occluded depth outlier must not strand a completed cycle away from home."""
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    demo.params = DemoParams()
    demo.camera_name = "head"
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
    locked = Localization(
        point_camera=[0.0, 0.0, 0.5],
        point_base=[0.0, 0.0, 0.5],
        pixel=[320.0, 240.0],
        depth_m=0.5,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[280, 100, 360, 400],
        confidence=0.9,
        frame_count=7,
    )
    calls = []

    def localize(*_args, **kwargs):
        calls.append(kwargs)
        return locked

    demo.localize = localize
    demo._build_head_scene = lambda target: calls.append({"scene": target})
    demo.stage = lambda *_args: None

    demo._refresh_head_scene_for_global_motion(locked)

    assert calls[0]["required_consensus_frames"] == 2
    assert calls[1] == {"scene": locked}


def test_finish_from_current_skips_localization_and_grasp():
    calls = []
    demo = _make_demo(calls)
    demo.args = type("Args", (), {})()
    demo.args.place_back = True
    demo.args.return_home = True
    demo.args.restore_teleop = False
    demo.stop_event = __import__("threading").Event()
    demo.stop_event.set()  # exit the hold loop immediately
    demo._place_back = lambda: calls.append(("place_back",))
    demo._return_home = lambda: calls.append(("return_home",))
    demo._finish_from_current()
    actions = [c for c in calls if c[0] in ("place_back", "return_home")]
    assert actions == [("place_back",), ("return_home",)]


def test_place_back_closes_empty_gripper_only_after_retreat():
    """The fingers stay open until every post-release retreat point completes."""
    calls = []
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    demo.params = demo_module.DemoParams()
    demo.stage = lambda name, msg="": calls.append(("stage", name))

    class FakeSafety:
        def assert_tcp_point(self, point, *, label):
            calls.append(("fence", label))

    class FakeRobot:
        def current_tcp(self):
            return np.eye(4)

        def escape_j4_singularity(self, params, safety_profile):
            calls.append(("escape_check",))
            return None  # 起点不在奇异带

        def move_linear(self, pose, speed):
            calls.append(("move", tuple(pose)))

        def open_gripper(self, params):
            calls.append(("open",))

        def close_empty_gripper(self, params):
            calls.append(("close_empty",))

    demo.safety = FakeSafety()
    demo.robot = FakeRobot()
    demo._plan_ik_avoiding_singularity = (
        lambda path, params, **kwargs: path
    )

    demo._place_back()

    retreat_stage = calls.index(("stage", "退开"))
    close_index = calls.index(("close_empty",))
    last_move = max(i for i, call in enumerate(calls) if call[0] == "move")
    completed_stage = calls.index(("stage", "放回完成"))
    assert retreat_stage < last_move < close_index < completed_stage


def test_complete_task_visually_confirms_release_before_empty_close():
    calls = []
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    demo.params = demo_module.DemoParams()
    demo.stage = lambda name, msg="": calls.append(("stage", name))

    class FakeSafety:
        @staticmethod
        def assert_tcp_point(_point, *, label):
            calls.append(("fence", label))

    class FakeRobot:
        @staticmethod
        def current_tcp():
            return np.eye(4)

        @staticmethod
        def escape_j4_singularity(_params, _safety):
            return None

        @staticmethod
        def move_linear(_pose, _speed):
            calls.append(("move",))

        @staticmethod
        def open_gripper(_params):
            calls.append(("open",))

        @staticmethod
        def close_empty_gripper(_params):
            calls.append(("close_empty",))

    locked = object()
    demo.safety = FakeSafety()
    demo.robot = FakeRobot()
    demo._plan_ik_avoiding_singularity = lambda path, params, **kwargs: path
    demo._confirm_released_target = lambda target, lifted=None: calls.append(
        ("release_confirm", target, lifted)
    )

    demo._place_back(locked)

    assert calls.index(("release_confirm", locked, None)) < calls.index(
        ("close_empty",)
    )
