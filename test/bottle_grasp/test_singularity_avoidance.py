"""Singularity handling without hardware: roll retries and elbow escape.

2026-07-17 real-hardware finding: the arm's resting pose can already sit
inside the J4≈0° singular band, so pure-translation legs abort on the very
first IK call. Roll about the tool z axis cannot fix that case — the elbow
magnitude is fixed by the shoulder-wrist distance and rolling does not move
the wrist center — so _plan_local_leg first performs a fence-checked
joint-space elbow escape (robot.escape_j4_singularity) and only then builds
the Cartesian path from the new pose.
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bottle_grasp.demo as demo_module
from bottle_grasp.core import DemoParams, SafetyAbort, matrix_pose, pose_matrix
from bottle_grasp.robot import RobotSession


def _straight_down_path():
    start = pose_matrix([0.3, 0.5, 0.0, 0.0, 0.0, 0.0])
    end = start.copy()
    end[2, 3] -= 0.05
    return [matrix_pose(start), matrix_pose(end)]


def test_zero_roll_succeeds_when_not_singular():
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    demo.stage = lambda name, msg="": None

    class FakeRobot:
        def plan_ik(self, poses, params, *, allow_first_jump=False):
            return [[0.0] * 7 for _ in poses]

    demo.robot = FakeRobot()
    path = _straight_down_path()
    result = demo._plan_ik_avoiding_singularity(path, DemoParams())
    assert result == path


def test_retries_with_roll_when_first_attempt_hits_singularity():
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    calls = []
    demo.stage = lambda name, msg="": calls.append((name, msg))

    class FlakyRobot:
        def __init__(self):
            self.attempts = 0

        def plan_ik(self, poses, params, *, allow_first_jump=False):
            self.attempts += 1
            if self.attempts < 3:  # fails for roll=0 and roll=8
                raise SafetyAbort("J4=-0.0°，进入奇异区")
            return [[0.0] * 7 for _ in poses]

    demo.robot = FlakyRobot()
    path = _straight_down_path()
    result = demo._plan_ik_avoiding_singularity(path, DemoParams())
    # 位置不变，姿态被旋转过（不再等于原始未旋转路径）
    assert len(result) == len(path)
    for original, rotated in zip(path, result):
        assert np.allclose(original[:3], rotated[:3])
    assert result != path
    assert demo.robot.attempts == 3
    assert any("避奇异" in name for name, _ in calls)


def test_raises_when_all_roll_angles_fail():
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    demo.stage = lambda name, msg="": None

    class AlwaysFailsRobot:
        def plan_ik(self, poses, params, *, allow_first_jump=False):
            raise SafetyAbort("J4=-0.0°，进入奇异区")

    demo.robot = AlwaysFailsRobot()
    path = _straight_down_path()
    with pytest.raises(SafetyAbort, match="奇异区"):
        demo._plan_ik_avoiding_singularity(path, DemoParams())


class _AcceptAllFence:
    def assert_tcp_point(self, point, *, label):
        pass


def _escape_session(joints, *, fence=None, tcp_x_from_j4=0.0):
    """Bare RobotSession with only what escape_j4_singularity touches."""
    session = RobotSession.__new__(RobotSession)
    session.take_control = True
    session.stop_event = threading.Event()
    session.joints_deg = lambda: list(joints)
    session.executed = []
    session.execute_planned_joints = (
        lambda points, speed, step: session.executed.append(
            (tuple(map(tuple, points)), speed, step)
        )
    )

    def tcp_from_joints(values):
        T = np.eye(4)
        T[0, 3] = tcp_x_from_j4 * values[3]
        return T

    session.tcp_from_joints = tcp_from_joints

    class FakeArm:
        def rm_get_joint_min_pos(self):
            return 0, [-175.0] * 7

        def rm_get_joint_max_pos(self):
            return 0, [175.0] * 7

    session.arm = FakeArm()
    return session


def test_escape_is_noop_outside_the_band():
    session = _escape_session([0, 0, 0, 30.0, 0, 0, 0])
    result = session.escape_j4_singularity(DemoParams(), _AcceptAllFence())
    assert result is None
    assert session.executed == []


def test_escape_bends_elbow_toward_same_sign_and_executes():
    params = DemoParams()
    session = _escape_session([0, 0, 0, 2.0, 0, 0, 0])
    target = session.escape_j4_singularity(params, _AcceptAllFence())
    assert target[3] == params.j4_escape_deg  # 起点+2° → 同号 +14°
    assert abs(target[3]) > params.j4_singularity_deg
    (points, speed, step), = session.executed
    assert points == (tuple(target),)
    assert speed == params.final_speed
    assert step == params.planned_joint_step_deg


def test_escape_falls_back_to_opposite_bend_when_fence_rejects():
    class OneSideBlockedFence:
        def assert_tcp_point(self, point, *, label):
            # tcp_x_from_j4=0.01 时 J4 越正 x 越大：正向弯到 +14° 的路径
            # 必然越过 x=0.03（J4>3°），反向路径 J4 最多 +2° 不会触发。
            if point[0] > 0.03:
                raise SafetyAbort(f"{label} 进入禁入区")

    params = DemoParams()
    session = _escape_session(
        [0, 0, 0, 2.0, 0, 0, 0], tcp_x_from_j4=0.01
    )
    target = session.escape_j4_singularity(params, OneSideBlockedFence())
    assert target[3] == -params.j4_escape_deg
    assert len(session.executed) == 1


def test_escape_aborts_when_both_bends_are_rejected():
    class BlockedFence:
        def assert_tcp_point(self, point, *, label):
            raise SafetyAbort(f"{label} 进入禁入区")

    session = _escape_session([0, 0, 0, 0.0, 0, 0, 0])
    with pytest.raises(SafetyAbort, match="两个弯肘方向均不可行"):
        session.escape_j4_singularity(DemoParams(), BlockedFence())
    assert session.executed == []


def test_escape_refuses_to_move_in_plan_only_session():
    session = _escape_session([0, 0, 0, 1.0, 0, 0, 0])
    session.take_control = False
    with pytest.raises(SafetyAbort, match="只规划会话"):
        session.escape_j4_singularity(DemoParams(), _AcceptAllFence())
    assert session.executed == []


def test_plan_local_leg_escapes_before_building_the_path():
    """路径必须在弯肘逃逸之后构建——逃逸会移动 TCP，先建的路径起点是错的。"""
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    order = []
    demo.stage = lambda name, msg="": order.append(("stage", name))
    demo.safety = _AcceptAllFence()

    class InBandRobot:
        def escape_j4_singularity(self, params, safety_profile):
            order.append(("escape",))
            return [0, 0, 0, 14.0, 0, 0, 0]

        def plan_ik(self, poses, params, *, allow_first_jump=False):
            return [[0.0] * 7 for _ in poses]

    demo.robot = InBandRobot()

    def build_path():
        order.append(("build",))
        return _straight_down_path()

    result = demo._plan_local_leg("抬升", build_path, DemoParams())
    assert result == _straight_down_path()
    assert order.index(("escape",)) < order.index(("build",))
    assert any(
        item[0] == "stage" and "弯肘逃逸" in item[1] for item in order
    )


def test_plan_local_leg_skips_escape_when_not_in_band():
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    stages = []
    demo.stage = lambda name, msg="": stages.append(name)
    demo.safety = _AcceptAllFence()

    class ClearRobot:
        def escape_j4_singularity(self, params, safety_profile):
            return None

        def plan_ik(self, poses, params, *, allow_first_jump=False):
            return [[0.0] * 7 for _ in poses]

    demo.robot = ClearRobot()
    result = demo._plan_local_leg(
        "退开", _straight_down_path, DemoParams()
    )
    assert result == _straight_down_path()
    assert not any("弯肘逃逸" in name for name in stages)


def test_complete_task_never_uses_unplanned_j4_escape_bypass():
    demo = demo_module.BottleDemo.__new__(demo_module.BottleDemo)
    demo.args = SimpleNamespace(task_mode="from-start")
    demo.safety = _AcceptAllFence()
    demo.stage = lambda *_args, **_kwargs: None

    class InBandTaskRobot:
        @staticmethod
        def joints_deg():
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        @staticmethod
        def escape_j4_singularity(*_args, **_kwargs):
            raise AssertionError("complete task must not call escape movej")

    demo.robot = InBandTaskRobot()
    built = []

    with pytest.raises(SafetyAbort, match="禁止用未经过场景规划"):
        demo._plan_local_leg(
            "抬升", lambda: built.append(True) or _straight_down_path(), DemoParams()
        )

    assert built == []
