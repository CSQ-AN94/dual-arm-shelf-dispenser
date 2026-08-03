import threading

import numpy as np
import pytest

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.arm import RobotSession


class _StoppedMoveArm:
    def rm_movel(self, pose, speed, radius, connect, block):
        return -6

    def rm_get_current_arm_state(self):
        return 0, {
            "pose": [0.01, 0.50, -0.10, 0.0, 0.0, 0.0],
            "arm_err": 0,
            "sys_err": 0,
        }

    def rm_get_joint_degree(self):
        return 0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    def rm_get_electronic_fence_enable(self):
        return 0, {"enable_state": True, "in_out_side": 0, "effective_region": 0}

    def rm_get_electronic_fence_config(self):
        return 0, {"form": 1, "name": "stale_table_fence"}

    def rm_get_electronic_fence_list_infos(self):
        return {
            "return_code": 0,
            "len": 1,
            "electronic_fence_list": ["stale_table_fence"],
        }


class _HiddenJ7FaultArm:
    def rm_get_arm_all_state(self):
        return 0, {
            "joint_en_flag": [1, 1, 1, 1, 1, 1, 1],
            "joint_err_code": [0, 0, 0, 0, 0, 0, 0xF000],
            "err": {"err_len": 1, "err": ["0"]},
        }

    def rm_get_controller_state(self):
        return {"return_code": 0, "system_error": 0}


class _RecoverableJ7FaultArm:
    def __init__(self):
        self.reads = 0
        self.cleared = []

    def rm_get_arm_all_state(self):
        self.reads += 1
        errors = [0, 0, 0, 0, 0, 0, 0xF000] if self.reads == 1 else [0] * 7
        return 0, {
            "joint_en_flag": [1] * 7,
            "joint_err_code": errors,
        }

    def rm_get_controller_state(self):
        return {"return_code": 0, "system_error": 0}

    def rm_set_joint_clear_err(self, joint):
        self.cleared.append(joint)
        return 0


def test_movel_minus_6_reports_external_stop_context():
    session = RobotSession.__new__(RobotSession)
    session.arm = _StoppedMoveArm()
    session.stop_event = threading.Event()
    target = [0.08, 0.51, -0.11, 0.0, 0.0, 0.0]

    with pytest.raises(SafetyAbort) as caught:
        session.move_linear(target, 3)

    message = str(caught.value)
    assert "外部停止" in message
    assert "target=[0.08, 0.51, -0.11" in message
    assert "stale_table_fence" in message


class _FeedbackArm:
    def __init__(self, actual_pose):
        self.actual_pose = actual_pose
        self.slow_stop_calls = 0

    def rm_movel(self, pose, speed, radius, connect, block):
        return 0

    def rm_set_arm_slow_stop(self):
        self.slow_stop_calls += 1
        return 0


def _feedback_session(actual_pose):
    session = RobotSession.__new__(RobotSession)
    session.arm = _FeedbackArm(actual_pose)
    session.stop_event = threading.Event()
    session.closed = False
    session.take_control = True
    session.assert_arm_healthy = lambda: {}
    session.current_tcp = lambda: np.asarray(actual_pose, dtype=float)
    return session


def test_movel_rejects_success_code_when_measured_tcp_missed_target():
    actual = np.eye(4)
    actual[0, 3] = 0.025
    session = _feedback_session(actual)

    with pytest.raises(SafetyAbort, match="执行反馈偏差过大"):
        session.move_linear([0.0] * 6, 3)

    assert session.arm.slow_stop_calls == 1


def test_movel_accepts_fresh_feedback_within_pose_tolerance():
    actual = np.eye(4)
    actual[0, 3] = 0.002
    session = _feedback_session(actual)

    session.move_linear([0.0] * 6, 3)

    assert session.arm.slow_stop_calls == 0


def test_stop_monitor_interrupts_a_blocking_sdk_move():
    entered = threading.Event()
    stopped = threading.Event()

    class BlockingArm(_StoppedMoveArm):
        def rm_movel(self, pose, speed, radius, connect, block):
            entered.set()
            assert stopped.wait(1.0), "slow-stop monitor did not interrupt movel"
            return -6

        def rm_set_arm_slow_stop(self):
            stopped.set()
            return 0

    session = RobotSession.__new__(RobotSession)
    session.arm = BlockingArm()
    session.stop_event = threading.Event()
    session.closed = False
    monitor = threading.Thread(target=session._monitor_stop, daemon=True)
    monitor.start()
    failures = []

    def run_move():
        try:
            session.move_linear([0.0] * 6, 3)
        except SafetyAbort as exc:
            failures.append(str(exc))

    mover = threading.Thread(target=run_move, daemon=True)
    mover.start()
    assert entered.wait(1.0)
    session.stop_event.set()
    mover.join(1.0)
    monitor.join(1.0)

    assert stopped.is_set()
    assert not mover.is_alive()
    assert failures and "外部停止" in failures[0]


def test_preflight_rejects_enabled_controller_native_fence():
    class FakeRobot:
        def assert_arm_healthy(self):
            return {
                "joints": {},
                "controller": {"return_code": 0, "system_error": 0},
            }

        def current_tcp(self):
            return None

        def controller_fence_status(self):
            return {
                "state": {"enable_state": True},
                "current": (0, {"name": "stale_table_fence"}),
                "saved": {"len": 1},
            }

        def gripper_state(self):
            raise AssertionError("must abort before commanding or reading gripper")

    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.args = type("Args", (), {"execute": True})()
    demo.robot = FakeRobot()
    demo.stage = lambda *args: None

    with pytest.raises(SafetyAbort, match="原生电子围栏仍处于启用状态"):
        demo._preflight()


def test_arm_health_rejects_joint_fault_hidden_by_summary_error():
    session = RobotSession.__new__(RobotSession)
    session.arm = _HiddenJ7FaultArm()

    with pytest.raises(
        SafetyAbort, match=r"J7=0xF000\(通信丢帧\)"
    ):
        session.assert_arm_healthy()


def test_read_only_session_computes_flange_from_joints_without_active_tcp():
    session = RobotSession.__new__(RobotSession)
    session.take_control = False
    session.joints_deg = lambda: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    expected = np.eye(4)
    expected[:3, 3] = [0.1, 0.2, 0.3]
    session.controller_flange_from_joints = lambda joints: expected.copy()

    actual = session.current_flange()

    np.testing.assert_allclose(actual, expected)


def _healthy_preflight_robot(gripper_pos, calls):
    class HealthyRobot:
        def assert_arm_healthy(self):
            return {
                "joints": {},
                "controller": {"return_code": 0, "system_error": 0},
            }

        def current_tcp(self):
            return None

        def controller_fence_status(self):
            return {
                "state": {"enable_state": False},
                "current": (0, {}),
                "saved": {"len": 0},
            }

        def gripper_state(self):
            return {"enable_state": True, "pos": [gripper_pos]}

        def close_empty_gripper(self, params):
            calls.append(("close_empty", gripper_pos))
            return {"dof_state": 3, "pos": [0]}

    return HealthyRobot()


def test_preflight_closes_an_open_gripper_before_transit():
    calls = []
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.args = type(
        "Args", (), {"execute": True, "finish_from_current": False}
    )()
    demo.params = DemoParams()
    demo.robot = _healthy_preflight_robot(902, calls)
    demo.stage = lambda name, msg="": calls.append(("stage", name))

    demo._preflight()

    assert ("close_empty", 902) in calls
    assert any(name == "夹爪预备闭合" for kind, name in calls if kind == "stage")


def test_preflight_leaves_an_already_closed_gripper_alone():
    calls = []
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.args = type(
        "Args", (), {"execute": True, "finish_from_current": False}
    )()
    demo.params = DemoParams()
    demo.robot = _healthy_preflight_robot(0, calls)
    demo.stage = lambda name, msg="": calls.append(("stage", name))

    demo._preflight()

    assert not any(kind == "close_empty" for kind, *_ in calls)


def test_preflight_skips_gripper_close_when_finishing_from_current():
    """--finish-from-current 假设夹爪已抓着水瓶，绝不能在这里被合上。"""
    calls = []

    class MustNotCloseRobot:
        def assert_arm_healthy(self):
            return {
                "joints": {},
                "controller": {"return_code": 0, "system_error": 0},
            }

        def current_tcp(self):
            return None

        def controller_fence_status(self):
            return {
                "state": {"enable_state": False},
                "current": (0, {}),
                "saved": {"len": 0},
            }

        def gripper_state(self):
            return {"enable_state": True, "pos": [902]}

        def close_empty_gripper(self, params):
            raise AssertionError(
                "must not close the gripper when a bottle may already be held"
            )

    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.args = type(
        "Args", (), {"execute": True, "finish_from_current": True}
    )()
    demo.params = DemoParams()
    demo.robot = MustNotCloseRobot()
    demo.stage = lambda name, msg="": calls.append(name)

    demo._preflight()

    assert "夹爪预备闭合" not in calls


def test_stop_after_observation_never_closes_an_open_gripper():
    """The supported task-mode observation endpoint must remain non-grasping."""
    calls = []
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.args = type(
        "Args",
        (),
        {
            "execute": True,
            "finish_from_current": False,
            "stop_after_observation": True,
        },
    )()
    demo.params = DemoParams()
    demo.robot = _healthy_preflight_robot(902, calls)
    demo.stage = lambda name, msg="": calls.append(("stage", name))

    demo._preflight()

    assert not any(kind == "close_empty" for kind, *_ in calls)
    assert ("stage", "观察后停止夹爪保护") in calls


def test_read_only_resume_check_skips_motion_preflight_even_with_execute_flag():
    class MustNotTouchRobot:
        def assert_arm_healthy(self):
            raise AssertionError("read-only visual check must not run motion preflight")

    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.args = type(
        "Args",
        (),
        {
            "execute": True,
            "resume_at_wrist": True,
            "stop_after_observation": True,
        },
    )()
    demo.robot = MustNotTouchRobot()

    demo._preflight()


def test_motion_preflight_clears_only_recoverable_frame_loss(monkeypatch):
    session = RobotSession.__new__(RobotSession)
    session.arm = _RecoverableJ7FaultArm()
    monkeypatch.setattr("shelf_dispenser.arm.time.sleep", lambda _: None)

    recovered = session.recover_transient_joint_frame_loss()

    assert recovered == [7]
    assert session.arm.cleared == [7]
    assert session.arm.reads == 3


def test_motion_preflight_never_clears_other_joint_faults(monkeypatch):
    arm = _RecoverableJ7FaultArm()
    arm.rm_get_arm_all_state = lambda: (
        0,
        {
            "joint_en_flag": [1] * 7,
            "joint_err_code": [0, 0, 0, 0, 0, 0, 0x0020],
        },
    )
    session = RobotSession.__new__(RobotSession)
    session.arm = arm
    monkeypatch.setattr("shelf_dispenser.arm.time.sleep", lambda _: None)

    assert session.recover_transient_joint_frame_loss() == []
    assert arm.cleared == []
