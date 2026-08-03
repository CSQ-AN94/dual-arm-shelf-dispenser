import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.arm import (
    CONNECTED_TRAJECTORY_MAX_ERROR_DEG,
    CONNECTED_TRAJECTORY_START_NOOP_DEG,
    RobotSession,
)


class _Arm:
    def __init__(self, *, follow: bool):
        self.follow = follow
        self.current = [0.0] * 7
        self.moves = []
        self.commands = []
        self.clears = 0

    def rm_movej(self, joints, speed, radius, trajectory_connect, block):
        self.moves.append(list(joints))
        self.commands.append(
            (list(joints), speed, radius, trajectory_connect, block)
        )
        if self.follow and trajectory_connect == 0:
            self.current = list(joints)
        return 0

    def rm_set_delete_current_trajectory(self):
        self.clears += 1
        return 0

    def rm_get_arm_all_state(self):
        return 0, {
            "joint_en_flag": [1] * 7,
            "joint_err_code": [0] * 7,
        }

    def rm_get_controller_state(self):
        return {"return_code": 0, "system_error": 0}

    def rm_get_joint_min_pos(self):
        return 0, [-180.0] * 7

    def rm_get_joint_max_pos(self):
        return 0, [180.0] * 7


def _session(*, follow: bool):
    session = RobotSession.__new__(RobotSession)
    session.take_control = True
    session.stop_event = threading.Event()
    session.arm = _Arm(follow=follow)
    session.joints_deg = lambda: list(session.arm.current)
    session.current_tcp = lambda: np.eye(4)
    return session


def test_planned_path_rejects_stale_real_robot_start_before_move():
    session = _session(follow=True)
    session.arm.current = [2.0] * 7

    with pytest.raises(SafetyAbort, match="轨迹已过期"):
        session.execute_planned_joints(
            [[3.0] * 7],
            3,
            1.5,
            expected_start_joints_deg=[0.0] * 7,
            start_tolerance_deg=0.8,
        )

    assert session.arm.moves == []


def test_planned_path_stops_when_blocking_move_feedback_does_not_follow():
    session = _session(follow=False)

    with pytest.raises(SafetyAbort, match="执行反馈偏差过大"):
        session.execute_planned_joints(
            [[1.0] * 7],
            3,
            1.5,
            expected_start_joints_deg=[0.0] * 7,
            tracking_tolerance_deg=0.5,
        )

    assert len(session.arm.moves) == 1


def test_planned_path_accepts_fresh_start_and_following_feedback():
    session = _session(follow=True)

    session.execute_planned_joints(
        [[2.0] * 7],
        3,
        1.5,
        expected_start_joints_deg=[0.0] * 7,
    )

    np.testing.assert_allclose(session.arm.current, [2.0] * 7)


def test_planned_path_recovers_frame_loss_that_appears_after_preflight(
    monkeypatch,
):
    """Replay 2026-07-21: J7 faults during planning, before first movej."""

    class LateJ7FrameLossArm(_Arm):
        def __init__(self):
            super().__init__(follow=True)
            self.frame_loss = True
            self.cleared = []

        def rm_get_arm_all_state(self):
            errors = [0] * 7
            if self.frame_loss:
                errors[6] = 0xF000
            return 0, {
                "joint_en_flag": [1] * 7,
                "joint_err_code": errors,
            }

        def rm_set_joint_clear_err(self, joint):
            self.cleared.append(joint)
            self.frame_loss = False
            return 0

        def rm_movej(
            self, joints, speed, trajectory_connect, radius, block
        ):
            if self.frame_loss:
                return -6
            return super().rm_movej(
                joints, speed, trajectory_connect, radius, block
            )

    session = _session(follow=True)
    session.arm = LateJ7FrameLossArm()
    session.joints_deg = lambda: list(session.arm.current)
    monkeypatch.setattr("shelf_dispenser.arm.time.sleep", lambda _: None)

    session.execute_planned_joints(
        [[1.0] * 7],
        75,
        1.5,
        expected_start_joints_deg=[0.0] * 7,
    )

    assert session.arm.cleared == [7]
    np.testing.assert_allclose(session.arm.current, [1.0] * 7)


def test_planned_path_minus_6_reports_late_joint_fault_context():
    class FaultOnFirstMoveArm(_Arm):
        def __init__(self):
            super().__init__(follow=True)
            self.failed = False

        def rm_movej(self, joints, *args):
            self.failed = True
            return -6

        def rm_get_arm_all_state(self):
            errors = [0] * 7
            if self.failed:
                errors[6] = 0xF000
            return 0, {
                "joint_en_flag": [1] * 7,
                "joint_err_code": errors,
            }

    session = _session(follow=True)
    session.arm = FaultOnFirstMoveArm()
    session.joints_deg = lambda: list(session.arm.current)

    with pytest.raises(SafetyAbort) as caught:
        session.execute_planned_joints(
            [[1.0] * 7],
            75,
            1.5,
            expected_start_joints_deg=[0.0] * 7,
        )

    message = str(caught.value)
    assert "外部停止" in message
    assert "J7=0xF000(通信丢帧)" in message


def test_planned_path_does_not_send_redundant_moveit_start_point():
    session = _session(follow=True)

    session.execute_planned_joints(
        [[0.0] * 7, [0.5] * 7],
        75,
        1.5,
        expected_start_joints_deg=[0.0] * 7,
    )

    assert session.arm.moves == [[0.5] * 7]


def test_planned_path_queues_intermediate_points_and_blocks_only_on_final(
    monkeypatch,
):
    monkeypatch.setenv("BOTTLE_GRASP_CONTINUOUS_TRAJECTORY", "1")
    session = _session(follow=True)

    session.execute_planned_joints(
        [[0.0] * 7, [0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5], [1.0] * 7],
        75,
        0.25,
        expected_start_joints_deg=[0.0] * 7,
    )

    assert session.arm.commands == [
        ([0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5], 75, 1, 1, 0),
        ([1.0] * 7, 75, 0, 0, 1),
    ]
    np.testing.assert_allclose(session.arm.current, [1.0] * 7)


def test_connected_path_is_compressed_below_controller_queue_limit():
    points = [
        [step / 100.0] * 7
        for step in range(1, 61)
    ]

    compressed = RobotSession._compress_connected_joint_path(
        [0.0] * 7, points
    )

    assert compressed == [[0.6] * 7]


def test_connected_path_command_count_survives_a_live_start_offset():
    """A snapshot pre-check must not pass what the live executor refuses.

    Regression for 2026-08-02 pick_19: the pre-flight fit ran on a snapshot,
    the executor re-read feedback ~0.004 deg away, and the extra lead-in
    command pushed a 30-command segment to 31 -- after the gripper had
    already opened.
    """
    points = [[step / 100.0] * 7 for step in range(1, 61)]
    planned_start = [0.0] * 7
    baseline = len(
        RobotSession._compress_connected_joint_path(planned_start, points)
    )

    for drift in (0.001, 0.004, CONNECTED_TRAJECTORY_MAX_ERROR_DEG):
        live_start = [drift] + [0.0] * 6
        assert (
            len(
                RobotSession._compress_connected_joint_path(live_start, points)
            )
            == baseline
        )


def test_connected_path_keeps_a_genuine_first_waypoint():
    # The lead-in drop must only absorb feedback noise.  A planned point
    # outside the compressor's path-error budget is a real waypoint.
    waypoint = [0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5]

    compressed = RobotSession._compress_connected_joint_path(
        [0.0] * 7, [waypoint, [1.0] * 7]
    )

    assert waypoint in compressed
    assert CONNECTED_TRAJECTORY_START_NOOP_DEG <= CONNECTED_TRAJECTORY_MAX_ERROR_DEG


def test_connected_path_rejects_an_overlarge_original_step():
    with pytest.raises(SafetyAbort, match="单段上限"):
        RobotSession._compress_connected_joint_path(
            [0.0] * 7, [[16.0] * 7]
        )


def test_connected_path_checks_steps_hidden_inside_a_valid_shortcut():
    middle = [-0.019, 0.019, 0.0, 0.0, 0.0, 0.0, 0.0]
    end = [15.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    with pytest.raises(SafetyAbort, match="单段上限"):
        RobotSession._compress_connected_joint_path(
            [0.0] * 7, [middle, end]
        )


def test_planned_path_rejects_controller_joint_limit_without_moving():
    session = _session(follow=True)
    session.tcp_from_joints = lambda _joints: np.eye(4)
    profile = SimpleNamespace(assert_tcp_path=lambda _points: 0)

    with pytest.raises(SafetyAbort, match="把关节推向控制器限位"):
        session.validate_planned_joints(
            [[178.0] * 7], 1.5, profile, start_joints_deg=[0.0] * 7
        )

    assert session.arm.moves == []


def test_planned_path_lets_the_arm_escape_a_joint_already_past_the_margin():
    """A margin must not become a trap.

    The margin stops a plan driving a joint toward its limit.  Judging the
    trajectory's first point -- the arm's own position -- by that same rule
    leaves no acceptable trajectory at all, since every one starts there: on
    2026-08-03 the arm sat at J5=175.32 deg against a 175.00 deg margin and
    four planners each found a way out, all refused on point 1 of 57.
    """
    session = _session(follow=True)
    session.tcp_from_joints = lambda _joints: np.eye(4)
    profile = SimpleNamespace(assert_tcp_path=lambda _points: 0)
    stuck = [176.0] + [0.0] * 6

    # Moving back toward the middle is allowed even though the start, and the
    # points interpolated near it, are past the margin.
    session.validate_planned_joints(
        [[120.0] + [0.0] * 6], 1.5, profile, start_joints_deg=stuck
    )

    # Going further into the limit from that same start is still refused.
    with pytest.raises(SafetyAbort, match="把关节推向控制器限位"):
        session.validate_planned_joints(
            [[177.5] + [0.0] * 6], 1.5, profile, start_joints_deg=stuck
        )


def test_planned_path_blocking_fallback_keeps_dense_checked_waypoints(
    monkeypatch,
):
    monkeypatch.setenv("BOTTLE_GRASP_CONTINUOUS_TRAJECTORY", "0")
    session = _session(follow=True)

    session.execute_planned_joints(
        [[0.5] * 7],
        75,
        0.25,
        expected_start_joints_deg=[0.0] * 7,
    )

    assert [command[0] for command in session.arm.commands] == [
        [0.25] * 7,
        [0.5] * 7,
    ]
    assert all(command[2:] == (0, 0, 1) for command in session.arm.commands)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_planned_path_rejects_nonfinite_joint_feedback(bad):
    session = _session(follow=True)
    original = session.joints_deg
    reads = 0

    def feedback():
        nonlocal reads
        reads += 1
        values = original()
        if reads >= 2:
            values[3] = bad
        return values

    session.joints_deg = feedback
    session.hold = lambda: None

    with pytest.raises(SafetyAbort, match="反馈含非有限数"):
        session.execute_planned_joints(
            [[1.0] * 7],
            3,
            1.5,
            expected_start_joints_deg=[0.0] * 7,
        )


def _execution_demo(*, left_joints=None):
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.args = SimpleNamespace(task_mode="from-start")
    demo.params = DemoParams()
    demo.stop_event = threading.Event()
    demo.stage = lambda *_args, **_kwargs: None

    class Left:
        @staticmethod
        def joints_deg():
            return list([0.0] * 7 if left_joints is None else left_joints)

    class Right:
        calls = 0

        @classmethod
        def execute_planned_joints(cls, *_args, **_kwargs):
            cls.calls += 1

        @staticmethod
        def hold():
            return None

    demo.left_robot = Left()
    demo.robot = Right()
    return demo


def _fresh_plan():
    return {
        "points_deg": [[0.0] * 7],
        "start_joints_deg": [0.0] * 7,
        "start_left_joints_deg": [0.0] * 7,
        "scene_captured_monotonic": time.monotonic(),
    }


def test_global_execution_rejects_stale_rgbd_scene_before_motion():
    demo = _execution_demo()
    plan = _fresh_plan()
    plan["scene_captured_monotonic"] -= demo.params.scene_max_age_s + 1.0

    with pytest.raises(SafetyAbort, match="规划场景已过期"):
        demo._execute_plan("test", plan)

    assert demo.robot.calls == 0


def test_global_execution_rejects_left_arm_snapshot_drift_before_motion():
    demo = _execution_demo(left_joints=[2.0] * 7)

    with pytest.raises(SafetyAbort, match="左臂已偏离"):
        demo._execute_plan("test", _fresh_plan())

    assert demo.robot.calls == 0


def test_scene_age_is_rechecked_after_blocking_left_arm_read(monkeypatch):
    """Regression: a scene just inside the window must not execute just outside it."""
    demo = _execution_demo()
    limit = demo.params.scene_max_age_s
    now = [100.0 + limit - 1.0]

    def delayed_left_read():
        now[0] = 100.0 + limit + 1.0
        return [0.0] * 7

    demo.left_robot.joints_deg = delayed_left_read
    monkeypatch.setattr("shelf_dispenser.orchestrator.time.monotonic", lambda: now[0])
    plan = _fresh_plan()
    plan["scene_captured_monotonic"] = 100.0

    with pytest.raises(SafetyAbort, match="规划场景已过期"):
        demo._execute_plan("test", plan)

    assert demo.robot.calls == 0


def test_local_task_path_uses_moveit_and_removes_only_target_cylinder():
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.args = SimpleNamespace(task_mode="from-observation")
    demo.params = DemoParams()
    demo.stage = lambda *_args, **_kwargs: None
    demo.scene_boxes = [{"id": "table"}]
    demo.cfg = SimpleNamespace(
        calibration=SimpleNamespace(
            T_base_right_to_camera_head=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -0.30],
                    [0.0, 0.0, 1.0, 0.30],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
    )
    demo.scene_voxels = [
        [0.01, 0.0, 0.05],  # locked bottle cylinder: contact is intentional
        [0.13, 0.0, 0.05],  # separate neighbouring object: must remain
        [0.0, 0.0, -0.13],  # table below cylinder: must remain
    ]
    calls = []

    class Safety:
        moveit_frame = "platform_base_link"

        @staticmethod
        def points_to_moveit(points):
            return points

    class Right:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

        @staticmethod
        def validate_planned_joints(*_args, **_kwargs):
            calls.append("sdk")
            return 1

    class Left:
        @staticmethod
        def joints_deg():
            return [0.0] * 7

    class Planner:
        @staticmethod
        def validate_exact_path(**kwargs):
            calls.append(("moveit", kwargs["obstacles"]))
            return {"success": True}

    demo.safety = Safety()
    demo.robot = Right()
    demo.left_robot = Left()
    demo.planner = Planner()

    demo._validate_local_joint_path(
        name="test",
        joints=[[1.0] * 7],
        target_base=np.zeros(3),
    )

    assert calls[0] == "sdk"
    assert calls[1] == (
        "moveit",
        [[0.13, 0.0, 0.05], [0.0, 0.0, -0.13]],
    )
