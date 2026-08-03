"""Regression tests for RealMan SDK tool-frame semantics."""

from types import SimpleNamespace

import numpy as np

from bottle_grasp.core import DemoParams, pose_matrix
from bottle_grasp.robot import RobotSession


class _Algo:
    def rm_algo_forward_kinematics(self, _joints, _flag):
        return [0.0] * 6

    def rm_algo_inverse_kinematics(self, _params):
        return 0, [0.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]

    def rm_algo_set_joint_min_limit(self, _limits):
        return None

    def rm_algo_set_joint_max_limit(self, _limits):
        return None


class _Arm:
    def rm_get_joint_min_pos(self):
        return 0, [-180.0] * 7

    def rm_get_joint_max_pos(self):
        return 0, [180.0] * 7


def _session():
    session = RobotSession.__new__(RobotSession)
    session.tcp_z_m = 0.151
    session.model_flange_offset_m = 0.0172
    session.link7_to_controller_flange = np.eye(4)
    session.link7_to_controller_flange[2, 3] = 0.0172
    session.algo = _Algo()
    session.arm = _Arm()
    session.joints_deg = lambda: [0.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    session.ik_params = lambda *args: args
    session._tool_offsets = []
    session._set_algo_tool_transform = lambda transform: (
        session._tool_offsets.append(np.asarray(transform, dtype=float).copy())
    )
    return session


def test_sdk_fk_composes_link7_flange_and_flange_tcp_as_distinct_segments():
    session = _session()

    session.controller_flange_from_joints([0.0] * 7)
    session.tcp_from_joints([0.0] * 7)

    T_flange_tcp = np.eye(4)
    T_flange_tcp[2, 3] = 0.151
    np.testing.assert_allclose(
        session._tool_offsets[0], session.link7_to_controller_flange
    )
    np.testing.assert_allclose(
        session._tool_offsets[1],
        session.link7_to_controller_flange @ T_flange_tcp,
    )


def test_sdk_ik_uses_the_same_two_segment_link7_tool_chain_as_fk():
    session = _session()

    session.solve_flange_ik(np.eye(4), DemoParams())
    session.plan_ik([[0.0] * 6], DemoParams())

    T_flange_tcp = np.eye(4)
    T_flange_tcp[2, 3] = 0.151
    np.testing.assert_allclose(
        session._tool_offsets[0], session.link7_to_controller_flange
    )
    np.testing.assert_allclose(
        session._tool_offsets[1],
        session.link7_to_controller_flange @ T_flange_tcp,
    )


def test_controller_tool_stays_flange_relative_while_algo_uses_link7_chain():
    session = _session()
    controller_frames = []
    session.rm_frame_t = lambda name, pose, *_args: (name, list(pose))

    class ControlArm(_Arm):
        @staticmethod
        def rm_set_manual_tool_frame(frame):
            controller_frames.append(frame)
            return 0

        @staticmethod
        def rm_change_tool_frame(_name):
            return 0

    session.arm = ControlArm()

    session.set_tcp()

    assert controller_frames[0][1][2] == 0.151
    assert session._tool_offsets[-1][2, 3] == 0.1682


def test_sdk_chain_preserves_noncommuting_mount_segments_at_every_boundary():
    """A full mount must not collapse to, or reverse, its scalar-Z fallback."""
    session = _session()
    session.link7_to_controller_flange = np.array(
        [
            [0.8660254, -0.5, 0.0, 0.0172],
            [0.5, 0.8660254, 0.0, -0.004],
            [0.0, 0.0, 1.0, 0.003],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    session.tcp_transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.006],
            [0.0, 0.9396926, -0.3420201, 0.012],
            [0.0, 0.3420201, 0.9396926, 0.151],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    controller_frames = []
    session.rm_frame_t = lambda name, pose, *_args: (name, list(pose))

    class ControlArm(_Arm):
        @staticmethod
        def rm_set_manual_tool_frame(frame):
            controller_frames.append(frame)
            return 0

        @staticmethod
        def rm_change_tool_frame(_name):
            return 0

    session.arm = ControlArm()
    expected_algo_tcp = (
        session.link7_to_controller_flange @ session.tcp_transform
    )
    reversed_algo_tcp = (
        session.tcp_transform @ session.link7_to_controller_flange
    )
    assert not np.allclose(expected_algo_tcp, reversed_algo_tcp)

    session.set_tcp()
    session.controller_flange_from_joints([0.0] * 7)
    session.tcp_from_joints([0.0] * 7)
    session.solve_flange_ik(np.eye(4), DemoParams())
    session.plan_ik([[0.0] * 6], DemoParams())

    assert controller_frames[0][0] == "bottleTCP"
    np.testing.assert_allclose(
        pose_matrix(controller_frames[0][1]), session.tcp_transform, atol=1e-6
    )
    assert len(session._tool_offsets) == 5
    for actual, expected in zip(
        session._tool_offsets,
        (
            expected_algo_tcp,
            session.link7_to_controller_flange,
            expected_algo_tcp,
            session.link7_to_controller_flange,
            expected_algo_tcp,
        ),
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_flange_ik_candidates_try_the_equivalent_j7_turn_without_changing_pose():
    """RM75 J7 is multi-turn but bounded: -170deg and +190deg are one pose.

    A single SDK seed keeps following the current turn until it runs into the
    +/-360deg software limit.  Observation planning needs both bounded turns
    so the complete continuation can choose the one with usable margin.
    """
    session = _session()
    session.joints_deg = lambda: [0.0, 20.0, 20.0, 20.0, 20.0, 20.0, -170.0]

    class MultiTurnArm(_Arm):
        def rm_get_joint_min_pos(self):
            return 0, [-178.0, -130.0, -178.0, -135.0, -178.0, -128.0, -360.0]

        def rm_get_joint_max_pos(self):
            return 0, [178.0, 130.0, 178.0, 135.0, 178.0, 128.0, 360.0]

    class SeedFollowingAlgo(_Algo):
        def __init__(self):
            self.seeds = []

        def rm_algo_inverse_kinematics(self, args):
            seed = list(map(float, args[0]))
            self.seeds.append(seed)
            solution = [0.0, 20.0, 20.0, 20.0, 20.0, 20.0, seed[6]]
            return 0, solution

    session.arm = MultiTurnArm()
    session.algo = SeedFollowingAlgo()

    candidates = session.solve_flange_ik_candidates(
        np.eye(4), DemoParams(), seed_joints_deg=session.joints_deg()
    )

    assert [candidate[6] for candidate in candidates] == [-170.0, 190.0]
    assert [seed[6] for seed in session.algo.seeds] == [-170.0, 190.0]


def test_flange_ik_candidates_include_distinct_rm75_arm_angle_solution():
    """The RM75 arm-angle API exposes elbow geometry, not just J7 turns."""
    session = _session()

    class ArmAngleAlgo(_Algo):
        def rm_algo_calculate_arm_angle_from_config_rm75(self, _seed):
            return 0, -60.0

        def rm_algo_inverse_kinematics_rm75_for_arm_angle(
            self, _params, arm_angle
        ):
            if arm_angle != -30.0:
                return -1, [0.0] * 7
            return 0, [40.0, 60.0, -25.0, 80.0, 10.0, -50.0, -20.0]

    session.algo = ArmAngleAlgo()

    candidates = session.solve_flange_ik_candidates(
        np.eye(4), DemoParams(), seed_joints_deg=session.joints_deg()
    )

    assert [0.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0] in candidates
    assert [40.0, 60.0, -25.0, 80.0, 10.0, -50.0, -20.0] in candidates
