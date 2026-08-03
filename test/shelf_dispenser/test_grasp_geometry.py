"""Shared local grasp/place geometry contracts; no hardware."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shelf_dispenser.core import (
    DemoParams,
    SafetyAbort,
    interpolate_poses,
    matrix_pose,
    pose_matrix,
)
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.grasp_orientation import (
    GraspFrameSpec,
    authored_tcp_rotation,
    installed_rmg24_tcp_rotation,
    link7_to_tcp_fixed_mount,
)
from shelf_dispenser.safe_planner import PlanTarget


def test_grasp_stops_short_and_post_release_retreat_is_independent():
    demo = object.__new__(RunOrchestrator)
    demo.params = DemoParams()
    start = np.eye(4)
    start[2, 3] = 0.20
    target = np.array([0.0, 0.0, 0.50])

    pregrasp, grasp, _transit, full_path = demo._local_pick_place_geometry(
        start, target, np.eye(3)
    )

    np.testing.assert_allclose(
        pose_matrix(grasp)[:3, 3],
        [0.0, 0.0, 0.50 - demo.params.grasp_stop_short_m],
    )
    # Pregrasp keeps its old target-relative hover distance; shortening the
    # final insertion must not move the observation/hover waypoint.
    np.testing.assert_allclose(
        pose_matrix(pregrasp)[:3, 3],
        [0.0, 0.0, 0.50 - demo.params.pregrasp_standoff_m],
    )
    # The path ends after release at a separately configured, longer retreat.
    np.testing.assert_allclose(
        np.asarray(full_path[-1][:3]),
        [
            0.0,
            0.0,
            0.50
            - demo.params.grasp_stop_short_m
            - demo.params.retreat_standoff_m,
        ],
    )


def test_shared_geometry_uses_conservative_first_site_adjustment():
    params = DemoParams()

    assert params.grasp_stop_short_m == 0.030
    assert params.pregrasp_standoff_m == 0.085
    assert params.retreat_standoff_m == 0.150
    assert params.local_clearance_lift_m == 0.050
    assert params.transit_speed == 100
    assert params.travel_speed == 15
    assert params.final_speed == 15


def test_observation_transit_lifts_vertically_before_rotating_toward_pregrasp():
    """A direct pose interpolation sweeps the turning hand into a low shelf.

    The first local leg must preserve the collision-free observation
    orientation while gaining vertical clearance; only then may it turn and
    translate toward the bottle.
    """
    demo = object.__new__(RunOrchestrator)
    demo.params = DemoParams()
    start = np.eye(4)
    start[:3, :3] = Rotation.from_euler("x", 25, degrees=True).as_matrix()
    start[:3, 3] = [0.05, 0.40, -0.125]
    target = np.array([-0.03, 0.62, -0.08])
    grasp_rotation = Rotation.from_euler("y", 80, degrees=True).as_matrix()

    _pregrasp, _grasp, transit, _full_path = (
        demo._local_pick_place_geometry(start, target, grasp_rotation)
    )

    clearance = pose_matrix(transit[1])
    np.testing.assert_allclose(
        clearance[:3, 3],
        start[:3, 3] + [0.0, 0.0, demo.params.local_clearance_lift_m],
    )
    np.testing.assert_allclose(clearance[:3, :3], start[:3, :3])
    # Rotation toward the grasp orientation starts only after the clearance
    # waypoint; the two vertical interpolation states retain the start pose.
    np.testing.assert_allclose(pose_matrix(transit[0])[:3, :3], start[:3, :3])
    assert not np.allclose(pose_matrix(transit[2])[:3, :3], start[:3, :3])


def test_large_optional_roll_is_split_into_small_rotation_steps():
    start = pose_matrix(
        [0.09857, 0.57603, -0.13667, 0.369, 1.522, 2.48]
    )
    rolled = start.copy()
    rolled[:3, :3] = (
        start[:3, :3]
        @ Rotation.from_euler("z", 89.0, degrees=True).as_matrix()
    )

    poses = interpolate_poses(matrix_pose(start), matrix_pose(rolled), 0.045)

    # Optional small-roll IK recovery must remain safely interpolated if a
    # caller ever requests a larger orientation change.
    total_deg = np.degrees(
        Rotation.from_matrix(start[:3, :3].T @ rolled[:3, :3]).magnitude()
    )
    assert len(poses) == int(np.ceil(total_deg / 10.0))
    assert len(poses) > 1
    previous = start[:3, :3]
    for pose in poses:
        current = pose_matrix(pose)
        np.testing.assert_allclose(current[:3, 3], start[:3, 3], atol=1e-12)
        step_deg = np.degrees(
            Rotation.from_matrix(previous.T @ current[:3, :3]).magnitude()
        )
        assert step_deg <= 10.0 + 1e-9
        previous = current[:3, :3]


def test_runtime_candidate_uses_one_authored_horizontal_orientation():
    demo = object.__new__(RunOrchestrator)
    demo.params = DemoParams()
    demo.grasp_rotation = None
    demo.stage = lambda *_args: None
    attempts = []

    class Safety:
        @staticmethod
        def assert_tcp_point(_point, *, label):
            return None

    class Robot:
        @staticmethod
        def current_tcp():
            return np.eye(4)

        @staticmethod
        def plan_ik(poses, params, *, allow_first_jump=False):
            attempts.append(list(poses))
            return [[0.0] * 7 for _ in poses]

    demo.safety = Safety()
    demo.robot = Robot()
    demo._validate_local_joint_path = lambda **_kwargs: None

    pregrasp, grasp, transit = demo.candidate_path(
        np.array([0.0, 0.0, 0.50])
    )

    assert len(attempts) == 1
    np.testing.assert_allclose(pose_matrix(pregrasp)[:3, :3], np.eye(3))
    np.testing.assert_allclose(pose_matrix(grasp)[:3, :3], np.eye(3))
    np.testing.assert_allclose(pose_matrix(transit[-1])[:3, :3], np.eye(3))
    assert all(
        np.allclose(pose_matrix(pose)[:3, :3], np.eye(3))
        for pose in transit
    )


@pytest.mark.parametrize("pitch_deg", (16.0, 25.0))
def test_shelf_candidate_does_not_inherit_sloped_observation_pitch(pitch_deg):
    demo = object.__new__(RunOrchestrator)
    demo.params = DemoParams()
    demo.grasp_rotation = None
    demo.stage = lambda *_args: None
    authored = GraspFrameSpec(
        opening_normal_base=(0.0, 1.0, 0.0),
        finger_axis_base=(1.0, 0.0, 0.0),
        palm_vertical_base=(0.0, 0.0, -1.0),
    )

    class Safety:
        grasp_frame = authored

        @staticmethod
        def assert_tcp_point(_point, *, label):
            return None

    observation = np.eye(4)
    observation[:3, :3] = Rotation.from_euler(
        "x", pitch_deg, degrees=True
    ).as_matrix()

    class Robot:
        @staticmethod
        def current_tcp():
            return observation

        @staticmethod
        def plan_ik(poses, params, *, allow_first_jump=False):
            return [[0.0] * 7 for _ in poses]

    demo.safety = Safety()
    demo.robot = Robot()
    demo._validate_local_joint_path = lambda **_kwargs: None

    pregrasp, grasp, path = demo.candidate_path(
        np.array([0.0, 0.5, 0.0])
    )

    expected = authored_tcp_rotation(authored)
    np.testing.assert_allclose(
        pose_matrix(pregrasp)[:3, :3], expected, atol=1e-12
    )
    np.testing.assert_allclose(
        pose_matrix(grasp)[:3, :3], expected, atol=1e-12
    )
    np.testing.assert_allclose(
        pose_matrix(path[-1])[:3, :3], expected, atol=1e-12
    )
    np.testing.assert_allclose(expected[:, 2], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(expected[:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(expected[:, 1], [0.0, 0.0, -1.0])


def test_installed_rmg24_controller_tcp_accounts_for_mount_yaw():
    spec = GraspFrameSpec(
        opening_normal_base=(0.0, 1.0, 0.0),
        finger_axis_base=(1.0, 0.0, 0.0),
        palm_vertical_base=(0.0, 0.0, -1.0),
    )

    rotation = installed_rmg24_tcp_rotation(spec)

    np.testing.assert_allclose(rotation[:, 2], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(rotation[:, 1], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotation[:, 0], [0.0, 0.0, -1.0])
    # The +90° mount puts controller TCP +X on the authored palm direction.
    # Rz(-90°) puts it on -palm, i.e. the same grasp rolled 180° about the
    # approach axis, which is what shipped between 4bc47cc and this test.
    np.testing.assert_allclose(rotation[:, 0], spec.palm_vertical_base)


def test_installed_rmg24_tcp_matches_demonstrated_grasp():
    """Pin the mount against the 2026-08-02 real-robot teleop demonstration.

    Evidence: outputs/field_evidence/2026-08-02_real_robot/remote_test_evidence/
    guided_demo/right_arm_demonstration.json, the sample where the operator
    closed the gripper on the bottle (gripper_events.jsonl "close").  Columns
    are that r_link7 orientation carried into the profile frame.
    """
    spec = GraspFrameSpec(
        opening_normal_base=(0.0, 1.0, 0.0),
        finger_axis_base=(1.0, 0.0, 0.0),
        palm_vertical_base=(0.0, 0.0, -1.0),
    )
    demonstrated = np.array(
        [
            [-0.054, -0.986, -0.155],
            [-0.093, -0.149, 0.984],
            [-0.994, 0.068, -0.084],
        ]
    )

    error_deg = Rotation.from_matrix(
        installed_rmg24_tcp_rotation(spec).T
        @ Rotation.from_matrix(demonstrated).as_matrix()
    ).magnitude() * 180.0 / np.pi

    # The operator's wrist was ~11° off square; a sign slip would be ~180°.
    assert error_deg < 12.0


def test_authored_grasp_axes_are_orthonormal_and_right_handed():
    spec = GraspFrameSpec(
        opening_normal_base=(0.0, 2.0, 0.0),
        finger_axis_base=(3.0, 0.001, 0.0),
        palm_vertical_base=(0.0, 0.0, -4.0),
    )

    rotation = authored_tcp_rotation(spec)

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.cross(rotation[:, 0], rotation[:, 1]), rotation[:, 2]
    )


def test_link7_to_tcp_mount_is_two_audited_collinear_offsets():
    transform = link7_to_tcp_fixed_mount(
        link7_to_controller_flange_m=0.0172,
        controller_flange_to_tcp_m=0.151,
    )

    np.testing.assert_allclose(transform[:3, :3], np.eye(3))
    np.testing.assert_allclose(transform[:3, 3], [0.0, 0.0, 0.1682])


def test_precheck_and_runtime_use_the_same_authored_pose_sequence():
    demo = object.__new__(RunOrchestrator)
    demo.params = DemoParams()
    demo.grasp_rotation = None
    demo.stage = lambda *_args: None
    demo.stop_event = None
    demo.safety = type(
        "Safety",
        (),
        {
            "grasp_frame": GraspFrameSpec(
                opening_normal_base=(0.0, 1.0, 0.0),
                finger_axis_base=(1.0, 0.0, 0.0),
                palm_vertical_base=(0.0, 0.0, -1.0),
            ),
            "assert_tcp_point": staticmethod(
                lambda _point, *, label: None
            ),
        },
    )()
    start_tcp = np.eye(4)
    start_tcp[:3, :3] = Rotation.from_euler(
        "x", 25.0, degrees=True
    ).as_matrix()
    start_flange = start_tcp @ np.linalg.inv(demo.T_flange_tcp)
    captured = []

    class Robot:
        @staticmethod
        def current_tcp():
            return start_tcp

        @staticmethod
        def plan_ik(
            poses,
            params,
            *,
            allow_first_jump=False,
            seed_joints_deg=None,
        ):
            captured.append([np.asarray(pose, dtype=float) for pose in poses])
            return [[0.0] * 7 for _ in poses]

    demo.robot = Robot()
    demo._validate_local_joint_path = lambda **_kwargs: None
    target_point = np.array([0.0, 0.50, 0.0])
    target = PlanTarget(
        label="same",
        flange=start_flange,
        goal_joints=tuple([0.0] * 7),
    )

    demo._plan_grasp_precheck(target, target_point, 3.0)
    demo.candidate_path(target_point)

    assert len(captured) == 2
    np.testing.assert_allclose(captured[0], captured[1], atol=1e-12)


def test_runtime_place_back_uses_the_longer_retreat_distance():
    demo = object.__new__(RunOrchestrator)
    demo.params = DemoParams()
    demo.stage = lambda *_args: None

    class Safety:
        @staticmethod
        def assert_tcp_point(_point, *, label):
            assert label == "放回后退开点"

    class Robot:
        def __init__(self):
            self.moves = []

        @staticmethod
        def current_tcp():
            return np.eye(4)

        @staticmethod
        def open_gripper(_params):
            pass

        @staticmethod
        def close_empty_gripper(_params):
            pass

        def move_linear(self, pose, _speed):
            self.moves.append((np.asarray(pose[:3], dtype=float), _speed))

    demo.safety = Safety()
    demo.robot = Robot()
    demo._plan_local_leg = lambda _name, build_path, _params: build_path()

    demo._place_back()

    np.testing.assert_allclose(
        demo.robot.moves[-1][0],
        [0.0, 0.0, -demo.params.retreat_standoff_m],
    )
    assert demo.robot.moves[0][1] == demo.params.final_speed
    assert demo.robot.moves[-1][1] == demo.params.travel_speed
