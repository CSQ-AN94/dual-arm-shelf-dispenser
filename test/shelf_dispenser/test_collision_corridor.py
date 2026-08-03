from __future__ import annotations

import numpy as np
import pytest

from shelf_dispenser.collision import check_approach_corridor
from shelf_dispenser.core import DemoParams, SafetyAbort


def _corridor_fixture(depth_m: float, *, patch=(30, 60, 30, 60)):
    depth = np.full((90, 90), np.nan, dtype=float)
    y1, y2, x1, x2 = patch
    depth[y1:y2, x1:x2] = depth_m
    intrinsics = np.array(
        [[250.0, 0.0, 45.0], [0.0, 250.0, 45.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    T_base_camera = np.eye(4)
    T_base_camera[2, 3] = -0.3

    class Camera:
        @staticmethod
        def get_latest_frames():
            return None, depth

        @staticmethod
        def get_camera_intrinsics():
            return intrinsics, None

    class Robot:
        @staticmethod
        def current_flange():
            return T_base_camera

        @staticmethod
        def current_tcp():
            return np.eye(4)

    return Camera(), Robot()


def _check(depth_m: float, target_box, *, patch=(30, 60, 30, 60)):
    camera, robot = _corridor_fixture(depth_m, patch=patch)
    return check_approach_corridor(
        camera=camera,
        robot=robot,
        target_box=target_box,
        target_base=np.array([0.0, 0.0, 0.085]),
        T_flange_camera=np.eye(4),
        params=DemoParams(),
    )


def _horizontal_corridor_check(depth_m: float):
    """Camera looks horizontally along base +Y; target is 30 cm away."""
    depth = np.full((90, 90), np.nan, dtype=float)
    depth[15:75, 15:75] = depth_m
    intrinsics = np.array(
        [[250.0, 0.0, 45.0], [0.0, 250.0, 45.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    T_base_camera = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.30],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    class Camera:
        @staticmethod
        def get_latest_frames():
            return None, depth

        @staticmethod
        def get_camera_intrinsics():
            return intrinsics, None

    class Robot:
        @staticmethod
        def current_flange():
            return T_base_camera

        @staticmethod
        def current_tcp():
            tcp = np.eye(4)
            tcp[:3, 3] = [0.0, -0.20, 0.0]
            return tcp

    return check_approach_corridor(
        camera=Camera(),
        robot=Robot(),
        target_box=(10, 10, 80, 80),
        target_base=np.zeros(3),
        T_flange_camera=np.eye(4),
        params=DemoParams(),
    )


def test_target_box_does_not_erase_a_foreground_obstacle_at_another_depth():
    # Base z=0.01 m: this patch lies in the TCP-to-target corridor but 75 mm
    # nearer than the locked target depth.  A 2-D whole-box mask hid it.
    with pytest.raises(SafetyAbort, match="夹爪前进通道被点云阻挡"):
        _check(0.31, (15, 15, 75, 75))


def test_target_box_removes_only_the_locked_bottle_depth_region():
    # Base z=0.055 m: 30 mm in front of the locked cylinder centre and inside
    # its current associated silhouette, representing the visible bottle skin.
    assert _check(0.355, (15, 15, 75, 75)) == 0


def test_target_box_depth_slab_covers_transparent_bottle_front_surface():
    # Real 2026-07-21 wrist frame: the locked robust depth was about 0.393 m
    # while supported bottle pixels reached 0.353 m.  With a horizontal view,
    # that 4 cm depth spread lies just outside a cylinder centred on the lock
    # and used to produce 103 false blockers at the final 8.5 cm pregrasp.
    assert _horizontal_corridor_check(0.26) == 0


def test_target_box_depth_slab_keeps_a_separate_foreground_obstacle():
    # Eight centimetres in front of the lock is outside the bounded bottle
    # depth slab even if it projects inside the detector box.
    with pytest.raises(SafetyAbort, match="夹爪前进通道被点云阻挡"):
        _horizontal_corridor_check(0.22)


def test_head_only_confirmation_does_not_reuse_or_invent_a_wrist_mask():
    # No box is available, but the explicit physical bottle cylinder is still
    # known and must let the target's own surface pass.
    assert _check(0.355, None) == 0


def test_head_only_target_cylinder_keeps_adjacent_obstacle_points():
    # Shift the patch just outside the physical bottle radius but keep enough
    # samples inside the 45 mm approach corridor.  A broad clearance hole would
    # erase this neighbour; a bounded occupancy cylinder must not.
    with pytest.raises(SafetyAbort, match="夹爪前进通道被点云阻挡"):
        _check(0.355, None, patch=(0, 90, 72, 78))


def test_corridor_gate_checks_the_actual_lifted_polyline_not_direct_chord():
    """A shelf can block the direct chord while the commanded dogleg is clear."""
    camera, robot = _corridor_fixture(0.355)
    params = DemoParams(
        target_occupancy_radius_m=0.001,
        target_occupancy_above_grasp_m=0.001,
        target_occupancy_below_grasp_m=0.001,
    )
    target = np.array([0.0, 0.0, 0.085])

    with pytest.raises(SafetyAbort, match="夹爪前进通道被点云阻挡"):
        check_approach_corridor(
            camera=camera,
            robot=robot,
            target_box=None,
            target_base=target,
            T_flange_camera=np.eye(4),
            params=params,
        )

    assert (
        check_approach_corridor(
            camera=camera,
            robot=robot,
            target_box=None,
            target_base=target,
            T_flange_camera=np.eye(4),
            params=params,
            corridor_waypoints_base=[
                [0.10, 0.0, 0.0],
                [0.10, 0.0, 0.085],
            ],
        )
        == 0
    )
