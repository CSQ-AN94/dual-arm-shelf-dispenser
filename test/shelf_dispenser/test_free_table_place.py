import numpy as np
import pytest

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.delivery_table import PlaceCandidate
from shelf_dispenser.free_table_place import (
    build_free_table_place_targets,
    demonstrated_place_from_trajectory,
)


BOTTLE_HEIGHT_M = 0.24
TABLE_HEIGHT_M = -0.3
CURRENT_JOINTS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
TAUGHT_JOINTS = [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0]

# Nearer to the arm, and nearer to where the bottle is being carried.
NEAR = PlaceCandidate((0.1, 0.5), TABLE_HEIGHT_M, 0.0, 60, 0.5)
# Farther out, and where the operator actually set a bottle down.
TAUGHT_PATCH = PlaceCandidate((0.4, 0.8), TABLE_HEIGHT_M, 1.0, 60, 0.5)


def tcp_from_bottle_transform() -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = [0.01, -0.02, 0.06]
    return transform


def taught_tcp(support_z: float = TABLE_HEIGHT_M) -> np.ndarray:
    """A taught endpoint whose bottle rests at ``support_z`` over TAUGHT_PATCH."""
    centre_z = support_z + BOTTLE_HEIGHT_M / 2.0
    transform = np.eye(4)
    transform[:3, 3] = [
        TAUGHT_PATCH.xy_base[0] - 0.01,
        TAUGHT_PATCH.xy_base[1] + 0.02,
        centre_z - 0.06,
    ]
    return transform


def demonstration_record(joints=TAUGHT_JOINTS) -> dict:
    sample = {"joints_deg": list(joints), "tcp_pose_moveit": {"xyz": [0.0, 0.0, 0.0]}}
    return {
        "schema_version": "grabber.demonstrated_joint_trajectory.v1",
        "pose_frame": "platform_base_link",
        "usable_for_planning_constraints": True,
        "fence_violations": [],
        "samples": [sample] * 4,
    }


def taught_place(support_z: float = TABLE_HEIGHT_M):
    return demonstrated_place_from_trajectory(
        demonstration_record(),
        tcp_from_joints=lambda joints: taught_tcp(support_z),
        tcp_from_bottle=tcp_from_bottle_transform(),
        bottle_height_m=BOTTLE_HEIGHT_M,
    )


def recording_solver(seeds: list):
    def solve(flange, seed_joints_deg):
        seeds.append(list(map(float, seed_joints_deg)))
        return [list(map(float, seed_joints_deg))]

    return solve


def build(**overrides):
    seeds: list = []
    kwargs = dict(
        candidates=[NEAR, TAUGHT_PATCH],
        current_tcp=np.eye(4),
        current_joints_deg=CURRENT_JOINTS,
        tcp_from_bottle=tcp_from_bottle_transform(),
        flange_from_tcp=np.eye(4),
        bottle_height_m=BOTTLE_HEIGHT_M,
        bottle_radius_m=0.03,
        release_gap_m=0.005,
        preplace_clearance_m=0.08,
        table_xy_min=[-1.0, -1.0],
        table_xy_max=[2.0, 2.0],
        solve_flange_ik_candidates=recording_solver(seeds),
    )
    kwargs.update(overrides)
    return build_free_table_place_targets(**kwargs), seeds


def test_place_height_and_upright_axis_come_from_the_measured_table():
    targets, _ = build()

    assert targets
    for target in targets:
        placed = target.place_tcp @ tcp_from_bottle_transform()
        # Bottle centre one half-body plus the release gap above the surface.
        assert np.isclose(placed[2, 3], TABLE_HEIGHT_M + 0.12 + 0.005)
        assert np.allclose(placed[:3, 2], [0.0, 0.0, 1.0])


def test_without_a_demonstration_the_search_starts_from_the_carrying_pose():
    targets, seeds = build()

    assert seeds and all(seed == CURRENT_JOINTS for seed in seeds)
    # Carried bottle sits at (0.01, -0.02), so the nearer patch is tried first.
    assert targets[0].candidate is NEAR


def test_a_demonstration_reseeds_the_ik_and_reorders_the_patches():
    targets, seeds = build(demonstration=taught_place())

    # The controller's IK is seed-dependent; every branch is now asked from
    # a configuration that has already placed a bottle on this table.
    assert seeds and all(seed == TAUGHT_JOINTS for seed in seeds)
    # And the patch the operator actually used is tried before the nearer one.
    assert targets[0].candidate is TAUGHT_PATCH


def test_a_demonstration_adds_no_endpoint_it_only_reorders():
    plain, _ = build()
    taught, _ = build(demonstration=taught_place())

    def endpoints(targets):
        return sorted(
            (
                target.candidate.xy_base,
                round(target.final_bottle_yaw_deg, 6),
                round(target.preplace_bottle_tilt_deg, 6),
                round(target.preplace_bottle_azimuth_deg, 6),
            )
            for target in targets
        )

    assert endpoints(plain) == endpoints(taught)


def test_a_taught_support_height_that_contradicts_the_table_is_refused():
    # The taught bottle reached its support by contact.  If the fitted plane
    # disagrees, the table or the chassis moved and neither height is usable.
    with pytest.raises(SafetyAbort, match="桌子或底盘可能已移动"):
        build(demonstration=taught_place(support_z=TABLE_HEIGHT_M + 0.06))


def test_a_tipped_taught_bottle_is_not_a_place_reference():
    tipped = np.eye(4)
    tipped[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    with pytest.raises(SafetyAbort, match="倾斜"):
        demonstrated_place_from_trajectory(
            demonstration_record(),
            tcp_from_joints=lambda joints: tipped,
            tcp_from_bottle=tcp_from_bottle_transform(),
            bottle_height_m=BOTTLE_HEIGHT_M,
        )


def test_a_demonstration_that_failed_its_own_fence_check_is_refused():
    record = demonstration_record()
    record["fence_violations"] = [{"index": 3}]
    with pytest.raises(SafetyAbort, match="示教未通过规划约束门禁"):
        demonstrated_place_from_trajectory(
            record,
            tcp_from_joints=lambda joints: taught_tcp(),
            tcp_from_bottle=tcp_from_bottle_transform(),
            bottle_height_m=BOTTLE_HEIGHT_M,
        )
