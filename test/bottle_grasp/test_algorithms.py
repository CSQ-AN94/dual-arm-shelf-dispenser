import json
from pathlib import Path

import threading
import numpy as np
import pytest

from bottle_grasp.core import (
    DemoParams,
    Detection,
    Localization,
    SafetyAbort,
    interpolate_poses,
    stop_reason,
)
from bottle_grasp.collision import classify_moveit_collision_probe
from bottle_grasp.perception import depth_point_for_detection, robust_near_cluster
from bottle_grasp.safety import load_safety_profile
from bottle_grasp.scene import (
    build_non_target_scene_voxels,
    build_scene_voxels,
    build_target_occupancy_voxels,
    union_scene_voxels,
)
from bottle_grasp.shelf_model import FACE_SPECS


def test_stop_reason_is_generic_without_a_recorded_source():
    event = threading.Event()
    event.set()
    assert stop_reason(event) == "用户停止"


def test_stop_reason_surfaces_a_guard_source_instead_of_the_generic_message():
    # 2026-07-19: a bare stop_event.is_set() check raised an indistinguishable
    # "用户停止" whether a person hit Ctrl+C or LeftArmStabilityGuard tripped
    # on left-arm drift mid-task — the guard's real, informative error was
    # separately swallowed by task.py's exception handler, so the operator
    # only ever saw the generic message and had no way to tell them apart.
    event = threading.Event()
    setattr(event, "source", "left_arm_drift")
    event.set()
    reason = stop_reason(event)
    assert reason != "用户停止"
    assert "left_arm_drift" in reason


def test_interpolate_respects_max_translation_step():
    start = [0, 0, 0, 0, 0, 0]
    end = [0.11, 0, 0, 0, 0, 0]
    poses = interpolate_poses(start, end, 0.045)
    assert len(poses) == 3
    last = np.array(start[:3], float)
    for pose in poses:
        assert np.linalg.norm(np.array(pose[:3]) - last) <= 0.045 + 1e-9
        last = np.array(pose[:3])


def test_robust_depth_prefers_supported_near_cluster():
    params = DemoParams()
    rng = np.random.default_rng(3)
    bottle = rng.normal(0.31, 0.003, 70)
    background = rng.normal(0.51, 0.005, 260)
    z, mad = robust_near_cluster(
        np.r_[bottle, background, 0, np.nan], params
    )
    assert abs(z - 0.31) < 0.01
    assert mad < 0.01


def test_depth_point_rejects_background_and_deprojects():
    params = DemoParams()
    depth = np.full((100, 100), 0.55, np.float32)
    depth[22:78, 44:57] = 0.30
    detection = Detection((35, 10, 65, 90), 0.9, "bottle")
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    point, z, _, pixel = depth_point_for_detection(depth, detection, K, params)
    assert abs(z - 0.30) < 0.015
    assert abs(point[2] - 0.30) < 1e-6
    assert abs(pixel[0] - 50) < 5


def test_grasp_pixel_height_is_deterministic_and_above_midline():
    # 纵向抓取点必须是检测框固定比例，不随有效深度像素分布漂移。
    # 2026-07-20 货架实测将它提到中线上方，给 r_hand 留出层板净空。
    params = DemoParams()
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    detection = Detection((35, 10, 65, 90), 0.9, "bottle")
    rng = np.random.default_rng(11)
    for seed in range(3):
        depth = np.full((100, 100), 0.55, np.float32)
        depth[22:78, 44:57] = 0.30 + rng.normal(0, 0.001, (56, 13))
        _, _, _, pixel = depth_point_for_detection(depth, detection, K, params)
        expected_v = 10 + params.grasp_height_fraction * 80
        assert pixel[1] == expected_v
    assert expected_v < 0.5 * (10 + 90)  # 高于框中线


def test_shelf_profile_uses_a_higher_grasp_band_without_changing_table():
    config = Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json"
    shelf = load_safety_profile(
        config, "shelf_template", require_verified=False
    )
    table = load_safety_profile(config, "table_demo", require_verified=False)

    assert shelf.grasp_height_fraction == pytest.approx(0.40)
    assert table.grasp_height_fraction is None
    # Same 80 px bottle box: shelf samples 4 px higher than the 0.45 default.
    assert 10 + shelf.grasp_height_fraction * 80 == pytest.approx(42.0)
    assert 10 + DemoParams().grasp_height_fraction * 80 == pytest.approx(46.0)


@pytest.mark.parametrize(
    "value", [0.0, 1.0, -0.1, 1.1, "nan", "not-a-number"]
)
def test_safety_profile_rejects_invalid_grasp_height_fraction(tmp_path, value):
    source = Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["profiles"]["shelf_template"]["grasp_height_fraction"] = value
    config = tmp_path / "profiles.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SafetyAbort, match="grasp_height_fraction"):
        load_safety_profile(
            config, "shelf_template", require_verified=False
        )


def test_scene_builds_generic_rgbd_voxels():
    params = DemoParams(
        head_min_depth_m=0.25,
        head_max_depth_m=1.0,
        scene_target_clearance_m=0.03,
    )
    depth = np.full((100, 100), 0.50, np.float32)
    depth[20:81, 40:61] = 0.60
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    localization = Localization(
        point_camera=[0.0, 0.0, 0.60],
        point_base=[0.0, 0.0, 0.60],
        pixel=[50.0, 50.0],
        depth_m=0.60,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[40, 20, 60, 80],
        confidence=0.9,
        frame_count=7,
    )
    voxels = build_scene_voxels(
        depth,
        K,
        np.eye(4),
        localization,
        params,
    )
    assert voxels


def test_global_scene_never_erases_obstacle_beside_target_or_inside_box():
    """Regression for the removed 14 cm target hole.

    Observation transfer is non-contact motion.  A point 5 cm beside the
    bottle must stay occupied even when it projects inside the detector box;
    otherwise MoveIt is given fabricated free space at exactly the task site.
    """
    params = DemoParams(
        head_min_depth_m=0.25,
        head_max_depth_m=1.0,
        scene_voxel_m=0.01,
        scene_target_clearance_m=0.14,
        scene_image_bottom_crop=100,
    )
    depth = np.full((100, 100), 0.60, np.float32)
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    localization = Localization(
        point_camera=[0.0, 0.0, 0.60],
        point_base=[0.0, 0.0, 0.60],
        pixel=[50.0, 50.0],
        depth_m=0.60,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[35, 20, 65, 80],
        confidence=0.9,
        frame_count=7,
    )

    voxels = np.asarray(
        build_scene_voxels(depth, K, np.eye(4), localization, params)
    )

    # Sample u=60, v=48 is 6.7 cm from the target and lies inside the box.
    expected = np.array([0.065, -0.015, 0.605])
    assert np.min(np.linalg.norm(voxels - expected, axis=1)) < 1e-9


def test_full_frame_scene_keeps_obstacles_in_the_legacy_bottom_crop():
    params = DemoParams(
        head_min_depth_m=0.25,
        head_max_depth_m=1.0,
        scene_voxel_m=0.01,
        scene_image_bottom_crop=60,
    )
    depth = np.full((100, 100), 0.50, np.float32)
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    localization = Localization(
        point_camera=[0.0, 0.0, 0.50],
        point_base=[0.0, 0.0, 0.50],
        pixel=[50.0, 50.0],
        depth_m=0.50,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[45, 40, 55, 60],
        confidence=0.9,
        frame_count=7,
    )

    cropped = np.asarray(
        build_scene_voxels(depth, K, np.eye(4), localization, params)
    )
    full = np.asarray(
        build_scene_voxels(
            depth,
            K,
            np.eye(4),
            localization,
            params,
            bottom_crop=depth.shape[0],
        )
    )
    lower_strip_voxel = np.array([-0.005, 0.165, 0.505])
    assert np.min(np.linalg.norm(cropped - lower_strip_voxel, axis=1)) > 0.05
    assert np.min(np.linalg.norm(full - lower_strip_voxel, axis=1)) < 0.011


def test_target_occupancy_voxels_do_not_claim_wide_box_edge():
    """A wide detector box cannot turn a separate object into target space."""
    params = DemoParams(
        head_min_depth_m=0.25,
        head_max_depth_m=1.0,
        scene_voxel_m=0.065,
        scene_image_bottom_crop=100,
    )
    depth = np.full((100, 100), 0.90, np.float32)
    depth[20:81, 35:66] = 0.60
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    localization = Localization(
        point_camera=[0.0, 0.0, 0.60],
        point_base=[0.0, 0.0, 0.60],
        pixel=[50.0, 50.0],
        depth_m=0.60,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[35, 20, 65, 80],
        confidence=0.9,
        frame_count=7,
    )

    target_voxels = np.asarray(
        build_target_occupancy_voxels(
            depth, K, np.eye(4), localization, params
        )
    )

    # u=36 is 8.4 cm from the centre ray at 60 cm. Treating the whole
    # detector box as target would create fabricated free space here.
    edge_voxel = np.array([-0.0975, -0.0325, 0.6175])
    assert np.min(np.linalg.norm(target_voxels - edge_voxel, axis=1)) > 1e-9


def test_target_occupancy_voxels_keep_foreground_object_inside_detector_box():
    """The silhouette mask is depth-bounded; it is not a whole-box hole."""
    params = DemoParams(
        head_min_depth_m=0.25,
        head_max_depth_m=1.0,
        scene_voxel_m=0.01,
        scene_image_bottom_crop=100,
    )
    depth = np.full((100, 100), 0.90, np.float32)
    depth[20:81, 35:66] = 0.60
    depth[45:56, 55:61] = 0.50
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    localization = Localization(
        point_camera=[0.0, 0.0, 0.60],
        point_base=[0.0, 0.0, 0.60],
        pixel=[50.0, 50.0],
        depth_m=0.60,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[35, 20, 65, 80],
        confidence=0.9,
        frame_count=7,
    )

    target_voxels = np.asarray(
        build_target_occupancy_voxels(
            depth, K, np.eye(4), localization, params
        )
    )

    foreground_voxel = np.array([0.025, -0.005, 0.505])
    assert np.min(
        np.linalg.norm(target_voxels - foreground_voxel, axis=1)
    ) > 1e-9


def test_target_occupancy_includes_transparent_bottle_rear_surface():
    params = DemoParams(
        head_min_depth_m=0.25,
        head_max_depth_m=1.0,
        scene_voxel_m=0.01,
        scene_image_bottom_crop=100,
    )
    depth = np.full((100, 100), 0.90, np.float32)
    depth[20:81, 35:66] = 0.60
    depth[45:56, 45:56] = 0.675
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    localization = Localization(
        point_camera=[0.0, 0.0, 0.60],
        point_base=[0.0, 0.0, 0.60],
        pixel=[50.0, 50.0],
        depth_m=0.60,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[35, 20, 65, 80],
        confidence=0.9,
        frame_count=7,
    )

    target_voxels = np.asarray(
        build_target_occupancy_voxels(
            depth, K, np.eye(4), localization, params
        )
    )

    rear_voxel = np.array([-0.015, -0.015, 0.675])
    assert np.min(np.linalg.norm(target_voxels - rear_voxel, axis=1)) < 1e-9


def test_target_occupancy_includes_lower_shelf_bottle_silhouette():
    params = DemoParams(
        head_min_depth_m=0.25,
        head_max_depth_m=1.0,
        scene_voxel_m=0.01,
        scene_image_bottom_crop=100,
    )
    depth = np.full((100, 100), 0.90, np.float32)
    depth[20:81, 35:66] = 0.60
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    T_base_camera = np.eye(4)
    T_base_camera[:3, :3] = np.array(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float
    )
    localization = Localization(
        point_camera=[-0.012, -0.048, 0.60],
        point_base=[-0.012, 0.60, 0.048],
        pixel=[48.0, 42.0],
        depth_m=0.60,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[35, 20, 65, 80],
        confidence=0.9,
        frame_count=7,
    )

    target_voxels = np.asarray(
        build_target_occupancy_voxels(
            depth, K, T_base_camera, localization, params
        )
    )

    # v=72 is 18 cm below the shelf lock in base Z, but still belongs to the
    # same detector silhouette and depth surface.
    lower_bottle_voxel = np.array([-0.015, 0.605, -0.135])
    assert np.min(
        np.linalg.norm(target_voxels - lower_bottle_voxel, axis=1)
    ) < 1e-9


def test_target_occupancy_keeps_same_depth_neighbour_inside_detector_box():
    params = DemoParams(
        head_min_depth_m=0.25,
        head_max_depth_m=1.0,
        scene_voxel_m=0.01,
        scene_image_bottom_crop=100,
    )
    depth = np.full((100, 100), 0.90, np.float32)
    depth[20:81, 25:81] = 0.60
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    localization = Localization(
        point_camera=[0.0, 0.0, 0.60],
        point_base=[0.0, 0.0, 0.60],
        pixel=[50.0, 50.0],
        depth_m=0.60,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[25, 20, 80, 80],
        confidence=0.9,
        frame_count=7,
    )

    target_voxels = np.asarray(
        build_target_occupancy_voxels(
            depth, K, np.eye(4), localization, params
        )
    )

    # u=74 is 14.4 cm sideways at the same depth and still inside the box.
    neighbour_voxel = np.array([0.145, -0.005, 0.605])
    assert np.min(
        np.linalg.norm(target_voxels - neighbour_voxel, axis=1)
    ) > 1e-9


def test_non_target_filter_keeps_mixed_coarse_voxel_as_obstacle():
    """Target and neighbour points sharing one 65 mm cell stay occupied."""
    params = DemoParams(
        head_min_depth_m=0.25,
        head_max_depth_m=1.0,
        scene_voxel_m=0.065,
        scene_image_bottom_crop=100,
    )
    depth = np.full((100, 100), 0.90, np.float32)
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    T_base_camera = np.eye(4)
    T_base_camera[0, 3] = 0.02
    localization = Localization(
        point_camera=[-0.018, -0.018, 0.90],
        point_base=[0.002, -0.018, 0.90],
        pixel=[48.0, 48.0],
        depth_m=0.90,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[40, 20, 70, 80],
        confidence=0.9,
        frame_count=7,
    )

    target = np.asarray(
        build_target_occupancy_voxels(
            depth, K, T_base_camera, localization, params
        )
    )
    non_target = np.asarray(
        build_non_target_scene_voxels(
            depth, K, T_base_camera, localization, params
        )
    )
    target_keys = {
        tuple(item)
        for item in np.floor(target / params.scene_voxel_m).astype(int)
    }
    non_target_keys = {
        tuple(item)
        for item in np.floor(non_target / params.scene_voxel_m).astype(int)
    }

    # Samples u=48 (target) and u=54 (5.4 cm lateral neighbour) land in
    # the same base-frame voxel. Point-level filtering must retain that cell.
    assert target_keys & non_target_keys


def test_scene_budget_overflow_aborts_instead_of_dropping_far_obstacles():
    params = DemoParams(
        head_min_depth_m=0.25,
        head_max_depth_m=1.0,
        scene_voxel_m=0.01,
        scene_image_bottom_crop=100,
        scene_max_voxels=5,
    )
    depth = np.full((100, 100), 0.60, np.float32)
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    localization = Localization(
        point_camera=[0.0, 0.0, 0.60],
        point_base=[0.0, 0.0, 0.60],
        pixel=[50.0, 50.0],
        depth_m=0.60,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[35, 20, 65, 80],
        confidence=0.9,
        frame_count=7,
    )

    with pytest.raises(SafetyAbort, match="拒绝丢弃远处障碍"):
        build_scene_voxels(depth, K, np.eye(4), localization, params)


def test_union_scene_voxels_keeps_a_voxel_seen_in_only_one_frame():
    """A surface that flickers in and out (dark/specular/grazing angle) is
    still a real obstacle. Union, not majority vote: 1-of-3 frames must be
    enough to keep it occupied."""
    params = DemoParams(scene_max_voxels=50)
    frame_a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    frame_b = [[0.0, 0.0, 0.0]]
    frame_c = [[0.0, 0.0, 0.0]]
    merged = union_scene_voxels([frame_a, frame_b, frame_c], params)
    assert [1.0, 0.0, 0.0] in merged
    assert [0.0, 0.0, 0.0] in merged
    assert len(merged) == 2


def test_union_scene_voxels_dedupes_identical_grid_cells():
    params = DemoParams(scene_max_voxels=50)
    merged = union_scene_voxels(
        [[[0.1, 0.2, 0.3]], [[0.1, 0.2, 0.3]], [[0.1, 0.2, 0.3]]], params
    )
    assert merged == [[0.1, 0.2, 0.3]]


def test_union_scene_voxels_budget_applies_to_the_union_not_one_frame():
    """A budget check that only looked at the smallest single frame would
    let the merged scene (what MoveIt actually receives) silently exceed
    the safety cap."""
    params = DemoParams(scene_max_voxels=3)
    frame_a = [[float(i), 0.0, 0.0] for i in range(2)]
    frame_b = [[float(i), 1.0, 0.0] for i in range(2)]
    with pytest.raises(SafetyAbort, match="拒绝丢弃远处障碍"):
        union_scene_voxels([frame_a, frame_b], params)


def test_union_scene_voxels_rejects_empty_frame_list():
    with pytest.raises(SafetyAbort, match="空的帧列表"):
        union_scene_voxels([], DemoParams())


def test_union_scene_voxels_rejects_malformed_point():
    with pytest.raises(SafetyAbort, match="第 2 帧"):
        union_scene_voxels([[[0.0, 0.0, 0.0]], [[float("nan"), 0.0, 0.0]]], DemoParams())


def test_observation_direction_normalization_does_not_mutate_target():
    target = np.array([-0.01, 0.65, -0.04])
    original = target.copy()
    horizontal = np.array(target[:2], dtype=float, copy=True)
    horizontal /= np.linalg.norm(horizontal)
    camera_position = target.copy()
    camera_position[:2] -= horizontal * 0.32
    assert np.array_equal(target, original)
    assert abs(np.linalg.norm(camera_position[:2] - target[:2]) - 0.32) < 1e-9


def test_electronic_fence_profile_accepts_task_and_home_zones():
    profile = load_safety_profile(
        Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json",
        "table_demo",
        require_verified=False,
    )
    profile.assert_tcp_point([0.31, -0.03, -0.59], label="home")
    profile.assert_tcp_point([0.00, 0.65, -0.04], label="bottle")
    # 2026-07-15: 桌面实测后加入 table_top 禁入盒，必须一并进入 MoveIt 世界
    assert [box["id"] for box in profile.moveit_collision_boxes()] == [
        "fence_table_top"
    ]


def test_shelf_profile_accepts_20260720_center_bottle_target():
    """The measured centre bottle must sit above, not inside, the shelf."""
    profile = load_safety_profile(
        Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json",
        "shelf_template",
        require_verified=False,
    )
    # 2026-07-20 site check: 7/7-frame localization after centring the bottle.
    # A follow-up read-only shelf survey measured the conservative shelf-bottom
    # bound at z=-0.2140 m (1683 inliers), so this z=-0.1366 m grasp point is
    # physically about 7.7 cm above the protected shelf surface.
    profile.assert_tcp_point(
        [-0.1133, 0.6976, -0.1366], label="measured centre bottle"
    )
    # The old narrow left/right bounds came from neighbouring bottles in one
    # early frame, not rigid shelf panels.  With one bottle on the shelf they
    # became stale obstacles only 7.5 cm from the target and blocked every
    # +15°/+30° grasp that otherwise cleared the real shelf bottom.  Current
    # neighbouring objects belong to the per-run RGB-D voxel scene instead.
    ids = {box.id for box in profile.keepout_boxes}
    assert "shelf_slot_left_guard" not in ids
    assert "shelf_slot_right_guard" not in ids
    dynamic_faces = {
        box.id for box in profile.keepout_boxes if box.id in FACE_SPECS
    }
    assert dynamic_faces == {"shelf_bottom", "shelf_top", "shelf_back"}


def test_rgbd_voxel_centres_already_inside_fence_are_not_double_inflated():
    profile = load_safety_profile(
        Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json",
        "shelf_template",
        require_verified=False,
    )
    # 2026-08-02 lift=647 mm repeat fit moved the conservative support top to
    # z=-0.2432; choose a point inside that measured shelf volume.
    represented_shelf = [-0.10, 0.70, -0.30]
    bottle_side_obstacle = [0.10, 0.70, -0.05]

    moveit_points = profile.moveit_obstacles_outside_fences(
        [represented_shelf, bottle_side_obstacle],
        profile.moveit_collision_boxes(),
    )

    assert len(moveit_points) == 1
    np.testing.assert_allclose(
        moveit_points[0], profile.point_to_moveit(bottle_side_obstacle)
    )


def test_shelf_profile_accepts_reused_home_pose_tcp():
    profile = load_safety_profile(
        Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json",
        "shelf_template",
        require_verified=False,
    )
    # 2026-07-20 site check FK at the real starting joints.  shelf_template
    # reuses table_demo's home_joints_deg, so its entry corridor must include
    # the same already-demonstrated low home TCP instead of rejecting point 1.
    profile.assert_tcp_point(
        [0.2361, 0.0245, -0.5727], label="reused real home TCP"
    )


def test_shelf_home_corridor_reaches_outer_workspace_clearance_boundary():
    profile = load_safety_profile(
        Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json",
        "shelf_template",
        require_verified=False,
    )
    # 2026-07-30 live MTC current→pregrasp path: this free-space TCP is still
    # 3.8 cm inside the outer workspace and far from every shelf keepout.
    profile.assert_tcp_point(
        [0.2054, -0.2615, -0.5273], label="live unfold corridor"
    )


def test_moveit_keepout_padding_is_horizontal_only_so_supports_stay_graspable():
    profile = load_safety_profile(
        Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json",
        "table_demo",
        require_verified=False,
    )
    item = profile.moveit_collision_boxes()[0]
    # 变换含 yaw 180° 旋转，转回 profile 系必须用完整刚体逆，不能只减平移
    T_profile_from_moveit = np.linalg.inv(profile.T_moveit_from_profile)
    center_profile = (
        T_profile_from_moveit @ np.r_[np.asarray(item["center"]), 1.0]
    )[:3]
    half_size = np.asarray(item["size"]) / 2

    # Horizontal padding stays. 2026-07-17 real-hardware `observe`: accepted
    # plans skimmed 1-1.7 cm past the padded boundary and were rejected by the
    # finer offline fence recheck. That root cause was tightened 4x at the
    # source (longest_valid_segment_fraction), and the padding was dialed back
    # from +5cm to +2cm on 2026-07-18.
    padding = profile.clearance_m + 0.02
    assert np.allclose(
        center_profile - half_size,
        [-1 - padding, 0.36 - padding, -0.75],
    )
    # But the TOP face must stay at the measured surface. It only ever grew
    # upward, which does no protective work anywhere: shelf_top is a ceiling
    # (growing up moves it away), shelf_back is vertical, and a support surface
    # is the one thing the hand must reach *above* to grasp anything standing
    # on it. Measured from r_hand.STL, the hand is strongly asymmetric about
    # the tool axis: 97 mm one side, 34 mm the other, and roll 0 vs roll 180
    # decides which faces the shelf floor. For the 2026-07-30 target
    # (TCP z=-0.0547, shelf_bottom top -0.1830) the true clearance of 128 mm
    # fits 97 mm, while +5cm of upward padding left 78 mm: 78 < 97 < 128. That
    # was the whole reason roll 180 was refused for r_hand vs fence_shelf_bottom.
    assert np.allclose(
        center_profile + half_size,
        [1 + padding, 1.5 + padding, -0.2],
    )


def test_electronic_fence_rejects_outside_allowed_zone():
    config = Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json"
    profile = load_safety_profile(
        config, "table_demo", require_verified=False
    )
    # 桌面以下的点现在被桌子禁入盒拦截（比"不在允许区"更早、语义更准）
    with pytest.raises(SafetyAbort, match="进入禁入区 table_top"):
        profile.assert_tcp_point([0.0, 0.65, -0.24], label="table")
    # 允许区之外但不在任何禁入盒里的点，仍走"不在任何允许区"这条路径
    with pytest.raises(SafetyAbort, match="不在任何允许区"):
        profile.assert_tcp_point([0.0, 0.96, 0.30], label="beyond task volume")
    with pytest.raises(SafetyAbort, match="尚未现场测量确认"):
        load_safety_profile(config, "table_demo", require_verified=True)


def test_overlapping_allowed_zones_have_no_clearance_seam():
    profile = load_safety_profile(
        Path(__file__).parents[2] / "bottle_grasp" / "safety_profiles.json",
        "table_demo",
        require_verified=False,
    )
    # Inside the authored transit zone but close to its y edge. This must not
    # be rejected merely because per-zone clearance creates a fake seam.
    profile.assert_tcp_point([0.1815, 0.3291, -0.4886], label="transit seam")


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        ((False, False, False), "baseline_invalid"),
        ((True, True, True), "collision_missed"),
        ((True, False, False), "cleanup_failed"),
        ((True, False, True), "healthy"),
    ],
)
def test_moveit_collision_probe_classifies_each_failure_mode(states, expected):
    status, _ = classify_moveit_collision_probe(
        baseline_valid=states[0],
        boxed_valid=states[1],
        cleared_valid=states[2],
    )
    assert status == expected
