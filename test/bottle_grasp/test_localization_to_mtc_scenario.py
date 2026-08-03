#!/usr/bin/env python3
"""Small regression check for localization-to-MTC conversion."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "localization_to_mtc_scenario.py"
CAPTURE_SCRIPT = ROOT / "scripts" / "capture_mtc_direct_pick_scene.py"
SPEC = importlib.util.spec_from_file_location("localization_to_mtc", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_historical_localization_reproduces_real_trace_geometry():
    localization = {
        "point_base": [-0.0240067496, 0.59934360096, -0.0856880181],
        "depth_m": 0.36546,
        "depth_mad_m": 0.0,
        "position_spread_m": 0.00003157,
        "confidence": 0.54177,
        "frame_count": 7,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "localization.json"
        path.write_text(json.dumps(localization), encoding="utf-8")
        scenario = MODULE.build_scenario(
            path,
            ROOT
            / "mtc_ws/src/grabber_mtc_planner/scenarios"
            / "right_arm_placeback_trace.yaml",
            ROOT / "bottle_grasp/safety_profiles.json",
            "table_demo",
            max_age_s=300.0,
            allow_stale=False,
        )

    assert scenario["fixture_source"] is True
    assert scenario["start_state_source"] == "current_state"
    assert scenario["shelf_boxes"]
    assert scenario["source_support_surface_id"] == "fence_table_top"
    assert scenario["target_support_surface_id"] == "fence_table_top"
    assert scenario["localization_provenance"]["frame_count"] == 7
    assert (
        scenario["localization_provenance"]["freshness_source"]
        == "captured_at_utc"
    )
    expected_tcp = [-0.039145, -0.675478, -0.077272]
    actual_tcp = scenario["source_grasp_pose"]["xyz"]
    assert max(abs(a - b) for a, b in zip(actual_tcp, expected_tcp)) < 0.001
    yaml.safe_dump(scenario)

    del localization["captured_at_utc"]
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.json"
        path.write_text(json.dumps(localization), encoding="utf-8")
        try:
            MODULE.build_scenario(
                path,
                ROOT
                / "mtc_ws/src/grabber_mtc_planner/scenarios"
                / "right_arm_placeback_trace.yaml",
                ROOT / "bottle_grasp/safety_profiles.json",
                "table_demo",
                max_age_s=300.0,
                allow_stale=False,
            )
        except MODULE.SafetyAbort as exc:
            assert "缺少 captured_at_utc" in str(exc)
        else:
            raise AssertionError("legacy localization was treated as fresh")

    localization["captured_at_utc"] = "2020-01-01T00:00:00+00:00"
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "stale.json"
        path.write_text(json.dumps(localization), encoding="utf-8")
        try:
            MODULE.build_scenario(
                path,
                ROOT
                / "mtc_ws/src/grabber_mtc_planner/scenarios"
                / "right_arm_placeback_trace.yaml",
                ROOT / "bottle_grasp/safety_profiles.json",
                "table_demo",
                max_age_s=300.0,
                allow_stale=False,
            )
        except MODULE.SafetyAbort as exc:
            assert "已过期" in str(exc)
        else:
            raise AssertionError("stale embedded capture time was accepted")


def test_fixed_head_pick_only_keeps_non_target_obstacles_and_shelf_geometry():
    captured = datetime.now(timezone.utc).isoformat()
    target = [0.072, 0.627, -0.051]
    localization = {
        "point_base": target,
        "depth_m": 0.42,
        "depth_mad_m": 0.001,
        "position_spread_m": 0.002,
        "confidence": 0.8,
        "frame_count": 7,
        "captured_at_utc": captured,
    }
    profile = MODULE.load_safety_profile(
        ROOT / "bottle_grasp/safety_profiles.json",
        "shelf_template",
        require_verified=False,
    )
    adjacent_bottle_voxel = [0.18, 0.70, -0.05]
    scene = {
        "captured_at_utc": captured,
        "safety_profile": "shelf_template",
        "frame": "right_controller_base",
        "target_point_base": target,
        "image_height_px": 480,
        "observed_row_limit_px": 480,
        "voxel_size_m": MODULE.DemoParams().scene_voxel_m,
        "scene_voxels": [target, adjacent_bottle_voxel],
        "non_target_scene_voxels": [adjacent_bottle_voxel],
        "target_occupancy_voxels": [target],
        "collision_boxes": profile.moveit_collision_boxes(),
    }
    with TemporaryDirectory() as tmp:
        localization_path = Path(tmp) / "head_localization.json"
        scene_path = Path(tmp) / "head_scene.json"
        localization_path.write_text(json.dumps(localization), encoding="utf-8")
        scene_path.write_text(json.dumps(scene), encoding="utf-8")
        scenario = MODULE.build_scenario(
            localization_path,
            ROOT
            / "mtc_ws/src/grabber_mtc_planner/scenarios"
            / "shelf_transfer_fixture.yaml",
            ROOT / "bottle_grasp/safety_profiles.json",
            "shelf_template",
            max_age_s=300.0,
            allow_stale=False,
            scene_path=scene_path,
            pick_only=True,
        )

    assert scenario["mode"] == "pick_only"
    assert scenario["planning_arm_id"] == "right_arm"
    assert scenario["local_motion_planner"] == "pilz_lin"
    assert scenario["fixture_source"] is False
    assert scenario["source_lift_direction"] == [0.0, 0.0, 1.0]
    assert scenario["source_lift_distance_m"] > 0.0
    assert scenario["source_pregrasp_offset_m"] == pytest.approx(0.085)
    assert scenario["source_contact_distance_m"] == pytest.approx(0.072)
    assert "post_pick_carry_joints_deg" not in scenario
    assert "source_pregrasp_staging_joints_deg" not in scenario
    assert "source_pregrasp_staging_evidence_id" not in scenario
    assert scenario["tcp_path_workspace"]["id"] == "tcp_path_workspace"
    assert all(value > 0 for value in scenario["tcp_path_workspace"]["size"])
    # One authored orientation -- the one demonstrated on the real shelf --
    # plus a small pitch about the finger axis.  No roll about the approach
    # axis: it is equivalent only if the finger centre sits exactly on the
    # tool axis, and the transform claiming that is nominal, not measured.
    assert [item["id"] for item in scenario["source_grasp_candidates"]] == [
        "horizontal_fingers_roll_0",
        "horizontal_fingers_roll_0_pitch_plus_3",
        "horizontal_fingers_roll_0_pitch_minus_3",
    ]
    candidate_rotations = [
        Rotation.from_quat(item["pose"]["quat_xyzw"]).as_matrix()
        for item in scenario["source_grasp_candidates"]
    ]
    # The installed gripper opens along controller-TCP local Y because its
    # base is mounted +90 degrees about r_link7 Z.
    opening_axes = [rotation[:, 1] for rotation in candidate_rotations]
    assert all(abs(axis[2]) < 1e-6 for axis in opening_axes)
    assert all(axis == pytest.approx(opening_axes[0]) for axis in opening_axes)
    relative_rotvecs = np.asarray(
        [
            Rotation.from_matrix(
                candidate_rotations[0].T @ rotation
            ).as_rotvec()
            for rotation in candidate_rotations
        ]
    )
    assert np.degrees(relative_rotvecs[:, 1]) == pytest.approx([0.0, 3.0, -3.0])
    assert relative_rotvecs[:, 0] == pytest.approx([0.0, 0.0, 0.0])
    assert relative_rotvecs[:, 2] == pytest.approx([0.0, 0.0, 0.0])
    assert candidate_rotations[0][:, 2] == pytest.approx(
        scenario["source_approach_direction"]
    )
    assert scenario["source_grasp_pose"] == scenario[
        "source_grasp_candidates"
    ][0]["pose"]
    assert {item["id"] for item in scenario["shelf_boxes"]} >= {
        "fence_shelf_bottom",
        "fence_shelf_top",
        "fence_shelf_back",
    }
    assert scenario["source_support_surface_id"] == "fence_shelf_bottom"
    assert scenario["target_support_surface_id"] == "fence_shelf_bottom"
    surface_moveit = profile.point_to_moveit(target)
    approach = np.asarray(
        scenario["source_approach_direction"], dtype=float
    )
    # The collision cylinder is centred on the bottle axis, one radius past
    # the observed near surface.
    expected_center = (
        surface_moveit + approach * scenario["bottle"]["radius_m"]
    )
    assert max(
        abs(actual - expected)
        for actual, expected in zip(
            scenario["bottle"]["pose"]["xyz"], expected_center
        )
    ) < 1e-9
    # The grasp does NOT go to that axis.  It stops at the one working
    # distance that has held a real bottle; driving to the axis assumes the
    # modelled TCP is the finger centre, which nothing has measured.
    expected_grasp = (
        surface_moveit - approach * MODULE.DemoParams().grasp_stop_short_m
    )
    assert scenario["source_grasp_pose"]["xyz"] == pytest.approx(
        expected_grasp
    )
    # Whichever convention the grasp uses, the standoff the arm pauses at
    # must land on the same physical point.
    standoff = expected_grasp - approach * scenario["source_pregrasp_offset_m"]
    assert standoff == pytest.approx(surface_moveit - approach * 0.115)
    assert len(scenario["obstacle_voxels"]) == 1
    expected_obstacle = profile.point_to_moveit(adjacent_bottle_voxel)
    assert max(
        abs(actual - expected)
        for actual, expected in zip(
            scenario["obstacle_voxels"][0], expected_obstacle
        )
    ) < 1e-9
    assert scenario["scene_provenance"]["target_associated_voxel_count"] == 1
    assert (
        scenario["scene_provenance"]["target_filter_stage"]
        == "raw_depth_before_voxelization"
    )


def test_direct_pick_capture_uses_refined_scene_voxel_budget():
    source = CAPTURE_SCRIPT.read_text(encoding="utf-8")
    assert "scene_voxel_m=0.025," in source
    assert "scene_max_voxels=3000," in source


def test_fixed_head_pick_only_rejects_a_bottom_cropped_scene():
    captured = datetime.now(timezone.utc).isoformat()
    target = [0.072, 0.627, -0.051]
    profile = MODULE.load_safety_profile(
        ROOT / "bottle_grasp/safety_profiles.json",
        "shelf_template",
        require_verified=False,
    )
    localization = {
        "point_base": target,
        "depth_m": 0.42,
        "depth_mad_m": 0.001,
        "position_spread_m": 0.002,
        "confidence": 0.8,
        "frame_count": 7,
        "captured_at_utc": captured,
    }
    scene = {
        "captured_at_utc": captured,
        "safety_profile": "shelf_template",
        "frame": "right_controller_base",
        "target_point_base": target,
        "image_height_px": 480,
        "observed_row_limit_px": 405,
        "voxel_size_m": MODULE.DemoParams().scene_voxel_m,
        "scene_voxels": [target],
        "non_target_scene_voxels": [],
        "target_occupancy_voxels": [target],
        "collision_boxes": profile.moveit_collision_boxes(),
    }
    with TemporaryDirectory() as tmp:
        localization_path = Path(tmp) / "head_localization.json"
        scene_path = Path(tmp) / "head_scene.json"
        localization_path.write_text(json.dumps(localization), encoding="utf-8")
        scene_path.write_text(json.dumps(scene), encoding="utf-8")
        with pytest.raises(MODULE.SafetyAbort, match="完整深度帧"):
            MODULE.build_scenario(
                localization_path,
                ROOT
                / "mtc_ws/src/grabber_mtc_planner/scenarios"
                / "shelf_transfer_fixture.yaml",
                ROOT / "bottle_grasp/safety_profiles.json",
                "shelf_template",
                max_age_s=300.0,
                allow_stale=False,
                scene_path=scene_path,
                pick_only=True,
            )


def _build_pick_only(depth_m: float, planning_arm_id: str = "right_arm") -> dict:
    """Build a minimal valid pick-only scenario at one head-camera depth."""
    captured = datetime.now(timezone.utc).isoformat()
    target = [0.072, 0.627, -0.051]
    profile = MODULE.load_safety_profile(
        ROOT / "bottle_grasp/safety_profiles.json",
        "shelf_template",
        require_verified=False,
    )
    localization = {
        "point_base": target,
        "depth_m": depth_m,
        "depth_mad_m": 0.001,
        "position_spread_m": 0.002,
        "confidence": 0.8,
        "frame_count": 7,
        "captured_at_utc": captured,
    }
    scene = {
        "captured_at_utc": captured,
        "safety_profile": "shelf_template",
        "frame": "right_controller_base",
        "target_point_base": target,
        "image_height_px": 480,
        "observed_row_limit_px": 480,
        "voxel_size_m": MODULE.DemoParams().scene_voxel_m,
        "scene_voxels": [target, [0.18, 0.70, -0.05]],
        "non_target_scene_voxels": [[0.18, 0.70, -0.05]],
        "target_occupancy_voxels": [target],
        "collision_boxes": profile.moveit_collision_boxes(),
    }
    with TemporaryDirectory() as tmp:
        localization_path = Path(tmp) / "head_localization.json"
        scene_path = Path(tmp) / "head_scene.json"
        localization_path.write_text(json.dumps(localization), encoding="utf-8")
        scene_path.write_text(json.dumps(scene), encoding="utf-8")
        return MODULE.build_scenario(
            localization_path,
            ROOT
            / "mtc_ws/src/grabber_mtc_planner/scenarios"
            / "shelf_transfer_fixture.yaml",
            ROOT / "bottle_grasp/safety_profiles.json",
            "shelf_template",
            max_age_s=300.0,
            allow_stale=False,
            scene_path=scene_path,
            pick_only=True,
            planning_arm_id=planning_arm_id,
        )


def test_pick_only_uses_head_camera_depth_range_not_the_wrist_one():
    params = MODULE.DemoParams()
    # The exact boundary that used to reject shelf targets: beyond the wrist
    # camera's close-in limit but well inside what the fixed head resolves.
    assert params.max_depth_m < 0.9 < params.head_max_depth_m
    assert _build_pick_only(0.9)["mode"] == "pick_only"

    # The head gate is a real gate, not a removed one, on both ends.
    for out_of_range in (params.head_min_depth_m - 0.01, params.head_max_depth_m + 0.01):
        with pytest.raises(MODULE.SafetyAbort) as excinfo:
            _build_pick_only(out_of_range)
        assert "固定头部" in str(excinfo.value)


def test_left_pick_only_connects_directly_without_taught_staging_pose():
    scenario = _build_pick_only(0.42, planning_arm_id="left_arm")
    assert scenario["planning_arm_id"] == "left_arm"
    assert "source_pregrasp_staging_joints_deg" not in scenario
    assert "source_pregrasp_staging_evidence_id" not in scenario


def test_non_pick_flow_keeps_the_wrist_camera_depth_limit():
    params = MODULE.DemoParams()
    localization = {
        "point_base": [-0.024, 0.599, -0.086],
        "depth_m": 0.9,
        "depth_mad_m": 0.0,
        "position_spread_m": 0.001,
        "confidence": 0.8,
        "frame_count": 7,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    assert params.max_depth_m < localization["depth_m"] < params.head_max_depth_m
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "localization.json"
        path.write_text(json.dumps(localization), encoding="utf-8")
        with pytest.raises(MODULE.SafetyAbort) as excinfo:
            MODULE.build_scenario(
                path,
                ROOT
                / "mtc_ws/src/grabber_mtc_planner/scenarios"
                / "right_arm_placeback_trace.yaml",
                ROOT / "bottle_grasp/safety_profiles.json",
                "table_demo",
                max_age_s=300.0,
                allow_stale=False,
            )
    assert "腕部" in str(excinfo.value)


if __name__ == "__main__":
    test_historical_localization_reproduces_real_trace_geometry()
    test_fixed_head_pick_only_keeps_non_target_obstacles_and_shelf_geometry()
    test_pick_only_uses_head_camera_depth_range_not_the_wrist_one()
    test_left_pick_only_does_not_reuse_the_right_arm_taught_staging_pose()
    test_non_pick_flow_keeps_the_wrist_camera_depth_limit()
    print("localization_to_mtc_scenario: PASS")
