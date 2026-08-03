import json
from pathlib import Path

import pytest

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.safety import load_safety_profile


def _profile():
    return {
        "description": "measured test profile",
        "enabled": True,
        "verified_for_execution": True,
        "frame": "right_controller_base",
        "moveit_frame": "platform_base_link",
        "T_moveit_from_profile": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        "clearance_m": 0.03,
        "use_dynamic_rgbd": True,
        # Synthetic valid mount so each test below reaches the independent
        # side-table schema guard it is intended to exercise.
        "tool_mount_calibration": {
            "verified": True,
            "evidence_id": "offline-side-table-fixture",
            "measured_at_utc": "2026-07-24T00:00:00Z",
            "max_position_residual_m": 0.001,
            "max_orientation_residual_deg": 0.2,
            "T_link7_controller_flange": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            "T_controller_flange_tcp": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.151],
                [0, 0, 0, 1],
            ],
        },
        "home_joints_deg": [1, 2, 3, 4, 5, 6, 7],
        "tcp_workspace": {
            "id": "workspace",
            "min": [-1, -1, -1],
            "max": [1, 1, 1],
        },
        "allowed_tcp_zones": [
            {"id": "allowed", "min": [-1, -1, -1], "max": [1, 1, 1]}
        ],
        "keepout_boxes": [
            {
                "id": "measured_fixture",
                "min": [0.7, 0.7, -0.9],
                "max": [0.9, 0.9, 0.9],
            }
        ],
        "side_table_delivery": {
            "transport_joints_deg": [7, 6, 5, 4, 3, 2, 1],
            "transport_pose_verified": True,
            "shelf_ready": {
                "x_m": 0.25,
                "y_m": -0.15,
                "yaw_deg": 179.0,
                "lift_height_mm": 700,
                "xy_tolerance_m": 0.03,
                "yaw_tolerance_deg": 2.0,
                "lift_tolerance_mm": 5,
            },
            "shelf_ready_verified": True,
            "source_lift_height_mm": 700,
            "target_lift_height_mm": 900,
            "target_lift_tolerance_mm": 5,
            "lift_transition_verified": True,
            "body_lift_speed": 15,
            "body_rotation_yaw_deg": -90.0,
            "max_angular_speed_radps": 0.12,
            "rotation_tolerance_deg": 2.0,
            "rotation_timeout_s": 25.0,
            "max_base_translation_m": 0.035,
            "rotation_sweep": {
                "positive": {"clearance_m": 0.05, "verified": True},
                "negative": {"clearance_m": 0.05, "verified": True},
            },
            "table_roi": {
                "min": [0.2, 0.2, -0.2],
                "max": [0.8, 0.8, 0.3],
            },
            "table_roi_verified": True,
            "workspace_verified": True,
            "keepouts_verified": True,
            "bottle_bottom_below_tcp_m": 0.12,
            "held_bottle_height_m": 0.25,
            "held_bottle_diameter_m": 0.07,
            "held_bottle_guard_padding_m": 0.02,
            "bottle_tcp_verified": True,
            "preplace_clearance_m": 0.12,
            "retreat_standoff_m": 0.15,
            "table_height_bin_m": 0.01,
            "table_inlier_band_m": 0.012,
            "table_min_inliers": 50,
            "table_frame_agreement_m": 0.012,
            "table_edge_margin_m": 0.10,
            "table_support_radius_m": 0.06,
            "table_min_patch_points": 4,
            "place_clearance_radius_m": 0.10,
            "place_grid_m": 0.04,
            "obstacle_min_height_m": 0.025,
            "obstacle_max_height_m": 0.45,
            "max_place_candidates": 8,
            "refresh_height_tolerance_m": 0.012,
            "refresh_xy_tolerance_m": 0.04,
        },
    }


def _write(tmp_path, profile):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": {"output": profile}}))
    return path


def _load(tmp_path, raw, *, require_verified=True):
    return load_safety_profile(
        _write(tmp_path, raw), "output", require_verified=require_verified
    )


def test_side_table_profile_parses_only_after_all_measured_guards(tmp_path):
    profile = _load(tmp_path, _profile())

    delivery = profile.side_table_delivery
    assert delivery.shelf_ready.yaw_deg == pytest.approx(179.0)
    assert delivery.source_lift_height_mm == 700
    assert delivery.target_lift_height_mm == 900
    assert delivery.body_lift_height_mm == 900  # transition compatibility alias
    assert delivery.rotation_sweep.positive_verified is True
    assert delivery.rotation_sweep.negative_verified is True


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("rotation_sweep", "positive", "verified"), "rotation_sweep.positive"),
        (("rotation_sweep", "negative", "verified"), "rotation_sweep.negative"),
        (("transport_pose_verified",), "transport_pose"),
        (("shelf_ready_verified",), "shelf_ready"),
        (("lift_transition_verified",), "lift_transition"),
        (("table_roi_verified",), "table_roi"),
        (("workspace_verified",), "workspace"),
        (("keepouts_verified",), "keepouts"),
        (("bottle_tcp_verified",), "bottle_tcp"),
    ],
)
def test_side_table_profile_rejects_each_unverified_measurement(
    tmp_path, path, message
):
    raw = _profile()
    value = raw["side_table_delivery"]
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = False

    with pytest.raises(SafetyAbort, match=message):
        _load(tmp_path, raw)


def test_side_table_profile_rejects_missing_nested_shelf_ready_value(tmp_path):
    raw = _profile()
    del raw["side_table_delivery"]["shelf_ready"]["yaw_deg"]

    with pytest.raises(SafetyAbort, match="shelf_ready.*yaw_deg"):
        _load(tmp_path, raw)


def test_side_table_profile_rejects_missing_required_delivery_value(tmp_path):
    raw = _profile()
    del raw["side_table_delivery"]["target_lift_height_mm"]

    with pytest.raises(SafetyAbort, match="target_lift_height_mm"):
        _load(tmp_path, raw)


def test_side_table_profile_rejects_source_lift_that_disagrees_with_shelf_ready(
    tmp_path,
):
    raw = _profile()
    raw["side_table_delivery"]["source_lift_height_mm"] = 701

    with pytest.raises(SafetyAbort, match="source_lift_height_mm"):
        _load(tmp_path, raw)


def test_side_table_profile_rejects_zero_keepouts(tmp_path):
    raw = _profile()
    raw["keepout_boxes"] = []

    with pytest.raises(SafetyAbort, match="至少配置一个实测 keepout"):
        _load(tmp_path, raw)


def test_unverified_profile_is_never_accepted_for_execution(tmp_path):
    raw = _profile()
    raw["verified_for_execution"] = False

    with pytest.raises(SafetyAbort, match="尚未现场测量确认"):
        _load(tmp_path, raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_angular_speed_radps", 0.8, "max_angular_speed_radps"),
        ("rotation_tolerance_deg", 12, "rotation_tolerance_deg"),
        ("place_grid_m", 0, "place_grid_m"),
        ("table_height_bin_m", 0, "table_height_bin_m"),
        ("refresh_xy_tolerance_m", 0.5, "refresh_xy_tolerance_m"),
    ],
)
def test_side_table_profile_rejects_unsafe_numeric_limits(
    tmp_path, field, value, message
):
    raw = _profile()
    raw["side_table_delivery"][field] = value

    with pytest.raises(SafetyAbort, match=message):
        _load(tmp_path, raw)


def test_checked_in_side_table_template_is_disabled_and_has_no_fabricated_geometry():
    path = Path(__file__).parents[2] / "shelf_dispenser" / "safety_profiles.json"
    template = json.loads(path.read_text())["profiles"]["side_table_template"]

    assert template["enabled"] is False
    assert template["verified_for_execution"] is False
    assert template["tcp_workspace"]["min"] is None
    assert template["keepout_boxes"][0]["min"] is None
    assert template["side_table_delivery"]["shelf_ready"]["x_m"] is None
    assert template["side_table_delivery"]["rotation_sweep"]["positive"]["verified"] is False

    with pytest.raises(SafetyAbort, match="尚未启用"):
        load_safety_profile(path, "side_table_template", require_verified=True)
