import json
from pathlib import Path

import pytest

from bottle_grasp.core import DemoParams, SafetyAbort
from bottle_grasp.safety import load_safety_profile


ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "bottle_grasp" / "safety_profiles.json"
RIGHT = [
    45.7550010681,
    105.875,
    130.9279937744,
    -71.5419998169,
    146.2890014648,
    -114.7239990234,
    -9.3540000916,
]
# Re-taught 2026-08-03.  The specific values carry no geometry: the left arm
# only has to sit somewhere known and fixed so every plan can build its
# collision scene against it.  The 2026-08-02 values were simply where the arm
# happened to be that day, and once it was moved on site they were no longer
# reachable -- this package has no left-arm execution path.
LEFT = [
    -45.889,
    -110.092,
    50.921,
    -82.368,
    11.999,
    126.406,
    -0.922,
]


def test_shelf_grasp_start_is_recorded_separately_from_home():
    profile = load_safety_profile(
        PROFILES, "shelf_template", require_verified=False
    )

    assert profile.grasp_start_right_joints_deg == pytest.approx(RIGHT)
    assert profile.grasp_start_left_joints_deg == pytest.approx(LEFT)
    assert profile.grasp_start_lift_height_mm == 647
    assert profile.home_joints_deg != pytest.approx(RIGHT)
    shelf_bottom = next(
        box for box in profile.keepout_boxes if box.id == "shelf_bottom"
    )
    assert shelf_bottom.maximum[2] == pytest.approx(-0.2432)


def test_shelf_grasp_start_is_atomic_and_required(tmp_path):
    payload = json.loads(PROFILES.read_text(encoding="utf-8"))
    del payload["profiles"]["shelf_template"][
        "grasp_start_left_joints_deg"
    ]
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SafetyAbort, match="必须同时配置双臂关节和升降高度"):
        load_safety_profile(path, "shelf_template", require_verified=False)


def test_live_grasp_start_gate_checks_both_arms_lift_and_motion_state():
    profile = load_safety_profile(
        PROFILES, "shelf_template", require_verified=False
    )
    kwargs = {
        "right_joints_deg": RIGHT,
        "left_joints_deg": LEFT,
        "lift_height_mm": 647,
        "lift_mode": 0,
        "joint_tolerance_deg": DemoParams().planned_start_tolerance_deg,
    }
    profile.assert_grasp_start(**kwargs)

    with pytest.raises(SafetyAbort, match="左臂未回到"):
        profile.assert_grasp_start(
            **{**kwargs, "left_joints_deg": [LEFT[0] + 1.0, *LEFT[1:]]}
        )
    with pytest.raises(SafetyAbort, match="升降未回到"):
        profile.assert_grasp_start(**{**kwargs, "lift_height_mm": 653})
    with pytest.raises(SafetyAbort, match="升降仍在运动"):
        profile.assert_grasp_start(**{**kwargs, "lift_mode": 1})


def test_gripper_calibration_uses_profile_grasp_start_not_a_fixed_default():
    source = (ROOT / "scripts" / "calibrate_mtc_gripper.py").read_text(
        encoding="utf-8"
    )
    assert "ArmJointReader" in source
    assert "profile.assert_grasp_start(" in source
    assert "profile.grasp_start_lift_height_mm" in source
    assert "profile.home_joints_deg" not in source
    assert "--expected-lift-mm" in source
    assert "cli.expected_lift_mm is not None" in source
    assert source.index("recover_transient_joint_frame_loss()") < source.index(
        "validate_hardware_preflight(robot)"
    )
    assert "left.close()" in source


def test_mtc_pick_entry_consumes_grasp_start_gate_before_execute_pick():
    source = (ROOT / "scripts" / "execute_mtc_trajectory.py").read_text(
        encoding="utf-8"
    )
    gate = source.index("profile.assert_grasp_start(")
    dispatch = source.index("run = execute_pick if cli.mode")
    assert gate < dispatch
