"""Offline contract tests for the explicit right-tool installation chain.

Except for the checked-in G0.6 shelf calibration, rotations below are synthetic
regression fixtures and are not claims about the installed robot.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shelf_dispenser.core import DemoParams, SafetyAbort, matrix_pose
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.grasp_orientation import (
    ToolMountCalibration,
    tool_mount_chain_residual,
)
from shelf_dispenser.arm import RobotSession
from shelf_dispenser.safe_planner import PlanTarget, SafeMotionPlanner
from shelf_dispenser.safety import load_safety_profile


PROFILE_PATH = (
    Path(__file__).parents[2] / "shelf_dispenser" / "safety_profiles.json"
)


def _profile_path(tmp_path, tool_mount):
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    raw = payload["profiles"]["table_demo"]
    # Production profiles are deliberately offline until field evidence
    # exists. This temporary fixture isolates the schema/propagation contract.
    raw["verified_for_execution"] = True
    raw["grasp_frame"] = {
        "opening_normal_base": [0.0, 1.0, 0.0],
        "finger_axis_base": [1.0, 0.0, 0.0],
        "palm_vertical_base": [0.0, 0.0, -1.0],
    }
    raw["tool_mount_calibration"] = tool_mount
    destination = tmp_path / "profiles.json"
    destination.write_text(
        json.dumps({"profiles": {"shelf": raw}}), encoding="utf-8"
    )
    return destination


def _rigid(rotation_degrees, translation):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", rotation_degrees, degrees=True
    ).as_matrix()
    transform[:3, 3] = translation
    return transform


def _verified_mount(link7_to_flange, flange_to_tcp):
    return {
        "verified": True,
        "evidence_id": "offline-fixture-only",
        "measured_at_utc": "2026-07-24T00:00:00Z",
        "max_position_residual_m": 0.001,
        "max_orientation_residual_deg": 0.2,
        "T_link7_controller_flange": link7_to_flange.tolist(),
        "T_controller_flange_tcp": flange_to_tcp.tolist(),
    }


def test_public_from_start_shelf_profile_reaches_planning_admission():
    raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"][
        "shelf_template"
    ]

    assert raw["verified_for_execution"] is True
    mount = raw["tool_mount_calibration"]
    assert mount["verified"] is True
    # These offsets are the nominal installation, kept because
    # grasp_stop_short_m absorbs the residual on the real arm.  Nothing was
    # measured, so the record must not carry residuals -- claiming them is
    # how the nominal fallback came to read as a 0.042 mm measurement.
    assert mount["provenance"] == "nominal_functionally_validated"
    assert mount["max_position_residual_m"] is None
    assert mount["max_orientation_residual_deg"] is None

    expected_link7_flange = np.eye(4)
    expected_link7_flange[2, 3] = 0.0172
    expected_flange_tcp = np.eye(4)
    expected_flange_tcp[2, 3] = 0.151
    np.testing.assert_allclose(
        mount["T_link7_controller_flange"], expected_link7_flange
    )
    np.testing.assert_allclose(
        mount["T_controller_flange_tcp"], expected_flange_tcp
    )

    # This is the pure configuration gate used by the public
    # ``from-start --execute`` path immediately before hardware/planner setup.
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.args = SimpleNamespace(
        task_mode="from-start",
        safety_config=str(PROFILE_PATH),
        safety_profile="shelf_template",
        delivery_safety_profile=None,
        dispense=False,
        execute=True,
    )
    demo.params = DemoParams()
    demo.safety = None
    demo.source_safety = None
    demo.delivery_safety = None
    demo._load_safety_profiles()

    loaded = demo.safety.tool_mount_calibration
    assert loaded is not None
    np.testing.assert_allclose(
        loaded.T_link7_tcp,
        expected_link7_flange @ expected_flange_tcp,
    )
    assert loaded.T_link7_tcp[2, 3] == pytest.approx(0.1682)


def test_only_shelf_template_is_enabled_for_execution():
    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"]

    assert profiles["table_demo"]["verified_for_execution"] is False
    assert profiles["table_demo"]["tool_mount_calibration"]["verified"] is False
    assert profiles["side_table_template"]["enabled"] is False
    assert (
        profiles["side_table_template"]["verified_for_execution"] is False
    )


def test_authored_shelf_axes_require_a_verified_full_mount_before_execution(tmp_path):
    path = _profile_path(
        tmp_path,
        {
            "verified": False,
            "evidence_id": None,
            "measured_at_utc": None,
            "max_position_residual_m": None,
            "max_orientation_residual_deg": None,
            "T_link7_controller_flange": None,
            "T_controller_flange_tcp": None,
        },
    )

    with pytest.raises(SafetyAbort, match="tool_mount_calibration"):
        load_safety_profile(path, "shelf", require_verified=True)

    offline = load_safety_profile(path, "shelf", require_verified=False)
    assert offline.tool_mount_calibration is not None
    assert offline.tool_mount_calibration.verified is False


def test_every_executing_right_arm_profile_requires_mount_even_without_grasp_axes(
    tmp_path,
):
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    raw = payload["profiles"]["table_demo"]
    raw["verified_for_execution"] = True
    raw.pop("grasp_frame", None)
    raw.pop("tool_mount_calibration", None)
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": {"table": raw}}), encoding="utf-8")

    with pytest.raises(SafetyAbort, match="右臂执行需要已实测"):
        load_safety_profile(path, "table", require_verified=True)


def test_verified_full_mount_flows_to_demo_sdk_and_moveit_boundaries(tmp_path):
    # Synthetic non-identity fixtures prove the code does not silently reduce
    # the installation chain to a scalar +Z. They are not a physical claim.
    T_link7_flange = _rigid([0.0, 0.0, -90.0], [0.0172, -0.004, 0.001])
    T_flange_tcp = _rigid([5.0, 0.0, 0.0], [0.0, 0.012, 0.151])
    path = _profile_path(
        tmp_path, _verified_mount(T_link7_flange, T_flange_tcp)
    )
    profile = load_safety_profile(path, "shelf", require_verified=True)

    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.params = DemoParams()
    demo.safety = profile
    np.testing.assert_allclose(demo.T_link7_controller_flange, T_link7_flange)
    np.testing.assert_allclose(demo.T_flange_tcp, T_flange_tcp)
    np.testing.assert_allclose(
        demo.T_link7_tcp, T_link7_flange @ T_flange_tcp
    )

    session = RobotSession.__new__(RobotSession)
    session.tcp_z_m = demo.params.tcp_z_m
    session.tcp_transform = T_flange_tcp
    T_base_flange = _rigid([0.0, 0.0, 35.0], [0.2, -0.1, 0.4])
    session.take_control = True
    session.current_tcp = lambda: T_base_flange @ T_flange_tcp
    np.testing.assert_allclose(
        session.current_flange(), T_base_flange, atol=1e-12
    )
    assert session._tool_pose_matches(matrix_pose(T_flange_tcp), T_flange_tcp)

    class IdentitySafety:
        @staticmethod
        def pose_to_moveit(pose):
            return np.asarray(pose, dtype=float)

    planner = SafeMotionPlanner(
        moveit=object(),
        robot=object(),
        left_robot=object(),
        safety=IdentitySafety(),
        params=DemoParams(),
        link7_to_controller_flange=T_link7_flange,
    )
    target = PlanTarget(
        label="fixture",
        flange=np.eye(4),
        goal_joints=tuple([0.0] * 7),
    )
    np.testing.assert_allclose(
        planner._target_link7_in_moveit(target),
        np.linalg.inv(T_link7_flange),
    )


def test_verified_mount_rejects_an_invalid_rotation_before_execution(tmp_path):
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    path = _profile_path(
        tmp_path, _verified_mount(invalid, np.eye(4))
    )

    with pytest.raises(SafetyAbort, match="旋转部分必须正交"):
        load_safety_profile(path, "shelf", require_verified=True)


def test_static_sample_residual_compares_tcp_to_the_full_link7_chain():
    # G0.6 2026-07-24 raw sample: SDK Algo zero-tool is r_link7 and the
    # controller state is the active TCP. Comparing their relative transform
    # directly to the 151 mm flange tool would create a false 17.2 mm RED.
    T_base_link7 = np.eye(4)
    T_base_tcp = _rigid(
        np.degrees([0.000749, -0.000976, -0.000669]),
        [0.000001, -0.000023, 0.168201],
    )
    T_link7_flange = _rigid([0.0, 0.0, 0.0], [0.0, 0.0, 0.0172])
    T_flange_tcp = _rigid([0.0, 0.0, 0.0], [0.0, 0.0, 0.151])

    position_m, orientation_deg = tool_mount_chain_residual(
        T_base_link7=T_base_link7,
        T_base_tcp=T_base_tcp,
        T_link7_controller_flange=T_link7_flange,
        T_controller_flange_tcp=T_flange_tcp,
    )

    assert position_m < 0.005
    assert orientation_deg == pytest.approx(0.0802, abs=0.001)


@pytest.mark.parametrize(
    ("position_residual_m", "orientation_residual_deg"),
    ((0.006, 0.2), (0.001, 1.1)),
)
def test_verified_mount_rejects_measurements_above_admission_quality(
    tmp_path, position_residual_m, orientation_residual_deg
):
    raw = _verified_mount(np.eye(4), np.eye(4))
    raw["max_position_residual_m"] = position_residual_m
    raw["max_orientation_residual_deg"] = orientation_residual_deg
    path = _profile_path(tmp_path, raw)

    with pytest.raises(SafetyAbort, match="残差超出工具安装准入上限"):
        load_safety_profile(path, "shelf", require_verified=True)


def test_delivery_profile_must_use_the_same_physical_tool_mount_chain():
    T_link7_flange = _rigid([0.0, 0.0, -90.0], [0.0172, 0.0, 0.0])
    T_flange_tcp = _rigid([0.0, 3.0, 0.0], [0.0, 0.0, 0.151])

    def mount(*, evidence_id="shared", offset=0.0):
        modified = T_flange_tcp.copy()
        modified[0, 3] += offset
        return ToolMountCalibration(
            verified=True,
            evidence_id=evidence_id,
            measured_at_utc="2026-07-24T00:00:00Z",
            max_position_residual_m=0.001,
            max_orientation_residual_deg=0.2,
            T_link7_controller_flange=T_link7_flange,
            T_controller_flange_tcp=modified,
        )

    home = tuple([0.0] * 7)
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.safety = SimpleNamespace(
        home_joints_deg=home,
        tool_mount_calibration=mount(),
    )
    demo.delivery_safety = SimpleNamespace(
        home_joints_deg=home,
        tool_mount_calibration=mount(),
    )
    demo._validate_side_table_profile_pair()

    demo.delivery_safety = SimpleNamespace(
        home_joints_deg=home,
        tool_mount_calibration=mount(offset=0.001),
    )
    with pytest.raises(SafetyAbort, match="工具安装标定不一致"):
        demo._validate_side_table_profile_pair()


def test_dispense_rejects_reusing_the_source_profile_name():
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.args = SimpleNamespace(
        safety_config=str(PROFILE_PATH),
        safety_profile="table_demo",
        delivery_safety_profile="table_demo",
        dispense=True,
        execute=False,
    )
    demo.safety = None
    demo.source_safety = None
    demo.delivery_safety = None
    demo.params = DemoParams()

    with pytest.raises(SafetyAbort, match="必须不同于 source"):
        demo._load_safety_profiles()


def test_profile_loader_does_not_trust_an_injected_safety_attribute():
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.args = SimpleNamespace(
        safety_config=str(PROFILE_PATH),
        safety_profile="table_demo",
        delivery_safety_profile=None,
        dispense=False,
        execute=False,
    )
    demo.params = DemoParams()
    demo.safety = SimpleNamespace(name="injected")
    demo.source_safety = None
    demo.delivery_safety = None

    demo._load_safety_profiles()

    assert demo.safety.name == "table_demo"
    assert demo.source_safety is demo.safety


def _nominal_mount():
    mount = _verified_mount(np.eye(4), np.eye(4))
    mount["provenance"] = "nominal_functionally_validated"
    mount["max_position_residual_m"] = None
    mount["max_orientation_residual_deg"] = None
    return mount


def test_nominal_provenance_is_accepted_without_residuals(tmp_path):
    path = _profile_path(tmp_path, _nominal_mount())

    profile = load_safety_profile(path, "shelf", require_verified=True)

    mount = profile.tool_mount_calibration
    assert mount.verified is True
    assert mount.provenance == "nominal_functionally_validated"
    assert mount.max_position_residual_m is None


def test_nominal_provenance_may_not_claim_residuals(tmp_path):
    """A record with no metrology behind it must not report residuals."""
    mount = _nominal_mount()
    mount["max_position_residual_m"] = 0.0000417213
    path = _profile_path(tmp_path, mount)

    with pytest.raises(SafetyAbort, match="没有实测就没有残差"):
        load_safety_profile(path, "shelf", require_verified=True)


def test_measured_provenance_still_requires_residuals(tmp_path):
    mount = _nominal_mount()
    mount["provenance"] = "measured"
    path = _profile_path(tmp_path, mount)

    with pytest.raises(SafetyAbort):
        load_safety_profile(path, "shelf", require_verified=True)


def test_unknown_provenance_is_rejected(tmp_path):
    mount = _nominal_mount()
    mount["provenance"] = "looks_about_right"
    path = _profile_path(tmp_path, mount)

    with pytest.raises(SafetyAbort, match="provenance"):
        load_safety_profile(path, "shelf", require_verified=True)
