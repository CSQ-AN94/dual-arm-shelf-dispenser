import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shelf_dispenser.core import SafetyAbort

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "solve_left_arm_model.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("solve_left_arm_model", SCRIPT)
SOLVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOLVER)


def _transform(rotvec, translation):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    result[:3, 3] = translation
    return result


def test_hand_eye_solver_recovers_bridge_and_tool_without_joint_remapping():
    rng = np.random.default_rng(7)
    bridge = _transform([0.2, -0.1, 2.8], [0.06, -0.11, -0.01])
    link7_to_flange = _transform([0.0, 0.0, 3.13], [0.0, 0.0, 0.046])
    flanges = [
        _transform(rng.normal(size=3), rng.normal(size=3))
        for _ in range(16)
    ]
    moveit = [
        bridge @ flange @ SOLVER.rigid_inverse(link7_to_flange)
        for flange in flanges
    ]

    solved_bridge, solved_tool, position_error, angle_error = SOLVER.solve(
        flanges, moveit
    )

    assert SOLVER.JOINT_SIGNS == (1,) * 7
    assert solved_bridge == pytest.approx(bridge, abs=1e-9)
    assert solved_tool == pytest.approx(link7_to_flange, abs=1e-9)
    assert position_error < 1e-9
    assert angle_error < 1e-9


def test_hand_eye_solver_rejects_an_underdetermined_sample_set():
    with pytest.raises(SafetyAbort, match="至少需要"):
        SOLVER.solve([np.eye(4)], [np.eye(4)])

    repeated = [np.eye(4) for _ in range(SOLVER.MIN_KINEMATIC_SAMPLES)]
    with pytest.raises(SafetyAbort, match="激励不足"):
        SOLVER.solve(repeated, repeated)


def test_profile_write_updates_the_tool_and_model_as_one_record():
    shelf = {"left_tool_mount_calibration": {"provenance": "nominal_unvalidated"}}
    bridge = _transform([0.0, 0.0, 0.1], [0.1, 0.2, 0.3])
    link7_to_flange = _transform([0.0, 0.0, 3.1], [0.0, 0.0, 0.04])
    tool_offset = _transform([0.0, 0.0, -3.1], [0.0, 0.0, -0.02])

    SOLVER.update_profile_records(
        shelf,
        bridge=bridge,
        link7_to_flange=link7_to_flange,
        tool_offset=tool_offset,
        measured_at="2026-08-04T12:00:00Z",
        samples=24,
        position_residual=0.0007,
        angle_residual=0.19,
    )

    tool = shelf["left_tool_mount_calibration"]
    model = shelf["left_arm_model"]
    assert tool["T_link7_controller_flange"] == pytest.approx(link7_to_flange)
    assert tool["measured_at_utc"] == model["measured_at_utc"]
    assert model["T_moveit_from_left_profile"] == pytest.approx(bridge)
    assert model["T_left_link7_tool_offset"] == pytest.approx(tool_offset)
    assert model["joint_signs"] == [1] * 7
    assert model["samples"] == 24
