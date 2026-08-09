from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "apply_demonstrated_grasp_to_scenario.py"
SPEC = importlib.util.spec_from_file_location("apply_row_template", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _point(point_id: str, x: float, joint: float) -> dict:
    return {
        "point_id": point_id,
        "layer_id": "upper",
        "row_position_m": 0.0,
        "operations": {
            "pick": {
                "pose_semantics": "demonstrated_grasp_curve",
                "arm_candidates": {
                    "right_arm": {
                        "tcp_pose_moveit": {
                            "xyz": [x, -0.72, -0.13],
                            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                        "reference_joints_deg": [joint] * 7,
                    }
                },
            }
        },
    }


def test_selects_nearest_learned_pick_pose_by_measured_tcp_x():
    templates = {
        "schema_version": "grabber.shelf_row_templates.v1",
        "pose_frame": "platform_base_link",
        "execution_verified": False,
        "points": [
            _point("far", -0.25, 1.0),
            _point("nearest", -0.12, 2.0),
        ],
    }

    selected = MODULE.select_row_pick_candidate(
        templates,
        layer_id="upper",
        arm_id="right_arm",
        target_tcp_x_m=-0.11,
        max_x_error_m=0.04,
    )

    assert selected["point_id"] == "nearest"
    assert selected["candidate"]["reference_joints_deg"] == [2.0] * 7
    assert selected["x_error_m"] == pytest.approx(0.01)


def test_rejects_localization_outside_learned_row_coverage():
    templates = {
        "schema_version": "grabber.shelf_row_templates.v1",
        "pose_frame": "platform_base_link",
        "execution_verified": False,
        "points": [_point("only", -0.25, 1.0)],
    }

    with pytest.raises(MODULE.SafetyAbort, match="超出已学习横向范围"):
        MODULE.select_row_pick_candidate(
            templates,
            layer_id="upper",
            arm_id="right_arm",
            target_tcp_x_m=-0.10,
            max_x_error_m=0.04,
        )


def test_row_pose_expands_to_mtc_redundant_orientation_candidates():
    pose = {"xyz": [-0.12, -0.72, -0.13], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]}

    candidates = MODULE.build_grasp_candidates(
        pose,
        Rotation.identity(),
        id_prefix="learned_row_nearest",
    )

    assert len(candidates) == 3
    assert len({candidate["id"] for candidate in candidates}) == 3
    assert all(candidate["pose"]["xyz"] == pose["xyz"] for candidate in candidates)
    assert len(
        {
            tuple(np.round(candidate["pose"]["quat_xyzw"], 8))
            for candidate in candidates
        }
    ) == 3


def test_levels_taught_finger_axis_without_changing_approach_direction():
    taught = Rotation.from_quat(
        [0.5051933981485576, 0.5314610879399054, -0.48562787945332847, 0.4759141783667606]
    )

    leveled, correction_deg = MODULE.level_finger_axis(taught)

    assert correction_deg == pytest.approx(2.0245, abs=1e-3)
    assert abs(leveled.as_matrix()[2, 1]) < 1e-12
    np.testing.assert_allclose(
        leveled.as_matrix()[:, 2], taught.as_matrix()[:, 2], atol=1e-12
    )


def test_taught_curve_keeps_live_depth_and_height_and_only_supplies_orientation():
    live_pose = {
        "xyz": [-0.1003, -0.6840, -0.1552],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    taught_rotation = Rotation.from_euler("xyz", [2.0, -4.0, 1.0], degrees=True)

    pose = MODULE.merge_live_translation_with_taught_orientation(
        live_pose, taught_rotation
    )

    assert pose["xyz"] == pytest.approx(live_pose["xyz"])
    assert pose["quat_xyzw"] == pytest.approx(taught_rotation.as_quat())
