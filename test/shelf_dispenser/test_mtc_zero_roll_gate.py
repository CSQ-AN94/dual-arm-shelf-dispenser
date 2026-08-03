from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.mtc_execution import (
    _assert_planned_tcp_matches,
    _pick_candidate_pose,
)


BASE_QUATERNION = [0.5, -0.5, 0.5, 0.5]


def _scenario(candidate_rotation: Rotation) -> dict:
    base_pose = {"xyz": [0.0, 0.0, 0.0], "quat_xyzw": BASE_QUATERNION}
    return {
        "source_grasp_pose": base_pose,
        "source_approach_direction": [0.0, -1.0, 0.0],
        "source_grasp_candidates": [
            {
                "id": "candidate",
                "pose": {
                    "xyz": [0.0, 0.0, 0.0],
                    "quat_xyzw": candidate_rotation.as_quat().tolist(),
                },
            }
        ],
    }


def test_pick_candidate_accepts_local_pitch_with_zero_roll():
    base = Rotation.from_quat(BASE_QUATERNION)
    pose = _pick_candidate_pose(
        _scenario(base * Rotation.from_euler("y", 3.0, degrees=True)),
        "candidate",
    )
    assert pose[:3, 1] == pytest.approx(base.as_matrix()[:, 1])


def test_pick_candidate_rejects_rotation_about_approach_axis():
    base = Rotation.from_quat(BASE_QUATERNION)
    with pytest.raises(SafetyAbort, match="左右滚转"):
        _pick_candidate_pose(
            _scenario(base * Rotation.from_euler("z", 3.0, degrees=True)),
            "candidate",
        )


def test_planned_attach_fk_has_quarter_degree_finger_tilt_gate():
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_quat(BASE_QUATERNION).as_matrix()
    actual = expected.copy()
    actual[:3, :3] = expected[:3, :3] @ Rotation.from_euler(
        "z", 2.0, degrees=True
    ).as_matrix()

    class Robot:
        def tcp_from_joints(self, _joints):
            return actual

    with pytest.raises(SafetyAbort, match="左右滚转超限"):
        _assert_planned_tcp_matches(
            Robot(),
            SimpleNamespace(T_moveit_from_profile=np.eye(4)),
            [0.0] * 7,
            expected,
            label="attach",
            params=DemoParams(),
            max_finger_tilt_deg=0.25,
        )
