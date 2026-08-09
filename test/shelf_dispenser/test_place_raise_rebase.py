"""Replay the 2026-08-08 split-place failure without robot hardware."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/_patch_after_raise.py"
SPEC = importlib.util.spec_from_file_location("patch_after_raise", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
EVIDENCE = ROOT / "outputs/runs/20260808_failed_pick_place_evidence/remote_home/place_split"


def _transform(pose: dict) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = pose["xyz"]
    if "quat_xyzw" in pose:
        rotation = Rotation.from_quat(pose["quat_xyzw"])
    else:
        rotation = Rotation.from_euler("xyz", pose["rpy_deg"], degrees=True)
    transform[:3, :3] = rotation.as_matrix()
    return transform


def test_physical_raise_keeps_the_bottle_rigidly_attached_to_the_tcp():
    before = yaml.safe_load((EVIDENCE / "place.yaml").read_text(encoding="utf-8"))
    raised = yaml.safe_load(
        (EVIDENCE / "place_raised.yaml").read_text(encoding="utf-8")
    )
    expected_bottle_in_tcp = np.linalg.inv(
        _transform(before["source_grasp_pose"])
    ) @ _transform(before["bottle"]["pose"])

    rebased = MODULE.rebase_held_scenario(
        copy.deepcopy(before), raised["source_grasp_pose"]
    )

    actual_bottle_in_tcp = np.linalg.inv(
        _transform(rebased["source_grasp_pose"])
    ) @ _transform(rebased["bottle"]["pose"])
    np.testing.assert_allclose(actual_bottle_in_tcp, expected_bottle_in_tcp, atol=1e-9)

    target_bottle = _transform(rebased["target_place_pose"]) @ actual_bottle_in_tcp
    support = next(
        box for box in rebased["shelf_boxes"] if box["id"] == "fence_shelf_bottom"
    )
    support_top = support["pose"]["xyz"][2] + support["size"][2] / 2.0
    cylinder_axis = target_bottle[:3, 2]
    z_extent = (
        abs(cylinder_axis[2]) * rebased["bottle"]["height_m"] / 2.0
        + np.linalg.norm(cylinder_axis[:2]) * rebased["bottle"]["radius_m"]
    )
    assert target_bottle[2, 3] - z_extent > support_top
