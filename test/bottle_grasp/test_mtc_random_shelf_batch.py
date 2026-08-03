from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from bottle_grasp.core import SafetyAbort

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/mtc_random_shelf_batch.py"
SPEC = importlib.util.spec_from_file_location("mtc_random_shelf_batch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_random_shelf_cases_share_geometry_but_remain_non_executable(tmp_path):
    right, right_provenance = MODULE.generate_case(
        17, "right_arm", local_motion_planner="cartesian"
    )
    left, left_provenance = MODULE.generate_case(
        17, "left_arm", local_motion_planner="cartesian"
    )

    assert right_provenance["bottle_centers_moveit_m"] == left_provenance[
        "bottle_centers_moveit_m"
    ]
    for scenario, arm_id in ((right, "right_arm"), (left, "left_arm")):
        assert scenario["scenario_id"].startswith("synthetic_shelf_pick_v2_")
        assert scenario["scene_version"] == "synthetic_shelf_v2@seed:17"
        assert scenario["planning_arm_id"] == arm_id
        assert scenario["simulation_source"] is True
        assert scenario["fixture_source"] is True
        assert scenario["mode"] == "pick_only"
        assert scenario["obstacle_voxels"]
        assert {box["id"] for box in scenario["shelf_boxes"]} >= {
            "fence_shelf_bottom",
            "fence_shelf_top",
            "fence_shelf_back",
        }
    for scenario in (right, left):
        assert "source_pregrasp_staging_joints_deg" not in scenario
        assert "source_pregrasp_staging_evidence_id" not in scenario

    centers = np.asarray(right_provenance["bottle_centers_moveit_m"])
    assert centers[0, 2] - right["bottle"]["height_m"] / 2 == pytest.approx(
        -0.193
    )
    assert (
        np.min(np.diff(np.sort(centers[:, 0])))
        >= MODULE.MIN_LATERAL_SEPARATION_M - 1e-12
    )
    target_index = right_provenance["target_index"]
    assert right["simulation_obstacle_bottles"] == [
        {"xyz": center.tolist()}
        for index, center in enumerate(centers)
        if index != target_index
    ]

    unsafe = dict(left)
    unsafe["simulation_source"] = False
    path = tmp_path / "not_simulation.yaml"
    path.write_text(yaml.safe_dump(unsafe), encoding="utf-8")
    with pytest.raises(SafetyAbort, match="simulation_source"):
        MODULE.plan_case(path, tmp_path / "result.json", 1.0)

    valid_path = tmp_path / "simulation.yaml"
    valid_path.write_text(yaml.safe_dump(left), encoding="utf-8")
    stale_result = tmp_path / "stale_result.json"
    stale_result.write_text("{}", encoding="utf-8")
    with pytest.raises(SafetyAbort, match="旧 MTC 结果"):
        MODULE.plan_case(valid_path, stale_result, 1.0)
