"""The empty-slot map must say which occlusion regime produced it.

A map captured while the arm holds a bottle has a region the head camera could
not see, and in that region "empty" and "not visible" are indistinguishable --
which is exactly the failure mode that would drop a bottle onto one already on
the shelf.  A map captured before the pick, with the arm parked clear, has no
such region.  The two are not interchangeable, so the pipeline records which
one it has.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "empty_shelf_places_to_mtc_scenario.py"
SPEC = importlib.util.spec_from_file_location("empty_places", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _payload(**overrides) -> dict:
    payload = {
        "schema_version": "grabber.empty_shelf_places.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame": "right_controller_base",
        "lift_height_mm": 250,
        "head_angle": {"angle1": 398, "angle2": 520},
        "occlusion_regime": "arm_clear_of_view",
        "held_tcp_base_xyz_rpy_rad": None,
        "held_right_joints_deg": None,
        "support_source": "visible_rgbd",
        "roi_min": [-0.20, 0.50, -0.30],
        "roi_max": [0.20, 0.80, 0.10],
        "observation": {
            "table_height_m": -0.24,
            "candidates": [{"xy_base": [0.05, 0.62]}],
        },
        "voxel_size_m": 0.065,
        "scene_voxels": [[0.30, 0.60, -0.20]],
    }
    payload.update(overrides)
    return payload


HELD_TCP = [0.05, 0.55, -0.10, 2.46, 1.46, -2.10]
HELD_JOINTS = [74.1, 108.9, 162.9, -59.6, 164.2, -74.0, -22.5]


def _pick_record() -> dict:
    return {
        "schema_version": "grabber.mtc_execution.v1",
        "mode": "pick",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completion": {
            "final_tcp_base_xyz_rpy_rad": HELD_TCP,
            "final_right_joints_deg": HELD_JOINTS,
        },
    }


def _run(tmp_path, payload, *, pick_record=None) -> dict:
    observation = tmp_path / "places.json"
    output = tmp_path / "place.yaml"
    observation.write_text(json.dumps(payload), encoding="utf-8")
    argv = [str(observation), str(output)]
    if pick_record is not None:
        record_path = tmp_path / "pick.json"
        record_path.write_text(json.dumps(pick_record), encoding="utf-8")
        argv += ["--pick-execution-record", str(record_path)]
    MODULE.main(argv)
    return yaml.safe_load(output.read_text(encoding="utf-8"))


def test_empty_handed_map_removes_nothing_and_takes_the_pose_from_the_pick(
    tmp_path,
):
    scenario = _run(tmp_path, _payload(), pick_record=_pick_record())

    provenance = scenario["placement_provenance"]
    assert provenance["occlusion_regime"] == "arm_clear_of_view"
    # Nothing was in the way, so nothing was subtracted -- that is the whole
    # reason to capture before the pick rather than during it.
    assert provenance["held_voxels_removed"] == 0
    assert provenance["robot_tool_voxels_removed"] == 0
    # Where the shelf is empty and where the bottle is being held are two
    # different questions with two different sources.
    assert provenance["held_right_joints_deg"] == pytest.approx(HELD_JOINTS)


def test_empty_handed_map_refuses_to_guess_where_the_bottle_is(tmp_path):
    with pytest.raises(MODULE.SafetyAbort, match="pick-execution-record"):
        _run(tmp_path, _payload())


def test_held_map_still_requires_the_held_pose(tmp_path):
    payload = _payload(occlusion_regime="held_arm_subtracted")

    with pytest.raises((MODULE.SafetyAbort, TypeError, ValueError)):
        _run(tmp_path, payload)


def test_a_map_without_a_regime_is_treated_as_held(tmp_path):
    # Older captures predate the field.  They were all taken while holding a
    # bottle, so defaulting to the permissive regime would silently claim a
    # clean view that never existed.
    payload = _payload()
    del payload["occlusion_regime"]

    with pytest.raises((MODULE.SafetyAbort, TypeError, ValueError)):
        _run(tmp_path, payload)


def test_unknown_regime_is_rejected(tmp_path):
    payload = _payload(occlusion_regime="probably_fine")

    with pytest.raises(MODULE.SafetyAbort, match="occlusion_regime"):
        _run(tmp_path, payload)
