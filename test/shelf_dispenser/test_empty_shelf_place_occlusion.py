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
import numpy as np
from scipy.spatial.transform import Rotation


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
            "table_height_m": -0.2162,
            "candidates": [{"xy_base": [0.05, 0.62]}],
        },
        "voxel_size_m": 0.065,
        "scene_voxels": [[0.30, 0.60, -0.20]],
    }
    payload.update(overrides)
    return payload


HELD_TCP = [0.05, 0.55, -0.10, 2.46, 1.46, -2.10]
HELD_JOINTS = [74.1, 108.9, 162.9, -59.6, 164.2, -74.0, -22.5]
BOTTLE_CENTER_IN_TCP = [0.004, -0.006, -0.012]


def _pick_record() -> dict:
    return {
        "schema_version": "grabber.mtc_execution.v1",
        "mode": "pick",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completion": {
            "final_tcp_base_xyz_rpy_rad": HELD_TCP,
            "final_right_joints_deg": HELD_JOINTS,
            "bottle_center_in_tcp_m": BOTTLE_CENTER_IN_TCP,
        },
    }


def _run(tmp_path, payload, *, pick_record=None, product_code="P01") -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    observation = tmp_path / "places.json"
    output = tmp_path / "place.yaml"
    observation.write_text(json.dumps(payload), encoding="utf-8")
    argv = [
        str(observation),
        str(output),
        "--product-code",
        product_code,
    ]
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
    assert provenance["bottle_center_in_tcp_m"] == pytest.approx(
        BOTTLE_CENTER_IN_TCP
    )
    assert provenance["orientation_template_point_id"].startswith("lower_")
    assert provenance["place_demonstration_label"] == "place_B3_right"
    assert provenance["place_demonstration_usage"] == "relative_route"
    taught = json.loads(
        (
            ROOT
            / "outputs/demonstrated_trajectories/place_B3_right_20260808.json"
        ).read_text(encoding="utf-8")
    )["samples"][-1]
    assert scenario["target_place_reference_joints_deg"] == pytest.approx(
        taught["joints_deg"]
    )
    assert Rotation.from_quat(
        scenario["target_place_pose"]["quat_xyzw"]
    ).approx_equal(
        Rotation.from_quat(taught["tcp_pose_moveit"]["quat_xyzw"]),
        atol=1e-12,
    )
    assert len(scenario["target_place_reference_joints_deg"]) == 7
    profile = MODULE.load_safety_profile(
        ROOT / "shelf_dispenser/safety_profiles.json",
        "shelf_template",
        require_verified=True,
    )
    target = scenario["target_place_pose"]
    target_moveit = np.eye(4)
    target_moveit[:3, :3] = Rotation.from_quat(
        target["quat_xyzw"]
    ).as_matrix()
    target_moveit[:3, 3] = target["xyz"]
    target_base = np.linalg.inv(profile.T_moveit_from_profile) @ target_moveit
    bottle_center_base = (
        target_base[:3, 3]
        + target_base[:3, :3] @ np.asarray(BOTTLE_CENTER_IN_TCP)
    )
    assert bottle_center_base == pytest.approx([0.05, 0.62, -0.1037])
    assert provenance["expected_surface_z_base"] == pytest.approx(-0.2162)
    assert provenance["surface_z_error_m"] == pytest.approx(0.0)


def test_product_dimensions_change_the_collision_body_and_place_height(tmp_path):
    short = _run(
        tmp_path / "short", _payload(), pick_record=_pick_record(), product_code="P02"
    )
    tall = _run(
        tmp_path / "tall", _payload(), pick_record=_pick_record(), product_code="P06"
    )

    assert short["product_code"] == "P02"
    assert short["bottle"]["height_m"] == pytest.approx(0.190)
    assert short["bottle"]["radius_m"] == pytest.approx(0.0325)
    assert tall["product_code"] == "P06"
    assert tall["bottle"]["height_m"] == pytest.approx(0.255)
    assert tall["bottle"]["radius_m"] == pytest.approx(0.035)
    assert (
        tall["target_place_pose"]["xyz"][2]
        - short["target_place_pose"]["xyz"][2]
    ) == pytest.approx((0.255 - 0.190) / 2.0)
    assert short["placement_provenance"]["place_demonstration_usage"] == (
        "relative_route"
    )
    assert "target_preplace_pose" in short
    assert "target_preplace_reference_joints_deg" in short


def test_capture_and_plan_product_codes_must_match(tmp_path):
    with pytest.raises(MODULE.SafetyAbort, match="商品编号不一致"):
        _run(
            tmp_path,
            _payload(product_code="P02"),
            pick_record=_pick_record(),
            product_code="P06",
        )


def test_lower_surface_rejects_a_plane_that_breaks_physical_layer_geometry(
    tmp_path,
):
    payload = _payload()
    payload["observation"]["table_height_m"] = -0.10

    with pytest.raises(MODULE.SafetyAbort, match="下层底板高度与现场测量不一致"):
        _run(tmp_path, payload, pick_record=_pick_record())


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


def test_a_candidate_behind_the_taught_depth_is_pulled_back_to_it(tmp_path):
    """Perception picks the row position; the taught point fixes the depth.

    The empty-patch centroid sits behind the reachable part of the free
    surface, because the shelf lip and the neighbouring bottles hide the front
    of it.  On 2026-08-08 that put the place target 55 mm behind the taught
    depth -- 30 mm outside the right arm's workspace at that orientation -- and
    target_approach truncated at Cartesian fraction 0.655 with no colliding
    pair to find, because an unreachable goal does not have one.
    """
    payload = _payload()
    payload["observation"]["candidates"] = [{"xy_base": [0.05, 0.66]}]

    scenario = _run(tmp_path, payload, pick_record=_pick_record())
    provenance = scenario["placement_provenance"]

    assert provenance["depth_pullback_m"] > 0.0
    assert provenance["requested_depth_m"] < provenance["template_depth_m"]
    # Pulled back exactly to the taught depth, not merely nearer to it.
    assert scenario["target_place_pose"]["xyz"][1] == pytest.approx(
        provenance["template_depth_m"]
    )
    # The route the arm actually follows has to end at the corrected pose too,
    # or the scenario would carry two different placements.
    placed = provenance["relative_route"]["waypoints"][-1]
    assert placed["name"] == "placed"
    assert placed["pose"]["xyz"][1] == pytest.approx(
        provenance["template_depth_m"]
    )


def test_a_shallower_candidate_is_left_alone(tmp_path):
    """Only deeper is unsafe.  Nearer the shelf mouth is inside the taught
    reach by construction, so the pull-back must not push a candidate back."""
    payload = _payload()
    payload["observation"]["candidates"] = [{"xy_base": [0.05, 0.58]}]

    scenario = _run(tmp_path, payload, pick_record=_pick_record())
    provenance = scenario["placement_provenance"]

    assert provenance["depth_pullback_m"] == 0.0
    assert scenario["target_place_pose"]["xyz"][1] == pytest.approx(
        provenance["requested_depth_m"]
    )


def test_a_candidate_far_behind_the_taught_depth_is_refused(tmp_path):
    """Past some distance this stops being a correction and starts being a
    guess: a patch that far back is a different surface, not this slot seen
    imprecisely.  Dragging the bottle forward from there would drop it off the
    front edge, so refuse instead."""
    payload = _payload()
    payload["observation"]["candidates"] = [{"xy_base": [0.05, 0.74]}]

    with pytest.raises(MODULE.SafetyAbort, match="已示教深度"):
        _run(tmp_path, payload, pick_record=_pick_record())


def test_place_demonstration_selection_uses_only_a_verified_terminal_state(
    tmp_path,
):
    def write_demo(slot, arm, x, end_joint):
        path = tmp_path / f"place_{slot}_{arm}_20260808.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "grabber.demonstrated_joint_trajectory.v1",
                    "label": f"place_{slot}_{arm}",
                    "arm_id": f"{arm}_arm",
                    "pose_frame": "platform_base_link",
                    "lift_height_mm": 250,
                    "fence_violations": [],
                    "usable_for_planning_constraints": True,
                    "samples": [
                        {
                            "joints_deg": [0.0] * 7,
                            "tcp_pose_moveit": {
                                "xyz": [x, -0.50, -0.05],
                                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                            },
                        },
                        {
                            "joints_deg": [end_joint] * 7,
                            "tcp_pose_moveit": {
                                "xyz": [x, -0.70, -0.11],
                                "quat_xyzw": [0.5, 0.5, -0.5, 0.5],
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

    write_demo("B3", "right", -0.07, 3.0)
    write_demo("B4", "right", -0.21, 4.0)

    selected = MODULE.select_place_demonstration(
        tmp_path,
        layer_id="lower",
        arm_id="right_arm",
        target_tcp_x_m=-0.05,
        max_x_error_m=0.08,
    )

    assert selected["label"] == "place_B3_right"
    assert selected["reference_joints_deg"] == [3.0] * 7
    assert selected["tcp_pose_moveit"]["xyz"] == [-0.07, -0.70, -0.11]
    assert "samples" not in selected

    explicit = MODULE.select_place_demonstration(
        tmp_path,
        layer_id="lower",
        arm_id="right_arm",
        target_tcp_x_m=-0.10,
        max_x_error_m=0.15,
        target_slot="B4",
    )
    assert explicit["label"] == "place_B4_right"
    assert explicit["reference_joints_deg"] == [4.0] * 7
