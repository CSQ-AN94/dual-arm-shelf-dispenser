"""Append-only record of what actually executed; no hardware, no ROS."""

import json

from shelf_dispenser.grasp_ledger import (
    append_executed_grasp,
    executed_grasp_row,
    read_executed_grasps,
)

SCENARIO = {
    "scenario_id": "localized_bottle_03ebc29416f9",
    "scene_version": "localization@sha256:03ebc29416f9",
    "row_template_provenance": {
        "layer_id": "upper",
        "point_id": "upper_045cm",
        "row_position_m": 0.45,
    },
}

TRAJECTORY = {
    "phase_boundaries": [
        {"name": "pregrasp", "start_index": 0, "end_index": 2},
        {"name": "approach", "start_index": 2, "end_index": 3},
    ],
    "points": [
        {"positions_deg": [0, 0, 0, 0, 0, 0, 0]},
        {"positions_deg": [1, 1, 1, 1, 1, 1, 1]},
        {"positions_deg": [2, 2, 2, 2, 2, 2, 2]},
        {"positions_deg": [3, 3, 3, 3, 3, 3, 3]},
    ],
}

COMPLETION = {
    "lift_start_mm": 647,
    "right_start_deg": [0.0] * 7,
    "final_right_joints_deg": [3.0] * 7,
    "empty_close_pos": 0,
    "gripper_close_feedback": {"pos": [365], "current": [100]},
    "bottle_center_in_tcp_m": [0.0, 0.0, 0.063],
}


def test_row_is_keyed_the_way_the_templates_are():
    row = executed_grasp_row(
        mode="pick",
        scenario=SCENARIO,
        trajectory=TRAJECTORY,
        completion=COMPLETION,
    )
    assert row["row_template_provenance"]["point_id"] == "upper_045cm"
    assert row["row_template_provenance"]["row_position_m"] == 0.45
    assert row["lift_start_mm"] == 647
    # Only in the scene that produced it: the key that keeps a stale path from
    # being replayed into a rearranged shelf.
    assert row["scene_version"] == "localization@sha256:03ebc29416f9"


def test_row_captures_the_free_space_leg_up_to_the_pregrasp():
    row = executed_grasp_row(
        mode="pick",
        scenario=SCENARIO,
        trajectory=TRAJECTORY,
        completion=COMPLETION,
    )
    assert row["pregrasp_joints_deg"] == [2] * 7
    assert len(row["free_space_waypoints_deg"]) == 3
    assert row["free_space_waypoints_deg"][-1] == [2] * 7


def test_row_tolerates_a_trajectory_without_phase_boundaries():
    row = executed_grasp_row(
        mode="place",
        scenario=SCENARIO,
        trajectory={"points": []},
        completion=COMPLETION,
    )
    assert row["pregrasp_joints_deg"] is None
    assert row["free_space_waypoints_deg"] == []


def test_appending_never_rewrites_earlier_rows(tmp_path):
    ledger = tmp_path / "nested" / "executed_grasps.jsonl"
    for index in range(3):
        append_executed_grasp(ledger, {"n": index})
    assert [row["n"] for row in read_executed_grasps(ledger)] == [0, 1, 2]


def test_a_torn_final_line_does_not_lose_the_rows_before_it(tmp_path):
    """Runs here get interrupted by fence aborts; a half-written row is normal."""
    ledger = tmp_path / "executed_grasps.jsonl"
    append_executed_grasp(ledger, {"n": 0})
    with open(ledger, "a", encoding="utf-8") as stream:
        torn = json.dumps({"n": 1})[:5]
        assert torn == '{"n":'
        stream.write(torn)
    assert [row["n"] for row in read_executed_grasps(ledger)] == [0]


def test_missing_ledger_reads_as_empty(tmp_path):
    assert read_executed_grasps(tmp_path / "nope.jsonl") == []
