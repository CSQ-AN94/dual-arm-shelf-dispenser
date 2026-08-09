from datetime import datetime

from scripts.continue_manual_grasp import build_record


def test_manual_grasp_record_preserves_explicit_evidence():
    record = build_record(
        before_right=[1.0] * 7,
        before_left=[2.0] * 7,
        before_tcp=[0.0] * 6,
        reached_right=[3.0] * 7,
        tuck_error_deg=0.2,
        lift_height_mm=647,
        empty_close_pos=0,
        bottle_center_in_tcp_m=[0.026, -0.002, 0.016],
        gripper_feedback={"pos": [368], "dof_state": [3]},
    )

    assert record["schema_version"] == "grabber.mtc_execution.v1"
    assert record["mode"] == "pick"
    datetime.fromisoformat(record["completed_at_utc"])
    completion = record["completion"]
    assert completion["manual_confirmation"] is True
    assert completion["empty_close_pos"] == 0
    assert completion["post_pick_tuck"]["right_joints_deg"] == [3.0] * 7
    assert completion["bottle_center_in_tcp_m"] == [0.026, -0.002, 0.016]
