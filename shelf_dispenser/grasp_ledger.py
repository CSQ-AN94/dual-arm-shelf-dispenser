"""What the arm actually did, kept next to what a person demonstrated.

``outputs/row_templates.json`` holds 122 hand-taught points, keyed by layer and
row position, and every plan starts from the nearest one.  Nothing has ever
written back: a pick that worked on hardware left no trace in the table it came
from, so the tenth success taught the planner exactly as much as the first.

This appends one line per executed stage, keyed the same way the templates are
(``layer_id`` / ``point_id`` / ``row_position_m``), so the two can be looked up
together later -- the demonstration as the prior, the execution as evidence.

Deliberately append-only JSON Lines.  A ledger that has to be read, edited and
rewritten to add a row is a ledger that loses rows when a run is interrupted
mid-write, and runs here are interrupted by design (fence aborts, e-stop).

Deliberately NOT consumed by anything yet.  One success is not a dataset, and a
rule fitted to it would be a constant wearing a model's clothes -- the failure
mode this repo already has scars from.  Collect first.

``scene_version`` rides along on every row because these paths are only sound
in the scene that produced them: a neighbouring bottle moves and yesterday's
clear path is through it.  Whatever eventually reads this must seed a plan, not
replace the checks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def executed_grasp_row(
    *,
    mode: str,
    scenario: dict,
    trajectory: dict,
    completion: dict,
) -> dict:
    """One ledger row: where the target was, and the posture that reached it."""
    phases = {
        phase["name"]: phase for phase in trajectory.get("phase_boundaries") or []
    }
    points = trajectory.get("points") or []

    def joints_at(index: int | None) -> list[float] | None:
        if index is None or not 0 <= index < len(points):
            return None
        return list(points[index]["positions_deg"])

    pregrasp = phases.get("pregrasp") or {}
    free_space_end = pregrasp.get("end_index")
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "scenario_id": scenario.get("scenario_id"),
        "scene_version": scenario.get("scene_version"),
        "row_template_provenance": scenario.get("row_template_provenance"),
        "lift_start_mm": completion.get("lift_start_mm"),
        "start_joints_deg": completion.get("right_start_deg"),
        "pregrasp_joints_deg": joints_at(free_space_end),
        "final_joints_deg": completion.get("final_right_joints_deg"),
        # The whole free-space leg, unsimplified.  Simplifying now would bake in
        # a tolerance nobody has needed yet; the raw points keep that open.
        "free_space_waypoints_deg": [
            list(point["positions_deg"])
            for point in points[: (free_space_end or -1) + 1]
        ],
        "empty_close_pos": completion.get("empty_close_pos"),
        "gripper_close_feedback": completion.get("gripper_close_feedback"),
        "bottle_center_in_tcp_m": completion.get("bottle_center_in_tcp_m"),
    }


def append_executed_grasp(ledger_path: Path, row: dict) -> None:
    """Append one row; never rewrite what is already there."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_executed_grasps(ledger_path: Path) -> list[dict]:
    """Every complete row.  A torn last line is skipped, not raised on."""
    if not ledger_path.exists():
        return []
    rows = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
