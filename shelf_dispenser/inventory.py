"""What is in each shelf slot, and when the robot last actually looked.

The slots are not invented here.  ``outputs/row_templates.json`` already carries
61 taught points per layer at 10 mm spacing across a 600 mm row, each tagged
with the arms that can reach it, so this only names four 150 mm bins per layer
and reads their reach off that table.  The split falls out of the measurements
rather than being chosen: bin 1 is left-arm-only, bin 4 is right-arm-only, and
bins 2 and 3 are the overlap.  Two bins per arm, which is what the operator
wanted to record taught routes for.

Every record carries when it was observed and how.  Nothing here is a standing
truth about the world -- it is the last thing the robot saw or did, with a
timestamp, and callers that care must say how stale they will tolerate.  A
hand-maintained shelf map would be the same mistake this repo has already made
twice with copied constants: a value nobody re-checks goes wrong silently and
the failure surfaces somewhere else entirely.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROW_SPAN_M = 0.6
SLOTS_PER_LAYER = 4
SLOT_WIDTH_M = ROW_SPAN_M / SLOTS_PER_LAYER
LAYER_PREFIX = {"upper": "A", "lower": "B"}
SCHEMA = "grabber.shelf_inventory.v1"


def slot_id(layer: str, index: int) -> str:
    """``('upper', 1) -> 'A1'``.  Index is 1-based, left to right."""
    if layer not in LAYER_PREFIX:
        raise ValueError(f"未知层: {layer}")
    if not 1 <= index <= SLOTS_PER_LAYER:
        raise ValueError(f"槽位序号必须在 1..{SLOTS_PER_LAYER}: {index}")
    return f"{LAYER_PREFIX[layer]}{index}"


def slot_for_row_position(layer: str, row_position_m: float) -> str:
    """Which slot a detected candidate at this row position belongs to."""
    if not 0.0 <= row_position_m <= ROW_SPAN_M:
        raise ValueError(f"行位置超出货架宽度 0..{ROW_SPAN_M}: {row_position_m}")
    index = min(SLOTS_PER_LAYER, int(row_position_m / SLOT_WIDTH_M) + 1)
    return slot_id(layer, index)


def build_slots(row_templates: dict) -> dict:
    """Derive the slot table from the taught points, including reach."""
    points = row_templates.get("points") or []
    if not points:
        raise ValueError("行模板里没有点，无法推导槽位")
    slots: dict[str, dict] = {}
    for layer in LAYER_PREFIX:
        for index in range(1, SLOTS_PER_LAYER + 1):
            low = (index - 1) * SLOT_WIDTH_M
            high = index * SLOT_WIDTH_M
            inside = [
                p
                for p in points
                if p.get("layer_id") == layer
                and low <= float(p["row_position_m"]) <= high
            ]
            arms: set[str] = set()
            for p in inside:
                arms |= set(p.get("eligible_arms") or [])
            slots[slot_id(layer, index)] = {
                "layer": layer,
                "index": index,
                "row_position_range_m": [round(low, 3), round(high, 3)],
                "row_position_center_m": round((low + high) / 2.0, 3),
                "taught_point_count": len(inside),
                "reachable_by": sorted(arms),
            }
    return slots


def new_inventory(row_templates: dict) -> dict:
    return {
        "schema_version": SCHEMA,
        "created_at_utc": _now(),
        "row_templates_sha256": row_templates.get("source_sha256"),
        "slots": build_slots(row_templates),
        "state": {},
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(inventory: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def observe(
    inventory: dict,
    slot: str,
    *,
    occupied: bool,
    contents: str | None = None,
    scene_version: str | None = None,
    source: str = "perception",
) -> dict:
    """Record what the robot just saw.  Overwrites the previous observation."""
    if slot not in inventory["slots"]:
        raise ValueError(f"未知槽位: {slot}")
    if occupied is False and contents is not None:
        raise ValueError("空槽位不能同时记录内容物")
    record = {
        "occupied": bool(occupied),
        "contents": contents,
        "observed_at_utc": _now(),
        "scene_version": scene_version,
        "source": source,
    }
    inventory["state"][slot] = record
    return record


def record_place(inventory: dict, slot: str, contents: str, **kw) -> dict:
    """The robot put something here.  Stronger than seeing it: we did it."""
    return observe(
        inventory, slot, occupied=True, contents=contents, source="placed", **kw
    )


def record_pick(inventory: dict, slot: str, **kw) -> dict:
    """The robot took what was here, so the slot is empty -- unless it was not
    the only thing in it, which this schema cannot express and does not pretend
    to.  One object per slot."""
    return observe(inventory, slot, occupied=False, source="picked", **kw)


def age_s(record: dict, now: datetime | None = None) -> float:
    seen = datetime.fromisoformat(record["observed_at_utc"])
    return ((now or datetime.now(timezone.utc)) - seen).total_seconds()


def status(
    inventory: dict, slot: str, *, max_age_s: float | None = None
) -> dict | None:
    """The last observation, or None when there is none or it is too old.

    None means "unknown", never "empty".  A caller that treats a missing record
    as a free slot will eventually place a bottle into an occupied one.
    """
    record = inventory["state"].get(slot)
    if record is None:
        return None
    if max_age_s is not None and age_s(record) > max_age_s:
        return None
    return record


def free_slots(
    inventory: dict,
    *,
    layer: str | None = None,
    arm: str | None = None,
    max_age_s: float | None = None,
) -> list[str]:
    """Slots observed empty recently enough, optionally reachable by one arm."""
    out = []
    for slot, spec in sorted(inventory["slots"].items()):
        if layer is not None and spec["layer"] != layer:
            continue
        if arm is not None and arm not in spec["reachable_by"]:
            continue
        record = status(inventory, slot, max_age_s=max_age_s)
        if record is not None and not record["occupied"]:
            out.append(slot)
    return out


def find(
    inventory: dict, contents: str, *, max_age_s: float | None = None
) -> list[str]:
    """Where the robot last saw this.  Empty list means "not known to be here"."""
    out = []
    for slot in sorted(inventory["slots"]):
        record = status(inventory, slot, max_age_s=max_age_s)
        if record is not None and record["occupied"] and record["contents"] == contents:
            out.append(slot)
    return out
