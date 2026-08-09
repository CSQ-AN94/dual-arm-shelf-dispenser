"""Slot bookkeeping; no hardware, no ROS."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from shelf_dispenser import inventory as inv

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = json.loads(
    (ROOT / "outputs" / "row_templates.json").read_text(encoding="utf-8")
)


def fresh():
    return inv.new_inventory(TEMPLATES)


def test_slots_are_derived_from_the_taught_points():
    book = fresh()
    assert sorted(book["slots"]) == [
        "A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4",
    ]
    for spec in book["slots"].values():
        assert spec["taught_point_count"] > 0


def test_reach_comes_from_the_templates_not_from_a_guess():
    """The 600 mm row splits into left-only, shared, right-only -- measured."""
    book = fresh()
    assert book["slots"]["A1"]["reachable_by"] == ["left_arm"]
    assert book["slots"]["A2"]["reachable_by"] == ["left_arm", "right_arm"]
    assert book["slots"]["A3"]["reachable_by"] == ["left_arm", "right_arm"]
    assert book["slots"]["A4"]["reachable_by"] == ["right_arm"]
    # Both layers are taught the same way, and each arm owns two bins.
    assert book["slots"]["B1"]["reachable_by"] == ["left_arm"]
    assert book["slots"]["B4"]["reachable_by"] == ["right_arm"]


def test_row_position_maps_onto_a_slot():
    assert inv.slot_for_row_position("lower", 0.05) == "B1"
    assert inv.slot_for_row_position("lower", 0.20) == "B2"
    assert inv.slot_for_row_position("upper", 0.40) == "A3"
    assert inv.slot_for_row_position("upper", 0.55) == "A4"
    # The right edge belongs to the last slot rather than falling off it.
    assert inv.slot_for_row_position("upper", 0.60) == "A4"
    with pytest.raises(ValueError):
        inv.slot_for_row_position("upper", 0.75)


def test_unknown_is_not_empty():
    """The trap this schema exists to avoid: treating "no record" as "free"."""
    book = fresh()
    assert inv.status(book, "B2") is None
    assert inv.free_slots(book) == []


def test_a_stale_observation_stops_counting_as_known():
    book = fresh()
    inv.observe(book, "B2", occupied=False, scene_version="scene@abc")
    assert inv.free_slots(book, max_age_s=60) == ["B2"]
    book["state"]["B2"]["observed_at_utc"] = (
        datetime.now(timezone.utc) - timedelta(seconds=600)
    ).isoformat()
    assert inv.free_slots(book, max_age_s=60) == []
    assert inv.status(book, "B2", max_age_s=60) is None


def test_place_and_pick_move_the_slot_between_states():
    book = fresh()
    inv.record_place(book, "B1", "coke", scene_version="scene@1")
    assert inv.find(book, "coke") == ["B1"]
    assert inv.status(book, "B1")["source"] == "placed"
    inv.record_pick(book, "B1", scene_version="scene@2")
    assert inv.find(book, "coke") == []
    assert inv.free_slots(book) == ["B1"]


def test_free_slots_can_be_filtered_by_which_arm_can_reach_them():
    book = fresh()
    for slot in ("B1", "B2", "B3", "B4"):
        inv.observe(book, slot, occupied=False)
    assert inv.free_slots(book, layer="lower", arm="right_arm") == ["B2", "B3", "B4"]
    assert inv.free_slots(book, layer="lower", arm="left_arm") == ["B1", "B2", "B3"]


def test_an_empty_slot_cannot_carry_contents():
    book = fresh()
    with pytest.raises(ValueError):
        inv.observe(book, "B1", occupied=False, contents="coke")


def test_round_trips_through_disk(tmp_path):
    book = fresh()
    inv.record_place(book, "A4", "sprite")
    path = tmp_path / "nested" / "shelf_inventory.json"
    inv.save(book, path)
    again = inv.load(path)
    assert inv.find(again, "sprite") == ["A4"]
