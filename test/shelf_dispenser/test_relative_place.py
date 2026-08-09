"""Behaviour contract for demonstration-relative placement routes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.relative_place import (
    build_relative_place_route,
    product_geometry,
)


ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "outputs" / "demonstrated_trajectories"


def _demo(slot: str) -> dict:
    path = next(DEMO_DIR.glob(f"place_{slot}_*_20260808.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def test_product_geometry_comes_from_the_recorded_catalog():
    short = product_geometry("P02")
    tall = product_geometry("P06")

    assert short == {
        "product_code": "P02",
        "height_m": 0.190,
        "graspable_height_m": 0.050,
        "radius_m": 0.0325,
    }
    assert tall["height_m"] == 0.255
    assert tall["radius_m"] == 0.035

    with pytest.raises(SafetyAbort, match="未知商品编号"):
        product_geometry("P99")


@pytest.mark.parametrize(
    "slot", ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4")
)
def test_all_taught_slots_build_a_continuous_relative_route(slot):
    demonstration = _demo(slot)
    taught_start = np.asarray(
        demonstration["samples"][0]["tcp_pose_moveit"]["xyz"], dtype=float
    )
    taught_end = np.asarray(
        demonstration["samples"][-1]["tcp_pose_moveit"]["xyz"], dtype=float
    )
    live_start = taught_start + [0.020, 0.0, 0.015]
    live_target = taught_end + [0.035, -0.035, 0.019]

    route = build_relative_place_route(
        demonstration,
        current_tcp_moveit_xyz=live_start,
        target_tcp_moveit_xyz=live_target,
        target_tcp_moveit_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        contact_distance_m=0.010,
    )

    assert route["slot_id"] == slot
    assert route["arm_id"] == demonstration["arm_id"]
    assert route["waypoints"][0]["name"] == "held"
    assert route["waypoints"][-1]["name"] == "placed"
    assert route["waypoints"][0]["pose"]["xyz"] == pytest.approx(live_start)
    assert route["waypoints"][-1]["pose"]["xyz"] == pytest.approx(live_target)
    assert route["transit_raise_m"] >= 0.0
    assert route["preplace_offset_m"] >= 0.040
    assert route["preplace_offset_m"] <= 0.180
    direction = np.asarray(route["insert_direction"], dtype=float)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    gate = np.asarray(route["preplace_pose"]["xyz"], dtype=float)
    contact = np.asarray(route["contact_direction"], dtype=float)
    reached = (
        gate
        + direction * route["preplace_offset_m"]
        + contact * route["contact_distance_m"]
    )
    assert reached == pytest.approx(live_target)
    assert len(route["preplace_reference_joints_deg"]) == 7


def test_relative_route_refuses_a_target_outside_the_taught_slot():
    demonstration = _demo("B3")
    start = demonstration["samples"][0]["tcp_pose_moveit"]["xyz"]
    target = np.asarray(
        demonstration["samples"][-1]["tcp_pose_moveit"]["xyz"], dtype=float
    )
    target[0] += 0.121

    with pytest.raises(SafetyAbort, match="目标偏离示教超过 120 mm"):
        build_relative_place_route(
            demonstration,
            current_tcp_moveit_xyz=start,
            target_tcp_moveit_xyz=target,
            target_tcp_moveit_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            contact_distance_m=0.010,
        )
