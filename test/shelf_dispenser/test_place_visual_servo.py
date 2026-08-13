"""The table place must land on the *observed* bottle bottom.

The tool mount chain is nominal: ~30 mm of accumulated error, absorbed for
the shelf by grasp_stop_short_m, which is tuned for a horizontal entry and
compensates nothing on a top-down approach.  So the release height may not
come from the catalog half-height alone -- these tests pin the three ways
that could silently go wrong.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.orchestrator import RunOrchestrator


TABLE_Z = -0.300
# P01 外星人: 0.215 m tall, so the catalog prior is 0.1075 m.
CATALOG_OFFSET = 0.215 / 2.0


def _servo(measured_offset, *, start_gap=0.090, params=None):
    """A servo wired to a bottle whose real TCP offset is measured_offset."""
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.params = params or DemoParams(target_product_classes=("p01",))
    demo.stage = lambda *_a, **_k: None
    demo._abort_if_stopped = lambda: None

    state = {"tcp_z": TABLE_Z + measured_offset + start_gap, "steps": []}

    demo.robot = SimpleNamespace(
        current_tcp=lambda: np.array(
            [
                [1.0, 0.0, 0.0, 0.10],
                [0.0, 1.0, 0.0, 0.60],
                [0.0, 0.0, 1.0, state["tcp_z"]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )

    def measure(*, table_height_m, label):
        # The camera sees the true bottom, wherever the nominal chain thinks
        # the TCP is.
        return state["tcp_z"] - measured_offset, 400

    def step(distance_m, *, refreshed):
        state["steps"].append(distance_m)
        state["tcp_z"] -= distance_m

    demo._measure_held_bottle_bottom_z = measure
    demo._step_tcp_down = step
    return demo, state


def test_release_height_follows_the_measured_offset_not_the_catalog():
    """A grasp 30 mm off the catalog must still land on the table."""
    real = CATALOG_OFFSET + 0.030
    demo, state = _servo(real)
    observation = SimpleNamespace(table_height_m=TABLE_Z)

    record = demo._servo_bottle_onto_table(observation, refreshed=None)

    assert record["measured_offset_m"] == pytest.approx(real, abs=1e-9)
    assert record["catalog_offset_m"] == pytest.approx(CATALOG_OFFSET)
    final_bottom = state["tcp_z"] - real
    gap = final_bottom - TABLE_Z
    # Landed just above the table, never through it.
    assert gap == pytest.approx(demo.params.place_servo_release_gap_m, abs=1e-6)
    assert gap > 0.0


def test_trusting_the_catalog_would_have_driven_this_bottle_into_the_table():
    """Guards the regression this servo exists to prevent."""
    real = CATALOG_OFFSET + 0.030
    demo, state = _servo(real)
    demo._servo_bottle_onto_table(
        SimpleNamespace(table_height_m=TABLE_Z), refreshed=None
    )
    # Had the descent aimed the *catalog* offset at the release gap, the TCP
    # would have stopped 30 mm lower -- i.e. the bottle 30 mm under the top.
    catalog_target_tcp_z = (
        TABLE_Z + CATALOG_OFFSET + demo.params.place_servo_release_gap_m
    )
    assert state["tcp_z"] > catalog_target_tcp_z
    assert (state["tcp_z"] - real - TABLE_Z) > 0.0


def test_every_step_stays_within_the_bounded_leg_size():
    demo, state = _servo(CATALOG_OFFSET, start_gap=0.110)
    demo._servo_bottle_onto_table(
        SimpleNamespace(table_height_m=TABLE_Z), refreshed=None
    )
    assert state["steps"], "servo must actually descend"
    assert max(state["steps"]) <= demo.params.place_servo_max_step_m + 1e-9
    assert min(state["steps"]) > 0.0


def test_an_offset_far_from_the_catalog_aborts_instead_of_lowering():
    """A cylinder that caught the gripper, the table, or another object."""
    demo, _ = _servo(CATALOG_OFFSET + 0.080)
    with pytest.raises(SafetyAbort, match="实测 TCP-瓶底偏移与商品表预期相差过大"):
        demo._servo_bottle_onto_table(
            SimpleNamespace(table_height_m=TABLE_Z), refreshed=None
        )


def test_a_bottle_already_below_the_release_height_aborts():
    """Never answer 'already too low' by lowering further."""
    real = CATALOG_OFFSET
    demo, state = _servo(real, start_gap=0.090)
    # Drop the bottle under the table between the loop and the final commit.
    original = demo._measure_held_bottle_bottom_z

    def measure(*, table_height_m, label):
        bottom, support = original(table_height_m=table_height_m, label=label)
        return bottom, support

    demo._measure_held_bottle_bottom_z = measure
    demo._step_tcp_down = lambda distance_m, *, refreshed: state.__setitem__(
        "tcp_z", TABLE_Z + real - 0.050
    )
    with pytest.raises(SafetyAbort):
        demo._servo_bottle_onto_table(
            SimpleNamespace(table_height_m=TABLE_Z), refreshed=None
        )
