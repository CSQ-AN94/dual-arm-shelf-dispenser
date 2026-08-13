"""Selecting a shelf layer must move SHELF_READY and the return together.

SHELF_READY is the snapshot the whole dispense run is restored to.  If a run
picks from the lower layer but still calls 647 mm "home", the body either
refuses at admission or returns to the wrong height with a bottle already
delivered -- so the admission height and the return height are one decision,
not two.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.safety import _load_search_lift_heights, load_safety_profile

PROFILES = (
    Path(__file__).resolve().parents[2]
    / "shelf_dispenser"
    / "safety_profiles.json"
)


def _demo(requested):
    """A resolver wired to the real shipped side-table contract.

    Deliberately not a SimpleNamespace stand-in: the resolver rebuilds the
    config with dataclasses.replace, so a stub would pass a test the real
    frozen types would fail.
    """
    profile = load_safety_profile(
        PROFILES, "side_table_template", require_verified=True
    )
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.stage = lambda *_a, **_k: None
    demo.args = SimpleNamespace(shelf_layer_lift_mm=requested)
    demo.delivery_safety = profile
    return demo


def test_no_selection_leaves_the_profile_untouched():
    demo = _demo(None)
    config = demo._side_table_config()
    assert config.shelf_ready.lift_height_mm == 647
    assert config.source_lift_height_mm == 647
    assert config is demo.delivery_safety.side_table_delivery


def test_selecting_the_lower_layer_moves_admission_and_return_together():
    demo = _demo(250)
    config = demo._side_table_config()
    assert config.shelf_ready.lift_height_mm == 250
    assert config.source_lift_height_mm == 250
    # Untouched fields must survive the rewrite.
    assert config.shelf_ready.x_m == pytest.approx(-0.333925)
    assert config.target_lift_height_mm == 515
    # The original profile object is not mutated -- a second run must not
    # inherit the previous run's layer.
    assert (
        demo.delivery_safety.side_table_delivery.shelf_ready.lift_height_mm
        == 647
    )


def test_the_resolved_config_is_stable_within_a_run():
    demo = _demo(250)
    assert demo._side_table_config() is demo._side_table_config()


def test_a_height_the_profile_never_listed_is_refused():
    """The caller selects a layer; it may not invent one."""
    demo = _demo(400)
    with pytest.raises(SafetyAbort, match="不在 profile 的 search_lift_heights_mm"):
        demo._side_table_config()


def test_search_heights_must_start_at_the_profiles_own_shelf_ready():
    shelf_ready = SimpleNamespace(lift_height_mm=647)
    assert _load_search_lift_heights(
        [647, 250], shelf_ready=shelf_ready, label="x"
    ) == (647, 250)
    with pytest.raises(SafetyAbort, match="第一项必须等于"):
        _load_search_lift_heights([250, 647], shelf_ready=shelf_ready, label="x")


@pytest.mark.parametrize(
    "value, match",
    [
        ([], "非空"),
        ([647, 647], "重复"),
        ([647, -10], "负高度"),
    ],
)
def test_malformed_search_heights_are_refused(value, match):
    with pytest.raises(SafetyAbort, match=match):
        _load_search_lift_heights(
            value,
            shelf_ready=SimpleNamespace(lift_height_mm=647),
            label="x",
        )
