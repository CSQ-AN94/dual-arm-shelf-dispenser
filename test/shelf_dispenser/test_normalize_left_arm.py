"""The left-arm entry has to wire the right pieces together, without a robot."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts" / "normalize_left_arm.py"


def test_the_entry_uses_the_left_group_and_the_left_framed_fence():
    """Regression: planning the left arm against the right arm's fence.

    Both mistakes are silent -- the wrong planning group plans the wrong seven
    joints, and the right arm's fence describes a volume 120 mm away -- so the
    wiring is asserted on the source rather than waiting for hardware to show it.
    """
    source = ENTRY.read_text(encoding="utf-8")
    assert 'planning_group="left_arm"' in source
    assert "left_view(" in source
    assert "safety=left_profile" in source
    assert "T_base_right_to_base_left" in source
    # The other arm goes in as live collision scene, not as the planned arm.
    assert "left_robot=right" in source
    # Nothing moves without an explicit flag, and the dense re-check runs first.
    assert "validate_planned_joints" in source
    assert source.index("validate_planned_joints") < source.index(
        "execute_planned_joints"
    )


def test_the_entry_defaults_to_a_dry_run():
    import importlib.util

    spec = importlib.util.spec_from_file_location("normalize_left_arm", ENTRY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["normalize_left_arm"] = module
    spec.loader.exec_module(module)

    parser_defaults = {}
    import argparse

    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        parsed = original(self, [] if args is None else args, namespace)
        parser_defaults.update(vars(parsed))
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(SystemExit):
            module.main([])
    finally:
        argparse.ArgumentParser.parse_args = original
    assert parser_defaults["execute"] is False


def test_left_profile_and_right_profile_describe_the_same_place():
    """A sanity check on the pairing the entry relies on."""
    from shelf_dispenser.left_arm import left_view
    from shelf_dispenser.safety import load_safety_profile

    profile = load_safety_profile(
        ROOT / "shelf_dispenser" / "safety_profiles.json",
        "shelf_template",
        require_verified=False,
    )
    transform = np.array(
        [
            [0.999691058769, 0.016811128493, 0.018307730006, -0.119987534677],
            [-0.016767487109, 0.999856203057, -0.002534676056, 0.007718909353],
            [-0.018347708175, 0.002226918364, 0.999829186631, 0.014391259738],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    left = left_view(profile, transform)
    assert {zone.id for zone in left.allowed_tcp_zones} == {
        zone.id for zone in profile.allowed_tcp_zones
    }
    assert left.clearance_m == profile.clearance_m
    assert left.frame == profile.frame
