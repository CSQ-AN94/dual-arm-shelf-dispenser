from types import SimpleNamespace

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "capture_empty_shelf_places",
    ROOT / "scripts" / "capture_empty_shelf_places.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_empty_slot_search_ignores_shelf_geometry_above_the_bottle():
    cli = SimpleNamespace(
        roi_min=(-0.40, 0.50, -0.45),
        roi_max=(0.23, 0.85, 0.05),
        clearance_radius_m=0.10,
    )
    demo = SimpleNamespace(
        params=SimpleNamespace(held_bottle_height_m=0.25)
    )

    config = MODULE._candidate_config(cli, demo)

    assert config.obstacle_max_height_m == 0.25
