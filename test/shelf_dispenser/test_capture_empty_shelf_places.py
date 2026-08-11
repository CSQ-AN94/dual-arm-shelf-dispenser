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


def test_place_scene_drops_the_same_floating_fragments_the_pick_scene_does():
    """The transit to the gate is where an unfiltered fragment costs a run.

    2026-08-11: the held bottle swept the place scene's voxel cloud on the way
    to the demonstrated gate and MTC refused transport_to_demonstrated_preplace
    with "Waypoint is in collision!".  Every voxel the bottle could touch along
    that path was a floating fragment -- the pick scene had been dropping those
    since 2026-08-07, the place scene never did.  The demonstrated branch plans
    one deterministic joint interpolation with no sampled sibling, so there is
    nothing to fall back on.

    This pins the wiring, not the algorithm: drop_floating_fragments has its
    own tests.  What broke was that the place path never called it.
    """
    import inspect

    source = inspect.getsource(MODULE)
    assert "drop_floating_fragments(" in source, (
        "放置场景必须和抓取场景一样扣除悬空碎块"
    )

    support_z = -0.2141
    voxel = 0.025
    # A bottle standing on the panel, and a two-cell blob floating 60 mm above
    # it -- the shape the 2026-08-11 transit actually hit.
    bottle = [
        [0.10, 0.70, support_z + voxel * (index + 0.5)] for index in range(9)
    ]
    floating = [[0.30, 0.70, support_z + 0.060], [0.30, 0.70, support_z + 0.085]]

    kept, dropped = MODULE.drop_floating_fragments(
        bottle + floating,
        voxel,
        support_z=support_z,
        bottle_height_m=0.217,
    )

    assert dropped == 2
    assert [0.30, 0.70, support_z + 0.060] not in kept
    # The real bottle stays.  A filter that eats supported geometry is worse
    # than the refused plan it was meant to prevent.
    assert len(kept) == len(bottle)
