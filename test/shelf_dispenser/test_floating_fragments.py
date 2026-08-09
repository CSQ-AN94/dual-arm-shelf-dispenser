"""Only clusters that are both too short to be a bottle AND floating go."""

import numpy as np

from shelf_dispenser.scene import drop_floating_fragments

VOXEL = 0.025
BOTTLE_H = 0.217
FLOOR = -0.220


def column(x, y, z0, cells):
    """A vertical run of cells, like a bottle seen by the camera."""
    return [[x, y, z0 + i * VOXEL] for i in range(cells)]


def test_a_floating_speck_goes():
    noise = [[0.10, -0.60, FLOOR + 0.12]]
    kept, dropped = drop_floating_fragments(
        noise, VOXEL, support_z=FLOOR, bottle_height_m=BOTTLE_H
    )
    assert dropped == 1
    assert kept == []


def test_a_bottle_standing_on_the_panel_stays():
    bottle = column(0.10, -0.60, FLOOR + VOXEL, 9)
    kept, dropped = drop_floating_fragments(
        bottle, VOXEL, support_z=FLOOR, bottle_height_m=BOTTLE_H
    )
    assert dropped == 0
    assert len(kept) == 9


def test_short_but_supported_stays():
    """A bottle seen badly is short; it is still sitting on the shelf."""
    stub = column(0.10, -0.60, FLOOR + VOXEL, 2)
    kept, dropped = drop_floating_fragments(
        stub, VOXEL, support_z=FLOOR, bottle_height_m=BOTTLE_H
    )
    assert dropped == 0
    assert len(kept) == 2


def test_tall_but_floating_stays():
    """Could be a real object whose lower half the arm occluded."""
    tall = column(0.10, -0.60, FLOOR + 0.15, 9)
    kept, dropped = drop_floating_fragments(
        tall, VOXEL, support_z=FLOOR, bottle_height_m=BOTTLE_H
    )
    assert dropped == 0
    assert len(kept) == 9


def test_both_conditions_are_required_together():
    scene = (
        column(0.10, -0.60, FLOOR + VOXEL, 9)  # bottle: stays
        + [[0.30, -0.60, FLOOR + 0.13]]  # floating speck: goes
        + column(-0.20, -0.60, FLOOR + VOXEL, 2)  # short, supported: stays
    )
    kept, dropped = drop_floating_fragments(
        scene, VOXEL, support_z=FLOOR, bottle_height_m=BOTTLE_H
    )
    assert dropped == 1
    assert len(kept) == 11
    assert not any(abs(c[0] - 0.30) < 1e-9 for c in kept)


def test_without_a_fitted_support_nothing_is_deleted():
    """No measured floor means no way to tell floating from standing."""
    scene = [[0.10, -0.60, FLOOR + 0.12]]
    kept, dropped = drop_floating_fragments(
        scene, VOXEL, support_z=None, bottle_height_m=BOTTLE_H
    )
    assert dropped == 0
    assert len(kept) == 1


def test_the_real_2026_08_07_scene_keeps_its_structures_and_loses_the_blobs():
    """The two side structures survive; the blobs clear of the panel do not.

    The single cell 20 mm above the floor is deliberately KEPT: that is inside
    the 1.5-cell band where something small genuinely resting on the shelf and
    a speck of noise are indistinguishable, and the filter is not allowed to
    guess there.  It is the blobs 70 mm up -- the two that sat inside the open
    gripper's envelope beside the approach corridor -- that this exists for.
    """
    scene = []
    for i in range(30):  # tall side structures, standing on the panel
        scene.append([0.60, -1.05, FLOOR + i * VOXEL])
    for i in range(30):
        scene.append([-0.54, -1.09, FLOOR + i * VOXEL])
    floating = [
        [0.20, -0.585, FLOOR + 0.07],
        [0.225, -0.585, FLOOR + 0.07],
        [-0.25, -0.560, FLOOR + 0.07],
        [-0.275, -0.560, FLOOR + 0.07],
    ]
    hugging_the_panel = [[-0.025, -0.835, FLOOR + 0.02]]
    kept, dropped = drop_floating_fragments(
        scene + floating + hugging_the_panel,
        VOXEL,
        support_z=FLOOR,
        bottle_height_m=BOTTLE_H,
    )
    assert dropped == len(floating)
    assert len(kept) == 60 + len(hugging_the_panel)
    assert [-0.025, -0.835, FLOOR + 0.02] in kept
