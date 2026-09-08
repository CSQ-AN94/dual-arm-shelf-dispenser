from types import SimpleNamespace

import numpy as np
import pytest

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.delivery_table import (
    observations_agree,
    observe_output_table,
    placement_still_valid,
)


def _config():
    return SimpleNamespace(
        table_roi_min=(0.20, 0.20, 0.40),
        table_roi_max=(0.80, 0.80, 0.65),
        table_height_bin_m=0.01,
        table_inlier_band_m=0.012,
        table_min_inliers=80,
        table_frame_agreement_m=0.012,
        table_edge_margin_m=0.08,
        table_support_radius_m=0.07,
        table_min_patch_points=4,
        place_clearance_radius_m=0.11,
        place_grid_m=0.04,
        obstacle_min_height_m=0.025,
        obstacle_max_height_m=0.45,
        max_place_candidates=8,
    )


def _frame(height=0.50, obstacle=True):
    x, y = np.meshgrid(np.linspace(0.20, 0.80, 50), np.linspace(0.20, 0.80, 50))
    table = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, height)))
    if not obstacle:
        return table
    z = np.linspace(height + 0.04, height + 0.25, 20)
    column = np.column_stack((np.full(20, 0.50), np.full(20, 0.50), z))
    return np.vstack((table, column))


def test_output_table_is_fit_per_frame_and_place_avoids_obstacle_column():
    observation = observe_output_table(
        [_frame(0.500), _frame(0.503), _frame(0.497)], _config()
    )

    assert observation.table_height_m == pytest.approx(0.50, abs=0.003)
    assert len(observation.candidates) == 8
    assert np.linalg.norm(np.asarray(observation.best.xy_base) - [0.5, 0.5]) >= 0.11
    assert observation.best.nearest_obstacle_m >= 0.11


def test_output_table_selects_table_above_a_more_populated_floor():
    floor = _frame(0.42, False)
    floor = np.vstack((floor, floor))
    table = _frame(0.55, False)

    observation = observe_output_table([np.vstack((floor, table))] * 3, _config())

    assert observation.table_height_m == pytest.approx(0.55, abs=0.003)


def test_output_table_rejects_disagreeing_depth_frames():
    with pytest.raises(SafetyAbort, match="多帧高度不一致"):
        observe_output_table(
            [_frame(0.50, False), _frame(0.54, False), _frame(0.50, False)],
            _config(),
        )


def test_refresh_rejects_a_changed_table_or_selected_patch():
    planned = observe_output_table([_frame()] * 3, _config())
    changed = observe_output_table([_frame(0.53)] * 3, _config())

    with pytest.raises(SafetyAbort, match="刷新桌面发生变化"):
        observations_agree(
            planned,
            changed,
            height_tolerance_m=0.012,
            xy_tolerance_m=0.04,
        )


def test_refresh_checks_the_selected_patch_not_whichever_patch_ranks_best():
    planned = observe_output_table([_frame()] * 3, _config())
    selected = planned.candidates[-1]
    refreshed = observe_output_table([_frame()] * 3, _config())

    placement_still_valid(
        planned_table_height_m=planned.table_height_m,
        planned_xy_base=selected.xy_base,
        refreshed=refreshed,
        height_tolerance_m=0.012,
        xy_tolerance_m=0.001,
    )


def test_scene_only_refresh_can_return_without_an_unused_place_patch():
    config = _config()
    config.table_min_patch_points = 100000

    observation = observe_output_table(
        [_frame()] * 3, config, require_candidates=False
    )

    assert observation.candidates == ()


def test_ranking_defaults_to_the_roi_centre():
    observation = observe_output_table([_frame(), _frame(), _frame()], _config())

    centre = np.asarray([0.50, 0.50])
    best = np.asarray(observation.best.xy_base)
    for candidate in observation.candidates[1:]:
        assert np.linalg.norm(best - centre) <= np.linalg.norm(
            np.asarray(candidate.xy_base) - centre
        )


def test_a_preferred_place_point_reranks_the_patches_that_survive_truncation():
    # The ROI centre is an authored safety envelope, not a statement about
    # reach.  A caller holding a taught place point ranks from there instead,
    # so max_place_candidates keeps the reachable patches rather than the
    # ones nearest the middle of the box.
    preferred = (0.30, 0.30)
    observation = observe_output_table(
        [_frame(), _frame(), _frame()],
        _config(),
        preferred_xy=preferred,
    )

    assert observation.candidates
    target = np.asarray(preferred)
    best = np.asarray(observation.best.xy_base)
    assert float(np.linalg.norm(best - target)) < 0.10
    for candidate in observation.candidates[1:]:
        assert np.linalg.norm(best - target) <= np.linalg.norm(
            np.asarray(candidate.xy_base) - target
        )


def test_a_malformed_preferred_place_point_is_refused():
    with pytest.raises(SafetyAbort, match="首选放置点"):
        observe_output_table(
            [_frame(), _frame(), _frame()],
            _config(),
            preferred_xy=(0.3, float("nan")),
        )


def _tilted_frame(a=0.02, b=0.03, c=0.50):
    """A tabletop whose height varies more than the inlier band allows."""
    x, y = np.meshgrid(np.linspace(0.20, 0.80, 60), np.linspace(0.20, 0.80, 60))
    z = a * x.ravel() + b * y.ravel() + c
    return np.column_stack((x.ravel(), y.ravel(), z))


def test_a_tilted_tabletop_is_fitted_whole_rather_than_one_strip():
    # Span across this surface is ~30 mm against a 12 mm band, so a constant
    # height can only ever hold part of it.
    frames = [_tilted_frame(), _tilted_frame(), _tilted_frame()]
    config = _config()
    config.max_place_candidates = 400
    observation = observe_output_table(frames, config, preferred_xy=(0.50, 0.50))

    # Both ends of the tilt are represented, not just the strip that happened
    # to land inside a 12 mm band about one seed.
    ys = {round(candidate.xy_base[1], 3) for candidate in observation.candidates}
    assert min(ys) < 0.35 and max(ys) > 0.65
    # And each patch carries the surface height under itself.
    for candidate in observation.candidates:
        expected = 0.02 * candidate.xy_base[0] + 0.03 * candidate.xy_base[1] + 0.50
        assert candidate.table_height_m == pytest.approx(expected, abs=0.003)


def test_a_narrow_high_surface_is_not_mistaken_for_the_table():
    # The arm carrying the bottle is the highest thing in the ROI and can
    # clear the point count outright; only its extent gives it away.
    x, y = np.meshgrid(np.linspace(0.45, 0.55, 40), np.linspace(0.45, 0.55, 40))
    mast = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.62)))
    frames = [np.vstack((_frame(0.50, obstacle=False), mast))] * 3

    observation = observe_output_table(frames, _config())

    assert observation.table_height_m == pytest.approx(0.50, abs=0.005)


def test_a_vouched_candidate_survives_where_the_camera_sees_no_support():
    # Blank the near strip so no camera-supported patch can be generated there.
    seen = _frame(0.50, obstacle=False)
    seen = seen[seen[:, 1] >= 0.45]
    frames = [seen, seen, seen]
    vouched = ((0.35, 0.35), 0.497)

    without = observe_output_table(frames, _config(), preferred_xy=(0.35, 0.35))
    assert all(
        candidate.xy_base != pytest.approx((0.35, 0.35))
        for candidate in without.candidates
    )

    with_taught = observe_output_table(
        frames,
        _config(),
        preferred_xy=(0.35, 0.35),
        vouched_candidates=[vouched],
    )
    best = with_taught.best
    assert best.xy_base == pytest.approx((0.35, 0.35))
    assert best.table_height_m == pytest.approx(0.497)
    # Marked as not camera-supported so the evidence file can say so.
    assert best.support_points == 0


def test_a_vouched_candidate_is_still_refused_when_something_stands_on_it():
    frames = [_frame(0.50), _frame(0.50), _frame(0.50)]
    # _frame puts an obstacle column at (0.50, 0.50); the demonstration says
    # nothing about what is standing there now.
    observation = observe_output_table(
        frames,
        _config(),
        vouched_candidates=[((0.50, 0.50), 0.50)],
    )

    assert all(
        candidate.xy_base != pytest.approx((0.50, 0.50))
        for candidate in observation.candidates
    )
