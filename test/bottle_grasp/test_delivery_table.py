from types import SimpleNamespace

import numpy as np
import pytest

from bottle_grasp.core import SafetyAbort
from bottle_grasp.delivery_table import (
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
