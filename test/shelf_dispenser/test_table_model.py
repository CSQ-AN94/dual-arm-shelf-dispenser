"""Per-run table fitting and bounded fence adaptation; no hardware."""

import numpy as np
import pytest

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.safety import FenceBox, SafetyProfile
from shelf_dispenser.table_model import (
    TABLE_KEEPOUT_ID,
    TableFit,
    adapt_profile_to_table,
    combine_table_fits,
    fit_table_top,
)

TARGET = np.array([0.0, 0.55, -0.10])


def _cloud(table_z, *, table_points=400, seed=7):
    """Synthetic head cloud: table plane + bottle body + floor + noise."""
    rng = np.random.default_rng(seed)
    table = np.column_stack(
        (
            rng.uniform(-0.5, 0.5, table_points),
            rng.uniform(0.35, 1.1, table_points),
            rng.normal(table_z, 0.004, table_points),
        )
    )
    bottle = np.column_stack(
        (
            rng.normal(0.0, 0.02, 60),
            rng.normal(0.55, 0.02, 60),
            rng.uniform(table_z, TARGET[2] + 0.08, 60),
        )
    )
    floor = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, 200),
            rng.uniform(0.2, 1.5, 200),
            rng.normal(-0.95, 0.01, 200),
        )
    )
    stray = rng.uniform([-1, -0.3, -1], [1, 1.5, 0.6], (40, 3))
    return np.vstack((table, bottle, floor, stray))


def _profile(**overrides):
    values = dict(
        name="table_demo",
        description="",
        frame="right_controller_base",
        moveit_frame="platform_base_link",
        T_moveit_from_profile=np.eye(4),
        verified_for_execution=True,
        clearance_m=0.025,
        tcp_workspace=FenceBox(
            id="ws", minimum=(-0.65, -0.3, -0.75), maximum=(0.75, 1.0, 0.65)
        ),
        allowed_tcp_zones=(
            FenceBox(
                id="transit", minimum=(-0.6, -0.25, -0.7), maximum=(0.7, 0.58, 0.6)
            ),
            FenceBox(
                id="above_table",
                minimum=(-0.52, 0.2, -0.18),
                maximum=(0.5, 0.94, 0.55),
            ),
        ),
        keepout_boxes=(
            FenceBox(
                id=TABLE_KEEPOUT_ID,
                minimum=(-1.0, 0.36, -0.75),
                maximum=(1.0, 1.5, -0.20),
            ),
        ),
        use_dynamic_rgbd=True,
        home_joints_deg=None,
    )
    values.update(overrides)
    return SafetyProfile(**values)


def test_fit_finds_the_table_not_the_floor_or_bottle():
    fit = fit_table_top(_cloud(-0.205), TARGET, DemoParams())
    assert fit is not None
    assert abs(fit.height_m - (-0.205)) < 0.01
    assert fit.top_m > fit.height_m
    assert fit.inliers >= 60


def test_fit_returns_none_without_plane_support():
    rng = np.random.default_rng(3)
    sparse = rng.uniform([-1, -0.3, -1], [1, 1.5, 0.6], (30, 3))
    assert fit_table_top(sparse, TARGET, DemoParams()) is None


def test_lower_table_lowers_keepout_and_zone_floor():
    """桌子比配置低时，旧盒子会挡住真实桌面上方的可用空间——必须跟下来。"""
    params = DemoParams()
    fit = fit_table_top(_cloud(-0.28), TARGET, params)
    adapted = adapt_profile_to_table(_profile(), fit, params)
    table = next(
        box for box in adapted.keepout_boxes if box.id == TABLE_KEEPOUT_ID
    )
    assert table.maximum[2] == pytest.approx(fit.top_m)
    assert table.maximum[2] < -0.24
    zone = next(
        box for box in adapted.allowed_tcp_zones if box.id == "above_table"
    )
    # 允许区底面保持"桌面上方 2cm"的作者化余量
    assert zone.minimum[2] == pytest.approx(fit.top_m + 0.02)
    # 与桌面无关的转移走廊不动
    transit = next(
        box for box in adapted.allowed_tcp_zones if box.id == "transit"
    )
    assert transit.minimum[2] == -0.7


def test_higher_table_raises_protection():
    params = DemoParams()
    fit = fit_table_top(_cloud(-0.13), TARGET, params)
    adapted = adapt_profile_to_table(_profile(), fit, params)
    table = next(
        box for box in adapted.keepout_boxes if box.id == TABLE_KEEPOUT_ID
    )
    assert table.maximum[2] > -0.13


def test_horizontal_extent_only_grows():
    params = DemoParams()
    fit = fit_table_top(_cloud(-0.205), TARGET, params)
    adapted = adapt_profile_to_table(_profile(), fit, params)
    table = next(
        box for box in adapted.keepout_boxes if box.id == TABLE_KEEPOUT_ID
    )
    assert table.minimum[0] <= -1.0
    assert table.maximum[0] >= 1.0
    assert table.minimum[1] <= 0.36


def test_out_of_envelope_height_refuses_to_run():
    params = DemoParams()
    fit = fit_table_top(_cloud(-0.36), TARGET, params)
    assert fit is not None
    with pytest.raises(SafetyAbort, match="超出自适应容差"):
        adapt_profile_to_table(_profile(), fit, params)


def test_missing_plane_is_fatal_when_profile_expects_a_table():
    with pytest.raises(SafetyAbort, match="找不到有足够支撑的桌面"):
        adapt_profile_to_table(_profile(), None, DemoParams())


def test_profile_without_table_keepout_is_untouched():
    profile = _profile(
        keepout_boxes=(
            FenceBox(
                id="shelf_bottom",
                minimum=(-0.5, 0.3, -0.2),
                maximum=(0.5, 1.0, -0.15),
            ),
        )
    )
    assert adapt_profile_to_table(profile, None, DemoParams()) is profile


def _fit(*, height_m, top_m=None, x_range=(-0.4, 0.4), y_front=0.3, inliers=100):
    return TableFit(
        height_m=height_m,
        top_m=height_m + 0.01 if top_m is None else top_m,
        x_range=x_range,
        y_front=y_front,
        inliers=inliers,
    )


def test_combine_table_fits_uses_median_height_across_frames():
    params = DemoParams()
    combined = combine_table_fits(
        [_fit(height_m=-0.204), _fit(height_m=-0.205), _fit(height_m=-0.206)],
        params,
    )
    assert combined.height_m == pytest.approx(-0.205)


def test_combine_table_fits_grows_the_protected_volume_never_shrinks_it():
    """Height is median (robust to one outlier), but everything that sizes
    the keepout box takes the most protective value seen across frames."""
    params = DemoParams()
    combined = combine_table_fits(
        [
            _fit(height_m=-0.205, top_m=-0.195, x_range=(-0.4, 0.3), y_front=0.32, inliers=120),
            _fit(height_m=-0.204, top_m=-0.190, x_range=(-0.3, 0.5), y_front=0.28, inliers=90),
        ],
        params,
    )
    assert combined.top_m == pytest.approx(-0.190)  # highest surface wins
    assert combined.x_range == (-0.4, 0.5)  # widest extent wins
    assert combined.y_front == pytest.approx(0.28)  # nearest front edge wins
    assert combined.inliers == 90  # weakest evidence reported, not flattered


def test_combine_table_fits_rejects_frames_that_disagree():
    """A real table does not move between consecutive frames. A spread this
    large means the scene changed mid-capture (hand crossing the view,
    exposure switch) — trusting either frame would silently shift the fence."""
    params = DemoParams(table_fit_agreement_m=0.015)
    with pytest.raises(SafetyAbort, match="帧实测桌面高度不一致"):
        combine_table_fits(
            [_fit(height_m=-0.205), _fit(height_m=-0.240)], params
        )


def test_combine_table_fits_tolerates_spread_within_the_agreement_band():
    params = DemoParams(table_fit_agreement_m=0.015)
    combined = combine_table_fits(
        [_fit(height_m=-0.205), _fit(height_m=-0.212)], params
    )
    assert combined.height_m == pytest.approx((-0.205 + -0.212) / 2)


def test_combine_table_fits_aborts_if_any_frame_found_no_plane():
    """One bad frame in the batch must not be silently dropped and averaged
    around — it is evidence the capture window was not stable."""
    params = DemoParams()
    with pytest.raises(SafetyAbort, match="1 帧找不到桌面平面"):
        combine_table_fits([_fit(height_m=-0.205), None], params)


def test_combine_table_fits_rejects_empty_list():
    with pytest.raises(SafetyAbort, match="空的帧列表"):
        combine_table_fits([], DemoParams())
