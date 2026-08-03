"""Per-run shelf-panel fitting and bounded fence adaptation; no hardware.

Mirrors test_table_model.py's coverage style but exercises shelf_model.py's
axis/sign-generalized fitting, which has never run on real hardware (unlike
table_model.py, whose defaults trace back to an actual measured table).
"""

import numpy as np
import pytest

from bottle_grasp.core import DemoParams, SafetyAbort
from bottle_grasp.safety import FenceBox, SafetyProfile
from bottle_grasp.shelf_model import (
    FACE_SPECS,
    FaceFit,
    adapt_profile_to_shelf,
    combine_shelf_fits,
    fit_shelf_face,
)

TARGET = np.array([0.0, 0.55, -0.10])

# One representative plane value per face, chosen to sit inside that face's
# search band relative to TARGET (see shelf_model.fit_shelf_face).
PLANE_VALUES = {
    "shelf_bottom": -0.20,
    "shelf_top": 0.15,
    "shelf_back": 0.75,
    "shelf_left_panel": -0.20,
    "shelf_right_panel": 0.20,
}


def _plane_cloud(axis, plane_value, *, plane_points=400, seed=7):
    """Synthetic head cloud: a dominant plane on `axis` plus scattered noise."""
    rng = np.random.default_rng(seed)
    other = [a for a in (0, 1, 2) if a != axis]
    cloud = np.zeros((plane_points, 3))
    for a in other:
        cloud[:, a] = rng.uniform(TARGET[a] - 0.35, TARGET[a] + 0.35, plane_points)
    cloud[:, axis] = rng.normal(plane_value, 0.004, plane_points)
    stray = rng.uniform([-1, -0.3, -1], [1, 1.5, 0.6], (60, 3))
    return np.vstack((cloud, stray))


def _box_for(face, plane_value):
    """A keepout box whose configured tracked bound equals `plane_value`,
    consistent with FACE_SPECS' axis/tracked_bound convention."""
    spec = FACE_SPECS[face]
    minimum = [-0.5, 0.3, -0.75]
    maximum = [0.5, 1.0, 0.6]
    if spec.tracked_bound == "max":
        maximum[spec.axis] = plane_value
    else:
        minimum[spec.axis] = plane_value
    return FenceBox(id=face, minimum=tuple(minimum), maximum=tuple(maximum))


def _profile(keepout_boxes, **overrides):
    values = dict(
        name="shelf_test",
        description="",
        frame="right_controller_base",
        moveit_frame="platform_base_link",
        T_moveit_from_profile=np.eye(4),
        verified_for_execution=False,
        clearance_m=0.03,
        tcp_workspace=FenceBox(
            id="ws", minimum=(-0.65, -0.3, -0.75), maximum=(0.75, 1.0, 0.65)
        ),
        allowed_tcp_zones=(
            FenceBox(
                id="home_corridor",
                minimum=(-0.4, -0.1, -0.4),
                maximum=(0.4, 0.3, 0.5),
            ),
        ),
        keepout_boxes=keepout_boxes,
        use_dynamic_rgbd=True,
        home_joints_deg=None,
    )
    values.update(overrides)
    return SafetyProfile(**values)


@pytest.mark.parametrize("face", sorted(FACE_SPECS))
def test_fit_finds_the_face_plane_not_noise(face):
    plane_value = PLANE_VALUES[face]
    axis = FACE_SPECS[face].axis
    fit = fit_shelf_face(_plane_cloud(axis, plane_value), TARGET, face, DemoParams())
    assert fit is not None
    assert fit.face == face
    # Percentile + conservative margin biases away from the raw plane value,
    # but only by a small, bounded amount.
    assert abs(fit.plane_m - plane_value) < 0.03
    assert fit.inliers >= DemoParams().shelf_fit_min_inliers


def test_fit_rejects_unknown_face():
    with pytest.raises(SafetyAbort, match="未知的货架面"):
        fit_shelf_face(_plane_cloud(2, -0.2), TARGET, "shelf_ceiling_fan", DemoParams())


def test_fit_returns_none_without_plane_support():
    rng = np.random.default_rng(3)
    sparse = rng.uniform([-1, -0.3, -1], [1, 1.5, 0.6], (30, 3))
    assert fit_shelf_face(sparse, TARGET, "shelf_bottom", DemoParams()) is None


def _fit(face, *, plane_m, ranges=None, inliers=100):
    return FaceFit(
        face=face,
        plane_m=plane_m,
        in_plane_ranges=ranges or {a: (-0.3, 0.3) for a in (0, 1, 2) if a != FACE_SPECS[face].axis},
        inliers=inliers,
    )


def test_combine_shelf_fits_uses_median_plane_across_frames():
    params = DemoParams()
    combined = combine_shelf_fits(
        [
            _fit("shelf_bottom", plane_m=-0.204),
            _fit("shelf_bottom", plane_m=-0.205),
            _fit("shelf_bottom", plane_m=-0.206),
        ],
        params,
    )
    assert combined.plane_m == pytest.approx(-0.205)


def test_combine_shelf_fits_grows_in_plane_ranges_never_shrinks():
    params = DemoParams()
    combined = combine_shelf_fits(
        [
            _fit("shelf_back", plane_m=0.75, ranges={0: (-0.3, 0.2), 2: (-0.1, 0.3)}, inliers=120),
            _fit("shelf_back", plane_m=0.751, ranges={0: (-0.2, 0.4), 2: (-0.2, 0.2)}, inliers=90),
        ],
        params,
    )
    assert combined.in_plane_ranges[0] == (-0.3, 0.4)
    assert combined.in_plane_ranges[2] == (-0.2, 0.3)
    assert combined.inliers == 90


def test_combine_shelf_fits_rejects_frames_that_disagree():
    params = DemoParams(table_fit_agreement_m=0.015)
    with pytest.raises(SafetyAbort, match="面位置不一致"):
        combine_shelf_fits(
            [_fit("shelf_bottom", plane_m=-0.205), _fit("shelf_bottom", plane_m=-0.240)],
            params,
        )


def test_combine_shelf_fits_aborts_if_any_frame_found_no_plane():
    with pytest.raises(SafetyAbort, match="1 帧找不到货架面平面"):
        combine_shelf_fits([_fit("shelf_bottom", plane_m=-0.205), None], DemoParams())


def test_combine_shelf_fits_rejects_empty_list():
    with pytest.raises(SafetyAbort, match="空的帧列表"):
        combine_shelf_fits([], DemoParams())


def test_combine_shelf_fits_rejects_mixed_faces():
    with pytest.raises(SafetyAbort, match="不同货架面"):
        combine_shelf_fits(
            [_fit("shelf_bottom", plane_m=-0.2), _fit("shelf_top", plane_m=0.2)],
            DemoParams(),
        )


@pytest.mark.parametrize("face", sorted(FACE_SPECS))
def test_adapt_tracks_measured_bound_within_tolerance(face):
    params = DemoParams()
    old_plane = PLANE_VALUES[face]
    box = _box_for(face, old_plane)
    profile = _profile((box,))
    axis = FACE_SPECS[face].axis
    new_plane = old_plane + 0.02  # within shelf_fit_bound_tolerance_m default 0.05
    fit = _fit(face, plane_m=new_plane)
    adapted = adapt_profile_to_shelf(profile, {face: fit}, params)
    updated = next(b for b in adapted.keepout_boxes if b.id == face)
    tracked = (
        updated.maximum[axis]
        if FACE_SPECS[face].tracked_bound == "max"
        else updated.minimum[axis]
    )
    assert tracked == pytest.approx(new_plane)


def test_adapt_raises_when_bound_exceeds_tolerance():
    params = DemoParams()
    box = _box_for("shelf_bottom", -0.20)
    profile = _profile((box,))
    fit = _fit("shelf_bottom", plane_m=-0.40)
    with pytest.raises(SafetyAbort, match="超出自适应容差"):
        adapt_profile_to_shelf(profile, {"shelf_bottom": fit}, params)


def test_adapt_raises_when_recognized_face_has_no_measurement():
    params = DemoParams()
    box = _box_for("shelf_bottom", -0.20)
    profile = _profile((box,))
    with pytest.raises(SafetyAbort, match="找不到有足够支撑"):
        adapt_profile_to_shelf(profile, {}, params)


def test_adapt_in_plane_extent_only_grows_symmetric_axis():
    """shelf_bottom's in-plane x axis should grow both bounds, like table's
    x_range, since x is not the camera's occluded depth axis."""
    params = DemoParams()
    box = FenceBox(id="shelf_bottom", minimum=(-0.3, 0.3, -0.75), maximum=(0.3, 1.0, -0.20))
    profile = _profile((box,))
    fit = FaceFit(
        face="shelf_bottom",
        plane_m=-0.20,
        in_plane_ranges={0: (-0.5, 0.5), 1: (0.25, 0.9)},
        inliers=100,
    )
    adapted = adapt_profile_to_shelf(profile, {"shelf_bottom": fit}, params)
    updated = next(b for b in adapted.keepout_boxes if b.id == "shelf_bottom")
    assert updated.minimum[0] <= -0.5
    assert updated.maximum[0] >= 0.5


def test_adapt_depth_axis_only_grows_near_bound():
    """y is the camera's occluded depth axis: only the near (min) bound may
    grow; the far (max) bound must stay exactly as configured."""
    params = DemoParams()
    box = FenceBox(id="shelf_bottom", minimum=(-0.3, 0.3, -0.75), maximum=(0.3, 1.0, -0.20))
    profile = _profile((box,))
    fit = FaceFit(
        face="shelf_bottom",
        plane_m=-0.20,
        in_plane_ranges={0: (-0.2, 0.2), 1: (0.20, 0.95)},
        inliers=100,
    )
    adapted = adapt_profile_to_shelf(profile, {"shelf_bottom": fit}, params)
    updated = next(b for b in adapted.keepout_boxes if b.id == "shelf_bottom")
    assert updated.minimum[1] == pytest.approx(0.20 - params.shelf_fit_edge_margin_m)
    assert updated.maximum[1] == 1.0  # far bound untouched despite a 0.95 measurement


def test_adapt_profile_without_shelf_keepouts_is_untouched():
    box = FenceBox(id="table_top", minimum=(-1.0, 0.36, -0.75), maximum=(1.0, 1.5, -0.20))
    profile = _profile((box,))
    assert adapt_profile_to_shelf(profile, {}, DemoParams()) is profile


def test_adapt_unrecognized_box_id_passes_through():
    known = _box_for("shelf_bottom", -0.20)
    unknown = FenceBox(id="mystery_fixture", minimum=(-0.1, -0.1, -0.1), maximum=(0.1, 0.1, 0.1))
    profile = _profile((known, unknown))
    fit = _fit("shelf_bottom", plane_m=-0.20)
    adapted = adapt_profile_to_shelf(profile, {"shelf_bottom": fit}, DemoParams())
    assert any(b.id == "mystery_fixture" for b in adapted.keepout_boxes)
    mystery = next(b for b in adapted.keepout_boxes if b.id == "mystery_fixture")
    assert mystery == unknown
