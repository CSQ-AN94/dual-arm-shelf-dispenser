from pathlib import Path

import numpy as np
import pytest

from bottle_grasp.core import DemoParams, Localization, SafetyAbort
from bottle_grasp.demo import BottleDemo
from bottle_grasp.lift_evidence import LiftEvidenceKind, LiftVisualEvidence
from bottle_grasp.safety import load_safety_profile
from scripts.bottle_grasp_no_environment_avoidance import (
    NoEnvironmentAvoidanceDemo,
    PROFILE_NAME,
    PROFILE_PATH,
)


def _localization(point):
    return Localization(
        point_camera=list(point),
        point_base=list(point),
        pixel=[0.0, 0.0],
        depth_m=0.5,
        depth_mad_m=0.001,
        position_spread_m=0.002,
        box=[0, 0, 1, 1],
        confidence=0.8,
        frame_count=3,
    )


def test_no_environment_profile_has_an_empty_world():
    profile = load_safety_profile(
        PROFILE_PATH, PROFILE_NAME, require_verified=False
    )

    assert profile.use_dynamic_rgbd is False
    assert profile.keepout_boxes == ()
    assert profile.moveit_collision_boxes() == []
    assert profile.observation_staging_joints_deg == pytest.approx(
        [
            25.21,
            54.74,
            21.496,
            62.516,
            -40.634,
            -36.028,
            -23.643,
        ]
    )
    profile.assert_tcp_point([0.0, 0.5, -0.25], label="table overlap")
    with pytest.raises(SafetyAbort, match="尚未现场测量确认"):
        load_safety_profile(
            PROFILE_PATH, PROFILE_NAME, require_verified=True
        )


def test_environment_gate_override_does_not_read_the_camera():
    demo = object.__new__(NoEnvironmentAvoidanceDemo)
    stages = []
    demo.stage = lambda name, message: stages.append((name, message))

    demo.collision_gate(None, None)

    assert stages == [
        ("右腕点云通道检查", "由无环境避障入口显式跳过")
    ]


def test_lift_confirmation_accepts_real_occluded_bottle_measurement():
    """Replay 2026-07-19: the gripper hid the lower bottle after lifting.

    The box-relative 66% sample moved to a different physical point, making
    the measured Z rise 102 mm although the bottle/TCP rose 50 mm.  This is
    still strong evidence of a successful upward departure in this explicitly
    supervised no-environment mode.
    """
    locked = _localization(
        [0.09857135759227303, 0.5760282137587792, -0.13667134924986352]
    )
    measured = _localization(
        [0.07510898427487475, 0.5737326619494627, -0.03468321644760006]
    )
    demo = object.__new__(NoEnvironmentAvoidanceDemo)
    demo.params = DemoParams()
    demo._post_lift_visual_evidence = lambda _target: LiftVisualEvidence(
        LiftEvidenceKind.VISUAL_CONFIRMED,
        "fresh wrist measurement",
        3,
        3,
        measured,
    )
    stages = []
    demo.stage = lambda name, message: stages.append((name, message))

    result = demo._confirm_lifted_target(
        locked,
        prelift_tcp=np.eye(4),
        postlift_tcp=np.eye(4),
    )

    assert result is measured
    assert stages[-1][0] == "抬升真实视觉确认通过"


@pytest.mark.parametrize(
    "measured_point",
    (
        [0.10, 0.57, -0.125],  # only 12 mm upward: still on/near the table
        [0.17, 0.57, -0.086],  # lifted but jumped 70 mm sideways
    ),
)
def test_lift_confirmation_still_rejects_no_lift_or_wrong_target(
    measured_point,
):
    locked = _localization([0.10, 0.57, -0.137])
    measured = _localization(measured_point)
    demo = object.__new__(NoEnvironmentAvoidanceDemo)
    demo.params = DemoParams()
    demo._post_lift_visual_evidence = lambda _target: LiftVisualEvidence(
        LiftEvidenceKind.VISUAL_NEGATIVE,
        f"measured={measured.point_base}",
        1,
        1,
        measured,
    )
    demo.stage = lambda *_args: None

    with pytest.raises(SafetyAbort, match="视觉否定优先"):
        demo._confirm_lifted_target(
            locked,
            prelift_tcp=np.eye(4),
            postlift_tcp=np.eye(4),
        )


def test_normal_entrypoint_files_are_not_the_unsafe_entrypoint():
    root = Path(__file__).parents[2]
    assert PROFILE_PATH != root / "bottle_grasp" / "safety_profiles.json"
    assert (root / "scripts" / "run_bottle_grasp.sh").exists()


def test_all_entry_modes_share_grasp_geometry_orientation_and_place_back():
    # The no-environment subclass may skip only environmental checks.  Both
    # from-start/from-observation modes must inherit the same physical grasp,
    # grasp geometry, lift-confirmation and retreat implementation as normal mode.
    for method_name in (
        "candidate_path",
        "_local_pick_place_geometry",
        "_confirm_lifted_target",
        "_place_back",
    ):
        assert method_name not in NoEnvironmentAvoidanceDemo.__dict__
        assert getattr(NoEnvironmentAvoidanceDemo, method_name) is getattr(
            BottleDemo, method_name
        )
