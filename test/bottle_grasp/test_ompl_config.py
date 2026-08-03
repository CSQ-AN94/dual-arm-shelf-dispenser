"""OMPL collision-check discretization override; no ROS required."""

from bottle_grasp.ompl_config import (
    DIVERSE_PLANNER_CONFIGS,
    OMPL_LONGEST_VALID_SEGMENT_FRACTION,
    apply_collision_check_resolution,
)


def test_every_group_section_gets_the_denser_resolution():
    ompl = {
        "planner_configs": {"RRTConnect": {"type": "geometric::RRTConnect"}},
        "right_arm": {"planner_configs": ["RRTConnect"]},
        "head": {"planner_configs": ["RRTConnect"]},
    }
    result = apply_collision_check_resolution(ompl)
    for group in ("right_arm", "left_arm", "head"):
        assert (
            result[group]["longest_valid_segment_fraction"]
            == OMPL_LONGEST_VALID_SEGMENT_FRACTION
        )
    # 平台配置里的 planner_configs 定义不能被当成规划组覆盖。
    assert "longest_valid_segment_fraction" not in result["planner_configs"]
    assert set(DIVERSE_PLANNER_CONFIGS).issubset(
        result["right_arm"]["planner_configs"]
    )
    for planner_id in DIVERSE_PLANNER_CONFIGS:
        assert planner_id in result["planner_configs"]


def test_planned_arms_are_forced_even_if_yaml_omits_them():
    result = apply_collision_check_resolution({"planner_configs": {}})
    assert (
        result["right_arm"]["longest_valid_segment_fraction"]
        == OMPL_LONGEST_VALID_SEGMENT_FRACTION
    )


def test_resolution_matches_offline_recheck_density():
    # 复核密度是 planned_joint_step_deg=1.5°；规划期离散化必须同量级，
    # 否则会回到"规划说无碰、复核说有碰"的 narrow band 拒绝循环。
    assert OMPL_LONGEST_VALID_SEGMENT_FRACTION <= 0.005
