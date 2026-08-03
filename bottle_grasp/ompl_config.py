"""ROS-free OMPL pipeline configuration overrides.

Imported by moveit_headless.py inside the ROS environment (script directory
is on sys.path) and by the vision-environment tests; keep it dependency-free.
"""

from __future__ import annotations

# OMPL 只在路径边上按这个"占构型空间总尺度的比例"离散采样做碰撞检测（无
# 连续碰撞检测）。默认 0.01 对 7 自由度 RM75 意味着相邻两次碰撞检测之间
# 关节可以走好几度、TCP 可以平移近 10cm——2026-07-17 真机 observe 实测就
# 是这个盲区让规划路径比垫大后的禁区盒还深入 1.7cm。0.0025 把每段最大
# 关节变化压到约 1.8°，与离线复核的 planned_joint_step_deg=1.5° 同量级，
# 让"规划时认为无碰"和"复核时确认无碰"的判定密度一致。
OMPL_LONGEST_VALID_SEGMENT_FRACTION = 0.0025

# Search diversity is part of the runtime contract.  Repeating the same
# RRTConnect request a couple of times is not a meaningful "no path" result.
# These are standard OMPL planners available through MoveIt's OMPL plugin.
DIVERSE_PLANNER_CONFIGS = {
    "RRTConnectkConfigDefault": {
        "type": "geometric::RRTConnect",
        "range": "0.0",
    },
    "LBKPIECEkConfigDefault": {
        "type": "geometric::LBKPIECE",
        "range": "0.0",
    },
    "RRTstarkConfigDefault": {
        "type": "geometric::RRTstar",
        "range": "0.0",
        "goal_bias": "0.05",
    },
    "PRMstarkConfigDefault": {"type": "geometric::PRMstar"},
}


def apply_collision_check_resolution(ompl: dict) -> dict:
    """Set the collision-check discretization for every planning group.

    The installed robot package's ompl_planning.yaml decides which group
    sections exist; both arms are forced in so the override cannot silently
    miss the group we actually plan with.
    """
    groups = {
        key
        for key, value in ompl.items()
        if isinstance(value, dict) and key != "planner_configs"
    }
    groups.update(("right_arm", "left_arm"))
    configs = ompl.setdefault("planner_configs", {})
    for name, config in DIVERSE_PLANNER_CONFIGS.items():
        configs.setdefault(name, dict(config))
    for group in sorted(groups):
        group_config = ompl.setdefault(group, {})
        group_config["longest_valid_segment_fraction"] = (
            OMPL_LONGEST_VALID_SEGMENT_FRACTION
        )
        enabled = list(group_config.get("planner_configs") or [])
        for name in DIVERSE_PLANNER_CONFIGS:
            if name not in enabled:
                enabled.append(name)
        group_config["planner_configs"] = enabled
    return ompl
