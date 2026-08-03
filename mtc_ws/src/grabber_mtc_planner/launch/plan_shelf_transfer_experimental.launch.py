"""Experimental plan-only MTC launch against the installed dual RM75 model.

This launch only starts the planner node.  It expects an already-running
move_group (the PlanningScene and the live joint states come from there); it
mirrors the parameter loading of bottle_grasp/moveit_headless.py so the planner
sees exactly the same robot model, kinematics, limits and OMPL settings.

No motion is possible from this launch: the node has no execution path.  The
50 ms KDL override below is intentionally confined to this experimental file.
"""

import math
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import xacro
import yaml

MOVEIT_CONFIG_PACKAGE = "dual_rm_75b_moveit_config"

# Keep in sync with bottle_grasp/ompl_config.py: the default 0.01 leaves a
# collision-check blind spot that let a planned path cut 1.7 cm into a keepout
# box on 2026-07-17.
OMPL_LONGEST_VALID_SEGMENT_FRACTION = 0.0025
# The controller reports J3's hard upper bound as 178.00 deg while the URDF
# says 178.19 deg.  Use 3.5 deg here so the plan remains at least 3.0 deg
# inside the controller's real bound after that model discrepancy.
JOINT_POSITION_MARGIN_RAD = math.radians(3.5)


def _load_yaml(package, relative_path):
    path = os.path.join(get_package_share_directory(package), relative_path)
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _load_text(package, relative_path):
    path = os.path.join(get_package_share_directory(package), relative_path)
    with open(path, "r", encoding="utf-8") as stream:
        return stream.read()


def generate_launch_description():
    package_path = get_package_share_directory(MOVEIT_CONFIG_PACKAGE)
    robot_xml = xacro.process_file(
        os.path.join(package_path, "config", "dual_rm_75b_description.urdf.xacro")
    ).toxml()
    semantic = _load_text(MOVEIT_CONFIG_PACKAGE, "config/dual_rm_75b_description.srdf")
    kinematics = _load_yaml(MOVEIT_CONFIG_PACKAGE, "config/kinematics.yaml")
    joint_limits = _load_yaml(MOVEIT_CONFIG_PACKAGE, "config/joint_limits.yaml")
    ompl = _load_yaml(MOVEIT_CONFIG_PACKAGE, "config/ompl_planning.yaml")

    # MoveIt otherwise accepts IK exactly on the URDF position bound (the
    # live controller rejected r_joint3=178.19° when blending).  Shrink both
    # arms only in this plan-only RobotModel.
    limit_overrides = joint_limits.setdefault("joint_limits", {})
    for joint in ET.fromstring(robot_xml).findall("joint"):
        name = joint.attrib.get("name", "")
        limit = joint.find("limit")
        if not name.startswith(("l_joint", "r_joint")) or limit is None:
            continue
        lower = float(limit.attrib["lower"]) + JOINT_POSITION_MARGIN_RAD
        upper = float(limit.attrib["upper"]) - JOINT_POSITION_MARGIN_RAD
        if lower >= upper:
            raise RuntimeError(f"joint position margin collapses {name}")
        limit_overrides.setdefault(name, {}).update(
            {
                "has_position_limits": True,
                "min_position": lower,
                "max_position": upper,
            }
        )

    # The installed kinematics.yaml gives KDL 5 ms, which is enough for the
    # existing single-shot service calls but makes MTC's multi-solution IK
    # flaky (the same grasp pose alternates between "solution found" and
    # "no IK found").  Overridden here only, for this node -- the production
    # dual_rm_75b_moveit_config is not touched.
    for group in kinematics.values():
        if isinstance(group, dict) and "kinematics_solver_timeout" in group:
            group["kinematics_solver_timeout"] = 0.05

    for key, value in ompl.items():
        if isinstance(value, dict) and key != "planner_configs":
            value["longest_valid_segment_fraction"] = OMPL_LONGEST_VALID_SEGMENT_FRACTION

    # MTC's PipelinePlanner builds a planning_pipeline under the "ompl"
    # parameter namespace, so the whole ompl_planning.yaml nests under it.
    ompl_pipeline = {
        "ompl": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": (
                "default_planner_request_adapters/AddTimeOptimalParameterization "
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ),
            "start_state_max_bounds_error": 0.1,
            **ompl,
        }
    }
    pilz_pipeline = {
        "pilz_industrial_motion_planner": {
            "planning_plugin": "pilz_industrial_motion_planner/CommandPlanner",
            "request_adapters": (
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ),
            "start_state_max_bounds_error": 0.1,
        }
    }
    joint_limits["cartesian_limits"] = {
        "max_trans_vel": 0.1,
        "max_trans_acc": 0.2,
        "max_trans_dec": -0.2,
        "max_rot_vel": 0.5,
    }

    default_scenario = os.path.join(
        get_package_share_directory("grabber_mtc_planner"),
        "scenarios",
        "shelf_transfer_fixture.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("scenario", default_value=default_scenario),
            DeclareLaunchArgument("out", default_value="mtc_plan_result.json"),
            DeclareLaunchArgument("hold_seconds", default_value="0"),
            Node(
                package="grabber_mtc_planner",
                executable="plan_shelf_transfer",
                output="screen",
                arguments=[
                    "--plan-only",
                    "--scenario",
                    LaunchConfiguration("scenario"),
                    "--out",
                    LaunchConfiguration("out"),
                    "--hold-seconds",
                    LaunchConfiguration("hold_seconds"),
                ],
                parameters=[
                    {"robot_description": robot_xml},
                    {"robot_description_semantic": semantic},
                    {"robot_description_kinematics": kinematics},
                    {"robot_description_planning": joint_limits},
                    ompl_pipeline,
                    pilz_pipeline,
                ],
            ),
        ]
    )
