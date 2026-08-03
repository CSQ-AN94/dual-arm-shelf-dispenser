#!/usr/bin/env python3
"""Launch a headless MoveIt move_group for the installed dual RM75 model."""

import os

import launch
import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from ompl_config import apply_collision_check_resolution


def load_yaml(package_name, relative_path):
    path = os.path.join(get_package_share_directory(package_name), relative_path)
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_text(package_name, relative_path):
    path = os.path.join(get_package_share_directory(package_name), relative_path)
    with open(path, "r", encoding="utf-8") as stream:
        return stream.read()


def description():
    package = "dual_rm_75b_moveit_config"
    package_path = get_package_share_directory(package)
    robot_xml = xacro.process_file(
        os.path.join(package_path, "config", "dual_rm_75b_description.urdf.xacro")
    ).toxml()
    semantic = load_text(package, "config/dual_rm_75b_description.srdf")
    kinematics = load_yaml(package, "config/kinematics.yaml")
    joint_limits = load_yaml(package, "config/joint_limits.yaml")
    ompl = apply_collision_check_resolution(
        load_yaml(package, "config/ompl_planning.yaml")
    )
    controllers = load_yaml(package, "config/moveit_controllers.yaml")
    pipeline = {
        "move_group": {
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
    robot_description = {"robot_description": robot_xml}
    return LaunchDescription(
        [
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=[
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "1",
                    "world",
                    "dummy_base_link",
                ],
                output="log",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[robot_description],
                output="log",
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[
                    robot_description,
                    {"robot_description_semantic": semantic},
                    {"robot_description_kinematics": kinematics},
                    {"robot_description_planning": joint_limits},
                    pipeline,
                    {
                        "moveit_simple_controller_manager": controllers,
                        "moveit_controller_manager": (
                            "moveit_simple_controller_manager/"
                            "MoveItSimpleControllerManager"
                        ),
                        "moveit_manage_controllers": False,
                        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
                        "trajectory_execution.allowed_goal_duration_margin": 0.5,
                        "trajectory_execution.allowed_start_tolerance": 0.1,
                    },
                    {
                        "planning_scene_monitor_options": {
                            "name": "planning_scene_monitor",
                            "robot_description": "robot_description",
                            "joint_state_topic": "/unused_joint_states",
                            "attached_collision_object_topic": "/attached_collision_object",
                            "publish_planning_scene_topic": "/planning_scene",
                            "monitored_planning_scene_topic": "/monitored_planning_scene",
                            "wait_for_initial_state_timeout": 0.0,
                        },
                        "publish_planning_scene": True,
                        "publish_geometry_updates": True,
                        "publish_state_updates": True,
                        "publish_transforms_updates": True,
                        "allow_trajectory_execution": False,
                    },
                ],
            ),
        ]
    )


if __name__ == "__main__":
    service = launch.LaunchService()
    service.include_launch_description(description())
    raise SystemExit(service.run())
