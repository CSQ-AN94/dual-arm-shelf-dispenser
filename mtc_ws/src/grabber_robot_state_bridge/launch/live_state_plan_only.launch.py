"""Plan-only stack driven by the robot's real joint states.

Starts, and only starts:

  world->dummy_base_link static TF, robot_state_publisher,
  grabber_robot_state_bridge (read-only), move_group (execution disabled),
  and optionally RViz.

Deliberately absent, so that nothing here can move the robot:

  * no controller manager and no joint trajectory controller -- the trajectory
    follow action server named in this model's moveit_controllers.yaml is never
    spawned, and moveit_manage_controllers stays False;
  * allow_trajectory_execution is False, so move_group refuses to execute even
    if something asked;
  * no gripper, lift or chassis node;
  * no MTC executable -- run that separately, plan-only.

The one substantive difference from bottle_grasp/moveit_headless.py is the
PlanningSceneMonitor's joint_state_topic: that file points move_group at
/unused_joint_states with wait_for_initial_state_timeout 0.0, i.e. it
deliberately never learns where the robot is.  Here it subscribes to the real
/joint_states and waits for it.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import xacro
import yaml

MOVEIT_CONFIG_PACKAGE = "dual_rm_75b_moveit_config"

# Keep in sync with bottle_grasp/ompl_config.py: the default 0.01 leaves a
# collision-check blind spot that let a planned path cut 1.7 cm into a keepout
# box on 2026-07-17.
OMPL_LONGEST_VALID_SEGMENT_FRACTION = 0.0025


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
    controllers = _load_yaml(MOVEIT_CONFIG_PACKAGE, "config/moveit_controllers.yaml")

    # Same node-local override as plan_shelf_transfer_experimental.launch.py:
    # the installed
    # 5 ms KDL timeout makes multi-solution IK flaky.  The production
    # dual_rm_75b_moveit_config is not modified.
    for group in kinematics.values():
        if isinstance(group, dict) and "kinematics_solver_timeout" in group:
            group["kinematics_solver_timeout"] = 0.05

    for key, value in ompl.items():
        if isinstance(value, dict) and key != "planner_configs":
            value["longest_valid_segment_fraction"] = OMPL_LONGEST_VALID_SEGMENT_FRACTION

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
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("publish_rate_hz", default_value="20.0"),
            DeclareLaunchArgument("right_arm_ip", default_value="169.254.128.19"),
            DeclareLaunchArgument("left_arm_ip", default_value="169.254.128.18"),
            DeclareLaunchArgument("bridge_status_file", default_value=""),
            DeclareLaunchArgument(
                "allow_faulted_lift_position", default_value="false"
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=["0", "0", "0", "0", "0", "0", "1", "world", "dummy_base_link"],
                output="log",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[robot_description],
                output="log",
            ),
            Node(
                package="grabber_robot_state_bridge",
                executable="joint_state_bridge",
                output="screen",
                parameters=[
                    {
                        "right_arm_ip": LaunchConfiguration("right_arm_ip"),
                        "left_arm_ip": LaunchConfiguration("left_arm_ip"),
                        "publish_rate_hz": LaunchConfiguration("publish_rate_hz"),
                        "status_file": LaunchConfiguration("bridge_status_file"),
                        "allow_faulted_lift_position": LaunchConfiguration(
                            "allow_faulted_lift_position"
                        ),
                    }
                ],
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
                        # move_group hard-requires a controller manager plugin
                        # name: leaving it unset makes TrajectoryExecutionManager
                        # log FATAL and then segfault in get_parameter (verified
                        # 2026-07-27, exit -11).  So it is declared, exactly as
                        # bottle_grasp/moveit_headless.py declares it, and
                        # neutered instead: manage_controllers off, execution
                        # not allowed, and no controller node is ever spawned,
                        # so the action servers this names do not exist.
                        "moveit_simple_controller_manager": controllers,
                        "moveit_controller_manager": (
                            "moveit_simple_controller_manager/"
                            "MoveItSimpleControllerManager"
                        ),
                        "moveit_manage_controllers": False,
                        "allow_trajectory_execution": False,
                        "planning_scene_monitor_options": {
                            "name": "planning_scene_monitor",
                            "robot_description": "robot_description",
                            "joint_state_topic": "/joint_states",
                            "attached_collision_object_topic": (
                                "/attached_collision_object"
                            ),
                            "publish_planning_scene_topic": "/planning_scene",
                            "monitored_planning_scene_topic": "/monitored_planning_scene",
                            "wait_for_initial_state_timeout": 10.0,
                        },
                        "publish_planning_scene": True,
                        "publish_geometry_updates": True,
                        "publish_state_updates": True,
                        "publish_transforms_updates": True,
                        "publish_robot_description": True,
                        "publish_robot_description_semantic": True,
                    },
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output="log",
                condition=IfCondition(LaunchConfiguration("rviz")),
                arguments=[
                    "-d",
                    os.path.join(package_path, "config", "moveit.rviz"),
                ],
                parameters=[
                    robot_description,
                    {"robot_description_semantic": semantic},
                    {"robot_description_kinematics": kinematics},
                    {"robot_description_planning": joint_limits},
                ],
            ),
        ]
    )
