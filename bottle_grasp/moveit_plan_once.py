#!/usr/bin/env python3
"""Apply voxel obstacles and request one collision-aware RM75 plan."""

from __future__ import annotations

import json
import math
import sys

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetMotionPlan,
    GetPositionFK,
    GetStateValidity,
)
from moveit_scene_helpers import (
    build_request_scene,
    live_scene_object_ids,
    wait,
)
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive


def quaternion_from_matrix(matrix):
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        return [
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
            0.25 * scale,
        ]
    index = max(range(3), key=lambda i: matrix[i][i])
    if index == 0:
        scale = math.sqrt(1 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2
        return [
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[2][1] - matrix[1][2]) / scale,
        ]
    if index == 1:
        scale = math.sqrt(1 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2
        return [
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
        ]
    scale = math.sqrt(1 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2
    return [
        (matrix[0][2] + matrix[2][0]) / scale,
        (matrix[1][2] + matrix[2][1]) / scale,
        0.25 * scale,
        (matrix[1][0] - matrix[0][1]) / scale,
    ]


def quaternion_to_matrix(quaternion):
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        raise RuntimeError("compute_fk returned a zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def pose_to_rotation_translation(pose):
    rotation = quaternion_to_matrix(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
    )
    translation = [pose.position.x, pose.position.y, pose.position.z]
    return rotation, translation


def compute_link7_fk(node, client, state, planning_frame):
    """Return the r_link7 pose expressed in planning_frame.

    /compute_fk 只有在 frame_id 为模型根坐标系时才不查 TF。headless 启动没
    有任何 /joint_states 发布者，platform_joint（prismatic，body_base_link
    -> platform_base_link）的 TF 永远缺失，任何非根 frame_id 都会确定性返
    回 FRAME_TRANSFORM_FAILURE(-21)。因此这里在根坐标系下同时求 r_link7
    和 planning_frame 两个 link 的 FK，再在本地合成相对位姿；两者出自同一
    份规划场景机器人状态，与规划器解释目标约束的坐标一致。
    """
    request = GetPositionFK.Request()
    request.header.frame_id = ""  # 空 frame_id = 模型根坐标系，不触发 TF
    request.fk_link_names = ["r_link7", planning_frame]
    request.robot_state = state
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10)
    response = future.result()
    if response is None or response.error_code.val != 1:
        code = None if response is None else response.error_code.val
        raise RuntimeError(f"compute_fk failed: {code}")
    poses = dict(zip(response.fk_link_names, response.pose_stamped))
    missing = [
        name for name in ("r_link7", planning_frame) if name not in poses
    ]
    if missing:
        raise RuntimeError(f"compute_fk missing link poses: {missing}")
    frame_rotation, frame_translation = pose_to_rotation_translation(
        poses[planning_frame].pose
    )
    link_rotation, link_translation = pose_to_rotation_translation(
        poses["r_link7"].pose
    )
    # T_rel = inv(T_frame) @ T_link7；刚体逆直接用旋转转置
    relative_rotation = [
        [
            sum(
                frame_rotation[k][i] * link_rotation[k][j]
                for k in range(3)
            )
            for j in range(3)
        ]
        for i in range(3)
    ]
    delta = [
        link_translation[i] - frame_translation[i] for i in range(3)
    ]
    relative_position = [
        sum(frame_rotation[k][i] * delta[k] for k in range(3))
        for i in range(3)
    ]
    return {
        "position": relative_position,
        "quaternion_xyzw": quaternion_from_matrix(relative_rotation),
    }


def clone_state_with_right_positions(state, names, positions):
    result = RobotState()
    result.is_diff = True
    result.joint_state = JointState()
    result.joint_state.name = list(state.joint_state.name)
    by_name = dict(zip(result.joint_state.name, state.joint_state.position))
    by_name.update(zip(names, positions))
    result.joint_state.position = [
        float(by_name[name]) for name in result.joint_state.name
    ]
    return result


def main():
    request_path, output_path = sys.argv[1:3]
    with open(request_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    rclpy.init()
    node = rclpy.create_node("bottle_moveit_plan_once")
    apply_client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    plan_client = node.create_client(GetMotionPlan, "/plan_kinematic_path")
    validity_client = node.create_client(
        GetStateValidity, "/check_state_validity"
    )
    fk_client = node.create_client(GetPositionFK, "/compute_fk")
    try:
        wait(apply_client)
        wait(plan_client)
        wait(validity_client)
        wait(fk_client)
        scene_request = ApplyPlanningScene.Request()
        scene = PlanningScene()
        planning_frame = data["planning_frame"]
        build_request_scene(scene, node, planning_frame, data)
        scene_request.scene = scene
        future = apply_client.call_async(scene_request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=20)
        if future.result() is None or not future.result().success:
            raise RuntimeError("failed to apply planning scene")

        world_collision_ids, attached_object_ids = live_scene_object_ids(
            node, timeout=10
        )

        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = "right_arm"
        planner_id = str(data["planner_id"])
        allowed_planning_time = float(data["allowed_planning_time_s"])
        num_planning_attempts = int(data["num_planning_attempts"])
        if (
            not planner_id
            or not math.isfinite(allowed_planning_time)
            or allowed_planning_time <= 0
            or num_planning_attempts < 1
        ):
            raise RuntimeError("invalid MoveIt planner search budget")
        motion.planner_id = planner_id
        motion.num_planning_attempts = num_planning_attempts
        motion.allowed_planning_time = allowed_planning_time
        motion.max_velocity_scaling_factor = 0.08
        motion.max_acceleration_scaling_factor = 0.08
        motion.workspace_parameters.header.frame_id = planning_frame
        workspace = data["workspace"]
        (
            motion.workspace_parameters.min_corner.x,
            motion.workspace_parameters.min_corner.y,
            motion.workspace_parameters.min_corner.z,
        ) = map(float, workspace["min"])
        (
            motion.workspace_parameters.max_corner.x,
            motion.workspace_parameters.max_corner.y,
            motion.workspace_parameters.max_corner.z,
        ) = map(float, workspace["max"])
        minimum_link7_z = data.get("minimum_link7_z")
        if minimum_link7_z is not None:
            minimum_link7_z = float(minimum_link7_z)
            workspace_min = list(map(float, workspace["min"]))
            workspace_max = list(map(float, workspace["max"]))
            if (
                not math.isfinite(minimum_link7_z)
                or minimum_link7_z < workspace_min[2]
                or minimum_link7_z >= workspace_max[2]
            ):
                raise RuntimeError("minimum_link7_z is outside workspace")
            path_position = PositionConstraint()
            path_position.header.frame_id = planning_frame
            path_position.link_name = "r_link7"
            path_box = SolidPrimitive()
            path_box.type = SolidPrimitive.BOX
            path_box.dimensions = [
                workspace_max[0] - workspace_min[0],
                workspace_max[1] - workspace_min[1],
                workspace_max[2] - minimum_link7_z,
            ]
            path_pose = Pose()
            path_pose.position.x = (workspace_min[0] + workspace_max[0]) / 2
            path_pose.position.y = (workspace_min[1] + workspace_max[1]) / 2
            path_pose.position.z = (minimum_link7_z + workspace_max[2]) / 2
            path_pose.orientation.w = 1.0
            path_position.constraint_region.primitives = [path_box]
            path_position.constraint_region.primitive_poses = [path_pose]
            path_position.weight = 1.0
            motion.path_constraints.position_constraints = [path_position]
        state = RobotState()
        # 关键：必须是diff状态。非diff的start_state会整体替换场景里的机器人
        # 状态，把上面刚附着的bottle_tool_guard防撞体清掉，规划时夹爪等于
        # 没有碰撞体积——2026-07-16实测规划出TCP距桌面2.7cm的擦桌路径。
        state.is_diff = True
        state.joint_state = JointState()
        right_names = [f"r_joint{i}" for i in range(1, 8)]
        right_positions = [
            math.radians(value) for value in data["start_joints_deg"]
        ]
        left_values = data.get("start_left_joints_deg")
        if left_values is None:
            state.joint_state.name = right_names
            state.joint_state.position = right_positions
        else:
            state.joint_state.name = [
                *[f"l_joint{i}" for i in range(1, 8)],
                *right_names,
            ]
            state.joint_state.position = [
                *[math.radians(value) for value in left_values],
                *right_positions,
            ]
        motion.start_state = state
        start_link7_fk = compute_link7_fk(
            node, fk_client, state, planning_frame
        )

        # There is deliberately no legacy default.  A missing field used to
        # turn a requested pose into a silent joint goal.
        goal_constraint = data["goal_constraint"]
        if goal_constraint not in {"pose", "joints"}:
            raise RuntimeError(
                f"invalid goal_constraint: {goal_constraint!r}"
            )

        # Always diagnose the supplied SDK IK candidate.  Pose-constrained
        # planning may choose another IK branch, but an invalid candidate must
        # no longer disappear behind MoveIt's unhelpful error=99999.
        goal_state = RobotState()
        goal_state.is_diff = True
        goal_state.joint_state = JointState()
        goal_state.joint_state.name = list(state.joint_state.name)
        goal_state.joint_state.position = list(state.joint_state.position)
        goal_joints = data.get("goal_joints_deg")
        if goal_joints is not None:
            right_goal = [math.radians(value) for value in goal_joints]
            if data.get("start_left_joints_deg") is None:
                goal_state.joint_state.position = right_goal
            else:
                goal_state.joint_state.position = [
                    *goal_state.joint_state.position[:7],
                    *right_goal,
                ]
        validity_request = GetStateValidity.Request()
        validity_request.robot_state = goal_state
        validity_request.group_name = "right_arm"
        validity_future = validity_client.call_async(validity_request)
        rclpy.spin_until_future_complete(
            node, validity_future, timeout_sec=10
        )
        validity = validity_future.result()
        goal_state_valid = None if validity is None else bool(validity.valid)
        goal_state_contacts = (
            []
            if validity is None
            else [
                [contact.contact_body_1, contact.contact_body_2]
                for contact in validity.contacts[:8]
            ]
        )
        goal_candidate_link7_fk = compute_link7_fk(
            node, fk_client, goal_state, planning_frame
        )

        constraint = Constraints()
        if goal_constraint == "joints":
            if goal_joints is None:
                raise RuntimeError("joint goal requested without goal_joints_deg")
            for name, value in zip(right_names, goal_joints):
                joint_constraint = JointConstraint()
                joint_constraint.joint_name = name
                joint_constraint.position = math.radians(value)
                joint_constraint.tolerance_above = math.radians(0.2)
                joint_constraint.tolerance_below = math.radians(0.2)
                joint_constraint.weight = 1.0
                constraint.joint_constraints.append(joint_constraint)
        else:
            target = data["target_flange"]
            rotation = target[:3]
            position = [row[3] for row in target[:3]]
            quaternion = quaternion_from_matrix(rotation)
            position_constraint = PositionConstraint()
            position_constraint.header.frame_id = planning_frame
            position_constraint.link_name = "r_link7"
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [0.008, 0.008, 0.008]
            region_pose = Pose()
            (
                region_pose.position.x,
                region_pose.position.y,
                region_pose.position.z,
            ) = position
            region_pose.orientation.w = 1.0
            position_constraint.constraint_region.primitives = [box]
            position_constraint.constraint_region.primitive_poses = [region_pose]
            position_constraint.weight = 1.0
            orientation_constraint = OrientationConstraint()
            orientation_constraint.header.frame_id = planning_frame
            orientation_constraint.link_name = "r_link7"
            (
                orientation_constraint.orientation.x,
                orientation_constraint.orientation.y,
                orientation_constraint.orientation.z,
                orientation_constraint.orientation.w,
            ) = quaternion
            orientation_constraint.absolute_x_axis_tolerance = 0.035
            orientation_constraint.absolute_y_axis_tolerance = 0.035
            orientation_constraint.absolute_z_axis_tolerance = 0.035
            orientation_constraint.weight = 1.0
            constraint.position_constraints = [position_constraint]
            constraint.orientation_constraints = [orientation_constraint]
        motion.goal_constraints = [constraint]

        future = plan_client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=20)
        response = future.result()
        if response is None:
            raise RuntimeError("MoveIt planning service timed out")
        result = response.motion_plan_response
        error_code = int(result.error_code.val)
        trajectory = result.trajectory.joint_trajectory
        endpoint_link7_fk = None
        if trajectory.points:
            endpoint_state = clone_state_with_right_positions(
                state,
                list(trajectory.joint_names),
                list(trajectory.points[-1].positions),
            )
            endpoint_link7_fk = compute_link7_fk(
                node, fk_client, endpoint_state, planning_frame
            )
        expected_joint_names = [f"r_joint{i}" for i in range(1, 8)]
        actual_joint_names = list(trajectory.joint_names)
        points_deg = []
        output_joint_names = actual_joint_names
        if trajectory.points:
            if (
                len(actual_joint_names) != len(expected_joint_names)
                or len(set(actual_joint_names)) != len(actual_joint_names)
                or set(actual_joint_names) != set(expected_joint_names)
            ):
                raise RuntimeError(
                    "right_arm trajectory joint contract violated: "
                    f"{actual_joint_names!r}"
                )
            source_index = {
                name: index for index, name in enumerate(actual_joint_names)
            }
            output_joint_names = expected_joint_names
            for point_index, point in enumerate(trajectory.points):
                if len(point.positions) != len(actual_joint_names):
                    raise RuntimeError(
                        "trajectory point position count does not match joint_names: "
                        f"point={point_index}"
                    )
                points_deg.append(
                    [
                        math.degrees(point.positions[source_index[name]])
                        for name in expected_joint_names
                    ]
                )

        output = {
            "success": error_code == 1 and bool(trajectory.points),
            "error_code": error_code,
            "planning_time": float(result.planning_time),
            "planner_id": planner_id,
            "num_planning_attempts": num_planning_attempts,
            "joint_names": output_joint_names,
            "points_deg": points_deg,
            "goal_constraint": goal_constraint,
            "goal_state_valid": goal_state_valid,
            "goal_state_contacts": goal_state_contacts,
            "start_link7_fk": start_link7_fk,
            "goal_candidate_link7_fk": goal_candidate_link7_fk,
            "endpoint_link7_fk": endpoint_link7_fk,
            "world_collision_ids": world_collision_ids,
            "attached_object_ids": attached_object_ids,
        }
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(output, stream, indent=2)
        return 0 if output["success"] else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
