#!/usr/bin/env python3
"""Validate every state of an exact demonstrated joint path in MoveIt."""

from __future__ import annotations

import json
import math
import sys

import rclpy
from moveit_msgs.msg import PlanningScene, RobotState
from moveit_msgs.srv import ApplyPlanningScene, GetStateValidity
from plan_once import arm_names
from scene_helpers import (
    build_request_scene,
    live_scene_object_ids,
    wait,
)
from sensor_msgs.msg import JointState


def main():
    request_path, output_path = sys.argv[1:3]
    with open(request_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    rclpy.init()
    node = rclpy.create_node("bottle_moveit_validate_path")
    apply_client = node.create_client(
        ApplyPlanningScene, "/apply_planning_scene"
    )
    validity_client = node.create_client(
        GetStateValidity, "/check_state_validity"
    )
    try:
        wait(apply_client)
        wait(validity_client)
        frame_id = data["planning_frame"]
        scene = PlanningScene()
        build_request_scene(scene, node, frame_id, data)
        apply_request = ApplyPlanningScene.Request()
        apply_request.scene = scene
        future = apply_client.call_async(apply_request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=20)
        if future.result() is None or not future.result().success:
            raise RuntimeError("failed to apply validation scene")
        world_collision_ids, attached_object_ids = live_scene_object_ids(
            node, timeout=10
        )

        # The trajectory belongs to whichever arm was planned, and the other
        # arm's live joints are scene.  This file used to assign the points to
        # r_joint* and the other arm to l_joint* unconditionally, so validating
        # a left-arm trajectory put the left arm's path on the right arm and
        # the right arm's real position on the left -- every point collided,
        # starting with index 0.
        group = str(data.get("planning_group", "right_arm"))
        names = arm_names(group)
        planned_names = names["joints"]
        other_names = names["other_joints"]
        other_positions = [
            math.radians(value)
            for value in (
                data.get("start_other_joints_deg")
                or data["start_left_joints_deg"]
            )
        ]
        invalid = []
        for index, planned_values in enumerate(data["points_deg"]):
            request = GetStateValidity.Request()
            request.group_name = group
            state = RobotState()
            # 同 moveit_plan_once.py：非diff状态会把附着的工具防撞体清掉，
            # 校验时夹爪没有碰撞体积，等于没校验。
            state.is_diff = True
            state.joint_state = JointState()
            state.joint_state.name = [*other_names, *planned_names]
            state.joint_state.position = [
                *other_positions,
                *[math.radians(value) for value in planned_values],
            ]
            request.robot_state = state
            future = validity_client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=5)
            response = future.result()
            if response is None:
                raise RuntimeError(f"state validity timeout at {index}")
            if not response.valid:
                invalid.append(
                    {
                        "index": index,
                        "contacts": [
                            [item.contact_body_1, item.contact_body_2]
                            for item in response.contacts[:12]
                        ],
                    }
                )
                break
        output = {
            "success": not invalid,
            "checked_states": (
                len(data["points_deg"])
                if not invalid
                else invalid[0]["index"] + 1
            ),
            "invalid": invalid,
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
