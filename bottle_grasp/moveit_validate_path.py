#!/usr/bin/env python3
"""Validate every state of an exact demonstrated joint path in MoveIt."""

from __future__ import annotations

import json
import math
import sys

import rclpy
from moveit_msgs.msg import PlanningScene, RobotState
from moveit_msgs.srv import ApplyPlanningScene, GetStateValidity
from moveit_scene_helpers import (
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

        right_names = [f"r_joint{i}" for i in range(1, 8)]
        left_names = [f"l_joint{i}" for i in range(1, 8)]
        left_positions = [
            math.radians(value)
            for value in data["start_left_joints_deg"]
        ]
        invalid = []
        for index, right_values in enumerate(data["points_deg"]):
            request = GetStateValidity.Request()
            request.group_name = "right_arm"
            state = RobotState()
            # 同 moveit_plan_once.py：非diff状态会把附着的工具防撞体清掉，
            # 校验时夹爪没有碰撞体积，等于没校验。
            state.is_diff = True
            state.joint_state = JointState()
            state.joint_state.name = [*left_names, *right_names]
            state.joint_state.position = [
                *left_positions,
                *[math.radians(value) for value in right_values],
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
