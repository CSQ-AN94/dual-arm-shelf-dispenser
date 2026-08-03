#!/usr/bin/env python3
"""Explicitly detach one object and verify the live PlanningScene read-back."""

from __future__ import annotations

import json
import sys

import rclpy
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene

from moveit_scene_helpers import live_scene_object_ids, wait


def main():
    request_path, output_path = sys.argv[1:3]
    with open(request_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    object_id = str(data["object_id"])
    rclpy.init()
    node = rclpy.create_node("bottle_moveit_detach_object")
    client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    try:
        wait(client)
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        attached = AttachedCollisionObject()
        attached.link_name = str(data.get("link_name", "r_link7"))
        attached.object.header.frame_id = attached.link_name
        attached.object.id = object_id
        attached.object.operation = CollisionObject.REMOVE
        scene.robot_state.attached_collision_objects = [attached]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=20)
        applied = future.result() is not None and future.result().success
        world_ids, attached_ids = live_scene_object_ids(node, timeout=10)
        result = {
            "success": bool(applied and object_id not in attached_ids),
            "world_collision_ids": world_ids,
            "attached_object_ids": attached_ids,
            "removed_object_id": object_id,
        }
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        return 0 if result["success"] else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
