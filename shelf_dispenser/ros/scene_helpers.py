"""Shared planning-scene construction for the ROS helper scripts.

Runs only inside the ROS 2 environment (`python3 shelf_dispenser/<helper>.py`
puts this directory on sys.path). Keep vision-environment code out of here.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningSceneComponents,
)
from moveit_msgs.srv import GetPlanningScene
from scene_ids import RGBD_VOXELS_ID, managed_scene_ids
from shape_msgs.msg import SolidPrimitive


def wait(client, timeout=20.0):
    if not client.wait_for_service(timeout_sec=timeout):
        raise RuntimeError(f"service unavailable: {client.srv_name}")


def live_world_object_ids(node, timeout=20.0):
    """Ask move_group which world collision objects currently exist."""
    client = node.create_client(GetPlanningScene, "/get_planning_scene")
    wait(client)
    request = GetPlanningScene.Request()
    request.components.components = (
        PlanningSceneComponents.WORLD_OBJECT_NAMES
    )
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    response = future.result()
    if response is None:
        raise RuntimeError("get_planning_scene timed out")
    return [item.id for item in response.scene.world.collision_objects]


def live_scene_object_ids(node, timeout=20.0):
    """Read back world and attached object IDs after applying a scene diff."""
    client = node.create_client(GetPlanningScene, "/get_planning_scene")
    wait(client)
    request = GetPlanningScene.Request()
    request.components.components = (
        PlanningSceneComponents.WORLD_OBJECT_NAMES
        | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
    )
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    response = future.result()
    if response is None:
        raise RuntimeError("get_planning_scene timed out")
    return (
        [item.id for item in response.scene.world.collision_objects],
        [
            item.object.id
            for item in response.scene.robot_state.attached_collision_objects
        ],
    )


def remove_stale_objects(scene, frame_id, existing_ids):
    """Delete every demo-owned object before re-adding the current request.

    The previous design tracked which ids the last request created and sent
    per-id clears; a helper that crashed between "request written" and
    "scene applied" desynchronized that bookkeeping and left stale voxels
    in the live scene. Querying the scene makes cleanup stateless.
    """
    for object_id in managed_scene_ids(existing_ids):
        collision = CollisionObject()
        collision.header.frame_id = frame_id
        collision.id = object_id
        collision.operation = CollisionObject.REMOVE
        scene.world.collision_objects.append(collision)


def add_voxel_object(scene, frame_id, centers, voxel_size):
    """Add all RGB-D voxels as one CollisionObject with many box primitives.

    One object with N primitives keeps ApplyPlanningScene and collision
    manager bookkeeping O(1) in object count instead of N separate objects,
    and removal is a single REMOVE of RGBD_VOXELS_ID.
    """
    if not centers:
        return
    collision = CollisionObject()
    collision.header.frame_id = frame_id
    collision.id = RGBD_VOXELS_ID
    for center in centers:
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(voxel_size)] * 3
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, center)
        pose.orientation.w = 1.0
        collision.primitives.append(primitive)
        collision.primitive_poses.append(pose)
    collision.operation = CollisionObject.ADD
    scene.world.collision_objects.append(collision)


def add_box(scene, frame_id, object_id, center, size):
    collision = CollisionObject()
    collision.header.frame_id = frame_id
    collision.id = str(object_id)
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(map(float, size))
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, center)
    pose.orientation.w = 1.0
    collision.primitives = [primitive]
    collision.primitive_poses = [pose]
    collision.operation = CollisionObject.ADD
    scene.world.collision_objects.append(collision)


def _attached_box(*, object_id, link_name, size, center, quaternion, touch_links):
    attached = AttachedCollisionObject()
    attached.link_name = str(link_name)
    attached.touch_links = list(touch_links)
    attached.object.header.frame_id = str(link_name)
    attached.object.id = str(object_id)
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(map(float, size))
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, center)
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = map(float, quaternion)
    attached.object.primitives = [primitive]
    attached.object.primitive_poses = [pose]
    attached.object.operation = CollisionObject.ADD
    return attached


def attach_tool_guard(scene, guard, held_object=None):
    """Attach the gripper stand-in volume to r_link7.

    The enclosing scene's robot_state must be a diff: a non-diff robot state
    replaces the whole scene robot state and silently drops this attachment
    (2026-07-16 real-arm incident: planned TCP passed 2.7cm above the table).
    """
    if not guard and not held_object:
        return
    attached_objects = []
    if guard:
        attached_objects.append(
            _attached_box(
                object_id="bottle_tool_guard",
                link_name="r_link7",
                size=[guard["xy"], guard["xy"], guard["length"]],
                center=[0.0, 0.0, guard["center_z"]],
                quaternion=[0.0, 0.0, 0.0, 1.0],
                touch_links=["r_link6", "r_link7", "r_hand"],
            )
        )
    if held_object:
        attached_objects.append(
            _attached_box(
                object_id="held_bottle_guard",
                link_name="r_link7",
                size=held_object["size"],
                center=held_object["center"],
                quaternion=held_object["quaternion_xyzw"],
                touch_links=["r_link6", "r_link7", "r_hand"],
            )
        )
    scene.robot_state.is_diff = True
    scene.robot_state.attached_collision_objects = attached_objects


def build_request_scene(scene, node, frame_id, data):
    """Populate a diff PlanningScene from one helper request dict."""
    scene.is_diff = True
    remove_stale_objects(scene, frame_id, live_world_object_ids(node))
    add_voxel_object(
        scene, frame_id, data.get("obstacles", []), data["voxel_size"]
    )
    for item in data.get("boxes", []):
        add_box(scene, frame_id, item["id"], item["center"], item["size"])
    attach_tool_guard(scene, data.get("tool_guard"), data.get("held_object"))
