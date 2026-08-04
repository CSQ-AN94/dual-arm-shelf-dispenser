#!/usr/bin/env python3
"""Batch MoveIt link7 FK for one planning group, at joint states given to it.

Used to measure an arm's bridge from its controller base frame to the MoveIt
frame: the same joint state, evaluated by both kinematics, gives one equation
for the constant between them.  Nothing has to move -- both sides are pure
forward kinematics -- so this never commands the robot.

Request JSON:  {"planning_group": "left_arm", "joint_states_deg": [[7 values], ...],
                "planning_frame": "platform_base_link"}
Reply JSON:    {"poses": [{"position": [...], "quaternion_xyzw": [...]}, ...]}
"""

import json
import math
import sys

import rclpy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from plan_once import arm_names, compute_link7_fk
from scene_helpers import wait
from sensor_msgs.msg import JointState


def main():
    request_path, output_path = sys.argv[1:3]
    with open(request_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    group = str(data.get("planning_group", "right_arm"))
    names = arm_names(group)
    planning_frame = data["planning_frame"]
    states = data["joint_states_deg"]
    if not states:
        raise RuntimeError("joint_states_deg is empty")

    rclpy.init()
    node = rclpy.create_node("shelf_dispenser_link7_fk")
    fk_client = node.create_client(GetPositionFK, "/compute_fk")
    try:
        wait(fk_client)
        poses = []
        for values in states:
            if len(values) != 7:
                raise RuntimeError("each joint state must hold seven degrees")
            state = RobotState()
            state.is_diff = True
            state.joint_state = JointState()
            state.joint_state.name = names["joints"]
            state.joint_state.position = [math.radians(v) for v in values]
            poses.append(
                compute_link7_fk(
                    node, fk_client, state, planning_frame, names["ik_link"]
                )
            )
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump({"poses": poses}, stream)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
