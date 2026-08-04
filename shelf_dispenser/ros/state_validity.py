#!/usr/bin/env python3
"""Ask MoveIt whether given joint states collide, and with what.

Written to answer one question with data instead of argument: the left arm is
physically sitting somewhere touching nothing, and the planner refuses a three
degree move from it as a collision with the head.  Either the model disagrees
with reality, or the arm really is that close.  Asking the model directly about
the pose the arm is actually in settles it.

Request JSON:  {"planning_group": "left_arm",
                "states": [{"label": "...", "joints_deg": [7 values],
                            "other_joints_deg": [7 values] | null,
                            "extra_joints": {"head_joint1": 0.0, ...}}]}
Reply JSON:    {"results": [{"label": "...", "valid": bool,
                             "contacts": [[link_a, link_b], ...]}]}
"""

import json
import math
import sys

import rclpy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity
from plan_once import arm_names
from scene_helpers import wait
from sensor_msgs.msg import JointState


def main():
    request_path, output_path = sys.argv[1:3]
    with open(request_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    group = str(data.get("planning_group", "right_arm"))
    names = arm_names(group)

    rclpy.init()
    node = rclpy.create_node("shelf_dispenser_state_validity")
    client = node.create_client(GetStateValidity, "/check_state_validity")
    try:
        wait(client)
        results = []
        for item in data["states"]:
            joint_names = list(names["joints"])
            positions = [math.radians(v) for v in item["joints_deg"]]
            other = item.get("other_joints_deg")
            if other is not None:
                joint_names = [*names["other_joints"], *joint_names]
                positions = [math.radians(v) for v in other] + positions
            for name, value in (item.get("extra_joints") or {}).items():
                joint_names.append(name)
                positions.append(float(value))

            state = RobotState()
            state.is_diff = True
            state.joint_state = JointState()
            state.joint_state.name = joint_names
            state.joint_state.position = positions

            request = GetStateValidity.Request()
            request.robot_state = state
            request.group_name = group
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=15)
            response = future.result()
            if response is None:
                raise RuntimeError("check_state_validity timed out")
            results.append(
                {
                    "label": item.get("label", ""),
                    "valid": bool(response.valid),
                    "contacts": [
                        [c.contact_body_1, c.contact_body_2]
                        for c in response.contacts
                    ],
                }
            )
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump({"results": results}, stream)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
