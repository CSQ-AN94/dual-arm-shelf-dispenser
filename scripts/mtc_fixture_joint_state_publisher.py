#!/usr/bin/env python3
"""Publish an explicit simulation-only joint-state fixture for plan-only MTC."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


NAMES = [
    "platform_joint",
    "head_joint1",
    "head_joint2",
    *[f"l_joint{i}" for i in range(1, 8)],
    *[f"r_joint{i}" for i in range(1, 8)],
    "joint_left_wheel",
    "joint_right_wheel",
    *[f"joint_swivel_wheel_{i}_{j}" for i in range(1, 5) for j in (1, 2)],
]


class Publisher(Node):
    def __init__(self, positions: list[float]):
        super().__init__("grabber_simulation_fixture_joint_state_publisher")
        self._positions = positions
        self._publishers = [
            self.create_publisher(JointState, topic, qos_profile_sensor_data)
            for topic in ("/joint_states", "/unused_joint_states")
        ]
        self.create_timer(0.05, self.publish_fixture)

    def publish_fixture(self) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = NAMES
        message.position = self._positions
        for publisher in self._publishers:
            publisher.publish(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    args, ros_args = parser.parse_known_args()
    payload = json.loads(args.state.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "grabber.mtc_fixture_joint_state.v1"
        or payload.get("simulation_only") is not True
        or payload.get("hardware_connections") != 0
    ):
        parser.error("state is not an explicit simulation-only fixture")
    platform_m = float(payload["platform_height_mm"]) / 1000.0
    head = np.asarray(payload["head_joints_rad"], dtype=float)
    left = np.radians(np.asarray(payload["left_joints_deg"], dtype=float))
    right = np.radians(np.asarray(payload["right_joints_deg"], dtype=float))
    if (
        not math.isfinite(platform_m)
        or not 0.0 <= platform_m <= 1.0
        or head.shape != (2,)
        or left.shape != (7,)
        or right.shape != (7,)
        or not np.all(np.isfinite(np.r_[head, left, right]))
    ):
        parser.error("fixture joint dimensions or values are invalid")
    positions = [platform_m, *head.tolist(), *left.tolist(), *right.tolist(), *([0.0] * 10)]
    rclpy.init(args=ros_args)
    rclpy.spin(Publisher(positions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
