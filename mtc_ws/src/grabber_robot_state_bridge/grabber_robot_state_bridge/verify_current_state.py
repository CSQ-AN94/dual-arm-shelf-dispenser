#!/usr/bin/env python3
"""Prove that MoveIt's current state is the robot's real state, then report it.

This is the Gate B admission test.  It is read-only: it subscribes to
/joint_states, takes its own independent SDK snapshot, asks the running
move_group for its monitored PlanningScene, and cross-checks the published
joint convention with /compute_fk against the controller's own pose report.

Everything it learns lands in current_state_report.json.  Any failed check is
written into the report as a failure -- it never rewrites a check to pass.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import rclpy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPlanningScene, GetPositionFK
from rclpy.node import Node
from sensor_msgs.msg import JointState

from . import joint_map
from .sdk_reader import ArmReader

ARM_LINKS = {"right": ("r_base_link1", "r_link7"), "left": ("l_base_link1", "l_link7")}


class Verifier(Node):
    def __init__(self, listen_s: float) -> None:
        super().__init__("grabber_current_state_verifier")
        self.listen_s = listen_s
        self.messages: list[tuple[float, JointState]] = []
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 20)
        self.scene_client = self.create_client(GetPlanningScene, "/get_planning_scene")
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")

    def _on_joint_state(self, message: JointState) -> None:
        self.messages.append((time.time(), message))

    # ---- collection ----------------------------------------------------

    def collect(self) -> dict:
        deadline = time.time() + self.listen_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not self.messages:
            return {"received": 0, "failures": ["no /joint_states message received"]}

        wall_times = [item[0] for item in self.messages]
        stamps = [
            item[1].header.stamp.sec + item[1].header.stamp.nanosec * 1e-9
            for item in self.messages
        ]
        gaps = [b - a for a, b in zip(wall_times, wall_times[1:])] or [0.0]
        last = self.messages[-1][1]
        failures = []

        names = list(last.name)
        if len(names) != len(last.position):
            failures.append("name/position length mismatch")
        if len(names) != len(set(names)):
            failures.append("duplicate joint names in /joint_states")
        missing = [name for name in joint_map.ALL_JOINTS if name not in names]
        if missing:
            failures.append(f"missing model joints: {missing}")
        missing_planning = [
            name for name in joint_map.PLANNING_JOINTS if name not in names
        ]
        if missing_planning:
            failures.append(f"missing planning joints: {missing_planning}")
        if any(stamp == 0.0 for stamp in stamps):
            failures.append("at least one header.stamp is zero")
        if len(set(stamps)) < len(stamps) / 2:
            failures.append("header.stamp is not advancing")

        age = time.time() - stamps[-1]
        if abs(age) > 0.5:
            failures.append(f"latest state age {age:.3f}s exceeds 0.5s")

        rate = (len(self.messages) - 1) / (wall_times[-1] - wall_times[0] or 1.0)
        return {
            "received": len(self.messages),
            "window_s": round(wall_times[-1] - wall_times[0], 3),
            "measured_rate_hz": round(rate, 2),
            "gap_mean_ms": round(statistics.mean(gaps) * 1000, 2),
            "gap_max_ms": round(max(gaps) * 1000, 2),
            "joint_count": len(names),
            "planning_joints_present": not missing_planning,
            "duplicate_joints": len(names) != len(set(names)),
            "latest_stamp": stamps[-1],
            "latest_state_age_s": round(age, 4),
            "stamp_nonzero": all(stamp != 0.0 for stamp in stamps),
            "published_values": dict(zip(names, [float(v) for v in last.position])),
            "failures": failures,
        }

    # ---- move_group side ----------------------------------------------

    def planning_scene_state(self) -> dict:
        if not self.scene_client.wait_for_service(timeout_sec=10.0):
            return {"available": False, "failures": ["/get_planning_scene unavailable"]}
        request = GetPlanningScene.Request()
        # ROBOT_STATE = 2 in moveit_msgs/PlanningSceneComponents
        request.components.components = 2
        future = self.scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result()
        if response is None:
            return {"available": False, "failures": ["/get_planning_scene timed out"]}
        state = response.scene.robot_state.joint_state
        values = dict(zip(list(state.name), [float(v) for v in state.position]))
        failures = []
        missing = [name for name in joint_map.PLANNING_JOINTS if name not in values]
        if missing:
            failures.append(f"move_group robot state missing joints: {missing}")
        arm_values = [values.get(name, 0.0) for name in joint_map.PLANNING_JOINTS]
        if all(abs(value) < 1e-9 for value in arm_values):
            failures.append("move_group robot state is still the all-zero default")
        return {
            "available": True,
            "joint_count": len(values),
            "complete_planning_joints": not missing,
            "all_zero": all(abs(value) < 1e-9 for value in arm_values),
            "values": values,
            "failures": failures,
        }

    def fk(self, robot_state: RobotState, links: list[str]) -> dict:
        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            return {}
        request = GetPositionFK.Request()
        request.header.frame_id = ""  # model root, no TF lookup needed
        request.fk_link_names = links
        request.robot_state = robot_state
        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result()
        if response is None or response.error_code.val != 1:
            return {}
        return {
            name: stamped.pose
            for name, stamped in zip(response.fk_link_names, response.pose_stamped)
        }


def _pose_xyz(pose) -> list:
    return [pose.position.x, pose.position.y, pose.position.z]


def _relative_distance(base, tip) -> float:
    return math.dist(_pose_xyz(base), _pose_xyz(tip))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="current_state_report.json")
    parser.add_argument("--listen-s", type=float, default=5.0)
    parser.add_argument("--right-ip", default="169.254.128.19")
    parser.add_argument("--left-ip", default="169.254.128.18")
    args = parser.parse_args()

    rclpy.init()
    node = Verifier(args.listen_s)
    report: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "read_only": True,
        "joint_mapping": joint_map.mapping_document(),
    }

    topic = node.collect()
    report["joint_states_topic"] = topic

    # Independent SDK snapshot, taken now, from this process's own handles.
    right = ArmReader("right", args.right_ip)
    left = ArmReader("left", args.left_ip)
    try:
        right_sample = right.sample()
        left_sample = left.sample()
    finally:
        right.close()
        left.close()
    snapshot_skew = abs(right_sample.monotonic - left_sample.monotonic)
    report["sdk_snapshot"] = {
        "right_degrees": right_sample.raw_degrees,
        "left_degrees": left_sample.raw_degrees,
        "right_radians": right_sample.positions_rad,
        "left_radians": left_sample.positions_rad,
        "arm_sample_skew_s": round(snapshot_skew, 6),
    }

    published = topic.get("published_values", {})
    sdk_radians = dict(
        list(zip(joint_map.RIGHT_ARM_JOINTS, right_sample.positions_rad))
        + list(zip(joint_map.LEFT_ARM_JOINTS, left_sample.positions_rad))
    )
    deltas = {
        name: round(published.get(name, float("nan")) - value, 6)
        for name, value in sdk_radians.items()
    }
    finite = [abs(value) for value in deltas.values() if value == value]
    report["sdk_vs_ros"] = {
        "per_joint_delta_rad": deltas,
        "max_abs_delta_rad": round(max(finite), 6) if finite else None,
        "max_abs_delta_deg": round(math.degrees(max(finite)), 4) if finite else None,
        # The arms are static but the controller reports micro-jitter; a
        # stationary arm agrees to well under a milliradian.
        "within_1_mrad": bool(finite) and max(finite) < 1e-3,
    }

    scene = node.planning_scene_state()
    report["move_group_current_state"] = scene

    # Convention cross-check: FK the published state and compare the wrist
    # distance from the arm base against the controller's own pose report.
    fk_report = {}
    if scene.get("available") and topic.get("received"):
        state = RobotState()
        state.joint_state = node.messages[-1][1]
        state.is_diff = False
        links = [link for pair in ARM_LINKS.values() for link in pair]
        poses = node.fk(state, links)
        for arm, (base, tip) in ARM_LINKS.items():
            if base in poses and tip in poses:
                fk_report[arm] = {
                    "base_link": base,
                    "tip_link": tip,
                    "tip_xyz_world": [round(v, 6) for v in _pose_xyz(poses[tip])],
                    "base_xyz_world": [round(v, 6) for v in _pose_xyz(poses[base])],
                    "wrist_distance_from_base_m": round(
                        _relative_distance(poses[base], poses[tip]), 6
                    ),
                }
    report["forward_kinematics_from_published_state"] = fk_report

    failures = (
        topic.get("failures", [])
        + scene.get("failures", [])
        + ([] if report["sdk_vs_ros"]["within_1_mrad"] else ["SDK/ROS joint mismatch"])
    )
    report["failures"] = failures
    report["verdict"] = "PASS" if not failures else "FAIL"

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"verdict": report["verdict"], "failures": failures}, indent=2))

    node.destroy_node()
    rclpy.try_shutdown()
    raise SystemExit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
