#!/usr/bin/env python3
"""Publish a live, read-only /joint_states for the dual RM75 platform.

Why this node exists: this robot drives its arms over the RealMan SDK and used
MoveIt as a stateless planning service, so nothing ever published joint states.
MTC's ``stages::CurrentState`` therefore started every task from the all-zero
default RobotState.  This node closes that seam by publishing what the
controllers actually report, at a steady rate, with a real ROS timestamp.

Guarantees:

* Only read-only SDK/controller queries are used (see sdk_reader.py).
* A failed or stale read stops publication -- the last good sample is never
  republished to keep CurrentStateMonitor looking healthy.
* name/position always have the same length, cover all 27 model joints exactly
  once, and are in ROS units (radian / metre).
"""

from __future__ import annotations

import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from . import joint_map
from .sdk_reader import (
    ArmReader,
    LiftReader,
    StateReadError,
    validate_lift_sample,
)

READ_ONLY_BANNER = (
    "本节点只读取状态，不发送机器人运动命令。 "
    "(This node only reads state; it sends no robot motion commands.)"
)


class JointStateBridge(Node):
    def __init__(self) -> None:
        super().__init__("grabber_joint_state_bridge")
        joint_map.validate()

        self.declare_parameter("right_arm_ip", "169.254.128.19")
        self.declare_parameter("left_arm_ip", "169.254.128.18")
        self.declare_parameter("arm_port", 8080)
        self.declare_parameter("lift_host", "169.254.128.18")
        self.declare_parameter("lift_port", 8080)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("lift_period_s", 1.0)
        # Both arms are separate controllers, so a cycle reads them back to
        # back.  Measured skew is 4-9 ms; 50 ms is a generous ceiling that
        # still catches a hung controller.
        self.declare_parameter("max_arm_skew_s", 0.05)
        self.declare_parameter("max_read_age_s", 0.5)
        self.declare_parameter("allow_faulted_lift_position", False)
        self.declare_parameter("status_file", "")

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.lift_period_s = float(self.get_parameter("lift_period_s").value)
        self.max_arm_skew_s = float(self.get_parameter("max_arm_skew_s").value)
        self.max_read_age_s = float(self.get_parameter("max_read_age_s").value)
        self.allow_faulted_lift_position = bool(
            self.get_parameter("allow_faulted_lift_position").value
        )
        self.status_file = str(self.get_parameter("status_file").value)

        self.get_logger().warn(READ_ONLY_BANNER)
        self.get_logger().info(
            "unmeasured joints published as declared constant "
            f"{joint_map.UNMEASURED_VALUE}: {', '.join(joint_map.UNMEASURED_JOINTS)}"
        )

        port = int(self.get_parameter("arm_port").value)
        self.right = ArmReader("right", str(self.get_parameter("right_arm_ip").value), port)
        self.left = ArmReader("left", str(self.get_parameter("left_arm_ip").value), port)
        self.lift = LiftReader(
            str(self.get_parameter("lift_host").value),
            int(self.get_parameter("lift_port").value),
        )

        # Sensor-style QoS: best effort, keep last, volatile -- what
        # CurrentStateMonitor subscribes with.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(JointState, "/joint_states", qos)
        self.status_publisher = self.create_publisher(String, "~/status", 1)

        self.published = 0
        self.read_failures = 0
        self.skew_violations = 0
        self.skew_retries = 0
        self.last_error = ""
        self.max_skew_seen = 0.0
        self.started_monotonic = time.monotonic()
        self._lift_sample = None
        self._last_publish_monotonic = None

        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.status_timer = self.create_timer(1.0, self._publish_status)

    # ---- reading -------------------------------------------------------

    def _refresh_lift(self) -> None:
        stale = (
            self._lift_sample is None
            or time.monotonic() - self._lift_sample.monotonic >= self.lift_period_s
        )
        if stale:
            sample = self.lift.sample()
            validate_lift_sample(
                sample,
                allow_faulted_position=self.allow_faulted_lift_position,
            )
            if not sample.motion_ready:
                self.get_logger().warn(
                    "diagnostic plan-only override: publishing measured lift "
                    f"position {sample.position_m:.3f}m while motion is blocked "
                    f"(enabled={sample.enabled}, err_flag={sample.error_flag})"
                )
            self._lift_sample = sample

    def _tick(self) -> None:
        try:
            right = self.right.sample()
            left = self.left.sample()
            self._refresh_lift()
        except StateReadError as exc:
            self.read_failures += 1
            self.last_error = str(exc)
            self.get_logger().error(f"state read failed, not publishing: {exc}")
            return

        skew = abs(right.monotonic - left.monotonic)
        if skew > self.max_arm_skew_s:
            # The two arms are separate controllers read back to back, so one
            # scheduling hiccup between the reads should cost a re-read, not a
            # blanked /joint_states: on 2026-08-02 eight cycles were dropped
            # outright and the consumer had no way to see the gap.  Re-read the
            # pair once; a genuinely hung controller still fails below.
            self.skew_retries += 1
            try:
                right = self.right.sample()
                left = self.left.sample()
            except StateReadError as exc:
                self.read_failures += 1
                self.last_error = str(exc)
                self.get_logger().error(
                    f"state re-read after skew failed, not publishing: {exc}"
                )
                return
            skew = abs(right.monotonic - left.monotonic)
        self.max_skew_seen = max(self.max_skew_seen, skew)
        if skew > self.max_arm_skew_s:
            self.skew_violations += 1
            self.last_error = f"arm sample skew {skew * 1000:.1f} ms over limit"
            self.get_logger().error(
                f"left/right sample skew {skew * 1000:.1f} ms exceeds "
                f"{self.max_arm_skew_s * 1000:.0f} ms, not publishing"
            )
            return

        now = time.monotonic()
        oldest = min(right.monotonic, left.monotonic, self._lift_sample.monotonic)
        age = now - oldest
        # The lift is polled at lift_period_s, so its own age is allowed to
        # reach that period; the arms must be fresh within max_read_age_s.
        arm_age = now - min(right.monotonic, left.monotonic)
        if arm_age > self.max_read_age_s:
            self.read_failures += 1
            self.last_error = f"arm sample age {arm_age:.3f}s over limit"
            self.get_logger().error(f"stale arm sample ({arm_age:.3f}s), not publishing")
            return

        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(joint_map.ALL_JOINTS)
        message.position = (
            list(right.positions_rad)
            + list(left.positions_rad)
            + [self._lift_sample.position_m]
            + [joint_map.UNMEASURED_VALUE] * len(joint_map.UNMEASURED_JOINTS)
        )
        if len(message.name) != len(message.position):
            raise RuntimeError("joint name/position length mismatch")
        self.publisher.publish(message)
        self.published += 1
        self._last_publish_monotonic = now
        self._last_snapshot = {
            "right_deg": right.raw_degrees,
            "left_deg": left.raw_degrees,
            "lift_mm": self._lift_sample.height_mm,
            "skew_s": skew,
            "lift_age_s": now - self._lift_sample.monotonic,
            "oldest_age_s": age,
        }

    # ---- diagnostics ---------------------------------------------------

    def _status(self) -> dict:
        uptime = time.monotonic() - self.started_monotonic
        return {
            "read_only": True,
            "uptime_s": round(uptime, 3),
            "published": self.published,
            "average_rate_hz": round(self.published / uptime, 3) if uptime > 0 else 0.0,
            "read_failures": self.read_failures,
            "skew_violations": self.skew_violations,
            "skew_retries": self.skew_retries,
            "max_arm_skew_s": round(self.max_skew_seen, 6),
            "last_error": self.last_error,
            "last_snapshot": getattr(self, "_last_snapshot", None),
            "allow_faulted_lift_position": self.allow_faulted_lift_position,
            "lift_motion_ready": (
                self._lift_sample is not None and self._lift_sample.motion_ready
            ),
            "publishing": (
                self._last_publish_monotonic is not None
                and time.monotonic() - self._last_publish_monotonic < 1.0
            ),
        }

    def _publish_status(self) -> None:
        status = self._status()
        message = String()
        message.data = json.dumps(status)
        self.status_publisher.publish(message)
        if self.status_file:
            with open(self.status_file, "w", encoding="utf-8") as handle:
                json.dump(status, handle, indent=2)

    def destroy_node(self) -> bool:
        self.right.close()
        self.left.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = JointStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
