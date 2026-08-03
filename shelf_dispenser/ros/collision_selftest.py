#!/usr/bin/env python3
"""判定 MoveIt 的碰撞检查是否真的在工作。

2026-07-17 重新诊断发现，旧探针使用的关节姿态本身会让腕部撞到底盘，因而
无盒子基线永远 invalid，并被旧脚本误诊成"碰撞几何没加载"。当前探针改用已
真机示教的安全垂下姿态，并严格验证 valid → invalid → valid 三段状态。

本脚本用两个探针给出确定结论，不依赖机械臂 SDK、不需要任何姿态先验：

  探针A（世界碰撞）：往场景里放一个把整台机器人都罩住的巨型盒子，再查一个
    普通关节状态。碰撞几何正常的话，机器人连杆必然和盒子相交 → invalid。
    若仍报 valid → 机器人碰撞几何没加载，这就是根因。
  探针B（基线）：撤掉盒子再查同一状态 → 应为 valid。

脚本会复用已有 move_group；若服务不存在，会自动启动并在结束时清理自己的
只规划 move_group。只需先加载 ROS 环境：

  source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
  python3 shelf_dispenser/ros/collision_selftest.py

退出码 0 = 基线、世界碰撞拒绝、清场恢复全部正常；
退出码 2 = 自检不通过，输出会区分基线无效、漏碰撞、场景清理失败。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rclpy
from moveit_msgs.msg import (
    CollisionObject,
    PlanningScene,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetStateValidity
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

from shelf_dispenser.collision import classify_moveit_collision_probe

PLANNING_FRAME = "platform_base_link"
GROUP = "right_arm"
RIGHT_JOINTS = [f"r_joint{i}" for i in range(1, 8)]
LEFT_JOINTS = [f"l_joint{i}" for i in range(1, 8)]
# 2026-07-15 真机示教并记录在 table_demo.home_joints_deg 的安全垂下姿态。
# 旧探针 [0,90,0,90,0,0,0] 会让腕部与底盘/车身相交，导致无盒子基线
# 永远 invalid，进而把正常工作的世界碰撞检测误诊成“碰撞几何没加载”。
PROBE_RIGHT_DEG = [
    -10.368000030517578,
    121.64600372314453,
    55.37200164794922,
    7.206999778747559,
    22.586999893188477,
    5.623000144958496,
    -147.39100646972656,
]


def _wait(client, timeout=15.0):
    if not client.wait_for_service(timeout_sec=timeout):
        raise RuntimeError(f"服务不可用: {client.srv_name}")


def _ensure_moveit(validity_client):
    """Start a private plan-only move_group only when none is available."""
    if validity_client.wait_for_service(timeout_sec=1.0):
        return None
    log_path = Path("/tmp/bottle_moveit_selftest.log")
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["python3", str(ROOT / "shelf_dispenser" / "ros" / "headless.py")],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(
                f"MoveIt 自检服务启动失败，查看 {log_path}"
            )
        if validity_client.wait_for_service(timeout_sec=1.0):
            return process, log_handle
    os.killpg(process.pid, signal.SIGTERM)
    log_handle.close()
    raise RuntimeError(f"MoveIt 自检服务启动超时，查看 {log_path}")


def _stop_owned_moveit(owned):
    if owned is None:
        return
    process, log_handle = owned
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
    log_handle.close()


def _probe_state(node, validity_client):
    from math import radians

    request = GetStateValidity.Request()
    request.group_name = GROUP
    state = RobotState()
    state.is_diff = True
    state.joint_state = JointState()
    state.joint_state.name = [*LEFT_JOINTS, *RIGHT_JOINTS]
    state.joint_state.position = [0.0] * 7 + [radians(v) for v in PROBE_RIGHT_DEG]
    request.robot_state = state
    future = validity_client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10)
    response = future.result()
    if response is None:
        raise RuntimeError("check_state_validity 超时")
    return {
        "valid": bool(response.valid),
        "contacts": [
            [item.contact_body_1, item.contact_body_2]
            for item in response.contacts[:12]
        ],
    }


def _set_giant_box(
    node, apply_client, present: bool, *, allow_missing: bool = False
):
    scene = PlanningScene()
    scene.is_diff = True
    collision = CollisionObject()
    collision.header.frame_id = PLANNING_FRAME
    collision.id = "collision_selftest_box"
    if present:
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [4.0, 4.0, 4.0]  # 把整台机器人罩住
        pose = Pose()
        pose.orientation.w = 1.0
        collision.primitives = [primitive]
        collision.primitive_poses = [pose]
        collision.operation = CollisionObject.ADD
    else:
        collision.operation = CollisionObject.REMOVE
    scene.world.collision_objects.append(collision)
    request = ApplyPlanningScene.Request()
    request.scene = scene
    future = apply_client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=15)
    if future.result() is None:
        raise RuntimeError("apply_planning_scene 失败")
    if not future.result().success and not (allow_missing and not present):
        raise RuntimeError("apply_planning_scene 失败")


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("moveit_collision_selftest")
    apply_client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    validity_client = node.create_client(GetStateValidity, "/check_state_validity")
    owned_moveit = None
    try:
        owned_moveit = _ensure_moveit(validity_client)
        _wait(apply_client)
        _wait(validity_client)

        # A previous interrupted self-test must not poison the new baseline.
        _set_giant_box(node, apply_client, present=False, allow_missing=True)
        baseline = _probe_state(node, validity_client)
        _set_giant_box(node, apply_client, present=True)
        boxed = _probe_state(node, validity_client)
        _set_giant_box(node, apply_client, present=False)
        cleared = _probe_state(node, validity_client)

        print(f"探针B 基线（无盒子）: {baseline}  期望 valid=True")
        print(f"探针A 巨型盒子罩住整臂: {boxed}  期望 valid=False")
        print(f"撤盒子后恢复: {cleared}  期望 valid=True")

        status, reason = classify_moveit_collision_probe(
            baseline_valid=baseline["valid"],
            boxed_valid=boxed["valid"],
            cleared_valid=cleared["valid"],
        )
        if status == "healthy":
            print("\n结论：MoveIt 碰撞检查工作正常。")
            return 0
        print(f"\n结论：MoveIt 碰撞自检未通过 [{status}]：{reason}。")
        print("在诊断通过前，禁止把 MoveIt 的碰撞结论当作执行放行条件。")
        return 2
    finally:
        _stop_owned_moveit(owned_moveit)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
