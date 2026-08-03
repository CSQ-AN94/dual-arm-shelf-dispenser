#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯运动学交叉验证左右臂基座坐标系变换 —— 不涉及相机/棋盘格，独立于视觉标定。

原理：让两条手臂的末端依次触碰同一批真实世界里的物理点（几个螺丝钉尖、桌面
上的记号都行），每碰一个点就用只读连接分别记录两条手臂各自汇报的末端位置。
只要末端位置对上了，两条手臂摆什么关节姿态去够到这个点完全不重要。

采集到 >=3 个不共线的点后，用 Kabsch 算法对两组点做刚体配准，直接解出
T_base_right_to_base_left（不依赖视觉标定的任何假设），跟
scripts/compose_dual_arm_transform.py 算出来的结果对比，看是否一致。
如果只有1-2个点，只能做粗略的平移量级检查，给不出旋转。

只读连接，不经过 ArmController，不碰遥操/头部舵机，可以放心跟拖动示教
（长按末端绿色按钮）同时使用。

用法:
  python3 scripts/verify_dual_arm_by_touch.py
  跟着提示，交替把右臂/左臂末端移到同一个物理点，回车确认，至少3组后输入 done。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.config import load_config

try:
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
except ImportError:
    print("[FATAL] pip install robotic-arm")
    sys.exit(1)

def read_position(ip: str, port: int, label: str):
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(ip, port)
    if handle.id == -1:
        print(f"[FATAL] 连接 {label}({ip}) 失败")
        return None
    try:
        code, state = arm.rm_get_current_arm_state()
        if code != 0:
            print(f"[FATAL] 读 {label} 状态失败，错误码 {code}")
            return None
        pose = state.get("pose")
        return np.array(pose[:3], dtype=float)
    finally:
        arm.rm_delete_robot_arm()


def kabsch(P_left: np.ndarray, P_right: np.ndarray):
    """求刚体变换 R,t 使得 R @ P_left_i + t ≈ P_right_i（最小二乘）。"""
    left_mean = P_left.mean(axis=0)
    right_mean = P_right.mean(axis=0)
    left_c = P_left - left_mean
    right_c = P_right - right_mean

    H = left_c.T @ right_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = right_mean - R @ left_mean
    return R, t


def rotation_angle_deg(R_a, R_b) -> float:
    cos_angle = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def main():
    config = load_config()
    right_ip = config.connections.right_arm_ip
    left_ip = config.connections.left_arm_ip
    port = config.connections.arm_port
    vision_transform = config.calibration.T_base_right_to_base_left

    right_points = []
    left_points = []

    print("=== 纯运动学交叉验证（不涉及相机/棋盘格）===")
    print("交替把右臂、左臂末端移到同一个物理点（螺丝钉尖/桌面记号都行），")
    print("关节姿态不用管，只要末端碰到的是同一个点就行。至少采集3组不共线的点。\n")

    idx = 1
    while True:
        raw = input(f"[点 {idx}] 按 Enter 开始这一组（或输入 done 结束）: ").strip().lower()
        if raw == "done":
            break

        input(f"  把右臂末端移到点 {idx}，按 Enter 记录右臂位置...")
        p_right = read_position(right_ip, port, "右臂")
        if p_right is None:
            continue
        print(f"  右臂位置: {np.round(p_right, 4)}")

        input(f"  把左臂末端移到同一个点 {idx}，按 Enter 记录左臂位置...")
        p_left = read_position(left_ip, port, "左臂")
        if p_left is None:
            continue
        print(f"  左臂位置: {np.round(p_left, 4)}")

        right_points.append(p_right)
        left_points.append(p_left)
        idx += 1

    n = len(right_points)
    if n == 0:
        print("没有采集到任何点，退出。")
        return

    P_right = np.array(right_points)
    P_left = np.array(left_points)

    print(f"\n共采集 {n} 组点。")

    if n < 3:
        print("点数不够3个，只能做粗略平移量级检查，给不出旋转对比：")
        for i in range(n):
            print(f"  点{i+1}: 右臂={np.round(P_right[i],4)}  左臂={np.round(P_left[i],4)}  "
                  f"直线距离={np.linalg.norm(P_right[i]-P_left[i])*1000:.1f}mm")
        print("\n(这个距离不是两臂基座间距，只是两个读数的欧氏距离，量级用来感觉一下用)")
        return

    R_touch, t_touch = kabsch(P_left, P_right)
    T_touch = np.eye(4)
    T_touch[:3, :3] = R_touch
    T_touch[:3, 3] = t_touch

    # 拟合残差：R@P_left+t 应该约等于 P_right
    residuals = (R_touch @ P_left.T).T + t_touch - P_right
    rms_mm = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))) * 1000

    print("\n" + "=" * 60)
    print("纯运动学解出的 T_base_right_to_base_left：")
    print("=" * 60)
    for row in np.round(T_touch, 6).tolist():
        print(f"  - {row}")
    print(f"\n拟合RMS残差: {rms_mm:.1f}mm （越小说明这几个点触碰得越准，建议<20mm）")
    print(f"平移量: |t| = {np.linalg.norm(t_touch):.4f} m")

    print("\n" + "=" * 60)
    print("跟视觉标定结果（compose_dual_arm_transform.py）对比：")
    print("=" * 60)
    t_diff_mm = np.linalg.norm(t_touch - vision_transform[:3, 3]) * 1000
    r_diff_deg = rotation_angle_deg(R_touch, vision_transform[:3, :3])
    print(f"视觉标定平移: |t|={np.linalg.norm(vision_transform[:3,3]):.4f} m")
    print(f"运动学验证平移: |t|={np.linalg.norm(t_touch):.4f} m")
    print(f"两者平移差异: {t_diff_mm:.1f}mm")
    print(f"两者旋转差异: {r_diff_deg:.2f}deg")
    if t_diff_mm < 30 and r_diff_deg < 5:
        print("=> 一致，视觉标定结果可信。")
    else:
        print("=> 差异较大，视觉标定结果存疑，建议重新标定或检查标定板尺寸参数。")


if __name__ == "__main__":
    main()
