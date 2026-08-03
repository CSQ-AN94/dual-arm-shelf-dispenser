#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把头部相机对右臂、对左臂两轮 eye-to-hand 标定的结果组合起来，
解出左右臂基座坐标系之间的固定变换，从而把两条手臂的坐标系打通。

前提：
  T_BASE_RIGHT_TO_CAMERA_HEAD 来自 run_eye_to_hand_calibration.py（右臂那轮）的输出
  T_BASE_LEFT_TO_CAMERA_HEAD  来自 run_eye_to_hand_calibration_left.py（左臂那轮）的输出
  这两轮标定必须是头部舵机角度完全没变过的情况下做的，否则这里组合出来的结果是错的。

推导：
  两轮分别满足 P_base_right = T_base_right_to_camera_head @ P_cam_head
              P_base_left  = T_base_left_to_camera_head  @ P_cam_head
  对同一个物理点，两边算出的 P_cam_head 相同，消去后得到：
    P_base_right = (T_base_right_to_camera_head @ inv(T_base_left_to_camera_head)) @ P_base_left
  即 T_base_right_to_base_left = T_base_right_to_camera_head @ inv(T_base_left_to_camera_head)

用法:
  python3 scripts/compose_dual_arm_transform.py
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config


def main():
    calibration = load_config().calibration
    T_base_right_to_base_left = (
        calibration.T_base_right_to_camera_head
        @ np.linalg.inv(calibration.T_base_left_to_camera_head)
    )
    T_base_left_to_base_right = np.linalg.inv(T_base_right_to_base_left)

    right_error = np.max(np.abs(
        T_base_right_to_base_left - calibration.T_base_right_to_base_left
    ))
    left_error = np.max(np.abs(
        T_base_left_to_base_right - calibration.T_base_left_to_base_right
    ))

    print("=" * 60)
    print("T_base_right_to_base_left（把左臂基座坐标系下的点转到右臂基座坐标系）：")
    print("=" * 60)
    for row in np.round(T_base_right_to_base_left, 6).tolist():
        print(f"  - {row}")

    print()
    print("=" * 60)
    print("T_base_left_to_base_right（反过来，把右臂坐标转到左臂坐标系，备用）：")
    print("=" * 60)
    for row in np.round(T_base_left_to_base_right, 6).tolist():
        print(f"  - {row}")

    print()
    print("平移量粗略检查（两条手臂物理安装间距，单位米，可以和实际卷尺测量对比一下量级是否合理）：")
    print(f"  |t| = {np.linalg.norm(T_base_right_to_base_left[:3, 3]):.4f} m")
    print(f"配置闭环误差 max: right<-left={right_error:.2e}, left<-right={left_error:.2e}")


if __name__ == "__main__":
    main()
