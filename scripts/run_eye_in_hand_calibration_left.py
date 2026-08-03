#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
左腕相机（left_wrist_0）eye-in-hand 手眼标定驱动脚本。

目标：解出 T_end_left_to_camera_leftwrist（左腕相机在左臂末端坐标系下的位姿）。
跟 run_eye_in_hand_calibration_right.py 是同一种方法（eye-in-hand，标定板固定
不动放环境里），可以直接复用同一个标定板摆放位置，只是这次移动左臂。

前提条件（缺一不可）：
  1. ChArUco标定板固定不动（可以沿用标右腕相机那一轮的摆放位置，不用重新放）。
  2. WRIST_CAMERA_SERIAL 换成左腕相机的真实序列号（跟右腕那轮是另一个序列号）。
  3. LEFT_POSES_DEG 需要用 pose_reader.py 遥操左臂 + 看左腕相机画面采集。

用法:
  python3 scripts/run_eye_in_hand_calibration_left.py --poses left_wrist_poses.json
"""

import sys
import os
import time
import dataclasses
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from utils.config import load_config
from utils.handeye_calibrator import HandEyeCalibrator
from sensors.camera_thread import CameraThread
from controllers.arm_controller import ArmController

TARGET_ARM = "left"

# ChArUco板固定在环境中，重量不影响eye-in-hand标定。下面尺寸必须按实物核对：
# squares_x/y是方格数，square/marker length单位均为米。
BOARD_TYPE = "charuco"
# 与右腕使用同一块实物板：横向12格、纵向9格（54个marker）。
CHARUCO_SQUARES_X = 12
CHARUCO_SQUARES_Y = 9
CHARUCO_SQUARE_LENGTH = 0.030
CHARUCO_MARKER_LENGTH = 0.0225
CHARUCO_DICT_ID = cv2.aruco.DICT_5X5_250
MIN_CHARUCO_CORNERS = 12
MAX_REPROJECTION_ERROR_PX = 2.0

# !! 占位值：必须用 pose_reader.py 现场采集，替换成左腕相机能看到标定板的左臂姿态 !!
LEFT_POSES_DEG = [
    # [j1, j2, j3, j4, j5, j6, j7],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poses",
        default="left_wrist_poses.json",
        help="collect_calibration_poses.py生成的关节姿态JSON",
    )
    parser.add_argument("--output-dir", help="保存原图和标定位姿；默认按时间创建目录")
    args = parser.parse_args()
    app_config = load_config()
    wrist_camera_serial = app_config.camera.left_wrist_serial

    poses_deg = LEFT_POSES_DEG
    if os.path.exists(args.poses):
        with open(args.poses, encoding="utf-8") as f:
            poses_deg = json.load(f)
        print(f"=== 从 {args.poses} 加载 {len(poses_deg)} 组左腕标定姿态 ===")

    if wrist_camera_serial.startswith("REPLACE_ME"):
        print("[FATAL] WRIST_CAMERA_SERIAL 还是占位值，先现场确认左腕相机真实序列号。")
        sys.exit(1)
    if len(poses_deg) < 5:
        print("[FATAL] 左腕标定姿态不够（至少5组，建议10-15组）。")
        print("        请先运行 collect_calibration_poses.py 的左腕ChArUco采集命令。")
        sys.exit(1)
    pose_array = np.asarray(poses_deg, dtype=float)
    if pose_array.ndim != 2 or pose_array.shape[1] != 7 or not np.all(np.isfinite(pose_array)):
        print("[FATAL] 姿态JSON必须是有限数值组成的 N×7 关节角数组（单位：度）")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(
        "outputs", "handeye", f"left_wrist_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    conn_config = dataclasses.replace(app_config.connections, active_arm=TARGET_ARM)

    print(f"=== 启动左腕相机 ({wrist_camera_serial}) ===")
    camera_thread = CameraThread(
        serial=wrist_camera_serial,
        width=app_config.camera.width,
        height=app_config.camera.height,
        fps=app_config.camera.fps,
        strict_serial=True,
    )
    if not camera_thread.initialization_successful:
        print("[FATAL] 指定的左腕相机初始化失败，拒绝继续标定")
        sys.exit(1)
    camera_thread.start()
    time.sleep(3)
    if camera_thread.get_latest_frames()[0] is None:
        print("[FATAL] 相机数据获取失败")
        camera_thread.stop()
        camera_thread.join(timeout=3)
        sys.exit(1)

    arm_controller = None
    try:
        print(f"=== 初始化机械臂（强制 {TARGET_ARM} 臂，会停遥操）===")
        arm_controller = ArmController(conn_config, app_config.arm, app_config.gripper)

        calibrator = HandEyeCalibrator(
            arm_controller, camera_thread,
            board_type=BOARD_TYPE,
            squares_x=CHARUCO_SQUARES_X,
            squares_y=CHARUCO_SQUARES_Y,
            square_length=CHARUCO_SQUARE_LENGTH,
            marker_length=CHARUCO_MARKER_LENGTH,
            aruco_dict_id=CHARUCO_DICT_ID,
            min_charuco_corners=MIN_CHARUCO_CORNERS,
            max_reprojection_error_px=MAX_REPROJECTION_ERROR_PX,
        )
        T_end_to_camera = calibrator.run_calibration_process(
            poses_deg, output_dir=output_dir
        )

        if T_end_to_camera is None:
            print(f"\n标定失败；原图和有效样本已保存在 {output_dir}，请查看上方具体质量门禁。")
            return

        print("\n" + "=" * 60)
        print("标定成功。这是 T_end_left_to_camera_leftwrist，抄进 config.yaml：")
        print("=" * 60)
        for row in np.round(T_end_to_camera, 6).tolist():
            print(f"  - {row}")

    finally:
        if arm_controller is not None:
            arm_controller.close()
        camera_thread.stop()
        camera_thread.join(timeout=3)


if __name__ == "__main__":
    main()
