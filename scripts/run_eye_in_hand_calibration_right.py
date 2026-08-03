#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
右腕相机（right_wrist_0）eye-in-hand 手眼标定驱动脚本。

目标：解出 T_end_right_to_camera_rightwrist（右腕相机在右臂末端坐标系下的位姿）。
跟头部相机那两轮（run_eye_to_hand_calibration[_left].py）方法完全不同：这次
标定板要固定不动放在环境里（不是绑在夹爪上），移动右臂让右腕相机从不同角度看
固定的标定板。

前提条件（缺一不可）：
  1. ChArUco标定板牢固固定在桌面/支架上，标定过程中绝对不能挪动。
  2. WRIST_CAMERA_SERIAL 必须是已确认的右腕相机序列号；严格模式下找不到它会
     直接退出，不会误用另外一台RealSense。
  3. RIGHT_POSES_DEG 需要用 pose_reader.py 遥操右臂 + 看右腕相机画面采集，
     保证标定板在每个姿态下都能被右腕相机看到（近景，姿态范围跟头部相机那轮
     完全不同，不能沿用）。

用法:
  python3 scripts/run_eye_in_hand_calibration_right.py --poses right_wrist_poses.json
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

TARGET_ARM = "right"

# ChArUco板固定在环境中，重量不影响eye-in-hand标定。下面尺寸必须按实物核对：
# squares_x/y是方格数，square/marker length单位均为米。
BOARD_TYPE = "charuco"
# 实物板经右腕原图验证：横向12格、纵向9格（共54个 marker，ID 0-53）。
# 这里不能写成9x12；虽然marker仍能被检测到，但角点几何布局不匹配会导致
# PnP重投影误差约5.8px并被质量门禁拒绝。
CHARUCO_SQUARES_X = 12
CHARUCO_SQUARES_Y = 9
CHARUCO_SQUARE_LENGTH = 0.030
CHARUCO_MARKER_LENGTH = 0.0225
CHARUCO_DICT_ID = cv2.aruco.DICT_5X5_250
MIN_CHARUCO_CORNERS = 12
MAX_REPROJECTION_ERROR_PX = 2.0

# !! 占位值：必须用 pose_reader.py 现场采集，替换成右腕相机能看到标定板的右臂姿态 !!
RIGHT_POSES_DEG = [
    # [j1, j2, j3, j4, j5, j6, j7],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poses",
        default="right_wrist_poses.json",
        help="collect_calibration_poses.py生成的关节姿态JSON",
    )
    parser.add_argument("--output-dir", help="保存原图和标定位姿；默认按时间创建目录")
    args = parser.parse_args()
    app_config = load_config()
    wrist_camera_serial = app_config.camera.right_wrist_serial

    poses_deg = RIGHT_POSES_DEG
    if os.path.exists(args.poses):
        with open(args.poses, encoding="utf-8") as f:
            poses_deg = json.load(f)
        print(f"=== 从 {args.poses} 加载 {len(poses_deg)} 组右腕标定姿态 ===")

    if wrist_camera_serial.startswith("REPLACE_ME"):
        print("[FATAL] WRIST_CAMERA_SERIAL 还是占位值，先现场确认右腕相机真实序列号。")
        sys.exit(1)
    if len(poses_deg) < 5:
        print("[FATAL] 右腕标定姿态不够（至少5组，建议10-15组）。")
        print("        请先运行 collect_calibration_poses.py 的右腕ChArUco采集命令。")
        sys.exit(1)
    pose_array = np.asarray(poses_deg, dtype=float)
    if pose_array.ndim != 2 or pose_array.shape[1] != 7 or not np.all(np.isfinite(pose_array)):
        print("[FATAL] 姿态JSON必须是有限数值组成的 N×7 关节角数组（单位：度）")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(
        "outputs", "handeye", f"right_wrist_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    conn_config = dataclasses.replace(app_config.connections, active_arm=TARGET_ARM)

    print(f"=== 启动右腕相机 ({wrist_camera_serial}) ===")
    camera_thread = CameraThread(
        serial=wrist_camera_serial,
        width=app_config.camera.width,
        height=app_config.camera.height,
        fps=app_config.camera.fps,
        strict_serial=True,
    )
    if not camera_thread.initialization_successful:
        print("[FATAL] 指定的右腕相机初始化失败，拒绝继续标定")
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
        print("标定成功。这是 T_end_right_to_camera_rightwrist，抄进 config.yaml：")
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
