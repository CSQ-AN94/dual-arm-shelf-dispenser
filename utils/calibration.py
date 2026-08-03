# utils/calibration.py

import numpy as np
import cv2

class Calibration:
    """
    坐标变换核心类 - 简洁版本
    
    职责：像素坐标 → 基座坐标系3D点
    """
    def __init__(self, T_end_to_camera, camera_matrix, distortion_coeffs):
        """
        Args:
            T_end_to_camera (np.ndarray): 腕部相机到当前手臂末端的4x4外参
            camera_matrix (np.ndarray): 相机内参矩阵 K
            distortion_coeffs (np.ndarray): 相机畸变系数
        """
        self.T_end_to_camera = np.asarray(T_end_to_camera, dtype=float)
        self.K = camera_matrix
        self.dist = distortion_coeffs
        if self.T_end_to_camera.shape != (4, 4):
            raise ValueError("T_end_to_camera must be a 4x4 matrix")
    
    def _pixel_to_camera_point(self, u, v, d):
        """像素坐标 + 深度 → 相机坐标系下的3D点变换矩阵（4x4，仅含平移）。"""
        if d <= 0:
            return None

        # 1. 像素去畸变
        undist = cv2.undistortPoints(
            np.array([[[u, v]]], dtype=np.float32),
            self.K, self.dist, P=self.K
        )[0, 0]

        # 2. 反投影到相机坐标系
        fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]
        cam_x = (undist[0] - cx) * d / fx
        cam_y = (undist[1] - cy) * d / fy
        cam_z = d

        T_cam_target = np.eye(4)
        T_cam_target[:3, 3] = [cam_x, cam_y, cam_z]
        return T_cam_target

    def get_point_base(self, u, v, d, T_end_base):
        """
        像素坐标转基座坐标 —— eye-in-hand 场景（相机装在末端上，跟手臂一起动，
        比如腕部相机 left_wrist_0/right_wrist_0）。

        Args:
            u (float): 像素坐标 x
            v (float): 像素坐标 y
            d (float): 深度值
            T_end_base (np.ndarray): end_effector到base的4x4变换矩阵

        Returns:
            np.ndarray: 基座坐标系中的3D点 [x, y, z]
        """
        T_cam_target = self._pixel_to_camera_point(u, v, d)
        if T_cam_target is None:
            return None

        # 坐标变换链：base ← end_effector ← cam ← target
        T_base_target = T_end_base @ self.T_end_to_camera @ T_cam_target
        return T_base_target[:3, 3]

    def get_point_base_fixed_camera(self, u, v, d, T_base_to_camera):
        """
        像素坐标转基座坐标 —— eye-to-hand 场景（相机固定不跟手臂动，比如头部相机
        base_0；T_base_to_camera 是标定得到的固定矩阵，前提是标定后头部俯仰/旋转
        舵机没有再被移动过，否则这个矩阵已经失效）。

        Args:
            u, v (float): 像素坐标
            d (float): 深度值（米）
            T_base_to_camera (np.ndarray): 标定得到的相机在基座坐标系下的4x4位姿矩阵

        Returns:
            np.ndarray: 基座坐标系中的3D点 [x, y, z]
        """
        T_cam_target = self._pixel_to_camera_point(u, v, d)
        if T_cam_target is None:
            return None

        # 坐标变换链：base ← cam（固定）← target，不经过末端
        T_base_target = T_base_to_camera @ T_cam_target
        return T_base_target[:3, 3]
