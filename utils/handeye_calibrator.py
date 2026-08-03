from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from controllers.arm_controller import ArmController
    from sensors.camera_thread import CameraThread

class HandEyeCalibrator:
    """
    一键式自动化手眼标定工具

    支持两种标定板：
    - "chessboard"（默认）：普通棋盘格，纯黑白方格，无marker。用
      cv2.findChessboardCorners + solvePnP 检测。不需要猜字典型号，但要求
      每次拍照必须完整看到所有内角点（不像ChArUco能容忍部分遮挡）。
    - "charuco"：棋盘格+ArUco标记组合，能容忍部分遮挡，但要额外确认ArUco字典型号。

    不管哪种板子，chessboard_square_length / chessboard_corners（或 charuco 对应
    的几个尺寸参数）必须跟实际打印/测量的板子完全一致，否则要么检测不到，要么
    解出来的位置整体按比例偏。
    """
    def __init__(self,
                 arm_controller: ArmController,
                 camera_thread: CameraThread,
                 board_type: str = "chessboard",
                 # --- 普通棋盘格参数（board_type="chessboard"时用）---
                 chessboard_corners: tuple = (6, 9),   # (内角点列数, 内角点行数) = (方格列数-1, 方格行数-1)，实测板子7x10格
                 chessboard_square_length: float = 0.024,  # 单个方格边长，米
                 # --- ChArUco参数（board_type="charuco"时用，保留兼容）---
                 squares_x: int = 9,
                 squares_y: int = 12,
                 square_length: float = 0.030,
                 marker_length: float = 0.0225,
                 aruco_dict_id: int = None,
                 min_charuco_corners: int = 12,
                 max_reprojection_error_px: Optional[float] = None):

        self.arm_controller = arm_controller
        self.camera_thread = camera_thread
        self.board_type = board_type
        self.camera_matrix = None
        self.dist_coeffs = None
        self.min_charuco_corners = min_charuco_corners
        self.max_reprojection_error_px = max_reprojection_error_px
        self.last_reprojection_error_px = None

        if board_type == "chessboard":
            self.chessboard_corners = tuple(chessboard_corners)
            cols, rows = self.chessboard_corners
            objp = np.zeros((cols * rows, 3), dtype=np.float32)
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * chessboard_square_length
            self.chessboard_objp = objp
        elif board_type == "charuco":
            if aruco_dict_id is None:
                aruco_dict_id = cv2.aruco.DICT_5X5_250
            # 同时兼容机器人上的 OpenCV 4.5.4 旧 API 和开发机上的新版 API。
            if hasattr(cv2.aruco, "Dictionary_get"):
                self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)
            else:
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
            if hasattr(cv2.aruco, "DetectorParameters_create"):
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            else:
                self.aruco_params = cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, "CharucoBoard_create"):
                self.board = cv2.aruco.CharucoBoard_create(
                    squares_x, squares_y, square_length, marker_length, self.aruco_dict,
                )
            else:
                self.board = cv2.aruco.CharucoBoard(
                    (squares_x, squares_y), square_length, marker_length, self.aruco_dict,
                )
            # 机器人使用 OpenCV 4.5.4；新版 OpenCV 对偶数行 ChArUco 的生成规则
            # 有过不兼容变更。开发机验证时强制沿用旧规则，与现有实物板保持一致。
            if hasattr(self.board, "setLegacyPattern"):
                self.board.setLegacyPattern(True)
            self.charuco_detector = None
            if hasattr(cv2.aruco, "CharucoDetector"):
                self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        else:
            raise ValueError(f"未知 board_type: {board_type!r}，只能是 'chessboard' 或 'charuco'")

    def _find_pattern_in_image(self, image):
        """在图像中定位标定板，返回标定板在相机坐标系下的位姿（target2cam）。"""
        if self.board_type == "chessboard":
            return self._find_chessboard(image)
        return self._find_charuco(image)

    def _reprojection_is_acceptable(self, object_points, image_points, rvec, tvec) -> bool:
        """计算当前 PnP 的 RMS 重投影误差，并按配置拒绝明显错误的单帧位姿。"""
        projected, _ = cv2.projectPoints(
            np.asarray(object_points, dtype=np.float32),
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        observed = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        projected = projected.reshape(-1, 2)
        error_px = float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1))))
        self.last_reprojection_error_px = error_px
        if self.max_reprojection_error_px is not None and error_px > self.max_reprojection_error_px:
            print(
                f"Pattern rejected: reprojection RMS={error_px:.2f}px > "
                f"{self.max_reprojection_error_px:.2f}px"
            )
            return False
        return True

    def _find_chessboard(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # 优先用更鲁棒的 findChessboardCornersSB（新算法，内部自带高精度角点定位，
        # 实测对光照/清晰度不理想的真实照片明显比经典 findChessboardCorners 更容易
        # 成功检测）。找不到该函数（老版本OpenCV）时回退到经典方法+cornerSubPix。
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                gray, self.chessboard_corners,
                flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY,
            )
        else:
            found, corners = cv2.findChessboardCorners(
                gray, self.chessboard_corners,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if not found:
            return None

        ok, rvec, tvec = cv2.solvePnP(
            self.chessboard_objp, corners, self.camera_matrix, self.dist_coeffs,
        )
        if not ok:
            return None
        if not self._reprojection_is_acceptable(self.chessboard_objp, corners, rvec, tvec):
            return None

        R, _ = cv2.Rodrigues(rvec)
        T_target_to_cam = np.eye(4)
        T_target_to_cam[:3, :3] = R
        T_target_to_cam[:3, 3] = tvec.flatten()
        return T_target_to_cam

    def _find_charuco(self, image):
        if hasattr(cv2.aruco, "detectMarkers"):
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                image, self.aruco_dict, parameters=self.aruco_params
            )
            if marker_ids is None or len(marker_ids) < 4:
                return None
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                image,
                self.board,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
            )
        else:
            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                self.charuco_detector.detectBoard(image)
            )
            if marker_ids is None or len(marker_ids) < 4:
                return None

        num_corners = 0 if charuco_corners is None else len(charuco_corners)
        if num_corners < self.min_charuco_corners or charuco_ids is None:
            print(
                f"ChArUco rejected: only {int(num_corners)} corners, "
                f"need at least {self.min_charuco_corners}"
            )
            return None

        if hasattr(self.board, "chessboardCorners"):
            board_corners = np.asarray(self.board.chessboardCorners)
        else:
            board_corners = np.asarray(self.board.getChessboardCorners())
        object_points = board_corners[np.asarray(charuco_ids).reshape(-1)]
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            charuco_corners,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        if not self._reprojection_is_acceptable(
            object_points, charuco_corners, rvec, tvec
        ):
            return None

        R, _ = cv2.Rodrigues(rvec)
        T_target_to_cam = np.eye(4)
        T_target_to_cam[:3, :3] = R
        T_target_to_cam[:3, 3] = tvec.flatten()
        return T_target_to_cam

    @staticmethod
    def _is_valid_rigid_transform(T, atol: float = 1e-3) -> bool:
        """检查矩阵是否属于 SE(3)，避免把反射矩阵或 NaN 当成标定结果。"""
        if T is None:
            return False
        T = np.asarray(T)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            return False
        if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
            return False
        R = T[:3, :3]
        return (
            np.allclose(R.T @ R, np.eye(3), atol=atol)
            and abs(float(np.linalg.det(R)) - 1.0) <= atol
        )

    @classmethod
    def _solve_handeye(cls, gripper_to_base_transforms, target_to_cam_transforms):
        """调用 OpenCV PARK 解算并拒绝不是合法刚体变换的输出。"""
        try:
            R, t = cv2.calibrateHandEye(
                [T[:3, :3] for T in gripper_to_base_transforms],
                [T[:3, 3] for T in gripper_to_base_transforms],
                [T[:3, :3] for T in target_to_cam_transforms],
                [T[:3, 3] for T in target_to_cam_transforms],
                method=cv2.CALIB_HAND_EYE_PARK,
            )
        except cv2.error as exc:
            print(f"Hand-eye solver failed: {exc}")
            return None

        result = np.eye(4)
        result[:3, :3] = R
        result[:3, 3] = np.asarray(t).reshape(3)
        if not cls._is_valid_rigid_transform(result):
            det = float(np.linalg.det(result[:3, :3])) if np.all(np.isfinite(result)) else float("nan")
            print(
                "Hand-eye result rejected: output is not a valid SE(3) transform "
                f"(det(R)={det:.6f})."
            )
            return None
        return result

    @staticmethod
    def _rotation_distance_deg(R_a, R_b) -> float:
        cos_angle = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    @classmethod
    def _print_transform_consistency(cls, title, transforms) -> str:
        """用两两刚体距离检查理论上应保持不变的一组变换。"""
        translation_diffs = []
        rotation_diffs = []
        for i in range(len(transforms)):
            for j in range(i + 1, len(transforms)):
                translation_diffs.append(
                    float(np.linalg.norm(transforms[i][:3, 3] - transforms[j][:3, 3]))
                )
                rotation_diffs.append(
                    cls._rotation_distance_deg(transforms[i][:3, :3], transforms[j][:3, :3])
                )

        max_t = max(translation_diffs)
        mean_t = float(np.mean(translation_diffs))
        max_r = max(rotation_diffs)
        mean_r = float(np.mean(rotation_diffs))
        print(f"\n--- {title} ---")
        print(f"平移两两差异: max={max_t*1000:.1f}mm, mean={mean_t*1000:.1f}mm")
        print(f"旋转两两差异: max={max_r:.2f}deg, mean={mean_r:.2f}deg")
        if max_t < 0.01 and max_r < 2.0:
            print("=> 一致性良好（平移<1cm，旋转<2度），标定结果可信。")
            return "good"
        if max_t < 0.03 and max_r < 5.0:
            print("=> 一致性中等（平移<3cm，旋转<5度），仅建议低精度试验。")
            return "medium"
        print("=> 一致性较差，拒绝输出本次标定结果。")
        return "poor"
    
    def run_calibration_process(self, calibration_poses_deg, output_dir=None):
        """
        Eye-in-hand 标定：相机装在手腕上跟末端一起动（比如 left_wrist_0/right_wrist_0），
        标定板固定不动放在环境里，移动手臂让腕部相机从不同角度看到固定的标定板。

        calibration_poses_deg: 关节角度列表（度），需保证每个姿态下标定板都能被
        该手臂的腕部相机看到。建议先用 pose_reader.py 手动遥操/拖动示教采集。
        """
        print("--- Starting Eye-in-Hand Hand-Eye Calibration (camera on wrist) ---")

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Raw calibration data will be saved to: {output_dir}")

        self.camera_matrix, self.dist_coeffs = self.camera_thread.get_camera_intrinsics()
        if self.camera_matrix is None:
            print("Error: Could not get camera intrinsics. Aborting.")
            return None

        base_to_end_transforms = []
        target_to_cam_transforms = []
        valid_joint_poses = []
        for i, pose in enumerate(calibration_poses_deg):
            print(f"\nMoving to calibration pose {i+1}/{len(calibration_poses_deg)}...")
            move_result = self.arm_controller.move_to_joints(pose)
            if move_result != 0:
                print(f"Pose {i+1}: arm movement failed (code={move_result}). Aborting calibration.")
                return None
            time.sleep(3.5)  # 确保机械臂（带着腕部相机）完全静止再拍照

            color_image, _ = self.camera_thread.get_latest_frames()
            if color_image is None:
                print(f"Pose {i+1}: Could not get image. Skipping.")
                continue
            if output_dir is not None:
                cv2.imwrite(
                    os.path.join(output_dir, f"pose_{i+1:02d}.png"), color_image
                )
            T_base_to_end = self.arm_controller.get_base_to_end_pose_matrix()
            if not self._is_valid_rigid_transform(T_base_to_end):
                print(f"Pose {i+1}: invalid robot end pose matrix. Aborting calibration.")
                return None
            # _find_pattern_in_image 直接返回 PnP 解出的 target2cam（标定板在相机坐标系下的位姿）
            T_target_to_cam = self._find_pattern_in_image(color_image)

            if T_target_to_cam is not None:
                reproj = self.last_reprojection_error_px
                reproj_text = f", reprojection RMS={reproj:.2f}px" if reproj is not None else ""
                print(f"Pose {i+1}: Pattern found{reproj_text}\nT_target_to_cam:{T_target_to_cam}\n")
                base_to_end_transforms.append(T_base_to_end)
                target_to_cam_transforms.append(T_target_to_cam)
                valid_joint_poses.append(pose)
            else:
                print(f"Pose {i+1}: Pattern NOT found (标定板不在相机视野内或被遮挡). Skipping.")

        if output_dir is not None:
            sample_path = os.path.join(output_dir, "samples.npz")
            np.savez_compressed(
                sample_path,
                valid_joint_poses_deg=np.asarray(valid_joint_poses, dtype=float),
                base_to_end=np.asarray(base_to_end_transforms, dtype=float),
                target_to_cam=np.asarray(target_to_cam_transforms, dtype=float),
                camera_matrix=np.asarray(self.camera_matrix, dtype=float),
                distortion_coeffs=np.asarray(self.dist_coeffs, dtype=float),
            )
            print(f"Saved {len(base_to_end_transforms)} valid data pairs to {sample_path}")

        if len(base_to_end_transforms) < 5:
            print("Error: Not enough valid poses collected. Need at least 5. Aborting.")
            return None

        print(f"\nCollected {len(base_to_end_transforms)} valid data pairs. Solving eye-in-hand AX=XB...")

        # OpenCV 约定：第一组是 gripper->base，第二组是 target->camera，
        # 输出是 camera->gripper，即这里需要的 T_end_to_camera。
        T_end_to_camera = self._solve_handeye(
            base_to_end_transforms, target_to_cam_transforms
        )
        if T_end_to_camera is None:
            return None

        # Eye-in-hand 中标定板固定在环境里，因此每组数据反推得到的
        # base<-target 应是同一个常量：base<-end<-camera<-target。
        base_to_targets = [
            T_base_end @ T_end_to_camera @ T_target_cam
            for T_base_end, T_target_cam in zip(
                base_to_end_transforms, target_to_cam_transforms
            )
        ]
        quality = self._print_transform_consistency(
            "Eye-in-hand 自洽性验证（标定板在基座下的位姿应保持不变）",
            base_to_targets,
        )
        if quality == "poor":
            return None

        if output_dir is not None:
            result_path = os.path.join(output_dir, "T_end_to_camera.npy")
            np.save(result_path, T_end_to_camera)
            print(f"Saved calibration result to {result_path}")

        print("\nEye-in-Hand Calibration successful!")
        print("Resulting T_end_to_camera (4x4 Transformation Matrix):\n", T_end_to_camera)
        return T_end_to_camera

    def run_eye_to_hand_calibration(self, calibration_poses_deg, output_dir=None):
        """
        Eye-to-hand 标定：相机固定不动（例如头部相机，标定期间及标定后都不能再动头部
        俯仰/旋转舵机，否则本次标定结果失效），标定板固定装在夹爪/末端上，移动手臂让
        板子依次出现在相机视野的不同位置/角度。

        与 run_calibration_process()（eye-in-hand，相机装在手腕上跟末端一起动）的
        区别只在于：这里动的是标定板（跟着手臂），不动的是相机；数学上通过把
        T_base_to_end 换成其逆矩阵喂给 cv2.calibrateHandEye 来切换到 eye-to-hand 模式，
        解出来的是固定的 T_base_to_camera（相机在基座坐标系下的位姿），而不是 T_end_to_camera。

        calibration_poses_deg: 关节角度列表（度），需保证每个姿态下标定板都能被相机看到。
        建议先用 pose_reader.py 手动遥操，边看相机画面边记录合适的姿态。
        output_dir: 传了的话，每个姿态拍到的原图 + 最终的位姿数据都会存盘，方便
        标定失败后离线复查是哪几组姿态有问题，不用重新跑一遍机械臂。
        """
        print("--- Starting Eye-to-Hand Hand-Eye Calibration (static camera) ---")

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Raw calibration data will be saved to: {output_dir}")

        self.camera_matrix, self.dist_coeffs = self.camera_thread.get_camera_intrinsics()
        if self.camera_matrix is None:
            print("Error: Could not get camera intrinsics. Aborting.")
            return None

        base_to_end_transforms = []
        target_to_cam_transforms = []
        valid_joint_poses = []
        reprojection_errors = []
        for i, pose in enumerate(calibration_poses_deg):
            print(f"\nMoving to calibration pose {i+1}/{len(calibration_poses_deg)}...")
            move_result = self.arm_controller.move_to_joints(pose)
            if move_result != 0:
                print(f"Pose {i+1}: arm movement failed (code={move_result}). Aborting calibration.")
                return None
            time.sleep(3.5)  # 确保机械臂（带着标定板）完全静止再拍照

            color_image, _ = self.camera_thread.get_latest_frames()
            if color_image is None:
                print(f"Pose {i+1}: Could not get image. Skipping.")
                continue
            if output_dir is not None:
                cv2.imwrite(
                    os.path.join(output_dir, f"pose_{i+1:02d}.png"), color_image
                )
            T_base_to_end = self.arm_controller.get_base_to_end_pose_matrix()
            if not self._is_valid_rigid_transform(T_base_to_end):
                print(f"Pose {i+1}: invalid robot end pose matrix. Aborting calibration.")
                return None
            # _find_pattern_in_image 直接返回 PnP 解出的 target2cam（标定板在相机坐标系下的位姿）
            T_target_to_cam = self._find_pattern_in_image(color_image)

            if T_target_to_cam is not None:
                reproj = self.last_reprojection_error_px
                reproj_text = f", reprojection RMS={reproj:.2f}px" if reproj is not None else ""
                print(f"Pose {i+1}: Pattern found{reproj_text}\nT_target_to_cam:{T_target_to_cam}\n")
                base_to_end_transforms.append(T_base_to_end)
                target_to_cam_transforms.append(T_target_to_cam)
                valid_joint_poses.append(pose)
                reprojection_errors.append(reproj if reproj is not None else float("nan"))
            else:
                print(f"Pose {i+1}: Pattern NOT found (标定板不在相机视野内或被遮挡). Skipping.")

        if output_dir is not None:
            sample_path = os.path.join(output_dir, "samples.npz")
            np.savez_compressed(
                sample_path,
                valid_joint_poses_deg=np.asarray(valid_joint_poses, dtype=float),
                base_to_end=np.asarray(base_to_end_transforms, dtype=float),
                target_to_cam=np.asarray(target_to_cam_transforms, dtype=float),
                reprojection_errors_px=np.asarray(reprojection_errors, dtype=float),
                camera_matrix=np.asarray(self.camera_matrix, dtype=float),
                distortion_coeffs=np.asarray(self.dist_coeffs, dtype=float),
            )
            print(f"Saved {len(base_to_end_transforms)} valid data pairs to {sample_path}")

        if len(base_to_end_transforms) < 5:
            print("Error: Not enough valid poses collected. Need at least 5. Aborting.")
            return None

        print(f"\nCollected {len(base_to_end_transforms)} valid data pairs. Solving eye-to-hand AX=XB...")

        # eye-to-hand 技巧：把 base2gripper（T_base_to_end 的逆）喂进第一组参数位置，
        # target2cam 直接用 PnP 的测量值（不取逆）。输出的 R/t 此时代表 cam2base。
        base_to_gripper_transforms = [
            np.linalg.inv(T) for T in base_to_end_transforms
        ]

        # calibrateHandEye 内部固定把输出叫"cam2gripper"，但因为我们把 base2gripper
        # （而不是常规的 gripper2base）喂给了第一组参数，这个输出直接就是 cam2base，
        # 即 P_base = output @ P_cam —— 正好是我们要的 T_base_to_camera，不需要再取逆。
        # （之前这里多取了一次逆，是我自己重新推导时发现的bug，已修正。）
        T_base_to_camera = self._solve_handeye(
            base_to_gripper_transforms, target_to_cam_transforms
        )
        if T_base_to_camera is None:
            return None

        quality = self._print_consistency_check(
            T_base_to_camera, base_to_end_transforms, target_to_cam_transforms
        )
        if quality == "poor":
            return None

        if output_dir is not None:
            result_path = os.path.join(output_dir, "T_base_to_camera.npy")
            np.save(result_path, T_base_to_camera)
            print(f"Saved calibration result to {result_path}")

        print("\nEye-to-Hand Calibration successful!")
        print("Resulting T_base_to_camera (4x4 Transformation Matrix):\n", T_base_to_camera)
        return T_base_to_camera

    def _print_consistency_check(self, T_base_to_camera, base_to_end_transforms, target_to_cam_transforms):
        """自洽性验证：标定板刚性固定在末端上，所以对每一组姿态反推出的
        T_gripper_target（标定板相对末端的位姿）理论上应该是同一个常数，
        跟具体姿态无关。这里算出每组姿态反推的T_gripper_target，看它们
        彼此之间的平移/旋转分散程度——分散得越小，标定越可信；分散很大
        说明标定不准（哪怕每张图都成功检测到了棋盘格）。
        """
        gripper_targets = []
        for T_base_end, T_cam_target in zip(base_to_end_transforms, target_to_cam_transforms):
            T_gripper_target = np.linalg.inv(T_base_end) @ T_base_to_camera @ T_cam_target
            gripper_targets.append(T_gripper_target)

        return self._print_transform_consistency(
            "Eye-to-hand 自洽性验证（标定板相对末端的位姿应保持不变）",
            gripper_targets,
        )
