"""
RealSense 相机线程（替换原 Orbbec Gemini 版本）
依赖 pyrealsense2；对外接口与旧版完全一致，上层代码无需修改。
"""

import json
import os
import threading
import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


DEFAULT_SHARED_DIR = "/tmp/grabber_camera_frames"
DEFAULT_SHARED_FPS = 10.0
SERIAL_SHARED_NAMES = {
    "153122071777": "head",
    "405622073249": "right_wrist",
    "335522072194": "left_wrist",
}


class DisplayMode(Enum):
    COLOR = "color"
    DEPTH = "depth"
    COLOR_DEPTH = "color_depth"
    COLOR_VISION = "color_vision"
    ALL = "all"
    NONE = "none"


class CameraThread(threading.Thread):
    """
    后台 RealSense 相机线程。
    通过序列号指定相机；默认为双臂机器人头部相机 153122071777。
    """

    def __init__(
        self,
        serial: str = "153122071777",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        strict_serial: bool = False,
        shared_name: Optional[str] = None,
        shared_dir: Optional[str] = None,
        shared_fps: float = DEFAULT_SHARED_FPS,
    ):
        super().__init__()
        self.daemon = True
        self.stop_event = threading.Event()

        self._serial = serial
        self._width = width
        self._height = height
        self._fps = fps
        self._strict_serial = strict_serial
        self._shared_name = (
            shared_name
            or os.environ.get("CAMERA_SHARED_NAME")
            or SERIAL_SHARED_NAMES.get(serial)
        )
        self._shared_dir = shared_dir or os.environ.get("CAMERA_SHARE_DIR", DEFAULT_SHARED_DIR)
        self._shared_interval = 1.0 / max(0.1, float(shared_fps))
        self._last_shared_publish = 0.0

        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._depth_scale_m: Optional[float] = None
        self.color_intrinsics: Optional[rs.intrinsics] = None

        self._frame_lock = threading.Lock()
        self._latest_color: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._frame_timestamp: float = 0.0

        self._display_thread: Optional[threading.Thread] = None
        self._display_mode = DisplayMode.NONE
        self._vision_analyzer = None
        self.initialization_error: Optional[str] = None

        self.initialization_successful = self._initialize_camera()
        if not self.initialization_successful:
            print("[CameraThread] CRITICAL: Camera initialization failed.")

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _initialize_camera(self) -> bool:
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
            if not serials:
                self.initialization_error = "未检测到任何 RealSense 设备"
                print("[CameraThread] CRITICAL: 未检测到任何 RealSense 设备")
                return False

            if self._serial not in serials:
                if self._strict_serial:
                    self.initialization_error = (
                        f"指定序列号 {self._serial} 未找到；可用设备: {serials}"
                    )
                    print(
                        f"[CameraThread] CRITICAL: 指定序列号 {self._serial} 未找到，"
                        f"可用设备: {serials}。严格模式下拒绝切换到其他相机。"
                    )
                    return False
                print(
                    f"[CameraThread] 序列号 {self._serial} 未找到，"
                    f"可用设备: {serials}，使用第一个"
                )
                self._serial = serials[0]

            cfg = rs.config()
            cfg.enable_device(self._serial)
            cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
            cfg.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)

            self._pipeline = rs.pipeline()
            profile = self._pipeline.start(cfg)
            self._depth_scale_m = float(
                profile.get_device().first_depth_sensor().get_depth_scale()
            )
            if not np.isfinite(self._depth_scale_m) or self._depth_scale_m <= 0:
                raise RuntimeError(
                    f"RealSense 返回无效 depth_scale={self._depth_scale_m}"
                )

            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.color_intrinsics = color_stream.get_intrinsics()
            self._align = rs.align(rs.stream.color)

            print(
                f"[CameraThread] RealSense ({self._serial}) 初始化成功 "
                f"{self._width}x{self._height}@{self._fps}fps "
                f"depth_scale={self._depth_scale_m:.9f}m/unit"
            )
            print(
                f"[CameraThread] 内参: fx={self.color_intrinsics.fx:.1f} "
                f"fy={self.color_intrinsics.fy:.1f} "
                f"cx={self.color_intrinsics.ppx:.1f} "
                f"cy={self.color_intrinsics.ppy:.1f}"
            )
            return True

        except Exception as e:
            self.initialization_error = str(e)
            print(f"[CameraThread] 初始化失败: {e}")
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
            import traceback
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # 线程主循环
    # ------------------------------------------------------------------

    def run(self) -> None:
        if not self.initialization_successful:
            return

        while not self.stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                if not frames:
                    continue

                aligned = self._align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())       # BGR uint8
                depth_data = np.asanyarray(depth_frame.get_data())
                depth_meters = (
                    depth_data.astype(np.float32) * self._depth_scale_m
                )

                with self._frame_lock:
                    self._latest_color = color_image.copy()
                    self._latest_depth = depth_meters.copy()
                    self._frame_timestamp = time.time()
                self._publish_shared_frame(color_image)

            except Exception as e:
                print(f"[CameraThread] 主循环错误: {e}")
                time.sleep(1)

    # ------------------------------------------------------------------
    # 对外接口（与旧版完全一致）
    # ------------------------------------------------------------------

    def get_latest_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """返回最新的 (color_bgr, depth_meters)。"""
        with self._frame_lock:
            color = self._latest_color.copy() if self._latest_color is not None else None
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
        return color, depth

    def get_frame_timestamp(self) -> float:
        with self._frame_lock:
            return self._frame_timestamp

    def get_camera_intrinsics(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """返回 (K 3×3, dist 5) 或 (None, None)。"""
        if not self.initialization_successful or self.color_intrinsics is None:
            return None, None
        intr = self.color_intrinsics
        K = np.array([
            [intr.fx,    0,   intr.ppx],
            [0,       intr.fy, intr.ppy],
            [0,          0,      1     ],
        ], dtype=np.float64)
        dist = np.array(intr.coeffs, dtype=np.float64)
        return K, dist

    def _publish_shared_frame(self, color_image: np.ndarray) -> None:
        if not self._shared_name:
            return
        now = time.time()
        if now - self._last_shared_publish < self._shared_interval:
            return
        self._last_shared_publish = now
        try:
            os.makedirs(self._shared_dir, exist_ok=True)
            jpg_path = os.path.join(self._shared_dir, f"{self._shared_name}.jpg")
            json_path = os.path.join(self._shared_dir, f"{self._shared_name}.json")
            ok, encoded = cv2.imencode(
                ".jpg", color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if not ok:
                return
            tmp_jpg = f"{jpg_path}.tmp"
            with open(tmp_jpg, "wb") as fh:
                fh.write(encoded.tobytes())
            os.replace(tmp_jpg, jpg_path)

            meta = {
                "camera": self._shared_name,
                "serial": self._serial,
                "timestamp": now,
                "width": int(color_image.shape[1]),
                "height": int(color_image.shape[0]),
                "fps": self._fps,
            }
            tmp_json = f"{json_path}.tmp"
            with open(tmp_json, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False)
            os.replace(tmp_json, json_path)
        except Exception as exc:
            print(f"[CameraThread] 发布共享帧失败: {exc}")

    def generate_pointcloud(self, max_depth: float = 3.0) -> Optional[np.ndarray]:
        """返回 [N, 6] 点云数组 [x, y, z, r, g, b]。"""
        if not self.initialization_successful or self.color_intrinsics is None:
            return None
        color, depth = self.get_latest_frames()
        if color is None or depth is None:
            return None

        intr = self.color_intrinsics
        fx, fy = intr.fx, intr.fy
        cx, cy = intr.ppx, intr.ppy
        h, w = depth.shape

        v, u = np.mgrid[0:h:4, 0:w:4]
        v_flat, u_flat = v.flatten(), u.flatten()
        z_vals = depth[v_flat, u_flat]

        valid = (z_vals > 0) & (z_vals < max_depth)
        v_v, u_v, z_v = v_flat[valid], u_flat[valid], z_vals[valid]
        if len(z_v) == 0:
            return None

        x = (u_v - cx) * z_v / fx
        y = (v_v - cy) * z_v / fy
        colors_rgb = color[v_v, u_v][:, [2, 1, 0]]

        return np.column_stack([x, y, z_v, colors_rgb])

    def save_pointcloud_ply(self, filepath: str, max_depth: float = 3.0) -> bool:
        pc = self.generate_pointcloud(max_depth)
        if pc is None:
            return False
        try:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(pc)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write("end_header\n")
                for pt in pc:
                    f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                            f"{int(pt[3])} {int(pt[4])} {int(pt[5])}\n")
            print(f"[CameraThread] 点云已保存 {filepath} ({len(pc)} 点)")
            return True
        except Exception as e:
            print(f"[CameraThread] 保存 PLY 失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 显示
    # ------------------------------------------------------------------

    def start_display(self, mode: DisplayMode = DisplayMode.COLOR, vision_analyzer=None):
        if self._display_thread and self._display_thread.is_alive():
            print("[CameraThread] Display already running")
            return
        self._display_mode = mode
        self._vision_analyzer = vision_analyzer
        self._display_thread = threading.Thread(target=self._display_worker, daemon=True)
        self._display_thread.start()
        print(f"[CameraThread] Display started: {mode.value}")

    def stop_display(self):
        if self._display_thread:
            self._display_mode = DisplayMode.NONE
            if self._display_thread.is_alive():
                self._display_thread.join(timeout=2)
            self._display_thread = None
            cv2.destroyAllWindows()

    def is_display_running(self) -> bool:
        return self._display_thread is not None and self._display_thread.is_alive()

    def _display_worker(self):
        try:
            while self._display_mode != DisplayMode.NONE:
                color, depth = self.get_latest_frames()
                if color is None:
                    time.sleep(0.01)
                    continue
                if self._display_mode in (DisplayMode.COLOR, DisplayMode.COLOR_DEPTH, DisplayMode.ALL):
                    cv2.imshow("Color Feed", color)
                if depth is not None and self._display_mode in (
                    DisplayMode.DEPTH, DisplayMode.COLOR_DEPTH, DisplayMode.ALL
                ):
                    vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imshow("Depth Feed", cv2.applyColorMap(vis, cv2.COLORMAP_JET))
                if self._display_mode in (DisplayMode.COLOR_VISION, DisplayMode.ALL) and self._vision_analyzer:
                    try:
                        dets = self._vision_analyzer.analyze_image(color, depth)
                        annotated = self._vision_analyzer.draw_detections(color, dets)
                        cv2.imshow("Color + Vision", annotated)
                    except Exception:
                        cv2.imshow("Color + Vision", color)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            print(f"[CameraThread] Display error: {e}")
        finally:
            self._display_mode = DisplayMode.NONE
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # 停止
    # ------------------------------------------------------------------

    def stop(self):
        self.stop_display()
        self.stop_event.set()

    def join(self, timeout=None):
        if self._pipeline and self.initialization_successful:
            try:
                self._pipeline.stop()
                print("[CameraThread] Pipeline stopped")
            except Exception as e:
                print(f"[CameraThread] 停止 pipeline 失败: {e}")
        super().join(timeout)

    def get_export_info(self) -> Dict[str, Any]:
        return {
            "camera_ready": self.initialization_successful,
            "has_frames": self._latest_color is not None,
            "frame_timestamp": self.get_frame_timestamp(),
            "serial": self._serial,
        }
