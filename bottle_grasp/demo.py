"""Autonomous head-camera to wrist-camera bottle grasp state machine."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from uuid import uuid4

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera_access import (
    CameraAccessError,
    hardware_reset_camera,
    prepare_camera_access,
)
from . import console
from .collision import check_approach_corridor
from .core import (
    BottleDetectionLost,
    CameraFrameUnavailable,
    DemoParams,
    InsufficientDepth,
    Localization,
    SafetyAbort,
    interpolate_joint_path,
    interpolate_poses,
    look_at_camera_pose,
    matrix_pose,
    pose_matrix,
    stop_reason,
)
from .grasp_orientation import (
    resolve_tcp_grasp_rotation,
)
from .lift_evidence import LiftEvidenceKind, LiftVisualEvidence
from .dashboard import Dashboard, PreviewWorker, SharedState
from .delivery_table import (
    OutputTableObservation,
    observe_output_table,
    placement_still_valid,
)
from . import head_lock
from .mobile_body import (
    BodySnapshot,
    LiftSocketAdapter,
    MobileBodyCoordinator,
    ReturnAuthorization,
    WooshChassisAdapter,
)
from .model_assets import (
    ModelAssetContractError,
    inspect_model_asset_contract,
    require_model_asset_contract,
    verified_asset_path,
)
from .perception import BottleDetector, depth_point_for_detection
from .planner import MoveItPlanner
from .robot import ArmJointReader, RobotSession
from .run_manifest import manifest_profile_expectations
from .safe_planner import PlanTarget, SafeMotionPlanner, VerifiedPlan
from .safety import FenceBox, SafetyProfile, load_safety_profile
from .scene import (
    build_non_target_scene_voxels,
    build_scene_voxels,
    build_target_occupancy_voxels,
    conservative_scene_union,
    head_scene_points,
    union_scene_voxels,
    voxelize_scene_points,
)
from .shelf_model import (
    FACE_SPECS as SHELF_FACE_SPECS,
    adapt_profile_to_shelf,
    combine_shelf_fits,
    fit_shelf_face,
)
from .table_model import (
    TABLE_KEEPOUT_ID,
    adapt_profile_to_table,
    combine_table_fits,
    fit_table_top,
)
from .target_guard import GuardResult, LockedTargetGuard, ProjectedTargetAssociation
from .target_guard import PostLiftTargetAssociation

LOG = logging.getLogger("bottle_demo")

# How long a taught-pose move may take to settle before its arrival error is
# judged.  Bounded so an arm that never arrives still fails.
ARRIVAL_SETTLE_TIMEOUT_S = 2.0


class BottleDemo:
    def __init__(self, args, config):
        self.args = args
        self.cfg = config
        self.params = DemoParams()
        target_product = getattr(args, "target_product", None)
        if target_product:
            self.params = replace(
                self.params,
                target_product_classes=tuple(
                    item.strip()
                    for item in target_product.split(",")
                    if item.strip()
                ),
            )
        commissioning_speed = getattr(args, "commissioning_speed", None)
        if commissioning_speed is not None:
            if (
                not isinstance(commissioning_speed, int)
                or not 1 <= commissioning_speed <= 100
            ):
                raise SafetyAbort("commissioning 速度必须是 1-100 的整数")
            # Commissioning is an explicit, reversible cap.  Do not introduce
            # a second motion algorithm or change any waypoint; each existing
            # regime simply uses the lower requested percentage.
            self.params = replace(
                self.params,
                transit_speed=min(self.params.transit_speed, commissioning_speed),
                travel_speed=min(self.params.travel_speed, commissioning_speed),
                final_speed=min(self.params.final_speed, commissioning_speed),
                gripper_speed=min(self.params.gripper_speed, commissioning_speed),
            )
        self.stop_event = threading.Event()
        self.state = SharedState(self.stop_event)
        self.project_root = Path(args.config).resolve().parent
        run_name = (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            + f"_{uuid4().hex[:8]}"
        )
        self.run_dir = Path(args.output_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_handler = logging.FileHandler(self.run_dir / "run.log")
        # Evidence file: keeps DEBUG-level per-frame detail that the console
        # deliberately hides.
        self.run_log_handler.setLevel(logging.DEBUG)
        self.run_log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logging.getLogger().addHandler(self.run_log_handler)
        self.camera: Optional[Any] = None
        self.camera_name = ""
        self.robot: Optional[RobotSession] = None
        self.left_robot: Optional[ArmJointReader] = None
        self.planner: Optional[MoveItPlanner] = None
        self.detector: Optional[BottleDetector] = None
        self.wrist_detector: Optional[BottleDetector] = None
        self.model_asset_contract: Optional[dict] = None
        self.dashboard: Optional[Dashboard] = None
        self.preview: Optional[PreviewWorker] = None
        self.safety: Optional[SafetyProfile] = None
        self.source_safety: Optional[SafetyProfile] = None
        self.delivery_safety: Optional[SafetyProfile] = None
        # These distinguish profiles loaded through the manifest-checked
        # configuration path from arbitrary attributes an embedding might set.
        self._source_safety_profile_loaded = False
        self._delivery_safety_profile_loaded = False
        self.mobile_body: Optional[MobileBodyCoordinator] = None
        self.shelf_ready_body_snapshot: Optional[BodySnapshot] = None
        self.held_object_guard: Optional[dict] = None
        self.head_scene_voxels: list[list[float]] = []
        self.non_target_scene_voxels: list[list[float]] = []
        self.target_occupancy_voxels: list[list[float]] = []
        self.scene_voxels: list[list[float]] = []
        self.scene_boxes: list[dict] = []
        self.head_scene_captured_monotonic: Optional[float] = None
        self._base_pose_for_scene = np.eye(4)
        self._head_scene_base_pose = np.eye(4)
        self.grasp_rotation: Optional[np.ndarray] = None
        # Set by the entry point when console presentation is installed; the
        # workflow must stay runnable (tests, embedding) without it.
        self.timeline: Optional[console.RunTimeline] = None

    @property
    def T_flange_wrist_camera(self) -> np.ndarray:
        return np.asarray(
            self.cfg.calibration.T_end_right_to_camera_rightwrist, dtype=float
        )

    @property
    def T_base_head_camera(self) -> np.ndarray:
        return np.asarray(
            self.cfg.calibration.T_base_right_to_camera_head, dtype=float
        )

    @property
    def T_flange_tcp(self) -> np.ndarray:
        calibration = getattr(
            getattr(self, "safety", None), "tool_mount_calibration", None
        )
        if calibration is not None and calibration.verified:
            _link7_to_flange, flange_to_tcp = calibration.require_transforms()
            return flange_to_tcp
        # Legacy offline/table fixtures have no shelf grasp frame.  A real
        # shelf execution cannot reach this fallback: profile loading requires
        # an explicit verified tool_mount_calibration before RobotSession is
        # created.  Do not reinterpret this as an installation measurement.
        transform = np.eye(4)
        transform[2, 3] = self.params.tcp_z_m
        return transform

    @property
    def T_link7_controller_flange(self) -> np.ndarray:
        calibration = getattr(
            getattr(self, "safety", None), "tool_mount_calibration", None
        )
        if calibration is not None and calibration.verified:
            link7_to_flange, _flange_to_tcp = calibration.require_transforms()
            return link7_to_flange
        transform = np.eye(4)
        transform[2, 3] = self.params.moveit_link7_to_controller_flange_m
        return transform

    @property
    def T_link7_tcp(self) -> np.ndarray:
        return self.T_link7_controller_flange @ self.T_flange_tcp

    def _is_read_only_vision_check(self) -> bool:
        return bool(
            getattr(self.args, "resume_at_wrist", False)
            and getattr(self.args, "stop_after_observation", False)
        )

    def _verify_detector_assets(self) -> dict:
        """Fail closed before safety, head, camera, SDK, or MoveIt setup."""
        vision = getattr(self.cfg, "vision", None)
        model_path = getattr(vision, "model_path", None)
        if not model_path:
            raise SafetyAbort("vision.model_path 未配置，拒绝初始化检测器")
        contract = inspect_model_asset_contract(
            self.project_root, model_path
        )
        try:
            require_model_asset_contract(contract)
        except ModelAssetContractError as exc:
            raise SafetyAbort(str(exc)) from exc
        return contract

    def _wrist_detector_from_contract(self) -> BottleDetector:
        """Build wrist detection from audited paths only, never a dirty tree."""
        fallback_path = verified_asset_path(
            self.model_asset_contract or {}, "fallback_detector"
        )
        if fallback_path is not None:
            return BottleDetector(str(fallback_path), 0.05)
        primary_path = verified_asset_path(
            self.model_asset_contract or {}, "primary_detector"
        )
        if primary_path is None:
            raise SafetyAbort("主模型资产未校验，拒绝初始化右腕检测器")
        LOG.warning("右腕检测复用已校验主模型；optional fallback 未提供")
        return BottleDetector(str(primary_path), 0.05)

    def _target_classes(self) -> Optional[set]:
        """Requested-product class filter for the detector, or None to keep
        the detector's built-in generic bottle-alias behaviour unchanged."""
        classes = self.params.target_product_classes
        return set(classes) if classes else None

    def stage(self, name: str, message: str = ""):
        # The stage flag lets the console formatter render workflow phases
        # differently from detail lines; file handlers ignore it.
        LOG.info(
            "%s %s" if message else "%s%s",
            name,
            message,
            extra={console.STAGE_FLAG: True},
        )
        if self.timeline is not None:
            self.timeline.mark(name)
        self.state.update(stage=name, message=message)

    @staticmethod
    def _plausible_close_bottle(detection, image_shape) -> bool:
        height, width = image_shape[:2]
        x1, y1, x2, y2 = detection.box
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        area = box_width * box_height
        return (
            box_width >= 0.05 * width
            and box_height >= 0.25 * height
            and area >= 0.02 * width * height
            and box_height / box_width >= 1.15
        )

    def _start_camera(self, camera_name: str):
        from sensors.camera_thread import CameraThread

        if self.preview:
            self.preview.stop()
            if self.preview.is_alive():
                self.preview.join(timeout=2)
            if self.preview.is_alive():
                raise CameraFrameUnavailable(
                    "相机预览线程停止超时，拒绝切换相机"
                )
            self.preview = None
        if self.camera:
            self.camera.stop()
            if self.camera.is_alive():
                self.camera.join(timeout=3)
            if self.camera.is_alive():
                raise CameraFrameUnavailable(
                    f"{self.camera_name or '当前'} 相机线程停止超时，"
                    "拒绝启动第二个 RGB-D pipeline"
                )
            self.camera = None
        serial = self.cfg.camera.serial_for(camera_name)
        if camera_name == "head":
            width, height = self.params.head_width, self.params.head_height
        else:
            width, height = self.cfg.camera.width, self.cfg.camera.height
        last_failure = "未知错误"
        for attempt in range(1, 4):
            try:
                prepare_camera_access(serial)
            except CameraAccessError as exc:
                raise CameraFrameUnavailable(
                    f"{camera_name} 相机不可用: {exc}"
                ) from exc
            self.camera = CameraThread(
                serial=serial,
                width=width,
                height=height,
                fps=self.cfg.camera.fps,
                strict_serial=True,
                shared_name=camera_name,
            )
            if self.camera.initialization_successful:
                self.camera.start()
                deadline = time.time() + 5
                while (
                    self.camera.get_latest_frames()[0] is None
                    and time.time() < deadline
                ):
                    time.sleep(0.1)
                if self.camera.get_latest_frames()[0] is not None:
                    break
                last_failure = "pipeline 已启动但 5 秒内没有画面"
                self.camera.stop()
                if self.camera.is_alive():
                    self.camera.join(timeout=3)
                if self.camera.is_alive():
                    raise CameraFrameUnavailable(
                        f"{camera_name} 相机线程停止超时，拒绝重建 pipeline"
                    )
            else:
                detail = getattr(self.camera, "initialization_error", None)
                last_failure = detail or "pipeline 初始化失败"
                self.camera.stop()
            self.camera = None
            if attempt == 1:
                LOG.warning(
                    "%s 相机第一次打开失败（%s）；释放后重建 pipeline 一次",
                    camera_name,
                    last_failure,
                )
                time.sleep(1.0)
            elif attempt == 2:
                LOG.warning(
                    "%s 相机两次打开都无帧；执行一次相机硬件重启后最后重试",
                    camera_name,
                )
                try:
                    hardware_reset_camera(serial)
                except CameraAccessError as exc:
                    raise CameraFrameUnavailable(
                        f"{camera_name} 相机硬件恢复失败: {exc}"
                    ) from exc
        else:
            raise CameraFrameUnavailable(
                f"{camera_name} 相机重建及硬件重启后仍无画面: {last_failure}"
            )
        self.camera_name = camera_name
        if camera_name == "right_wrist" and self.wrist_detector is None:
            self.wrist_detector = self._wrist_detector_from_contract()
        self.preview = PreviewWorker(self.camera, self.state)
        self.preview.start()
        self.state.update(detection=None, depth_m=None)

    def _ensure_head_reference(self):
        """在一切开始前，强制把头部舵机拉回标定基准角度（俯仰最低、左右居中）。

        `T_base_right_to_camera_head` 只在头部处于这个角度时有效——头部可能
        被人手动摆过（现场调试 head_camera_control.py），或者被 SDK 初始化
        的未知副作用带偏（旧 ArmController 有过这个实测坑，RobotSession 是
        否也有暂未排除，见项目记忆）。不管原因是什么，每次运行都强制校正
        一遍，不假设"应该还在原位"。
        """
        if getattr(self.args, "finish_from_current", False) or self._is_read_only_vision_check():
            # Finish does not use vision.  `resume check` promises not to move
            # hardware, so it cannot correct the head servos.  Motion-capable
            # resume does use the fixed head camera as an independent fallback.
            return
        current = head_lock.read_current_angle()
        if head_lock.is_at_reference(current):
            self.stage("头部基准位确认", f"未漂移: {current}")
            return
        self.stage(
            "头部基准位校正",
            f"当前 {current}，目标 {head_lock.HEAD_REFERENCE}",
        )
        if not self.args.execute:
            LOG.warning(
                "头部偏离标定基准角度，但当前非 --execute 不实际驱动舵机；"
                "真机执行前必须先解决，否则头部相机定位不可信"
            )
            return
        result = head_lock.restore_reference()
        if not result["ok"]:
            raise SafetyAbort(
                f"头部无法回到标定基准角度: {result.get('reason')}"
            )
        self.stage(
            "头部基准位已校正",
            f"{result['angle']}，用了 {result['steps']} 步",
        )

    def _load_safety_profiles(self) -> None:
        """Load profile-only safety contracts without opening hardware.

        The dispense task invokes this before the SHELF_READY body gate.  It
        must stay pure configuration work: no camera, arm, gripper, planner,
        or head-servo action belongs here.
        """
        expected_profiles = None
        run_dir = getattr(self, "run_dir", None)
        project_root = getattr(self, "project_root", None)
        if run_dir is not None and project_root is not None:
            expected_profiles = manifest_profile_expectations(
                Path(run_dir) / "run_manifest.json",
                args=self.args,
                project_root=project_root,
            )
        source_expected_digest = (
            expected_profiles["source"] if expected_profiles is not None else None
        )
        delivery_expected_digest = (
            expected_profiles["delivery"]
            if expected_profiles is not None
            else None
        )
        if not getattr(self, "_source_safety_profile_loaded", False):
            self.safety = load_safety_profile(
                self.args.safety_config,
                self.args.safety_profile,
                require_verified=self.args.execute,
                expected_profile_sha256=source_expected_digest,
            )
            self.source_safety = self.safety
            self._source_safety_profile_loaded = True
            if self.safety.grasp_height_fraction is not None:
                self.params = replace(
                    self.params,
                    grasp_height_fraction=self.safety.grasp_height_fraction,
                )
        if not getattr(self.args, "dispense", False):
            return
        if getattr(self, "_delivery_safety_profile_loaded", False):
            return
        delivery_profile = getattr(self.args, "delivery_safety_profile", None)
        if not delivery_profile:
            raise SafetyAbort(
                "--dispense 必须指定独立的 --delivery-safety-profile；"
                "货架转向后的围栏不能复用转向前坐标"
            )
        if delivery_profile == self.args.safety_profile:
            raise SafetyAbort(
                "--delivery-safety-profile 必须不同于 source safety profile；"
                "货架转向后的围栏不得复用转向前坐标"
            )
        self.delivery_safety = load_safety_profile(
            self.args.safety_config,
            delivery_profile,
            require_verified=self.args.execute,
            expected_profile_sha256=delivery_expected_digest,
        )
        self._delivery_safety_profile_loaded = True
        if self.delivery_safety.side_table_delivery is None:
            raise SafetyAbort(
                f"送货 profile {delivery_profile} 未配置 side_table_delivery"
            )

    def _validate_side_table_profile_pair(self) -> None:
        if self.safety is None or self.delivery_safety is None:
            raise SafetyAbort("桌面送货 profile 尚未加载")
        source_home = np.asarray(self.safety.home_joints_deg, dtype=float)
        output_home = np.asarray(
            self.delivery_safety.home_joints_deg, dtype=float
        )
        if (
            source_home.shape != (7,)
            or output_home.shape != (7,)
            or not np.allclose(source_home, output_home, atol=1e-6, rtol=0.0)
        ):
            raise SafetyAbort(
                "货架与桌面 profile 必须配置同一个已示教、无遮挡的 home_joints_deg"
            )
        # The physical right-tool mounting chain cannot change when the body
        # turns. SDK TCP and MoveIt were initialized from the source profile;
        # reject a delivery profile that would make its fence/held-object
        # geometry describe a different tool.
        source_mount = getattr(self.safety, "tool_mount_calibration", None)
        delivery_mount = getattr(
            self.delivery_safety, "tool_mount_calibration", None
        )
        if (
            source_mount is None
            or delivery_mount is None
            or not source_mount.verified
            or not delivery_mount.verified
        ):
            raise SafetyAbort(
                "货架与桌面 profile 都必须引用同一份已验证工具安装标定"
            )
        source_link7_flange, source_flange_tcp = (
            source_mount.require_transforms()
        )
        delivery_link7_flange, delivery_flange_tcp = (
            delivery_mount.require_transforms()
        )
        if (
            source_mount.evidence_id != delivery_mount.evidence_id
            or source_mount.measured_at_utc != delivery_mount.measured_at_utc
            or not np.allclose(
                source_link7_flange, delivery_link7_flange, atol=1e-9, rtol=0.0
            )
            or not np.allclose(
                source_flange_tcp, delivery_flange_tcp, atol=1e-9, rtol=0.0
            )
        ):
            raise SafetyAbort(
                "货架与桌面 profile 的工具安装标定不一致，拒绝混用 SDK/MoveIt/TCP 几何"
            )

    def _ensure_mobile_body(self) -> MobileBodyCoordinator:
        """Construct only the chassis/lift adapters, never an arm session."""
        if self.mobile_body is None:
            self.mobile_body = MobileBodyCoordinator(
                chassis=WooshChassisAdapter(
                    diagnostic_path=getattr(
                        self.args,
                        "chassis_diagnostic_path",
                        "/home/rm/agv_debug_tools/agv_diag",
                    ),
                    pose_query_path=getattr(
                        self.args,
                        "chassis_pose_query_path",
                        "/home/rm/agv_debug_tools/agv_pose_query",
                    ),
                    init_helper_path=getattr(
                        self.args,
                        "chassis_init_helper",
                        "/home/rm/agv_debug_tools/agv_mode_init",
                    ),
                    rotate_helper_path=getattr(
                        self.args,
                        "chassis_rotate_helper",
                        "/home/rm/agv_debug_tools/grabber_rotate_relative",
                    ),
                    stop_event=self.stop_event,
                ),
                lift=LiftSocketAdapter(
                    self.cfg.connections.left_arm_ip,
                    self.cfg.connections.arm_port,
                ),
                stop_event=self.stop_event,
                evidence_dir=self.run_dir,
            )
        return self.mobile_body

    def _capture_shelf_ready_for_dispense(self) -> BodySnapshot:
        """Run the body-only admission gate before any arm/gripper command."""
        if not getattr(self.args, "dispense", False) or not self.args.execute:
            raise SafetyAbort("SHELF_READY 仅允许明确的 --execute --dispense 任务")
        self._load_safety_profiles()
        self._validate_side_table_profile_pair()
        config = self.delivery_safety.side_table_delivery
        if config is None:
            raise SafetyAbort("桌面送货 profile 缺少 side_table_delivery")
        snapshot = self._ensure_mobile_body().capture_shelf_ready(config)
        self.shelf_ready_body_snapshot = snapshot
        return snapshot

    def initialize(self):
        # Asset integrity is the first runtime gate.  A missing or altered
        # archive model must not reach profile/head/camera/SDK/MoveIt setup.
        self.model_asset_contract = self._verify_detector_assets()
        primary_model_path = verified_asset_path(
            self.model_asset_contract, "primary_detector"
        )
        if primary_model_path is None:
            raise SafetyAbort("主模型资产未校验，拒绝初始化检测器")
        task_mode = getattr(self.args, "task_mode", None)
        skip_head = bool(
            not task_mode
            and (
                self.args.resume_at_wrist
                or getattr(self.args, "finish_from_current", False)
            )
        )
        needs_head_fallback = bool(
            self.args.resume_at_wrist
            and self.args.execute
            and not self._is_read_only_vision_check()
        )
        # Profile/schema failure follows asset integrity and still precedes
        # head, arm, gripper, camera, and planner setup.  Dispense may already
        # have loaded profiles through the stricter SHELF_READY admission gate.
        self._load_safety_profiles()
        self._ensure_head_reference()
        self.scene_boxes = self.safety.moveit_collision_boxes()
        self.stage(
            "初始化",
            (
                f"电子围栏 profile={self.safety.name}；"
                "固定头部 RGB-D 搜索水瓶；"
                f"抓取框高度比例={self.params.grasp_height_fraction:.2f}"
            ),
        )
        # Both supported task modes create a fresh fixed-head lock.  In
        # particular, FROM_OBSERVATION is a physical starting condition, not a
        # request to resume from an old localization file.
        if not skip_head or needs_head_fallback:
            fallback_path = verified_asset_path(
                self.model_asset_contract or {}, "fallback_detector"
            )
            if fallback_path is None:
                LOG.warning(
                    "YOLO fallback 未随受审计 archive 提供；仅使用已校验主模型，"
                    "不会查找 dirty 目录或自动下载"
                )
            self.detector = BottleDetector(
                str(primary_model_path),
                self.params.confidence,
                fallback_model_path=(
                    str(fallback_path) if fallback_path is not None else None
                ),
                fallback_confidence=0.05,
            )
        self.dashboard = Dashboard(self.state, self.args.host, self.args.port)
        self.dashboard.start()
        self._start_camera("right_wrist" if skip_head else "head")

        read_only_vision_check = self._is_read_only_vision_check()
        needs_robot = (
            self.args.plan_only or self.args.execute or read_only_vision_check
        )
        if needs_robot:
            self.robot = RobotSession(
                self.cfg.connections.right_arm_ip,
                self.cfg.connections.arm_port,
                self.stop_event,
                self.params.tcp_z_m,
                self.params.moveit_link7_to_controller_flange_m,
                take_control=self.args.execute and not read_only_vision_check,
                tcp_transform=self.T_flange_tcp,
                link7_to_controller_flange=(
                    self.T_link7_controller_flange
                ),
            )
        needs_planner = self.args.plan_only or (
            self.args.execute and not read_only_vision_check
        )
        if needs_planner:
            self.left_robot = ArmJointReader(
                self.cfg.connections.left_arm_ip,
                self.cfg.connections.arm_port,
            )
            self.planner = MoveItPlanner(self.project_root, self.run_dir)
            self.planner.start()
        if getattr(self.args, "dispense", False) and self.args.execute:
            self._ensure_mobile_body()

    def _load_resume_localization(self) -> Localization:
        output_dir = Path(self.args.output_dir)
        candidates = [
            path
            for path in output_dir.glob("*/*_localization.json")
            if "右腕" in path.name or "预抓取" in path.name
        ]

        def resume_priority(path: Path):
            if "右腕续抓定位" in path.name:
                priority = 0
            elif "右腕精定位" in path.name:
                priority = 1
            else:
                priority = 2
            return priority, -path.stat().st_mtime

        candidates.sort(key=resume_priority)
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                localization = Localization(**payload)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            self.stage(
                "加载续抓先验",
                f"{path.parent.name}/{path.name}: {localization.point_base}",
            )
            return localization
        raise SafetyAbort("续抓模式找不到上一轮腕部稳定定位记录")

    def localize(
        self,
        label: str,
        transform_provider: Callable[[], np.ndarray],
        depth_params: DemoParams,
        depth_prior_base: Optional[np.ndarray] = None,
        *,
        allow_depth_prior_fallback: bool = True,
        required_consensus_frames: Optional[int] = None,
    ) -> Localization:
        self.stage(label, f"{self.camera_name} 连续采集 {depth_params.samples} 帧")
        K, _ = self.camera.get_camera_intrinsics()
        if K is None:
            raise SafetyAbort("相机内参不可用")
        camera_points, base_points = [], []
        depths, mads, detections, pixels = [], [], [], []
        last_timestamp = 0.0
        deadline = time.time() + max(10, depth_params.samples * 2.5)
        while len(camera_points) < depth_params.samples and time.time() < deadline:
            if self.stop_event.is_set():
                raise SafetyAbort(stop_reason(self.stop_event))
            timestamp = self.camera.get_frame_timestamp()
            if timestamp <= last_timestamp:
                time.sleep(0.03)
                continue
            last_timestamp = timestamp
            color, depth = self.camera.get_latest_frames()
            if color is None or depth is None:
                continue
            detector = (
                self.wrist_detector
                if self.camera_name == "right_wrist"
                else self.detector
            )
            predicate = None
            association = None
            T_base_camera = None
            if self.camera_name == "right_wrist":
                shape = color.shape
                if depth_prior_base is None:
                    predicate = lambda det: self._plausible_close_bottle(det, shape)
                else:
                    T_base_camera = transform_provider()
                    association = ProjectedTargetAssociation.from_view(
                        target_base=np.asarray(depth_prior_base, dtype=float),
                        T_base_camera=T_base_camera,
                        intrinsics=K,
                        image_shape=shape,
                    )
                    predicate = association.accepts
            elif depth_prior_base is not None:
                # Independent head confirmations must associate the same
                # locked object too; with multiple bottles, a generic class
                # detection is not evidence that the released target stayed
                # at its table location.
                T_base_camera = transform_provider()
                association = ProjectedTargetAssociation.from_view(
                    target_base=np.asarray(depth_prior_base, dtype=float),
                    T_base_camera=T_base_camera,
                    intrinsics=K,
                    image_shape=color.shape,
                )
                predicate = association.accepts
            detection = detector.detect(
                color, predicate, target_classes=self._target_classes()
            )
            if detection is None:
                self.state.update(
                    detection=None, message="未检测到符合形状的 bottle"
                )
                continue
            self.state.update(detection=detection)
            # A wrist view of a transparent cylinder measures a visible
            # surface, not its centre.  Worse, a box touching the image border
            # has no stable height semantics.  When the fixed head has already
            # locked the object, wrist data may refine horizontal centring but
            # never overwrite locked depth/grasp height.
            if self.camera_name == "right_wrist" and depth_prior_base is not None:
                assert association is not None and T_base_camera is not None
                try:
                    point_camera, point_base, pixel, z = (
                        association.refine_locked_depth(
                            detection=detection,
                            target_base=np.asarray(depth_prior_base, dtype=float),
                            T_base_camera=T_base_camera,
                            intrinsics=K,
                        )
                    )
                except SafetyAbort as exc:
                    self.state.update(message=str(exc))
                    continue
                mad = 0.0
                truncated = association.touches_image_border(
                    detection, color.shape
                )
                self.state.update(
                    message=(
                        "腕部截断框：保持头部锁定深度/抓取高度，仅修正横向"
                        if truncated
                        else "腕部关联：保持头部锁定深度，仅修正横向"
                    )
                )
            else:
                try:
                    point_camera, z, mad, pixel = depth_point_for_detection(
                        depth, detection, K, depth_params
                    )
                except SafetyAbort as exc:
                    if (
                        depth_prior_base is None
                        or not allow_depth_prior_fallback
                    ):
                        self.state.update(message=str(exc))
                        continue
                    T_base_camera = transform_provider()
                    prior_camera = (
                        np.linalg.inv(T_base_camera)
                        @ np.r_[np.asarray(depth_prior_base, dtype=float), 1.0]
                    )[:3]
                    z = float(prior_camera[2])
                    if not (
                        depth_params.min_depth_m
                        <= z
                        <= depth_params.max_depth_m
                    ):
                        self.state.update(message=f"先验深度越界: {z:.3f} m")
                        continue
                    x1, y1, x2, y2 = detection.box
                    u = 0.5 * (x1 + x2)
                    v = y1 + depth_params.grasp_height_fraction * (y2 - y1)
                    point_camera = np.array(
                        [
                            (u - K[0, 2]) * z / K[0, 0],
                            (v - K[1, 2]) * z / K[1, 1],
                            z,
                        ],
                        dtype=float,
                    )
                    mad = 0.0
                    pixel = (float(u), float(v))
                if T_base_camera is None:
                    T_base_camera = transform_provider()
                point_base = (T_base_camera @ np.r_[point_camera, 1])[:3]
            camera_points.append(point_camera)
            base_points.append(point_base)
            depths.append(z)
            mads.append(mad)
            detections.append(detection)
            pixels.append(pixel)
            # Per-frame detail is evidence, not something an operator needs
            # seven copies of on screen; the console shows the consensus line
            # below, the run log keeps every frame at DEBUG.
            LOG.debug(
                "%s 帧 %d box=%s conf=%.3f base=[%.3f, %.3f, %.3f]",
                label,
                len(camera_points),
                detection.box,
                detection.confidence,
                point_base[0],
                point_base[1],
                point_base[2],
            )
            self.state.update(
                depth_m=z,
                message=f"有效帧 {len(camera_points)}/{depth_params.samples}",
            )
            time.sleep(0.08)

        if len(camera_points) < depth_params.samples:
            raise SafetyAbort(
                f"检测/深度稳定帧不足: {len(camera_points)}/{depth_params.samples}"
            )
        base = np.asarray(base_points, dtype=float)
        distances = np.linalg.norm(base[:, None, :] - base[None, :, :], axis=2)
        support = distances <= depth_params.max_position_spread_m
        support_counts = np.count_nonzero(support, axis=1)
        seed = int(np.argmax(support_counts))
        if required_consensus_frames is None:
            required = max(3, int(np.ceil(0.70 * len(base))))
        else:
            required = int(required_consensus_frames)
            if not (2 <= required <= len(base)):
                raise SafetyAbort(
                    "多帧定位共识门槛配置无效: "
                    f"required={required}, samples={len(base)}"
                )
        inliers = np.flatnonzero(support[seed])
        if len(inliers) < required:
            raise SafetyAbort(
                "多帧定位没有稳定共识: "
                f"最大同簇 {len(inliers)}/{len(base)}"
            )
        center = np.median(base[inliers], axis=0)
        inliers = np.flatnonzero(
            np.linalg.norm(base - center, axis=1)
            <= depth_params.max_position_spread_m
        )
        if len(inliers) < required:
            raise SafetyAbort(
                "多帧定位离群过滤后不足: "
                f"{len(inliers)}/{len(base)}"
            )
        center = np.median(base[inliers], axis=0)
        spread = float(
            np.max(np.linalg.norm(base[inliers] - center, axis=1))
        )
        if spread > depth_params.max_position_spread_m:
            raise SafetyAbort(f"多帧三维位置过散: {spread * 1000:.1f} mm")
        best = max(
            (detections[index] for index in inliers),
            key=lambda item: item.confidence,
        )
        localization = Localization(
            np.median(np.asarray(camera_points)[inliers], axis=0).tolist(),
            center.tolist(),
            np.median(np.asarray(pixels)[inliers], axis=0).tolist(),
            float(np.median(np.asarray(depths)[inliers])),
            float(np.median(np.asarray(mads)[inliers])),
            spread,
            list(best.box),
            best.confidence,
            len(inliers),
            best.class_name,
        )
        self.stage(
            f"{label}稳定",
            (
                f"共识帧 {len(inliers)}/{len(base)}，"
                f"散布 {spread * 1000:.1f} mm"
            ),
        )
        self._save_localization(label, localization)
        return localization

    def _save_localization(self, label: str, localization: Localization):
        color, depth = self.camera.get_latest_frames()
        stem = label.replace(" ", "_")
        if color is not None:
            cv2.imwrite(str(self.run_dir / f"{stem}_color.jpg"), color)
        if depth is not None:
            np.save(self.run_dir / f"{stem}_depth_m.npy", depth)
        localization.captured_at_utc = datetime.now(timezone.utc).isoformat()
        payload = asdict(localization)
        (self.run_dir / f"{stem}_localization.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _observation_flange_candidates(
        self, target_base: np.ndarray
    ) -> list[np.ndarray]:
        # Make a copy: normalizing a NumPy slice in-place must not mutate the
        # detected bottle position used to construct the observation pose.
        horizontal = np.array(target_base[:2], dtype=float, copy=True)
        if np.linalg.norm(horizontal) < 0.1:
            horizontal = np.array([1.0, 0.0])
        horizontal /= np.linalg.norm(horizontal)
        lateral_axis = np.array([-horizontal[1], horizontal[0]])
        candidates = []
        # Raised shelf-clearance views first.  Lower views remain as fallbacks
        # and still need the exact same whole-arm validation.
        for height in (0.13, 0.08, 0.03, -0.02):
            for standoff in (0.30, 0.36, 0.40, 0.26):
                for lateral in (0.0, 0.06, -0.06):
                    camera_position = target_base.copy()
                    camera_position[:2] -= horizontal * standoff
                    camera_position[:2] += lateral_axis * lateral
                    camera_position[2] += height
                    T_base_camera = look_at_camera_pose(
                        target_base, camera_position
                    )
                    optical_pitch_deg = float(
                        np.degrees(
                            np.arcsin(
                                np.clip(T_base_camera[2, 2], -1.0, 1.0)
                            )
                        )
                    )
                    if not (
                        self.params.observation_camera_min_pitch_deg
                        <= optical_pitch_deg
                        <= self.params.observation_camera_max_pitch_deg
                    ):
                        continue
                    candidates.append(
                        T_base_camera
                        @ np.linalg.inv(self.T_flange_wrist_camera)
                    )
        return candidates

    def _grasp_precheck_margin(
        self, target: PlanTarget, target_base: np.ndarray
    ) -> float | None:
        """返回该观察位通过抓取预演的最宽限位余量档位。

        分级而不是一刀切：2026-07-18 晚真机 watch 实测，10° 软余量的二元
        筛选把 11 个端点砍到只剩 1 个（瓶子位置本身处在手臂舒适区边缘，
        多数观察姿态天然贴限位），而唯一幸存者又恰好是 MoveIt 规划不出
        路径的端点（error=99999），没有备胎直接中止。现在按
        宽(10°)→中(6.5°)→执行余量(3°) 三档降级预演：宽余量候选优先，
        窄余量的保留但排后。None = 连执行余量都过不了，真正不可行。
        """
        margins = (
            self.params.observation_grasp_margin_deg,
            (
                self.params.observation_grasp_margin_deg
                + self.params.joint_limit_margin_deg
            )
            / 2,
            self.params.joint_limit_margin_deg,
        )
        planned: list[list[float]] | None = None
        accepted_margin: float | None = None
        for margin in margins:
            self._abort_if_stopped()
            try:
                planned = self._plan_grasp_precheck(
                    target, target_base, margin
                )
                accepted_margin = float(margin)
                break
            except SafetyAbort as exc:
                self._abort_if_stopped()
                LOG.debug(
                    "%s 抓取预检 %.1f° 余量未通过 IK/围栏: %s",
                    target.label,
                    margin,
                    exc,
                )
        if planned is None or accepted_margin is None:
            return None
        try:
            self._validate_grasp_precheck_plan(
                target, target_base, planned
            )
        except SafetyAbort as exc:
            self._abort_if_stopped()
            # The joint-margin setting does not change this already solved
            # joint path.  Replaying the same exact MoveIt collision at two
            # narrower margins only made one bad candidate cost 15 checks.
            LOG.debug(
                "%s 抓取预检全链碰撞失败，不以更窄余量重复同一路径: %s",
                target.label,
                exc,
            )
            return None
        return accepted_margin

    def _abort_if_stopped(self) -> None:
        """Make a requested stop terminal inside nested candidate searches."""
        event = getattr(self, "stop_event", None)
        if event is not None and event.is_set():
            raise SafetyAbort(stop_reason(event))

    def _plan_grasp_precheck(
        self,
        target: PlanTarget,
        target_base: np.ndarray,
        limit_margin_deg: float,
    ) -> list[list[float]]:
        """Solve one authored-orientation grasp continuation for a margin."""
        tcp = target.flange @ self.T_flange_tcp
        base_rotation = resolve_tcp_grasp_rotation(self.safety, tcp)
        precheck_params = replace(
            self.params,
            joint_limit_margin_deg=limit_margin_deg,
            # The observation endpoint must leave enough elbow bend for the
            # complete continuation.  Merely staying outside the controller's
            # hard 8-degree band reproduced the real "arrived, then singular"
            # failure after wrist relocalization.
            j4_singularity_deg=max(
                self.params.j4_singularity_deg,
                self.params.j4_escape_deg,
            ),
        )
        _, _, _, full_path = self._local_pick_place_geometry(
            tcp, target_base, base_rotation
        )
        for index, pose in enumerate(full_path, 1):
            self.safety.assert_tcp_point(
                pose[:3], label=f"完整抓放预检路径点 {index}"
            )
        return self.robot.plan_ik(
            full_path,
            precheck_params,
            allow_first_jump=False,
            seed_joints_deg=target.goal_joints,
        )

    def _validate_grasp_precheck_plan(
        self,
        target: PlanTarget,
        target_base: np.ndarray,
        planned: Sequence[Sequence[float]],
    ) -> None:
        # The real local task performs both controller IK checks and MoveIt
        # whole-arm collision validation.  Rehearsal must do the same;
        # otherwise hand-vs-shelf can pass site-check and fail after transfer.
        self._validate_local_joint_path(
            name="observation_grasp_precheck",
            joints=planned,
            target_base=np.asarray(target_base, dtype=float),
            # This is a hypothetical continuation from the candidate
            # observation endpoint.  Live home would invent an unplanned
            # home->grasp interpolation through the shelf.
            start_joints_deg=target.goal_joints,
        )

    def _grasp_precheck_ok(
        self,
        target: PlanTarget,
        target_base: np.ndarray,
        limit_margin_deg: float,
    ) -> bool:
        """Check the single final-horizontal continuation for this endpoint.

        2026-07-18 真机 observe 实测的教训：观察位只按"转移代价+3°硬限位
        余量"选，选出了 J2 距限位 3.3° 的端点。观察位和抓取不是两个独立
        问题：这里用 candidate_path() 完全相同的单一几何，从候选关节角
        出发做纯离线预演，不再用固定 ±15°/±30° 掩盖最终姿态问题。
        更宽的余量档位吸收头部定位和腕部精定位之间约 3cm 的目标漂移。
        """
        try:
            planned = self._plan_grasp_precheck(
                target, target_base, limit_margin_deg
            )
            self._validate_grasp_precheck_plan(
                target, target_base, planned
            )
            return True
        except SafetyAbort as exc:
            self._abort_if_stopped()
            LOG.debug("%s 抓取预检不可行: %s", target.label, exc)
            return False

    def _observation_plan_targets(
        self,
        target_base: np.ndarray,
        current_joints_deg: Optional[Sequence[float]] = None,
    ) -> list[PlanTarget]:
        current = np.asarray(
            (
                self.robot.joints_deg()
                if current_joints_deg is None
                else current_joints_deg
            ),
            dtype=float,
        )
        if current.shape != (7,) or not np.all(np.isfinite(current)):
            raise SafetyAbort("观察位候选评分起点必须是 7 个有限关节角")
        endpoint_total = 0
        graded: list[tuple[float, PlanTarget]] = []
        candidate_index = 0
        enough_viable = False
        for flange_index, flange in enumerate(
            self._observation_flange_candidates(target_base), 1
        ):
            self._abort_if_stopped()
            try:
                tcp = flange @ self.T_flange_tcp
                self.safety.assert_tcp_point(
                    tcp[:3, 3],
                    label=f"右腕观察位候选 {flange_index}",
                )
                multi_solver = getattr(
                    self.robot, "solve_flange_ik_candidates", None
                )
                if callable(multi_solver):
                    joint_solutions = multi_solver(
                        flange,
                        self.params,
                        seed_joints_deg=current,
                    )
                else:
                    joint_solutions = [
                        self.robot.solve_flange_ik(
                            flange,
                            self.params,
                            seed_joints_deg=current,
                        )
                    ]
            except SafetyAbort as exc:
                LOG.debug("观察位候选 %d 被拒绝: %s", flange_index, exc)
                continue
            elbow_groups: list[tuple[np.ndarray, float]] = []
            for joints in joint_solutions:
                candidate_index += 1
                index = candidate_index
                group_index = next(
                    (
                        item_index
                        for item_index, (reference, _) in enumerate(
                            elbow_groups
                        )
                        if np.allclose(
                            np.asarray(joints[:6], dtype=float),
                            reference,
                            atol=1e-3,
                            rtol=0.0,
                        )
                    ),
                    None,
                )
                if group_index is None:
                    elbow_groups.append(
                        (
                            np.asarray(joints[:6], dtype=float),
                            float(joints[6]),
                        )
                    )
                    group_index = len(elbow_groups) - 1
                reference_j7 = elbow_groups[group_index][1]
                elbow_label = (
                    "主肘位"
                    if group_index == 0
                    else f"备选肘位 {group_index}"
                )
                turn_delta = int(
                    round((float(joints[6]) - reference_j7) / 360.0)
                ) * 360
                turn_label = (
                    "J7 当前圈"
                    if turn_delta == 0
                    else f"J7 换圈 {turn_delta:+d}°"
                )
                try:
                    # A later grasp-axis roll cannot repair an observation
                    # endpoint already in collision.  Validate every bounded
                    # J7 turn once before rehearsing the continuation.
                    self._validate_local_joint_path(
                        name="observation_endpoint_precheck",
                        joints=[joints],
                        target_base=np.asarray(target_base, dtype=float),
                        start_joints_deg=joints,
                    )
                except SafetyAbort as exc:
                    LOG.debug(
                        "观察位候选 %d（%s）被拒绝: %s",
                        index,
                        f"{elbow_label}，{turn_label}",
                        exc,
                    )
                    continue
                endpoint_total += 1
                plan_target = PlanTarget(
                    label=(
                        f"右腕观察位候选 {index}"
                        f"（{elbow_label}，{turn_label}）"
                    ),
                    flange=flange,
                    goal_joints=tuple(joints),
                    score=float(
                        np.linalg.norm(
                            np.asarray(joints, dtype=float) - current
                        )
                    ),
                    # Execute the exact controller-IK branch that passed the
                    # endpoint and continuation prechecks.
                    goal_constraint="joints",
                )
                # Grade immediately.  The former two-pass loop first checked
                # every endpoint branch (including equivalent J7 turns) and
                # only then tried a complete pick; after arm-angle IK this
                # spent minutes proving redundant endpoints safe.
                self._abort_if_stopped()
                margin = self._grasp_precheck_margin(
                    plan_target, np.asarray(target_base)
                )
                if margin is None:
                    LOG.info(
                        "%s 连执行余量也未通过抓取预演，淘汰",
                        plan_target.label,
                    )
                else:
                    LOG.info(
                        "%s 抓取预演可行，限位余量档位 %.1f°",
                        plan_target.label,
                        margin,
                    )
                    graded.append((margin, plan_target))
                    if (
                        len(graded)
                        >= self.params.observation_viable_candidate_limit
                    ):
                        enough_viable = True
                        break
            if enough_viable:
                break
        if endpoint_total == 0:
            raise SafetyAbort("所有右腕观察位候选均越界、近限位、逆解或碰撞失败")
        if not graded:
            raise SafetyAbort(
                f"{endpoint_total} 个观察位端点通过，但"
                "完整抓放预演全部失败（限位/奇异/围栏/碰撞）"
            )
        graded.sort(key=lambda item: (-item[0], item[1].score))
        roomy = sum(
            1
            for margin, _ in graded
            if margin >= self.params.observation_grasp_margin_deg
        )
        self.stage(
            "生成右腕观察位候选",
            (
                f"端点通过 {endpoint_total} 个；"
                f"抓取预演可行（完整抓放） {len(graded)} 个"
                f"（宽余量 {roomy} 个，优先尝试）；"
                f"最多尝试前 {self.params.global_plan_max_candidates} 个"
            ),
        )
        return [target for _, target in graded]

    def _select_observation_flange(
        self, target_base: np.ndarray
    ) -> tuple[np.ndarray, list[float]]:
        """Compatibility helper returning the best endpoint, without planning."""
        target = self._observation_plan_targets(target_base)[0]
        return target.flange, list(target.goal_joints)

    def _build_head_scene(
        self, localization: Localization, *, full_frame: bool = False
    ):
        if not self.safety.use_dynamic_rgbd:
            self.head_scene_voxels = []
            self.non_target_scene_voxels = []
            self.target_occupancy_voxels = []
            self.scene_voxels = []
            self.head_scene_captured_monotonic = time.monotonic()
            self.stage(
                "构建障碍场景",
                f"使用 {len(self.scene_boxes)} 个静态电子围栏禁入区",
            )
            return
        K, _ = self.camera.get_camera_intrinsics()
        if K is None:
            raise SafetyAbort("头部相机内参不可用")
        captured_monotonic = time.monotonic()
        captured_at_utc = datetime.now(timezone.utc).isoformat()
        depth_frames = self._collect_fresh_depth_frames(
            self.params.scene_samples, label="头部障碍场景"
        )
        image_height_px = int(depth_frames[0].shape[0])
        if any(int(depth.shape[0]) != image_height_px for depth in depth_frames):
            raise SafetyAbort("头部深度帧高度在同一次场景采集中发生变化")
        observed_row_limit_px = (
            image_height_px
            if full_frame
            else min(image_height_px, self.params.scene_image_bottom_crop)
        )
        per_frame_voxels = [
            build_scene_voxels(
                depth,
                K,
                self.T_base_head_camera,
                localization,
                self.params,
                min_depth_m=self.params.head_min_depth_m,
                max_depth_m=self.params.head_max_depth_m,
                bottom_crop=observed_row_limit_px,
            )
            for depth in depth_frames
        ]
        per_frame_target_voxels = [
            build_target_occupancy_voxels(
                depth,
                K,
                self.T_base_head_camera,
                localization,
                self.params,
                min_depth_m=self.params.head_min_depth_m,
                max_depth_m=self.params.head_max_depth_m,
                bottom_crop=observed_row_limit_px,
            )
            for depth in depth_frames
        ]
        per_frame_non_target_voxels = [
            build_non_target_scene_voxels(
                depth,
                K,
                self.T_base_head_camera,
                localization,
                self.params,
                min_depth_m=self.params.head_min_depth_m,
                max_depth_m=self.params.head_max_depth_m,
                bottom_crop=observed_row_limit_px,
            )
            for depth in depth_frames
        ]
        self.head_scene_voxels = union_scene_voxels(
            per_frame_voxels, self.params
        )
        self.non_target_scene_voxels = union_scene_voxels(
            per_frame_non_target_voxels, self.params
        )
        self.target_occupancy_voxels = union_scene_voxels(
            per_frame_target_voxels,
            self.params,
            max_voxels=self.params.scene_max_voxels,
        )
        self.scene_voxels = list(self.head_scene_voxels)
        table_fit = self._adapt_fence_to_measured_table(
            depth_frames, K, localization
        )
        shelf_fits = self._adapt_fence_to_measured_shelf(
            depth_frames, K, localization
        )
        self.head_scene_captured_monotonic = captured_monotonic
        self._head_scene_reference_frame = self.safety.frame
        self._head_scene_base_pose = np.asarray(
            getattr(self, "_base_pose_for_scene", np.eye(4)), dtype=float
        ).copy()
        (self.run_dir / "head_scene.json").write_text(
            json.dumps(
                {
                    "safety_profile": self.safety.name,
                    "frame": self.safety.frame,
                    "captured_at_utc": captured_at_utc,
                    "target_captured_at_utc": localization.captured_at_utc,
                    "target_point_base": localization.point_base,
                    "image_height_px": image_height_px,
                    "observed_row_limit_px": observed_row_limit_px,
                    "voxel_size_m": self.params.scene_voxel_m,
                    "voxel_count": len(self.scene_voxels),
                    "scene_voxels": self.scene_voxels,
                    "non_target_scene_voxel_count": len(
                        self.non_target_scene_voxels
                    ),
                    "non_target_scene_voxels": self.non_target_scene_voxels,
                    "target_occupancy_voxel_count": len(
                        self.target_occupancy_voxels
                    ),
                    "target_occupancy_voxels": self.target_occupancy_voxels,
                    "collision_boxes": self.scene_boxes,
                    "table_fit": (
                        None if table_fit is None else asdict(table_fit)
                    ),
                    "shelf_fits": {
                        face: asdict(fit) for face, fit in shelf_fits.items()
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        per_frame_counts = [len(item) for item in per_frame_voxels]
        self.stage(
            "构建障碍场景",
            (
                f"{len(self.scene_boxes)} 个电子围栏禁入区；"
                f"{len(self.scene_voxels)} 个动态 RGB-D 体素"
                f"（{len(depth_frames)} 帧并集，单帧 {per_frame_counts}）"
            ),
        )

    def _collect_fresh_depth_frames(
        self, count: int, *, label: str
    ) -> list[np.ndarray]:
        """Collect `count` distinct fresh depth frames, or refuse to proceed.

        Frames are keyed on the camera timestamp so this cannot silently
        return the same buffer N times — which would look like consensus
        while providing none.
        """
        if count < 1:
            raise SafetyAbort(f"{label} 的采样帧数必须至少为 1")
        frames: list[np.ndarray] = []
        last_timestamp = 0.0
        deadline = time.time() + max(6.0, count * 2.0)
        while len(frames) < count and time.time() < deadline:
            if self.stop_event.is_set():
                raise SafetyAbort(stop_reason(self.stop_event))
            timestamp = self.camera.get_frame_timestamp()
            if timestamp <= last_timestamp:
                time.sleep(0.03)
                continue
            last_timestamp = timestamp
            _, depth = self.camera.get_latest_frames()
            if depth is None:
                continue
            frames.append(depth)
        if len(frames) < count:
            raise SafetyAbort(
                f"{label} 只取到 {len(frames)}/{count} 个新鲜深度帧，"
                "RGB-D 流不稳定，拒绝用不足的采样构建避障场景"
            )
        return frames

    def _adapt_fence_to_measured_table(
        self, depth_frames: Sequence[np.ndarray], K, localization: Localization
    ):
        """每轮实测桌面，让电子围栏跟着真实桌子走（容差外拒跑）。

        MoveIt 的动态体素本来就每轮反映真实桌面；会过期的是静态配置的
        table_top 禁区和贴着旧桌面高度画的允许区下沿。桌子比配置低/远时，
        旧盒子会挡住真实桌面上方明明可用的空间（虚假拒绝）；比配置高/近时
        围栏漏保护。这里在 table_fit_height_tolerance_m 的信封内自适应，
        超出信封说明布置真的变了，fail-closed 拒跑并提示重新测量。

        高度来自多帧：单帧的统计学去噪（分箱众数+中位数）扛得住散点噪声，
        但扛不住整帧状态不对（有人手经过、曝光刚切换）。帧间不一致就拒跑，
        因为这个测量结果会直接决定本轮围栏怎么调。
        """
        # A profile without a table keepout never needed the measurement, so
        # skip the RGB-D work entirely and do not let frame disagreement
        # abort a run it cannot affect.
        if not any(
            box.id == TABLE_KEEPOUT_ID for box in self.safety.keepout_boxes
        ):
            return None
        target = np.asarray(localization.point_base, dtype=float)
        fits = [
            fit_table_top(
                head_scene_points(
                    depth,
                    K,
                    self.T_base_head_camera,
                    self.params,
                    min_depth_m=self.params.head_min_depth_m,
                    max_depth_m=self.params.head_max_depth_m,
                    bottom_crop=self.params.scene_image_bottom_crop,
                ),
                target,
                self.params,
            )
            for depth in depth_frames
        ]
        table_fit = combine_table_fits(fits, self.params)
        adapted = adapt_profile_to_table(self.safety, table_fit, self.params)
        if adapted is self.safety:
            return table_fit
        old_top = next(
            box
            for box in self.safety.keepout_boxes
            if box.id == TABLE_KEEPOUT_ID
        ).maximum[2]
        self.safety = adapted
        self.scene_boxes = self.safety.moveit_collision_boxes()
        new_top = next(
            box
            for box in self.safety.keepout_boxes
            if box.id == TABLE_KEEPOUT_ID
        ).maximum[2]
        self.stage(
            "桌面围栏自适应",
            (
                f"{len(fits)} 帧一致，实测桌面 z={table_fit.height_m:.3f}"
                f"（最少 {table_fit.inliers} 内点），禁区顶面 "
                f"{old_top:.3f} -> {new_top:.3f}"
            ),
        )
        return table_fit

    def _adapt_fence_to_measured_shelf(
        self, depth_frames: Sequence[np.ndarray], K, localization: Localization
    ) -> dict:
        """每轮实测货架各面，让电子围栏跟着真实货架走（容差外拒跑）。

        泛化自 `_adapt_fence_to_measured_table`：同一套"多帧一致性 + 容差内
        跟随 + 水平范围只扩不缩、超容差 fail-closed"规则，应用到
        `shelf_model.FACE_SPECS` 里任意存在于当前 profile 的货架面（bottom/
        top/back/left/right），而不是只处理 `table_top` 这一个固定 id。跟
        桌面路径完全独立、互不影响：没有货架面 box 的 profile（比如
        `table_demo`）这里直接返回空字典，`_adapt_fence_to_measured_table`
        的行为和结果不受任何影响。
        """
        present_faces = [
            box.id
            for box in self.safety.keepout_boxes
            if box.id in SHELF_FACE_SPECS
        ]
        if not present_faces:
            return {}
        target = np.asarray(localization.point_base, dtype=float)
        fits_by_face = {}
        for face in present_faces:
            per_frame_fits = [
                fit_shelf_face(
                    head_scene_points(
                        depth,
                        K,
                        self.T_base_head_camera,
                        self.params,
                        min_depth_m=self.params.head_min_depth_m,
                        max_depth_m=self.params.head_max_depth_m,
                        bottom_crop=self.params.scene_image_bottom_crop,
                    ),
                    target,
                    face,
                    self.params,
                )
                for depth in depth_frames
            ]
            fits_by_face[face] = combine_shelf_fits(per_frame_fits, self.params)
        adapted = adapt_profile_to_shelf(self.safety, fits_by_face, self.params)
        if adapted is not self.safety:
            self.safety = adapted
            self.scene_boxes = self.safety.moveit_collision_boxes()
            self.stage(
                "货架围栏自适应",
                f"已按本轮实测更新 {len(fits_by_face)} 个货架面: "
                f"{', '.join(sorted(fits_by_face))}",
            )
        return fits_by_face

    def _verified_plan_targets(
        self,
        name: str,
        targets: list[PlanTarget],
        start_right_joints_deg: Optional[Sequence[float]] = None,
        continuation_validator: Optional[
            Callable[[PlanTarget, dict], None]
        ] = None,
        trajectory_validator: Optional[
            Callable[[PlanTarget, dict], None]
        ] = None,
        enforce_endpoint_vertical_floor: bool = False,
    ) -> VerifiedPlan:
        safe_planner = SafeMotionPlanner(
            moveit=self.planner,
            robot=self.robot,
            left_robot=self.left_robot,
            safety=self.safety,
            params=self.params,
            report=self.stage,
            held_object=getattr(self, "held_object_guard", None),
            link7_to_controller_flange=self.T_link7_controller_flange,
        )
        verified = safe_planner.plan(
            name=name,
            targets=targets,
            obstacle_points=self.scene_voxels,
            collision_boxes=self.scene_boxes,
            start_right_joints_deg=start_right_joints_deg,
            continuation_validator=continuation_validator,
            trajectory_validator=trajectory_validator,
            enforce_endpoint_vertical_floor=enforce_endpoint_vertical_floor,
        )
        captured = getattr(self, "head_scene_captured_monotonic", None)
        if captured is not None:
            verified.trajectory["scene_captured_monotonic"] = float(
                captured
            )
        return verified

    def _plan_flange(
        self,
        name: str,
        target_flange: np.ndarray,
        goal_joints: Optional[list[float]] = None,
    ) -> dict:
        explicit_joint_goal = goal_joints is not None
        if goal_joints is None:
            goal_joints = self.robot.solve_flange_ik(
                target_flange, self.params
            )
        verified = self._verified_plan_targets(
            name,
            [
                PlanTarget(
                    label="固定目标",
                    flange=target_flange,
                    goal_joints=tuple(goal_joints),
                    goal_constraint=("joints" if explicit_joint_goal else "pose"),
                )
            ],
        )
        return verified.trajectory

    def _assert_no_vertical_undershoot(
        self,
        *,
        label: str,
        start_joints_deg: Sequence[float],
        trajectory: dict,
    ) -> None:
        """Reject a transfer that drops below both endpoints before rising."""
        points = trajectory.get("points_deg") or []
        if not points:
            raise SafetyAbort(f"{label}轨迹为空，无法检查垂直路线")
        start = np.asarray(start_joints_deg, dtype=float)
        if start.shape != (7,) or not np.all(np.isfinite(start)):
            raise SafetyAbort(f"{label}规划起点不是 7 个有限关节角")
        dense = interpolate_joint_path(
            start, points, self.params.planned_joint_step_deg
        )
        tcp_z = [
            float(self.robot.tcp_from_joints(start)[2, 3]),
            *[
                float(self.robot.tcp_from_joints(joints)[2, 3])
                for joints in dense
            ],
        ]
        if not np.all(np.isfinite(tcp_z)):
            raise SafetyAbort(f"{label}轨迹 TCP 高度含非有限值")
        lower_endpoint = min(tcp_z[0], tcp_z[-1])
        lowest = min(tcp_z)
        undershoot = lower_endpoint - lowest
        if (
            undershoot
            > self.params.observation_vertical_undershoot_tolerance_m
        ):
            raise SafetyAbort(
                f"{label}路线会先下探再回升: "
                f"最低点低于较低端点 {undershoot * 1000:.1f} mm "
                f"(上限 {self.params.observation_vertical_undershoot_tolerance_m * 1000:.0f} mm)"
            )

    def _plan_observation_staging(
        self,
        start_right_joints_deg: Optional[Sequence[float]] = None,
    ) -> Optional[dict]:
        """Plan the open/high departure leg configured for a low parked arm."""
        staging = self.safety.observation_staging_joints_deg
        if staging is None:
            return None
        start = np.asarray(
            (
                self.robot.joints_deg()
                if start_right_joints_deg is None
                else start_right_joints_deg
            ),
            dtype=float,
        )
        goal = np.asarray(staging, dtype=float)
        if (
            start.shape != (7,)
            or goal.shape != (7,)
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(goal))
        ):
            raise SafetyAbort("观察准备位的起点/终点关节角无效")
        max_delta = float(np.max(np.abs(goal - start)))
        if max_delta <= self.params.planned_start_tolerance_deg:
            self.stage(
                "观察准备位",
                f"当前姿态已在准备位容差内（最大差 {max_delta:.2f}°），无需移动",
            )
            return None

        target_flange = self.robot.controller_flange_from_joints(goal)
        target_tcp = self.robot.tcp_from_joints(goal)
        if (
            np.asarray(target_flange).shape != (4, 4)
            or np.asarray(target_tcp).shape != (4, 4)
            or not np.all(np.isfinite(target_flange))
            or not np.all(np.isfinite(target_tcp))
        ):
            raise SafetyAbort("观察准备位 SDK FK 无效")
        self.safety.assert_tcp_point(
            np.asarray(target_tcp)[:3, 3], label="抬高展开观察准备位"
        )

        def validate_transfer_shape(
            _target: PlanTarget, trajectory: dict
        ) -> None:
            self._assert_no_vertical_undershoot(
                label="到观察准备位",
                start_joints_deg=start,
                trajectory=trajectory,
            )

        verified = self._verified_plan_targets(
            "moveit_observation_staging",
            [
                PlanTarget(
                    label="抬高展开观察准备位",
                    flange=np.asarray(target_flange, dtype=float),
                    goal_joints=tuple(map(float, goal)),
                    score=float(np.linalg.norm(goal - start)),
                    goal_constraint="joints",
                )
            ],
            start_right_joints_deg=start,
            trajectory_validator=validate_transfer_shape,
            enforce_endpoint_vertical_floor=True,
        )
        self.stage(
            "选择观察准备位路线",
            (
                f"从当前姿态先抬高并展开；最大关节变化 {max_delta:.1f}°，"
                f"规划尝试 {verified.attempts} 次"
            ),
        )
        return verified.trajectory

    def _plan_observation(
        self,
        target_base: np.ndarray,
        start_right_joints_deg: Optional[Sequence[float]] = None,
    ) -> dict:
        target_point = np.asarray(target_base, dtype=float)
        planning_start = np.asarray(
            (
                self.robot.joints_deg()
                if start_right_joints_deg is None
                else start_right_joints_deg
            ),
            dtype=float,
        )
        if (
            planning_start.shape != (7,)
            or not np.all(np.isfinite(planning_start))
        ):
            raise SafetyAbort("观察位规划起点必须是 7 个有限关节角")

        def validate_transfer_shape(
            _target: PlanTarget, trajectory: dict
        ) -> None:
            self._assert_no_vertical_undershoot(
                label="观察位",
                start_joints_deg=planning_start,
                trajectory=trajectory,
            )

        def validate_actual_endpoint(
            target: PlanTarget, trajectory: dict
        ) -> None:
            points = trajectory.get("points_deg") or []
            if not points:
                raise SafetyAbort("观察位轨迹为空，无法预演后续抓放")
            endpoint = tuple(map(float, points[-1]))
            actual_flange = self.robot.controller_flange_from_joints(endpoint)
            actual_target = replace(
                target,
                flange=actual_flange,
                goal_joints=endpoint,
                goal_constraint="joints",
            )
            margin = self._grasp_precheck_margin(
                actual_target, target_point
            )
            if margin is None:
                raise SafetyAbort(
                    "MoveIt 实际观察终点无法完成后续接近/抓取/抬升/放回"
                )

        verified = self._verified_plan_targets(
            "moveit_observation",
            self._observation_plan_targets(
                target_point,
                current_joints_deg=planning_start,
            ),
            start_right_joints_deg=planning_start,
            continuation_validator=validate_actual_endpoint,
            trajectory_validator=validate_transfer_shape,
            enforce_endpoint_vertical_floor=True,
        )
        self.stage(
            "选择右腕观察位",
            (
                f"{verified.target.label}，关节变化评分 "
                f"{verified.target.score:.1f}；规划尝试 {verified.attempts} 次"
            ),
        )
        return verified.trajectory

    def _execute_plan(self, name: str, plan: dict) -> None:
        def assert_scene_fresh() -> None:
            scene_time = plan.get("scene_captured_monotonic")
            if scene_time is None and getattr(self.args, "task_mode", None):
                raise SafetyAbort("规划凭证缺少 RGB-D 场景时间戳，禁止执行")
            if scene_time is None:
                return
            scene_age = time.monotonic() - float(scene_time)
            if (
                not np.isfinite(scene_age)
                or scene_age < 0
                or scene_age > self.params.scene_max_age_s
            ):
                raise SafetyAbort(
                    "规划场景已过期，禁止按旧障碍快照运动: "
                    f"age={scene_age:.1f}s, limit={self.params.scene_max_age_s:.1f}s"
                )

        assert_scene_fresh()

        expected_left = plan.get("start_left_joints_deg")
        monitor_done = threading.Event()
        monitor_errors: list[SafetyAbort] = []
        monitor = None

        def assert_left_snapshot() -> None:
            if expected_left is None:
                if getattr(self.args, "task_mode", None):
                    raise SafetyAbort("规划凭证缺少左臂起点快照，禁止执行")
                return
            if self.left_robot is None:
                raise SafetyAbort("无法读取规划时参与碰撞场景的左臂状态")
            expected = np.asarray(expected_left, dtype=float)
            actual = np.asarray(self.left_robot.joints_deg(), dtype=float)
            if (
                expected.shape != (7,)
                or actual.shape != (7,)
                or not np.all(np.isfinite(expected))
                or not np.all(np.isfinite(actual))
            ):
                raise SafetyAbort("左臂规划快照或实时反馈含非有限数/维度无效")
            error = float(np.max(np.abs(actual - expected)))
            if error > self.params.planned_start_tolerance_deg:
                raise SafetyAbort(
                    "轨迹已过期：左臂已偏离碰撞规划快照，拒绝执行: "
                    f"最大关节差={error:.2f}°，"
                    f"上限={self.params.planned_start_tolerance_deg:.2f}°"
                )

        if expected_left is not None or getattr(self.args, "task_mode", None):
            assert_left_snapshot()
            # Reading a remote left arm can block for seconds.  Recheck the
            # RGB-D snapshot *after* that read so a 44-second scene cannot be
            # executed at 46 seconds (the previous TOCTOU window).
            assert_scene_fresh()

            def monitor_left_arm() -> None:
                while not monitor_done.wait(0.5):
                    try:
                        assert_left_snapshot()
                    except SafetyAbort as exc:
                        monitor_errors.append(exc)
                        setattr(self.stop_event, "source", "left_arm_drift")
                        self.stop_event.set()
                        return

            monitor = threading.Thread(
                target=monitor_left_arm,
                name="bottle-left-arm-snapshot-guard",
                daemon=True,
            )
            monitor.start()

        self.stage(
            name,
            f"{len(plan['points_deg'])} 个 MoveIt 轨迹点，SDK {self.params.transit_speed}%",
        )
        motion_error: BaseException | None = None
        try:
            self.robot.execute_planned_joints(
                plan["points_deg"],
                self.params.transit_speed,
                self.params.planned_joint_step_deg,
                expected_start_joints_deg=plan.get("start_joints_deg"),
                start_tolerance_deg=self.params.planned_start_tolerance_deg,
                tracking_tolerance_deg=self.params.planned_tracking_tolerance_deg,
            )
        except BaseException as exc:
            motion_error = exc
        finally:
            monitor_done.set()
            if monitor is not None:
                monitor.join(timeout=9.0)
        if monitor is not None and monitor.is_alive():
            self.stop_event.set()
            self.robot.hold()
            raise SafetyAbort("左臂状态监控线程未能及时退出，拒绝继续任务")
        if monitor_errors:
            raise monitor_errors[0]
        if motion_error is not None:
            raise motion_error
        if expected_left is not None:
            assert_left_snapshot()

    def _refresh_head_scene_for_global_motion(
        self, locked_target: Localization
    ) -> None:
        """Reacquire a fixed-head world snapshot before a later global leg."""
        if self.camera_name != "head":
            self._start_camera("head")
        params = replace(
            self.params,
            samples=self.params.wrist_relocalization_samples,
            min_depth_m=self.params.head_min_depth_m,
            max_depth_m=self.params.head_max_depth_m,
            max_position_spread_m=0.06,
        )
        target = self.localize(
            "返回前头部场景确认",
            lambda: self.T_base_head_camera,
            params,
            depth_prior_base=np.asarray(locked_target.point_base, dtype=float),
            required_consensus_frames=(
                self.params.post_action_confirmation_min_frames
            ),
        )
        shift = float(
            np.linalg.norm(
                np.asarray(target.point_base, dtype=float)
                - np.asarray(locked_target.point_base, dtype=float)
            )
        )
        if shift > self.params.head_confirmation_tolerance_m:
            raise SafetyAbort(
                "返回前瓶子位置已改变，禁止沿旧任务场景规划: "
                f"shift={shift * 1000:.1f}mm"
            )
        self._build_head_scene(target)
        self.stage(
            "返回前刷新障碍场景",
            f"固定头部重采场景；瓶子偏移 {shift * 1000:.1f} mm",
        )

    def _refresh_and_revalidate_plan(
        self,
        *,
        name: str,
        plan: dict,
        locked_target: Localization,
    ) -> None:
        """Refresh a slow global search and revalidate its exact trajectory."""
        self._refresh_head_scene_for_global_motion(locked_target)
        if self.planner is None or self.left_robot is None:
            raise SafetyAbort("刷新全局轨迹时 MoveIt/左臂读取器未初始化")
        planned_start = np.asarray(plan.get("start_joints_deg"), dtype=float)
        actual_start = np.asarray(self.robot.joints_deg(), dtype=float)
        if (
            planned_start.shape != (7,)
            or actual_start.shape != (7,)
            or not np.all(np.isfinite(planned_start))
            or not np.all(np.isfinite(actual_start))
        ):
            raise SafetyAbort("刷新轨迹复核的右臂起点无效")
        start_error = float(np.max(np.abs(actual_start - planned_start)))
        if start_error > self.params.planned_start_tolerance_deg:
            raise SafetyAbort(
                "刷新场景时右臂已偏离规划起点，必须重新规划: "
                f"最大关节差={start_error:.2f}°"
            )
        left = self.left_robot.joints_deg()
        dense = interpolate_joint_path(
            actual_start,
            plan.get("points_deg") or [],
            self.params.planned_joint_step_deg,
        )
        self.robot.validate_planned_joints(
            plan.get("points_deg") or [],
            self.params.planned_joint_step_deg,
            self.safety,
            start_joints_deg=actual_start,
        )
        self.planner.validate_exact_path(
            name=f"{name}_fresh_scene",
            start_left_joints_deg=left,
            points_deg=dense,
            obstacles=self._moveit_scene_obstacles(self.scene_voxels),
            boxes=self.scene_boxes,
            planning_frame=self.safety.moveit_frame,
            tool_guard={
                "xy": self.params.tool_guard_xy_m,
                "length": self.params.tool_guard_length_m,
                "center_z": self.params.tool_guard_center_z_m,
            },
            held_object=getattr(self, "held_object_guard", None),
            voxel_size=self.params.scene_voxel_m,
        )
        plan["start_joints_deg"] = actual_start.tolist()
        plan["start_left_joints_deg"] = list(map(float, left))
        plan["scene_captured_monotonic"] = float(
            self.head_scene_captured_monotonic
        )
        self.stage(
            "新鲜场景轨迹复核",
            f"{name}: 原轨迹在新采 RGB-D/左臂快照下仍有效",
        )

    def _approach_pregrasp(self, wrist_target: Localization):
        """一次规划、直线分段接近到预抓取位。

        原先每前进一段就向 MoveIt 重新规划一次（蠕动式走走停停）；全局转移
        已由 SafeMotionPlanner 统一做 MoveIt 碰撞规划、后验碰撞复核和电子围栏
        复核。近距离接近仍用更可预测的直线分段，并由候选姿态围栏校验 + plan_ik
        连续性/限位/奇异检查 + 腕部点云通道检查 + 每段 RGB-D 存活检查保护。
        目标已由 7 帧锁定；横移时暂时离开 YOLO 视野只警告，到达预抓取位后
        的复检仍必须重新检测到瓶子，才会进入最后接近。
        """
        target_base = np.asarray(wrist_target.point_base)
        pregrasp_pose, _, transit_path = self.candidate_path(target_base)
        start_distance = float(
            np.linalg.norm(self.robot.current_tcp()[:3, 3] - target_base)
        )
        distances = [
            float(np.linalg.norm(np.asarray(pose[:3]) - target_base))
            for pose in transit_path
        ]
        if not distances:
            raise SafetyAbort("预抓取转移路径为空")
        if any(
            distance > start_distance + 0.005 for distance in distances
        ):
            raise SafetyAbort(
                "预抓取路径没有朝锁定目标收敛: "
                f"start={start_distance:.3f}m path={np.round(distances, 3).tolist()}"
            )
        if abs(distances[-1] - self.params.pregrasp_standoff_m) > 0.012:
            raise SafetyAbort(
                "预抓取终点距锁定目标不等于预定悬停距离: "
                f"actual={distances[-1]:.3f}m "
                f"expected={self.params.pregrasp_standoff_m:.3f}m"
            )
        # candidate_path 已校验预抓取点与最终接近段；这里补上转移段逐点围栏。
        for index, pose in enumerate(transit_path, 1):
            self.safety.assert_tcp_point(
                pose[:3], label=f"预抓取转移路径点 {index}"
            )
        self.robot.plan_ik(transit_path, self.params, allow_first_jump=False)
        self.collision_gate(
            wrist_target.box,
            target_base,
            corridor_waypoints_base=[pose[:3] for pose in transit_path],
        )
        self.stage(
            "直线接近预抓取位",
            (
                f"{len(transit_path)} 段，速度 {self.params.travel_speed}%；"
                f"距锁定目标 {start_distance * 100:.1f}→"
                f"{distances[-1] * 100:.1f} cm"
            ),
        )
        guard = LockedTargetGuard(
            wrist_check=lambda point: self.ensure_bottle_visible(
                target_base=point
            ),
            head_confirm=self._confirm_locked_target_from_head,
        )
        for index, (pose, distance) in enumerate(
            zip(transit_path, distances), 1
        ):
            guard_result = guard.verify(target_base)
            if guard_result.source == "head":
                LOG.warning(
                    "腕部检测丢失；固定头部相机已确认锁定目标，继续预抓取转移"
                )
            elif guard_result.source == "head_cached":
                LOG.warning(
                    "腕部检测再次丢失；本段沿用本次转移内刚完成的头部确认"
                )
            LOG.info(
                "锁定目标接近 %d/%d：TCP 距目标 %.1f cm，视觉来源=%s",
                index,
                len(transit_path),
                distance * 100,
                guard_result.source,
            )
            self.robot.move_linear(pose, self.params.travel_speed)

    def collision_gate(
        self,
        target_box: Optional[Sequence[int]],
        target_base: np.ndarray,
        corridor_waypoints_base: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        count = check_approach_corridor(
            camera=self.camera,
            robot=self.robot,
            target_box=target_box,
            target_base=target_base,
            T_flange_camera=self.T_flange_wrist_camera,
            params=self.params,
            corridor_waypoints_base=corridor_waypoints_base,
        )
        self.stage("右腕点云通道检查", f"通过，疑似障碍点 {count}")

    def _scene_without_locked_target(
        self, target_base: np.ndarray
    ) -> list[list[float]]:
        """Remove only voxels intersecting the locked bottle cylinder.

        ``target_base`` is a depth point on the camera-facing bottle surface,
        not its axis.  Centre the cylinder one bottle radius farther along the
        head-camera ray.  Because ``scene_voxels`` stores cell centres, expand
        the centre test by half a voxel: otherwise a cell containing the
        bottle edge survives simply because its centre is outside the exact
        physical radius and MoveIt treats the target itself as an obstacle.
        """
        points = np.asarray(self.scene_voxels, dtype=float)
        target = np.asarray(target_base, dtype=float)
        if points.size == 0:
            return []
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or target.shape != (3,)
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(target))
        ):
            raise SafetyAbort("局部 MoveIt 场景点或锁定目标无效")
        camera_xy = np.asarray(self.T_base_head_camera[:2, 3], dtype=float)
        view_xy = target[:2] - camera_xy
        view_norm = float(np.linalg.norm(view_xy))
        cylinder_center = target.copy()
        if view_norm > 1e-9:
            cylinder_center[:2] += (
                view_xy
                / view_norm
                * self.params.target_occupancy_radius_m
            )
        # A square voxel intersects a horizontal cylinder when its centre is
        # within the physical radius plus the cell's half diagonal.
        radial_limit = (
            self.params.target_occupancy_radius_m
            + self.params.scene_voxel_m / np.sqrt(2.0)
        )
        vertical_half_cell = self.params.scene_voxel_m / 2.0
        radial = np.linalg.norm(
            points[:, :2] - cylinder_center[:2], axis=1
        )
        is_target = (
            (radial <= radial_limit)
            & (
                points[:, 2]
                >= target[2]
                - self.params.target_occupancy_below_grasp_m
                - vertical_half_cell
            )
            & (
                points[:, 2]
                <= target[2]
                + self.params.target_occupancy_above_grasp_m
                + vertical_half_cell
            )
        )
        recorded = np.asarray(
            getattr(self, "target_occupancy_voxels", []), dtype=float
        )
        if recorded.size:
            if (
                recorded.ndim != 2
                or recorded.shape[1] != 3
                or not np.all(np.isfinite(recorded))
            ):
                raise SafetyAbort("目标占据体素记录无效")
            point_keys = np.floor(
                points / self.params.scene_voxel_m
            ).astype(np.int32)
            recorded_keys = {
                tuple(key)
                for key in np.floor(
                    recorded / self.params.scene_voxel_m
                ).astype(np.int32)
            }
            is_target |= np.asarray(
                [tuple(key) in recorded_keys for key in point_keys],
                dtype=bool,
            )
        return points[~is_target].tolist()

    def _moveit_scene_obstacles(
        self, scene_points: Sequence[Sequence[float]]
    ) -> list[list[float]]:
        """Convert dynamic points without double-inflating fitted fences."""
        obstacle_filter = getattr(
            self.safety, "moveit_obstacles_outside_fences", None
        )
        if callable(obstacle_filter):
            return obstacle_filter(scene_points, self.scene_boxes)
        # Compatibility for small test doubles and legacy custom profiles.
        return self.safety.points_to_moveit(scene_points)

    def _validate_local_joint_path(
        self,
        *,
        name: str,
        joints: Sequence[Sequence[float]],
        target_base: Optional[np.ndarray],
        start_joints_deg: Optional[Sequence[float]] = None,
        scene_points_override: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """Validate an exact local IK chain with SDK fence and full MoveIt."""
        args = getattr(self, "args", None)
        if not (
            getattr(args, "task_mode", None)
            or bool(getattr(args, "plan_only", False))
        ):
            return
        if self.planner is None or self.left_robot is None:
            raise SafetyAbort("完整任务局部路径缺少 MoveIt/左臂碰撞状态")
        self._abort_if_stopped()
        start = np.asarray(
            (
                self.robot.joints_deg()
                if start_joints_deg is None
                else start_joints_deg
            ),
            dtype=float,
        )
        if start.shape != (7,) or not np.all(np.isfinite(start)):
            raise SafetyAbort("局部碰撞复核起点必须是 7 个有限关节角")
        left = self.left_robot.joints_deg()
        dense = interpolate_joint_path(
            start, joints, self.params.planned_joint_step_deg
        )
        self.robot.validate_planned_joints(
            joints,
            self.params.planned_joint_step_deg,
            self.safety,
            start_joints_deg=start,
        )
        sequence = int(getattr(self, "_local_validation_sequence", 0)) + 1
        self._local_validation_sequence = sequence
        scene_points = (
            list(scene_points_override)
            if scene_points_override is not None
            else (
                self.scene_voxels
                if target_base is None
                else self._scene_without_locked_target(target_base)
            )
        )
        self.planner.validate_exact_path(
            name=f"local_{sequence:02d}_{name}",
            start_left_joints_deg=left,
            points_deg=dense,
            obstacles=self._moveit_scene_obstacles(scene_points),
            boxes=self.scene_boxes,
            planning_frame=self.safety.moveit_frame,
            tool_guard={
                "xy": self.params.tool_guard_xy_m,
                "length": self.params.tool_guard_length_m,
                "center_z": self.params.tool_guard_center_z_m,
            },
            held_object=getattr(self, "held_object_guard", None),
            voxel_size=self.params.scene_voxel_m,
        )
        self._abort_if_stopped()
        self.stage(
            "局部全链碰撞复核",
            f"{name}: SDK 围栏与 MoveIt 全链/自碰/左臂/场景均通过",
        )

    def _plan_local_leg(
        self,
        name: str,
        build_path: Callable[[], list[list[float]]],
        params: DemoParams,
        *,
        allow_first_jump: bool = False,
        roll_degrees: Sequence[float] = (0, 8, -8, 15, -15, 25, -25),
    ) -> list[list[float]]:
        """本地笛卡尔小段（抬升/下降/退开）的统一规划入口。

        先处理"起点已在 J4 奇异带内"：绕接近轴 roll 改不了 |J4|（肘角
        大小由肩-腕距离唯一决定，绕工具 z 轴不移动腕心），2026-07-17 真机
        finish 就是这样把所有 roll 重试耗尽的。唯一干净的出路是关节空间
        弯肘逃逸（robot.escape_j4_singularity，逐点围栏校验后 movej），
        逃逸会移动 TCP，所以路径必须在逃逸之后再从新姿态构建——这就是
        这里收一个 build_path 回调而不是现成路径的原因。
        """
        if getattr(getattr(self, "args", None), "task_mode", None):
            current = np.asarray(self.robot.joints_deg(), dtype=float)
            if current.shape != (7,) or not np.all(np.isfinite(current)):
                raise SafetyAbort(f"{name} 前实时关节状态无效")
            if abs(float(current[3])) < params.j4_singularity_deg:
                raise SafetyAbort(
                    f"{name} 前 J4={current[3]:.1f}° 已进入奇异带；"
                    "完整任务禁止用未经过场景规划的临时弯肘 movej 旁路"
                )
            escaped = None
        else:
            escaped = self.robot.escape_j4_singularity(params, self.safety)
        if escaped is not None:
            self.stage(
                "J4 奇异带弯肘逃逸",
                f"{name}: 起点在奇异带内，已弯肘至 J4={escaped[3]:.1f}° 后重建路径",
            )
        return self._plan_ik_avoiding_singularity(
            build_path(),
            params,
            allow_first_jump=allow_first_jump,
            roll_degrees=roll_degrees,
        )

    def _plan_ik_avoiding_singularity(
        self,
        poses: list[list[float]],
        params: DemoParams,
        *,
        allow_first_jump: bool = False,
        roll_degrees: Sequence[float] = (0, 8, -8, 15, -15, 25, -25),
    ) -> list[list[float]]:
        """如 plan_ik，被拒绝时尝试绕接近轴（工具 z 轴）小角度重试。

        roll 重试能解决的是逆解分支/限位/关节跳变类拒绝；它改不了 |J4|
        （肘角大小由肩-腕距离唯一决定，绕工具 z 轴不移动腕心）。"起点已在
        J4 奇异带内"的场景由 _plan_local_leg 的关节空间弯肘逃逸处理，
        不要指望这里的 roll。返回值替换调用方原来的 poses 列表，因为真正
        被执行的姿态必须和通过逆解检查的姿态一致。
        """
        for roll_deg in roll_degrees:
            if roll_deg:
                roll = Rotation.from_euler(
                    "z", roll_deg, degrees=True
                ).as_matrix()
                rotated = []
                for pose in poses:
                    T = pose_matrix(pose).copy()
                    T[:3, :3] = T[:3, :3] @ roll
                    rotated.append(matrix_pose(T))
            else:
                rotated = list(poses)
            try:
                planned = self.robot.plan_ik(
                    rotated, params, allow_first_jump=allow_first_jump
                )
                target = getattr(self, "local_contact_target_base", None)
                if target is not None:
                    self._validate_local_joint_path(
                        name="local_leg",
                        joints=planned,
                        target_base=np.asarray(target, dtype=float),
                    )
                if roll_deg:
                    self.stage(
                        "绕接近轴避奇异",
                        f"当前姿态贴近 J4 奇异区，旋转 {roll_deg:+d}° 后逆解通过",
                    )
                return rotated
            except SafetyAbort as exc:
                LOG.warning("绕轴 %+.0f° 仍未通过逆解: %s", roll_deg, exc)
        raise SafetyAbort(
            "多个旋转角度均未能避开 J4 奇异区，放弃移动——若拒绝原因是"
            "路径中途进入奇异带，说明目标接近手臂最大伸展，roll 无法解决，"
            "需要调整目标高度/距离或移动底盘"
        )

    def candidate_path(
        self, target: np.ndarray
    ) -> tuple[list[float], list[float], list[list[float]]]:
        current_tcp = self.robot.current_tcp()
        current = matrix_pose(current_tcp)
        if self.grasp_rotation is None:
            self.grasp_rotation = resolve_tcp_grasp_rotation(
                self.safety, current_tcp
            )
        pregrasp_pose, grasp_pose, path, full_path = (
            self._local_pick_place_geometry(
                current_tcp,
                target,
                self.grasp_rotation,
            )
        )
        try:
            for index, pose in enumerate(full_path, 1):
                self.safety.assert_tcp_point(
                    pose[:3], label=f"完整局部抓放路径点 {index}"
                )
            planned = self.robot.plan_ik(
                full_path,
                self.params,
                allow_first_jump=False,
            )
            self._validate_local_joint_path(
                name="complete_pick_place",
                joints=planned,
                target_base=np.asarray(target, dtype=float),
            )
        except SafetyAbort as exc:
            raise SafetyAbort(
                "最终水平抓取的完整路径未通过逆解/限位/奇异/碰撞检查: "
                f"{exc}"
            ) from exc
        return pregrasp_pose, grasp_pose, path

    def _local_pick_place_geometry(
        self,
        start_tcp: np.ndarray,
        target: np.ndarray,
        rotation: np.ndarray,
    ) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
        """Build the complete local tail used to qualify an observation pose.

        The old precheck jumped directly from the observation joint seed to a
        pregrasp pose and only checked the final approach.  It could therefore
        approve an observation pose whose very next Cartesian segment crossed
        J4's singular band.  This geometry mirrors every later local motion:
        observation→pregrasp→grasp→lift→lower→retreat.
        """
        start = np.asarray(start_tcp, dtype=float)
        target = np.asarray(target, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if (
            start.shape != (4, 4)
            or target.shape != (3,)
            or rotation.shape != (3, 3)
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(target))
            or not np.all(np.isfinite(rotation))
        ):
            raise SafetyAbort("完整局部抓放预演收到无效几何")
        axis = rotation[:, 2]
        grasp = np.eye(4)
        grasp[:3, :3] = rotation
        grasp[:3, 3] = target - axis * self.params.grasp_stop_short_m
        pregrasp = grasp.copy()
        pregrasp[:3, 3] = target - axis * self.params.pregrasp_standoff_m
        lift = grasp.copy()
        lift[2, 3] += self.params.lift_m

        pregrasp_pose = matrix_pose(pregrasp)
        grasp_pose = matrix_pose(grasp)
        clearance = start.copy()
        clearance[2, 3] += self.params.local_clearance_lift_m
        clearance_pose = matrix_pose(clearance)
        # First rise in the already collision-checked observation attitude,
        # then interpolate directly to the authored final-horizontal grasp.
        # There is no discrete ±15°/±30° intermediate-angle catalogue.
        transit = [
            *interpolate_poses(
                matrix_pose(start), clearance_pose, self.params.segment_m
            ),
            *interpolate_poses(
                clearance_pose,
                pregrasp_pose,
                self.params.segment_m,
            ),
        ]
        approach = interpolate_poses(
            pregrasp_pose, grasp_pose, self.params.segment_m
        )
        lift_path = interpolate_poses(
            grasp_pose, matrix_pose(lift), self.params.segment_m
        )
        lower_path = interpolate_poses(
            matrix_pose(lift), grasp_pose, self.params.segment_m
        )
        retreat = grasp.copy()
        retreat[:3, 3] -= axis * self.params.retreat_standoff_m
        retreat_path = interpolate_poses(
            grasp_pose, matrix_pose(retreat), self.params.segment_m
        )
        return (
            pregrasp_pose,
            grasp_pose,
            transit,
            [*transit, *approach, *lift_path, *lower_path, *retreat_path],
        )

    def ensure_bottle_visible(
        self, target_base: Optional[np.ndarray] = None
    ):
        if self.camera.get_frame_timestamp() < time.time() - self.params.frame_timeout_s:
            raise CameraFrameUnavailable("RGB-D 画面中断")
        color, _ = self.camera.get_latest_frames()
        if color is None:
            raise CameraFrameUnavailable("RGB-D 彩色画面缺失")
        detector = (
            self.wrist_detector
            if self.camera_name == "right_wrist"
            else self.detector
        )
        predicate = None
        association = None
        if self.camera_name == "right_wrist":
            if target_base is None:
                shape = color.shape
                predicate = lambda det: self._plausible_close_bottle(det, shape)
            else:
                K, _ = self.camera.get_camera_intrinsics()
                if K is None:
                    raise CameraFrameUnavailable("RGB-D 相机内参缺失")
                association = ProjectedTargetAssociation.from_view(
                    target_base=np.asarray(target_base, dtype=float),
                    T_base_camera=(
                        self.robot.current_flange()
                        @ self.T_flange_wrist_camera
                    ),
                    intrinsics=K,
                    image_shape=color.shape,
                )
                predicate = association.accepts
        detection = detector.detect(
            color, predicate, target_classes=self._target_classes()
        )
        if detection is None:
            detail = ""
            if association is not None:
                detail = (
                    f"；锁定目标投影={np.round(association.pixel, 1).tolist()}"
                    f" in_image={association.in_image}"
                )
            raise BottleDetectionLost(
                "移动过程中与锁定目标关联的 bottle 检测丢失" + detail
            )
        return detection

    def _confirm_locked_target_from_head(
        self,
        target_base: np.ndarray,
        *,
        restore_wrist: bool = True,
    ) -> Localization:
        """Pause between segments and independently confirm via the head camera.

        The head result never overwrites the 7-frame wrist lock.  A meaningful
        shift means the bottle may have moved, so the current path is no longer
        valid and must stop instead of being patched in flight.
        """
        if self.detector is None:
            raise BottleDetectionLost("腕部检测丢失且头部检测器未初始化")
        target = np.asarray(target_base, dtype=float)
        head_target = None
        original_error = None
        try:
            self._start_camera("head")
            head_params = replace(
                self.params,
                samples=self.params.wrist_relocalization_samples,
                min_depth_m=self.params.head_min_depth_m,
                max_depth_m=self.params.head_max_depth_m,
                max_position_spread_m=0.06,
            )
            head_target = self.localize(
                "头部补充确认",
                lambda: self.T_base_head_camera,
                head_params,
                depth_prior_base=target,
            )
        except SafetyAbort as exc:
            original_error = exc
        finally:
            if restore_wrist:
                try:
                    self._start_camera("right_wrist")
                except SafetyAbort as restore_exc:
                    raise SafetyAbort(
                        f"头部补充确认后无法恢复右腕相机: {restore_exc}"
                    ) from restore_exc
        if original_error is not None:
            raise BottleDetectionLost(
                f"腕部检测丢失，头部相机也未能确认锁定目标: {original_error}"
            ) from original_error
        shift = float(
            np.linalg.norm(np.asarray(head_target.point_base) - target)
        )
        if shift > self.params.head_confirmation_tolerance_m:
            raise SafetyAbort(
                "头部相机确认瓶子已偏离锁定目标，当前路径作废: "
                f"shift={shift * 1000:.1f}mm "
                f"limit={self.params.head_confirmation_tolerance_m * 1000:.0f}mm"
            )
        self.stage(
            "头部补充确认通过",
            f"目标相对腕部锁定点偏移 {shift * 1000:.1f} mm",
        )
        return head_target

    def _confirm_target_at_pregrasp(self, locked: Localization) -> GuardResult:
        """Confirm presence without redefining the locked grasp target.

        At pregrasp distance the eye-in-hand view is commonly clipped or
        occluded by the fingers.  Re-running full wrist 3-D localization there
        caused the real 2026-07-18 ``0/3`` abort even though the fixed head had
        just confirmed the bottle.  A live wrist-associated detection is
        preferred; only detector loss (never RGB-D stream loss) may fall back
        to the independent fixed-head observer.
        """
        target = np.asarray(locked.point_base, dtype=float)
        guard = LockedTargetGuard(
            wrist_check=lambda point: self.ensure_bottle_visible(
                target_base=point
            ),
            head_confirm=self._confirm_locked_target_from_head,
        )
        result = guard.verify(target)
        self.stage(
            "预抓取目标确认",
            (
                "腕部关联检测通过，保持原锁定抓取点"
                if result.source == "wrist"
                else "腕部局部视野丢失，固定头部确认通过；保持原锁定抓取点"
            ),
        )
        return result

    def _run_pregrasp_visual_servo(
        self, locked: Localization
    ) -> Localization:
        """Run an opt-in, bounded eye-in-hand correction at the safe hover.

        This is not high-rate MoveIt Servo.  The deployed controller executes
        blocking commands, so the only honest closed loop available today is:
        fresh wrist consensus -> one small translation -> fresh consensus.
        Every translation is requalified by the existing complete grasp-chain,
        IK/singularity/joint-margin, MoveIt/SDK fence and RGB-D corridor checks.
        """
        args = getattr(self, "args", None)
        mode = getattr(args, "visual_servo_mode", None)
        if mode is None:
            # Compatibility for direct users of the old --visual-servo flag
            # and the small test/embedding argument objects.
            mode = (
                "active"
                if bool(getattr(args, "visual_servo", False))
                else "off"
            )
        if mode not in {"off", "shadow", "active"}:
            raise SafetyAbort(f"未知预抓取视觉闭环模式: {mode!r}")
        if mode == "off":
            return locked
        if mode == "shadow":
            return self._run_pregrasp_visual_shadow(locked)
        if (
            not bool(getattr(args, "execute", False))
            or bool(getattr(args, "plan_only", False))
        ):
            raise SafetyAbort("预抓取视觉闭环只允许在明确的真机执行模式中运行")

        max_corrections_arg = getattr(
            args, "visual_servo_max_corrections", None
        )
        step_mm_arg = getattr(args, "visual_servo_step_mm", None)
        total_mm_arg = getattr(args, "visual_servo_total_mm", None)
        convergence_mm_arg = getattr(
            args, "visual_servo_convergence_mm", None
        )
        max_corrections = int(
            self.params.visual_servo_max_corrections
            if max_corrections_arg is None
            else max_corrections_arg
        )
        max_step_m = float(
            self.params.visual_servo_max_step_m
            if step_mm_arg is None
            else step_mm_arg / 1000.0
        )
        max_total_m = float(
            self.params.visual_servo_max_total_m
            if total_mm_arg is None
            else total_mm_arg / 1000.0
        )
        convergence_m = float(
            self.params.visual_servo_convergence_m
            if convergence_mm_arg is None
            else convergence_mm_arg / 1000.0
        )
        if not (
            1 <= max_corrections <= 3
            and np.isfinite(max_step_m)
            and np.isfinite(max_total_m)
            and np.isfinite(convergence_m)
            and 0.0 < convergence_m <= max_step_m <= max_total_m
        ):
            raise SafetyAbort(
                "预抓取视觉闭环参数无效：要求 corrections=1..3 且 "
                "0 < convergence <= step <= total"
            )

        original = np.asarray(locked.point_base, dtype=float)
        if original.shape != (3,) or not np.all(np.isfinite(original)):
            raise SafetyAbort("预抓取视觉闭环收到无效的锁定目标")
        current = original.copy()
        current_localization = locked
        previous_residual = None
        servo_params = replace(
            self.params,
            samples=self.params.wrist_relocalization_samples,
            max_position_spread_m=(
                self.params.visual_servo_max_position_spread_m
            ),
        )
        self._start_camera("right_wrist")

        for observation_index in range(max_corrections + 1):
            self._abort_if_stopped()
            # ensure_bottle_visible distinguishes a stale/dead RGB-D stream from
            # a detector miss.  Neither may be converted into a motion command.
            self.ensure_bottle_visible(target_base=current)
            measured = self.localize(
                "预抓取视觉闭环复检",
                lambda: self.robot.current_flange()
                @ self.T_flange_wrist_camera,
                servo_params,
                depth_prior_base=current,
            )
            measured_point = np.asarray(measured.point_base, dtype=float)
            if (
                measured_point.shape != (3,)
                or not np.all(np.isfinite(measured_point))
            ):
                raise SafetyAbort("预抓取视觉闭环复检返回无效目标")
            residual_vector = measured_point - current
            residual = float(np.linalg.norm(residual_vector))
            requested_total = float(np.linalg.norm(measured_point - original))
            self.stage(
                "预抓取视觉闭环观测",
                f"第 {observation_index + 1} 次：残差 {residual * 1000:.1f} mm，"
                f"相对初始锁定 {requested_total * 1000:.1f} mm",
            )
            if requested_total > max_total_m + 1e-9:
                raise SafetyAbort(
                    "预抓取视觉闭环目标超过累计修正上限: "
                    f"requested={requested_total * 1000:.1f}mm "
                    f"limit={max_total_m * 1000:.1f}mm"
                )
            if residual <= convergence_m:
                self.stage(
                    "预抓取视觉闭环收敛",
                    f"修正 {observation_index} 次，最终残差 "
                    f"{residual * 1000:.1f} mm；保持当前位置进入最后接近",
                )
                return replace(
                    current_localization,
                    point_base=current.tolist(),
                    box=list(measured.box),
                    confidence=float(measured.confidence),
                    frame_count=int(measured.frame_count),
                    position_spread_m=float(measured.position_spread_m),
                )
            if (
                previous_residual is not None
                and residual
                > previous_residual
                + self.params.visual_servo_divergence_tolerance_m
            ):
                raise SafetyAbort(
                    "预抓取视觉闭环误差发散: "
                    f"{previous_residual * 1000:.1f} -> "
                    f"{residual * 1000:.1f} mm"
                )
            if observation_index >= max_corrections:
                raise SafetyAbort(
                    "预抓取视觉闭环达到最大修正次数仍未收敛: "
                    f"residual={residual * 1000:.1f}mm"
                )

            step_m = min(residual, max_step_m)
            applied = residual_vector * (step_m / residual)
            corrected = current + applied
            if float(np.linalg.norm(corrected - original)) > max_total_m + 1e-9:
                raise SafetyAbort("预抓取视觉闭环累计修正将超过安全包络")

            had_contact_target = hasattr(self, "local_contact_target_base")
            previous_contact_target = getattr(
                self, "local_contact_target_base", None
            )
            self.local_contact_target_base = corrected.copy()
            try:
                # Qualify the whole remaining grasp/lift/place chain for the
                # corrected target before qualifying the small correction leg.
                self.candidate_path(corrected)

                def build_correction_path() -> list[list[float]]:
                    start = np.asarray(self.robot.current_tcp(), dtype=float)
                    if start.shape != (4, 4) or not np.all(np.isfinite(start)):
                        raise SafetyAbort("视觉闭环修正前实时 TCP 无效")
                    end = start.copy()
                    end[:3, 3] += applied
                    path = interpolate_poses(
                        matrix_pose(start),
                        matrix_pose(end),
                        min(self.params.segment_m, max_step_m),
                    )
                    for index, pose in enumerate(path, 1):
                        self.safety.assert_tcp_point(
                            pose[:3], label=f"视觉闭环修正路径点 {index}"
                        )
                    return path

                correction_path = self._plan_local_leg(
                    "预抓取视觉闭环修正",
                    build_correction_path,
                    self.params,
                    # Do not rotate the wrist after measuring image error.  A
                    # roll fallback would invalidate the fresh observation and
                    # turn this translation-only controller into an unmeasured
                    # orientation change.
                    roll_degrees=(0,),
                )
                self.collision_gate(
                    measured.box,
                    corrected,
                    corridor_waypoints_base=[
                        pose[:3] for pose in correction_path
                    ],
                )
            except Exception:
                if had_contact_target:
                    self.local_contact_target_base = previous_contact_target
                else:
                    delattr(self, "local_contact_target_base")
                raise

            self.stage(
                "预抓取视觉闭环修正",
                f"低速平移 {step_m * 1000:.1f} mm；"
                f"单步上限 {max_step_m * 1000:.0f} mm",
            )
            for pose in correction_path:
                self.robot.move_linear(pose, self.params.final_speed)
            current = corrected
            current_localization = replace(
                measured, point_base=current.tolist()
            )
            previous_residual = residual

        raise AssertionError("unreachable visual-servo loop")

    def _run_pregrasp_visual_shadow(
        self, locked: Localization
    ) -> Localization:
        """Observe a proposed bounded correction without changing the motion path.

        Shadow is deliberately informational: a bad shadow frame cannot turn
        the established grasp transaction into a new rejection path, and it
        never calls IK, MoveIt or a robot motion method.  ``active`` remains
        the only mode allowed to alter the pregrasp target.
        """
        original = np.asarray(locked.point_base, dtype=float)
        if original.shape != (3,) or not np.all(np.isfinite(original)):
            self.stage("预抓取视觉闭环影子", "锁定目标无效，跳过影子观测")
            return locked
        try:
            self._abort_if_stopped()
            self._start_camera("right_wrist")
            self.ensure_bottle_visible(target_base=original)
            shadow_params = replace(
                self.params,
                samples=self.params.wrist_relocalization_samples,
                max_position_spread_m=(
                    self.params.visual_servo_max_position_spread_m
                ),
            )
            measured = self.localize(
                "预抓取视觉闭环影子复检",
                lambda: self.robot.current_flange()
                @ self.T_flange_wrist_camera,
                shadow_params,
                depth_prior_base=original,
            )
            measured_point = np.asarray(measured.point_base, dtype=float)
            if (
                measured_point.shape != (3,)
                or not np.all(np.isfinite(measured_point))
            ):
                raise SafetyAbort("影子视觉闭环返回无效目标")
            residual_m = float(np.linalg.norm(measured_point - original))
            proposed_m = min(residual_m, self.params.visual_servo_max_step_m)
            self.stage(
                "预抓取视觉闭环影子",
                f"观测残差 {residual_m * 1000:.1f} mm；"
                f"若启用 active 将申请不超过 {proposed_m * 1000:.1f} mm 的修正；"
                "影子模式未改变目标或下发运动",
            )
        except SafetyAbort as exc:
            stop_event = getattr(self, "stop_event", None)
            if stop_event is not None and stop_event.is_set():
                raise
            # A shadow experiment must have a one-command rollback: use off
            # semantics and retain the original locked target.
            self.stage(
                "预抓取视觉闭环影子", f"影子观测不可用，按 off 回退: {exc}"
            )
        return locked

    def _measure_target_from_head_3d(
        self,
        label: str,
        expected_base: np.ndarray,
    ) -> Localization:
        """Measure, never synthesize, a target associated with ``expected_base``."""
        if self.detector is None:
            raise BottleDetectionLost(f"{label}时固定头部检测器未初始化")
        if getattr(self, "camera_name", "") != "head":
            self._start_camera("head")
        head_params = replace(
            self.params,
            samples=self.params.wrist_relocalization_samples,
            min_depth_m=self.params.head_min_depth_m,
            max_depth_m=self.params.head_max_depth_m,
            max_position_spread_m=0.04,
        )
        return self.localize(
            label,
            lambda: self.T_base_head_camera,
            head_params,
            depth_prior_base=np.asarray(expected_base, dtype=float),
            allow_depth_prior_fallback=False,
            required_consensus_frames=(
                self.params.post_action_confirmation_min_frames
            ),
        )

    def _measure_target_from_wrist_3d(
        self,
        label: str,
        expected_base: np.ndarray,
    ) -> Localization:
        """Measure the held bottle from the camera that rises with the gripper."""
        if self.detector is None:
            raise BottleDetectionLost(f"{label}时腕部检测器未初始化")
        if getattr(self, "camera_name", "") != "right_wrist":
            self._start_camera("right_wrist")
        wrist_params = replace(
            self.params,
            samples=self.params.wrist_relocalization_samples,
            max_position_spread_m=0.025,
        )
        return self.localize(
            label,
            lambda: self.robot.current_flange() @ self.T_flange_wrist_camera,
            wrist_params,
            depth_prior_base=np.asarray(expected_base, dtype=float),
            allow_depth_prior_fallback=False,
            required_consensus_frames=(
                self.params.post_action_confirmation_min_frames
            ),
        )

    def _post_lift_visual_evidence(
        self, locked: Localization
    ) -> LiftVisualEvidence:
        """Collect typed evidence from distinct, fresh wrist RGB-D frames."""

        original = np.asarray(locked.point_base, dtype=float)
        expected = original.copy()
        expected[2] += self.params.lift_m
        if getattr(self, "camera_name", "") != "right_wrist":
            try:
                self._start_camera("right_wrist")
            except CameraFrameUnavailable as exc:
                return LiftVisualEvidence(
                    LiftEvidenceKind.CAMERA_UNAVAILABLE,
                    str(exc),
                    0,
                    0,
                )
        detector = getattr(self, "wrist_detector", None)
        if detector is None:
            return LiftVisualEvidence(
                LiftEvidenceKind.CAMERA_UNAVAILABLE,
                "腕部检测器未初始化",
                0,
                0,
            )
        K, _distortion = self.camera.get_camera_intrinsics()
        if K is None:
            return LiftVisualEvidence(
                LiftEvidenceKind.CAMERA_UNAVAILABLE,
                "腕部相机内参不可用",
                0,
                0,
            )
        required = int(self.params.lift_occlusion_required_frames)
        wrist_params = replace(
            self.params,
            samples=required,
            max_position_spread_m=0.025,
        )
        fresh_frames = 0
        missing_buffers = 0
        depth_usable_frames = 0
        occluded_frames = 0
        measurements: list[Localization] = []
        last_timestamp = float(self.camera.get_frame_timestamp())
        deadline = time.time() + max(6.0, required * 2.0)
        while fresh_frames < required and time.time() < deadline:
            self._abort_if_stopped()
            timestamp = self.camera.get_frame_timestamp()
            if (
                timestamp <= last_timestamp
                or timestamp
                < time.time() - self.params.frame_timeout_s
            ):
                time.sleep(0.03)
                continue
            last_timestamp = timestamp
            fresh_frames += 1
            color, depth = self.camera.get_latest_frames()
            if color is None or depth is None:
                missing_buffers += 1
                continue
            valid_depth = np.isfinite(depth) & (
                depth >= wrist_params.min_depth_m
            ) & (depth <= wrist_params.max_depth_m)
            if int(np.count_nonzero(valid_depth)) < (
                wrist_params.lift_occlusion_roi_min_valid_depth
            ):
                continue
            depth_usable_frames += 1
            T_base_camera = (
                self.robot.current_flange() @ self.T_flange_wrist_camera
            )
            association = PostLiftTargetAssociation.from_view(
                locked=locked,
                expected_base=expected,
                T_base_camera=T_base_camera,
                intrinsics=K,
                image_shape=color.shape,
            )
            detection = detector.detect(
                color,
                association.accepts,
                target_classes=self._target_classes(),
            )
            if detection is not None:
                try:
                    point_camera, z, mad, pixel = depth_point_for_detection(
                        depth,
                        detection,
                        K,
                        wrist_params,
                        measured_vertical=True,
                    )
                except InsufficientDepth:
                    continue
                point_base = (
                    T_base_camera @ np.r_[point_camera, 1.0]
                )[:3]
                measured = Localization(
                    point_camera=point_camera.tolist(),
                    point_base=point_base.tolist(),
                    pixel=list(pixel),
                    depth_m=float(z),
                    depth_mad_m=float(mad),
                    position_spread_m=0.0,
                    box=list(detection.box),
                    confidence=float(detection.confidence),
                    frame_count=1,
                    class_name=detection.class_name,
                )
                delta = point_base - original
                upward = float(delta[2])
                horizontal = float(np.linalg.norm(delta[:2]))
                # A fresh associated detection that is still at the original
                # height is stronger negative evidence than any gripper/TCP
                # fallback and therefore terminates immediately.
                vertically_truncated = bool(
                    detection.box[1] <= 4
                    or detection.box[3] >= color.shape[0] - 1 - 4
                )
                if (
                    upward
                    < self.params.lift_confirmation_min_displacement_m
                    or horizontal
                    > self.params.lift_confirmation_max_horizontal_m
                ):
                    if (
                        vertically_truncated
                        and horizontal
                        <= self.params.lift_confirmation_max_horizontal_m
                    ):
                        # A clipped box provides real depth but not a stable
                        # whole-bottle vertical extent.  It may contribute
                        # neither success nor a one-frame vertical veto.
                        continue
                    return LiftVisualEvidence(
                        LiftEvidenceKind.VISUAL_NEGATIVE,
                        "新鲜腕部视觉看见关联瓶子未按预期上升或水平跳变: "
                        f"up={upward * 1000:.1f}mm "
                        f"horizontal={horizontal * 1000:.1f}mm",
                        fresh_frames,
                        len(measurements) + 1,
                        measured,
                    )
                measurements.append(measured)
                continue

            # This is deliberately an *any-bottle* negative check.  Restricting
            # it to the requested product class would let a visible wrong
            # product masquerade as gripper occlusion and reach the held-object
            # fusion path.  The associated query above remains class-aware;
            # this second query only decides whether a mismatch is evidence of
            # a wrong bottle rather than a detector-free occlusion.
            any_bottle = detector.detect(color, target_classes=None)
            if any_bottle is not None:
                # The requested class is visible but its horizontal bearing or
                # width identifies another bottle.  Never relabel it as
                # gripper occlusion.
                return LiftVisualEvidence(
                    LiftEvidenceKind.VISUAL_NEGATIVE,
                    "新鲜腕部画面只看见水平位置/宽度不匹配的另一瓶",
                    fresh_frames,
                    len(measurements),
                )

            if not (
                np.isfinite(association.projected_u)
                and np.isfinite(association.projected_v)
                and 0 <= association.projected_u < color.shape[1]
                and 0 <= association.projected_v < color.shape[0]
            ):
                continue
            rows, columns = association.occlusion_roi()
            roi = np.asarray(depth[rows, columns], dtype=float)
            valid_mask = (
                np.isfinite(roi)
                & (roi >= wrist_params.min_depth_m)
                & (roi <= wrist_params.max_depth_m)
            )
            valid_roi = roi[valid_mask]
            valid_fraction = (
                float(valid_roi.size) / float(roi.size) if roi.size else 0.0
            )
            if (
                valid_roi.size
                < wrist_params.lift_occlusion_roi_min_valid_depth
                or valid_fraction
                < wrist_params.lift_occlusion_roi_min_valid_fraction
            ):
                continue
            # "Occluded" means a fresh depth image measures a surface in front
            # of the expected bottle projection.  A detector miss, stale frame
            # or all-NaN crop is not occlusion evidence.
            foreground = valid_roi <= (
                association.expected_depth_m
                - wrist_params.lift_occlusion_depth_margin_m
            )
            foreground_fraction = float(np.count_nonzero(foreground)) / float(
                valid_roi.size
            )
            if (
                foreground_fraction
                >= wrist_params.lift_occlusion_roi_min_foreground_fraction
            ):
                occluded_frames += 1

        if fresh_frames < required:
            return LiftVisualEvidence(
                LiftEvidenceKind.CAMERA_UNAVAILABLE,
                f"只收到 {fresh_frames}/{required} 个新鲜腕部帧",
                fresh_frames,
                len(measurements),
            )
        if len(measurements) >= self.params.post_action_confirmation_min_frames:
            points = np.asarray(
                [item.point_base for item in measurements], dtype=float
            )
            center = np.median(points, axis=0)
            spread = float(
                np.max(np.linalg.norm(points - center, axis=1))
            )
            if spread <= wrist_params.max_position_spread_m:
                best = max(measurements, key=lambda item: item.confidence)
                confirmed = replace(
                    best,
                    point_base=center.tolist(),
                    position_spread_m=spread,
                    frame_count=len(measurements),
                )
                return LiftVisualEvidence(
                    LiftEvidenceKind.VISUAL_CONFIRMED,
                    f"{len(measurements)}/{fresh_frames} 帧腕部三维共识",
                    fresh_frames,
                    len(measurements),
                    confirmed,
                )
        if missing_buffers:
            return LiftVisualEvidence(
                LiftEvidenceKind.CAMERA_UNAVAILABLE,
                f"{missing_buffers}/{fresh_frames} 个新鲜时间戳缺少 RGB-D 缓冲",
                fresh_frames,
                len(measurements),
            )
        if (
            fresh_frames == required
            and occluded_frames == required
            and depth_usable_frames == required
        ):
            return LiftVisualEvidence(
                LiftEvidenceKind.OCCLUDED_WITH_FRESH_FRAME,
                f"{occluded_frames}/{required} 个新鲜腕部深度帧确认前景完全遮挡",
                fresh_frames,
                len(measurements),
            )
        return LiftVisualEvidence(
            LiftEvidenceKind.INSUFFICIENT_DEPTH,
            "抬升腕部真实证据不足: "
            f"关联={len(measurements)}/{required}, "
            f"遮挡={occluded_frames}/{required}, "
            f"有效深度帧={depth_usable_frames}/{required}",
            fresh_frames,
            len(measurements),
        )

    def _validate_occluded_lift_fusion(
        self,
        *,
        prelift_tcp: np.ndarray,
        postlift_tcp: np.ndarray,
    ) -> None:
        """Fuse only fresh occlusion, three re-reads and measured TCP motion."""

        states = []
        for _index in range(3):
            states.append(self.robot.gripper_state())
            time.sleep(0.05)
        positions = []
        baseline = getattr(
            self.robot,
            "empty_close_pos",
            self.params.gripper_empty_closed_position,
        )
        threshold = baseline + self.params.gripper_object_margin
        for index, state in enumerate(states, 1):
            try:
                dof_state = int(state["dof_state"][0])
                position = int(state["pos"][0])
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise SafetyAbort(
                    f"遮挡融合第 {index}/3 帧夹爪反馈缺失，拒绝确认 held"
                ) from exc
            if dof_state != 3 or position <= threshold:
                raise SafetyAbort(
                    f"遮挡融合第 {index}/3 帧夹持证据丢失: "
                    f"state={dof_state}, pos={position}, threshold={threshold}"
                )
            positions.append(position)
        if (
            max(positions) - min(positions)
            > self.params.lift_gripper_stability_position_tolerance
        ):
            raise SafetyAbort(
                "遮挡融合三帧夹持位置不稳定: "
                f"positions={positions}"
            )
        before = np.asarray(prelift_tcp, dtype=float)
        after = np.asarray(postlift_tcp, dtype=float)
        if (
            before.shape != (4, 4)
            or after.shape != (4, 4)
            or not np.all(np.isfinite(before))
            or not np.all(np.isfinite(after))
        ):
            raise SafetyAbort("遮挡融合缺少有效的抬升前后实际 TCP")
        delta = after[:3, 3] - before[:3, 3]
        vertical_error = abs(float(delta[2]) - self.params.lift_m)
        horizontal = float(np.linalg.norm(delta[:2]))
        if (
            vertical_error > self.params.lift_tcp_vertical_tolerance_m
            or horizontal > self.params.lift_tcp_max_horizontal_m
        ):
            raise SafetyAbort(
                "遮挡融合实际 TCP 抬升不成立: "
                f"dz={delta[2] * 1000:.1f}mm "
                f"horizontal={horizontal * 1000:.1f}mm"
            )

    def _confirm_lifted_target(
        self,
        locked: Localization,
        *,
        prelift_tcp: np.ndarray,
        postlift_tcp: np.ndarray,
    ) -> Localization | None:
        """Confirm lift through typed wrist evidence, never exception guessing."""

        evidence = self._post_lift_visual_evidence(locked)
        self.last_lift_evidence = evidence
        if evidence.kind is LiftEvidenceKind.VISUAL_CONFIRMED:
            assert evidence.measurement is not None
            self.last_lift_confirmation_camera = "right_wrist"
            self.stage("抬升真实视觉确认通过", evidence.reason)
            return evidence.measurement
        if evidence.kind is LiftEvidenceKind.OCCLUDED_WITH_FRESH_FRAME:
            self._validate_occluded_lift_fusion(
                prelift_tcp=prelift_tcp,
                postlift_tcp=postlift_tcp,
            )
            self.last_lift_confirmation_camera = "occlusion_fusion"
            self.stage(
                "抬升遮挡融合确认通过",
                evidence.reason
                + "；三帧夹持稳定且实际 TCP 完成约 5 cm 垂直抬升",
            )
            return None
        if evidence.kind is LiftEvidenceKind.VISUAL_NEGATIVE:
            raise SafetyAbort("抬升视觉否定优先，拒绝 held: " + evidence.reason)
        if evidence.kind is LiftEvidenceKind.CAMERA_UNAVAILABLE:
            raise CameraFrameUnavailable(
                "抬升腕部相机不可用，不能当作遮挡: " + evidence.reason
            )
        raise SafetyAbort(
            "抬升证据不足（明确进入真实 0/3/深度分支）: "
            + evidence.reason
        )

    def _confirm_point_released(
        self,
        point_base,
        *,
        label: str,
        stage_name: str,
        error_point_description: str,
        stage_point_description: str,
        lifted_point_base=None,
        prefer_wrist: bool = False,
    ) -> str:
        """Confirm a released bottle using a fresh 3-D sample.

        When a lifted observation is available, split the evidence by axis:
        horizontal proximity associates the bottle with the intended place,
        while a meaningful Z drop from the lifted observation proves it came
        down.  Comparing one raw 3-D norm across wrist and head cameras is not
        valid because their boxes can sample different heights on the bottle.
        Without lifted evidence (the output-bin flow), retain the conservative
        full-3-D proximity check.
        """
        target = np.asarray(point_base, dtype=float)
        source = "head"
        if prefer_wrist:
            source = "right_wrist"
            try:
                measured = self._measure_target_from_wrist_3d(
                    label.replace("三维确认", "腕部三维确认"), target
                )
            except (
                BottleDetectionLost,
                CameraFrameUnavailable,
            ) as wrist_error:
                self.stage(
                    "放回腕部确认不可用",
                    f"{wrist_error}；仅在此情况下尝试固定头部辅助确认",
                )
                source = "head"
                measured = self._measure_target_from_head_3d(
                    label.replace("三维确认", "头部辅助确认"), target
                )
        else:
            measured = self._measure_target_from_head_3d(label, target)
        measured_point = np.asarray(measured.point_base, dtype=float)
        camera_label = "右腕" if source == "right_wrist" else "固定头部"
        if lifted_point_base is not None:
            lifted = np.asarray(lifted_point_base, dtype=float)
            if (
                target.shape != (3,)
                or measured_point.shape != (3,)
                or lifted.shape != (3,)
                or not np.all(np.isfinite(target))
                or not np.all(np.isfinite(measured_point))
                or not np.all(np.isfinite(lifted))
            ):
                raise SafetyAbort(f"{label}的锁定/抬升/释放三维点无效")
            horizontal_error = float(
                np.linalg.norm(measured_point[:2] - target[:2])
            )
            downward = float(lifted[2] - measured_point[2])
            if horizontal_error > self.params.release_confirmation_tolerance_m:
                raise SafetyAbort(
                    f"{label}失败: 水平距{error_point_description} "
                    f"{horizontal_error * 1000:.1f}mm，上限 "
                    f"{self.params.release_confirmation_tolerance_m * 1000:.1f}mm"
                )
            if downward < self.params.release_confirmation_min_drop_m:
                raise SafetyAbort(
                    f"{label}失败: 相对抬升观测只下降 "
                    f"{downward * 1000:.1f}mm，下限 "
                    f"{self.params.release_confirmation_min_drop_m * 1000:.1f}mm"
                )
            self.stage(
                stage_name,
                f"{camera_label}实测瓶子水平距{stage_point_description} "
                f"{horizontal_error * 1000:.1f} mm；"
                f"相对抬升观测下降 {downward * 1000:.1f} mm；"
                f"确认相机={source}",
            )
        else:
            error = float(np.linalg.norm(measured_point - target))
            if error > self.params.release_confirmation_tolerance_m:
                raise SafetyAbort(
                    f"{label}失败: "
                    f"距{error_point_description} {error * 1000:.1f}mm，"
                    f"上限 {self.params.release_confirmation_tolerance_m * 1000:.1f}mm"
                )
            self.stage(
                stage_name,
                f"{camera_label}实测瓶子距{stage_point_description} "
                f"{error * 1000:.1f} mm；确认相机={source}",
            )
        return source

    def _confirm_released_target(
        self,
        locked: Localization,
        lifted: Optional[Localization] = None,
    ) -> str:
        """Confirm release from the wrist view; use head only as fallback."""
        return self._confirm_point_released(
            locked.point_base,
            label="放回三维确认",
            stage_name="放回视觉确认",
            error_point_description="锁定点",
            stage_point_description="锁定放置点",
            lifted_point_base=(
                None if lifted is None else lifted.point_base
            ),
            prefer_wrist=True,
        )

    def _fresh_head_target(self) -> Localization:
        """Acquire a head target owned exclusively by the current run."""
        head_params = replace(
            self.params,
            min_depth_m=self.params.head_min_depth_m,
            max_depth_m=self.params.head_max_depth_m,
            max_position_spread_m=0.045,
        )
        target = self.localize(
            "头部粗定位", lambda: self.T_base_head_camera, head_params
        )
        self.safety.assert_tcp_point(
            target.point_base,
            label="头部定位的水瓶抓取点",
        )
        return target

    def _fresh_wrist_target(self, head_target: Localization) -> Localization:
        """Acquire a new wrist association without replacing locked depth/height."""
        self._start_camera("right_wrist")
        return self.localize(
            "右腕精定位",
            lambda: self.robot.current_flange() @ self.T_flange_wrist_camera,
            self.params,
            depth_prior_base=np.asarray(head_target.point_base, dtype=float),
        )

    def _verify_wrist_observation_start(self, target: Localization) -> None:
        """Reject a task started outside the calibrated wrist observation domain."""
        self.robot.assert_arm_healthy()
        current_tcp = self.robot.current_tcp()
        if (
            np.asarray(current_tcp).shape != (4, 4)
            or not np.all(np.isfinite(current_tcp))
        ):
            raise SafetyAbort("右腕观察位实时 TCP 无效")
        self.safety.assert_tcp_point(
            np.asarray(current_tcp)[:3, 3], label="右腕观察位当前 TCP"
        )
        point = np.asarray(target.point_base, dtype=float)
        point_camera = np.asarray(target.point_camera, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise SafetyAbort("右腕观察位目标坐标无效")
        if point_camera.shape != (3,) or not np.all(np.isfinite(point_camera)):
            raise SafetyAbort("右腕观察位相机坐标无效")
        depth = float(point_camera[2])
        minimum = self.params.pregrasp_standoff_m + 0.025
        maximum = min(self.params.max_depth_m, 0.55)
        if not minimum <= depth <= maximum:
            raise SafetyAbort(
                "当前右臂不在可用观察位：锁定目标的腕部纵深 "
                f"{depth:.3f} m，不在 [{minimum:.3f}, {maximum:.3f}] m"
            )
        self.safety.assert_tcp_point(point, label="右腕锁定的水瓶抓取点")
        self.local_contact_target_base = point.copy()
        # This is a no-motion IK/fence gate.  It proves the current physical
        # pose can enter the shared pregrasp recipe before any gripper action;
        # task mode additionally validates the exact IK chain in MoveIt.
        self.candidate_path(point)
        self.stage(
            "右腕观察位验证",
            f"新鲜目标纵深 {depth:.3f} m，完整抓放路径逆解/围栏/全链碰撞通过",
        )

    def _verify_wrist_pregrasp_start(self, target: Localization) -> None:
        """Accept only the already-reached 8.5 cm hover after a safe abort."""
        self.robot.assert_arm_healthy()
        current_tcp = np.asarray(self.robot.current_tcp(), dtype=float)
        point = np.asarray(target.point_base, dtype=float)
        if (
            current_tcp.shape != (4, 4)
            or point.shape != (3,)
            or not np.all(np.isfinite(current_tcp))
            or not np.all(np.isfinite(point))
        ):
            raise SafetyAbort("预抓取续跑收到无效的实时 TCP 或目标坐标")
        self.safety.assert_tcp_point(
            current_tcp[:3, 3], label="预抓取续跑当前 TCP"
        )
        self.safety.assert_tcp_point(point, label="预抓取续跑水瓶抓取点")
        distance = float(np.linalg.norm(current_tcp[:3, 3] - point))
        tolerance = 0.015
        if abs(distance - self.params.pregrasp_standoff_m) > tolerance:
            raise SafetyAbort(
                "当前右臂不在预抓取悬停位："
                f"TCP 距新鲜目标 {distance:.3f} m，"
                f"期望 {self.params.pregrasp_standoff_m:.3f}±{tolerance:.3f} m"
            )
        self.local_contact_target_base = point.copy()
        # No motion: qualify the complete remaining grasp/lift/place chain in
        # the fresh scene before the resumed entry may touch the gripper.
        self.candidate_path(point)
        self.stage(
            "预抓取悬停位验证",
            f"TCP 距新鲜目标 {distance:.3f} m；剩余抓放路径全链复核通过",
        )

    def run(self):
        task_mode = getattr(self.args, "task_mode", None)
        if task_mode:
            from .task import BottlePickPlaceTask, DeliverMode, StartMode

            deliver_mode = (
                DeliverMode.DISPENSE
                if getattr(self.args, "dispense", False)
                else DeliverMode.PLACE_BACK
            )
            return BottlePickPlaceTask(self).run(
                StartMode(task_mode), deliver_mode
            )

        self.initialize()
        self._preflight()
        if getattr(self.args, "finish_from_current", False):
            return self._finish_from_current()
        if self.args.resume_at_wrist:
            prior = self._load_resume_localization()
            self.safety.assert_tcp_point(
                prior.point_base,
                label="续抓保存的水瓶抓取点",
            )
            self.head_scene_voxels = []
            self.non_target_scene_voxels = []
            self.target_occupancy_voxels = []
            self.scene_voxels = []
            # 首次续抓定位必须全靠实测深度（不传先验），保证位置是独立
            # 测量：桌子/瓶子可能在两次运行之间被挪动过，陈旧先验会污染
            # 绝对位置。保存的先验只用作跳变门槛的参照。
            wrist_target = self.localize(
                "右腕续抓定位",
                lambda: self.robot.current_flange()
                @ self.T_flange_wrist_camera,
                self.params,
            )
            jump = float(
                np.linalg.norm(
                    np.asarray(wrist_target.point_base)
                    - np.asarray(prior.point_base)
                )
            )
            if jump > self.params.resume_prior_jump_m:
                raise SafetyAbort(
                    f"续抓目标相对保存位置跳变 {jump * 1000:.1f} mm"
                )
            if jump > self.params.max_relocalization_jump_m:
                LOG.warning(
                    "续抓目标相对保存先验偏移 %.1f mm"
                    "（桌子/瓶子可能被挪动过），以本次腕部定位为准",
                    jump * 1000,
                )
            if self.args.stop_after_observation:
                self.stage("续抓视觉确认完成", "本轮不发送运动命令")
                time.sleep(self.args.observe_seconds)
                return
            return self._finish_grasp_from_wrist(wrist_target)

        head_target = self._fresh_head_target()
        if not (self.args.plan_only or self.args.execute):
            self.stage("头部观察完成", "已自主发现水瓶；未连接或移动机械臂")
            time.sleep(self.args.observe_seconds)
            return

        self._build_head_scene(head_target)
        observation_plan = self._plan_observation(
            np.asarray(head_target.point_base)
        )
        if self.args.plan_only:
            self.stage(
                "自主规划完成",
                f"观察位轨迹 {len(observation_plan['points_deg'])} 点；未执行任何运动",
            )
            time.sleep(self.args.observe_seconds)
            return

        self._execute_plan("避障移动到右腕观察位", observation_plan)
        wrist_target = self._fresh_wrist_target(head_target)
        if self.args.stop_after_observation:
            self.stage(
                "观察位测试完成",
                (
                    f"右腕已检测水瓶，深度 {wrist_target.depth_m:.3f} m；"
                    "本轮不抓取"
                ),
            )
            time.sleep(self.args.observe_seconds)
            return
        if getattr(self.args, "confirm_before_grasp", False):
            self._wait_for_grasp_confirmation()
        self._finish_grasp_from_wrist(wrist_target)

    def _wait_for_grasp_confirmation(self):
        """观察位人工确认关卡：到位、检测到瓶子后暂停，Enter 继续/STOP 中止。

        跟"跑完 observe 停住、再另开一次 cycle"不同，这一步不重启进程——
        相机、YOLO 模型、MoveIt 只启动一次（这些初始化合计约 30-40 秒），
        操作者只是在同一次运行里对着已经到位的真实姿态按 Enter。这也避免
        了 2026-07-18 那次"observe 停住后接续抓脚本，走的是另一条跳过头部
        相机/抓取预检的代码路径"——确认后继续走的是同一个 _finish_grasp_
        from_wrist，跟 grasp/cycle 完全同一条路。
        """
        self.stage(
            "等待人工确认",
            "已到观察位并检测到水瓶；确认无误后按 Enter 继续抓取，Ctrl+C/STOP 中止",
        )
        confirmed = threading.Event()
        outcome: dict[str, str | None] = {"error": None}

        def _read_confirmation():
            try:
                line = sys.stdin.readline()
                if line == "":
                    outcome["error"] = "确认终端已关闭（EOF），拒绝自动继续抓取"
                elif line not in {"\n", "\r\n"}:
                    outcome["error"] = "确认只接受空 Enter；收到其他输入，拒绝继续抓取"
            except Exception as exc:
                outcome["error"] = f"读取人工确认失败，拒绝继续抓取: {exc}"
            confirmed.set()

        threading.Thread(target=_read_confirmation, daemon=True).start()
        while not confirmed.is_set():
            if self.stop_event.wait(0.2):
                raise SafetyAbort("确认抓取前收到停止请求")
        if outcome["error"] is not None:
            raise SafetyAbort(outcome["error"])
        self._abort_if_stopped()
        self.stage("人工确认通过", "继续执行抓取")

    def _finish_grasp_from_wrist(self, wrist_target: Localization):
        """续抓/普通模式收尾：抓取抬升后按 --place-back/--return-home 决定后续动作。"""
        lifted_target = self._grasp_and_lift(wrist_target)
        if getattr(self.args, "place_back", False):
            self._place_back(wrist_target, lifted_target)
            if getattr(self.args, "return_home", False):
                self._return_home()
                self.stage("完成并保持", "已放回并返回初始姿态；STOP/Ctrl+C 结束")
            else:
                self.stage("完成并保持", "已放回；STOP/Ctrl+C 结束")
        else:
            self.stage("完成并保持", "不搬运、不放置；STOP/Ctrl+C 只保持")
        while not self.stop_event.wait(0.5):
            pass
        if getattr(self.args, "restore_teleop", False):
            self._restore_teleop()

    def _finish_from_current(self):
        """从当前姿态直接收尾：假设夹爪已抓着水瓶（上一轮运行遗留、保持在原地），
        跳过头部/腕部定位与抓取，只做放回（可选）+返回初始姿态（可选）。
        """
        self.stage(
            "从当前姿态收尾",
            "假设夹爪已抓稳水瓶，跳过定位/抓取，直接进入放回/返回流程",
        )
        if getattr(self.args, "place_back", False):
            self._place_back()
        if getattr(self.args, "return_home", False):
            self._return_home()
        self.stage("完成", "STOP/Ctrl+C 结束")
        while not self.stop_event.wait(0.5):
            pass
        if getattr(self.args, "restore_teleop", False):
            self._restore_teleop()

    def _grasp_and_lift(
        self,
        wrist_target: Localization,
        *,
        already_at_pregrasp: bool = False,
    ) -> Localization | None:
        """从当前腕部姿态完成：直线接近 → 最后接近 → 力控夹取 → 抬升 5cm。

        返回腕部优先的抬升实测；不做放回、不阻塞——后续释放确认用它
        与放回后的新鲜腕部实测做方向性对比（腕部不可用时才回退头部）。
        """
        visual_mode = getattr(
            getattr(self, "args", None), "visual_servo_mode", None
        )
        if visual_mode is None:
            visual_mode = (
                "active"
                if bool(getattr(getattr(self, "args", None), "visual_servo", False))
                else "off"
            )
        servo_note = {
            "active": "受限预抓取视觉闭环已启用",
            "shadow": "预抓取视觉闭环影子观测（不改变运动路径）",
        }.get(visual_mode, "预抓取视觉闭环关闭（保持原路径）")
        self.stage(
            "从当前腕部姿态续抓",
            f"{servo_note}；电子围栏校验和直线分段接近",
        )
        # 观察位夹爪前方是自由空间：先实测今天的空夹闭合基线，抓取判定
        # 不再依赖写死常量（2026-07-15 常量阈值把真实成功误判成空夹）。
        self.stage("夹爪空夹标定", "自由空间闭合一次，实测空夹基线")
        baseline = self.robot.calibrate_empty_close(self.params)
        self.stage("打开夹爪", f"空夹基线实测 pos={baseline}")

        if already_at_pregrasp:
            self.stage(
                "预抓取续跑",
                "当前 TCP 已通过 8.5 cm 悬停位复核，跳过已完成的局部转移",
            )
        else:
            self._approach_pregrasp(wrist_target)

        wrist_target = self._run_pregrasp_visual_servo(wrist_target)

        # Presence only: the 3-D target remains the head-locked/wrist-refined
        # estimate from observation distance.  A clipped near view has no
        # authority to rewrite depth or physical grasp height.
        confirmation = self._confirm_target_at_pregrasp(wrist_target)
        current_wrist_box = (
            confirmation.detection.box
            if confirmation.detection is not None
            else None
        )
        refined = wrist_target
        _, final_grasp, _ = self.candidate_path(
            np.asarray(wrist_target.point_base)
        )
        final_path = interpolate_poses(
            matrix_pose(self.robot.current_tcp()),
            final_grasp,
            self.params.segment_m,
        )
        self.robot.plan_ik(final_path, self.params)
        self.collision_gate(
            current_wrist_box,
            np.asarray(wrist_target.point_base),
        )
        self.stage(
            "低速最后接近",
            f"速度 {self.params.final_speed}%；相对视觉目标提前停止 "
            f"{self.params.grasp_stop_short_m * 100:.0f} cm",
        )
        for pose in final_path:
            # 最后10cm夹爪手指必然逐渐挡住瓶子，检测丢失是预期现象：
            # 画面中断仍然致命，检测丢失降级为警告（目标已锁定+人守急停）。
            try:
                self.ensure_bottle_visible(
                    target_base=np.asarray(wrist_target.point_base)
                )
            except BottleDetectionLost as exc:
                LOG.warning("最后接近中检测丢失（预期为夹爪遮挡）: %s", exc)
            self.robot.move_linear(pose, self.params.final_speed)

        self.stage("夹紧水瓶")
        gripper = self.robot.close_gripper(self.params)
        prelift_tcp = np.asarray(self.robot.current_tcp(), dtype=float)
        # The gripper has supplied object-contact evidence, but lift is not yet
        # proven.  Attach conservatively now so the lift validation itself and
        # every later carrying request include held_bottle_guard.
        self._set_held_bottle_guard(wrist_target)
        self.stage(
            "provisional held guard",
            "闭夹证据通过；lift 前将 held_bottle_guard 附着到 r_link7",
        )

        def build_lift_path() -> list[list[float]]:
            lift = self.robot.current_tcp()
            lift[2, 3] += self.params.lift_m
            return interpolate_poses(
                matrix_pose(self.robot.current_tcp()),
                matrix_pose(lift),
                self.params.segment_m,
            )

        lift_path = self._plan_local_leg(
            "抬升",
            build_lift_path,
            self.params,
            roll_degrees=(0,),
        )
        self.stage("抬升 5 cm", "抓取后保持")
        for pose in lift_path:
            self.robot.move_linear(pose, self.params.final_speed)
        postlift_tcp = np.asarray(self.robot.current_tcp(), dtype=float)
        lifted_measurement = self._confirm_lifted_target(
            wrist_target,
            prelift_tcp=prelift_tcp,
            postlift_tcp=postlift_tcp,
        )
        (self.run_dir / "grasp_lift.json").write_text(
            json.dumps(
                {
                    "final_tcp": matrix_pose(postlift_tcp),
                    "target": refined.point_base,
                    "lift_measurement": (
                        None
                        if lifted_measurement is None
                        else lifted_measurement.point_base
                    ),
                    "lift_evidence": (
                        None
                        if not hasattr(self, "last_lift_evidence")
                        else {
                            **asdict(self.last_lift_evidence),
                            "kind": self.last_lift_evidence.kind.value,
                        }
                    ),
                    "lift_confirmation_camera": getattr(
                        self, "last_lift_confirmation_camera", "unknown"
                    ),
                    "gripper": gripper,
                },
                indent=2,
            )
        )
        return lifted_measurement

    def _grasp_and_lift_from_pregrasp(
        self, wrist_target: Localization
    ) -> Localization | None:
        """Resume the shared contact tail without replaying the completed hover."""
        return self._grasp_and_lift(
            wrist_target, already_at_pregrasp=True
        )

    def _place_back(
        self,
        locked_target: Optional[Localization] = None,
        lifted_target: Optional[Localization] = None,
    ):
        """把瓶子放回原位：放低→张开→退开→确认释放→空载收拢。"""
        def build_lower_path() -> list[list[float]]:
            lower = self.robot.current_tcp()
            lower[2, 3] -= self.params.lift_m
            return interpolate_poses(
                matrix_pose(self.robot.current_tcp()),
                matrix_pose(lower),
                self.params.segment_m,
            )

        lower_path = self._plan_local_leg("放低", build_lower_path, self.params)
        self.stage("放回桌面", f"下降 {self.params.lift_m * 100:.0f} cm")
        for pose in lower_path:
            self.robot.move_linear(pose, self.params.final_speed)

        self.stage("松开夹爪")
        self.robot.open_gripper(self.params)

        # 沿抓取接近轴反向退开独立配置的释放后距离，避免手指刮倒瓶子。
        # 这里不能复用 pregrasp_standoff：观察悬停和释放后净空是两个需求。
        def build_retreat_path() -> list[list[float]]:
            tcp = self.robot.current_tcp()
            axis = tcp[:3, 2]
            retreat = tcp.copy()
            retreat[:3, 3] -= axis * self.params.retreat_standoff_m
            self.safety.assert_tcp_point(retreat[:3, 3], label="放回后退开点")
            return interpolate_poses(
                matrix_pose(tcp),
                matrix_pose(retreat),
                self.params.segment_m,
            )

        retreat_path = self._plan_local_leg(
            "退开", build_retreat_path, self.params
        )
        self.stage(
            "退开",
            f"沿接近轴反向 {self.params.retreat_standoff_m * 100:.0f} cm",
        )
        for pose in retreat_path:
            self.robot.move_linear(pose, self.params.travel_speed)
        if locked_target is not None:
            self._confirm_released_target(locked_target, lifted_target)
            self._detach_held_bottle_guard()
        self.stage("收拢夹爪", "手臂已退开，空载闭合夹爪")
        self.robot.close_empty_gripper(self.params)
        self.stage("放回完成", "瓶子已放回，手臂已退开，夹爪已收拢")

    def _return_home(self):
        """MoveIt 规划返回 profile 里配置的初始/垂下姿态（关节空间目标）。

        用的是跟"移动到观察位"完全相同的 SafeMotionPlanner：MoveIt 规划、
        MoveIt 密集状态后验碰撞复核、电子围栏密集 TCP 复核，以及失败后的
        有限自动换路。风险等级跟去程一致，不是另一套旁路实现。
        """
        home = self.safety.home_joints_deg
        if not home:
            raise SafetyAbort(
                f"profile {self.safety.name} 未配置 home_joints_deg，无法自动返回初始姿态"
            )
        error = self._move_right_arm_to_taught_joints(
            home, plan_name="moveit_return_home", label="返回初始姿态"
        )
        self.stage("已返回初始姿态", f"距目标关节角最大偏差 {error:.2f}°")

    def _move_right_arm_to_taught_joints(
        self, joints, *, plan_name: str, label: str
    ) -> float:
        """Plan and run the right arm to one taught joint target.

        Shared by every taught-pose move so none of them becomes a direct
        joint-motion bypass: the same MoveIt plan, dense post-plan state
        collision recheck, dense electronic-fence TCP recheck and bounded
        replanning that the observation transfer uses.
        """
        target_flange = self.robot.controller_flange_from_joints(list(joints))
        plan = self._plan_flange(
            plan_name, target_flange, goal_joints=list(joints)
        )
        self._execute_plan(label, plan)
        # Blended execution returns once the controller accepts the last
        # command, not once the arm has settled on it.  Reading immediately
        # reported 0.90 deg against a 0.80 deg tolerance for a move that
        # settled 0.18 deg from its goal.
        target = np.asarray(joints, dtype=float)
        deadline = time.monotonic() + ARRIVAL_SETTLE_TIMEOUT_S
        while True:
            error = float(
                np.max(np.abs(np.asarray(self.robot.joints_deg(), dtype=float) - target))
            )
            if (
                error <= self.params.planned_start_tolerance_deg
                or time.monotonic() >= deadline
            ):
                return error
            time.sleep(0.05)

    def normalize_to_grasp_start(self) -> float:
        """Put the right arm on the taught shelf-pick admission posture.

        Every shelf pick is planned from wherever the arm happens to be, so an
        un-normalized start makes each run's transit leg a different,
        unvalidated path -- and the executor refuses the run outright, since
        the profile freezes this posture as an admission gate.  Reaching it is
        therefore a step of the task, not something an operator does by hand.

        This moves the right arm only.  The left arm is captured live into
        every plan's collision scene and separately required not to drift
        during execution, which is what actually makes the plan sound.
        """
        target = self.safety.grasp_start_right_joints_deg
        if not target:
            raise SafetyAbort(
                f"profile {self.safety.name} 未配置 "
                "grasp_start_right_joints_deg，无法归位到抓取起点"
            )
        current = np.asarray(self.robot.joints_deg(), dtype=float)
        expected = np.asarray(target, dtype=float)
        if current.shape != (7,) or expected.shape != (7,):
            raise SafetyAbort("右臂当前关节或抓取起点关节维度不是 7")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(expected)):
            raise SafetyAbort("右臂当前关节或抓取起点关节包含非有限数")
        error = float(np.max(np.abs(current - expected)))
        tolerance = float(self.params.planned_start_tolerance_deg)
        if error <= tolerance:
            self.stage(
                "抓取起点检查", f"已在示教抓取起点，最大关节偏差 {error:.2f}°"
            )
            return error
        self.stage(
            "抓取起点归位",
            f"当前不在示教抓取起点（最大关节偏差 {error:.2f}°），MoveIt 安全归位",
        )
        error = self._move_right_arm_to_taught_joints(
            target,
            plan_name="moveit_grasp_start",
            label="归位到抓取起点",
        )
        if error > tolerance:
            raise SafetyAbort(
                f"归位后仍偏离示教抓取起点 {error:.2f}° > {tolerance:.2f}°"
            )
        self.stage("已到抓取起点", f"距目标关节角最大偏差 {error:.2f}°")
        return error

    def _normalize_start_home(self) -> bool:
        """Put a ``from-start`` run on the taught, repeatable home manifold.

        The caller has already completed motion preflight, closed the known
        empty gripper envelope, captured the left-arm guard, and built a fresh
        head RGB-D scene.  Reuse the exact same MoveIt/fence/full-arm planning
        contract as the end-of-task return instead of adding a direct joint
        motion bypass.  ``True`` tells the task state machine that the camera
        target and scene must be reacquired after the arm moved.
        """
        home = self.safety.home_joints_deg
        if not home:
            raise SafetyAbort(
                f"profile {self.safety.name} 未配置 home_joints_deg，无法规范起点"
            )
        current = np.asarray(self.robot.joints_deg(), dtype=float)
        target = np.asarray(home, dtype=float)
        if current.shape != (7,) or target.shape != (7,):
            raise SafetyAbort("右臂当前关节或 home 关节维度不是 7")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(target)):
            raise SafetyAbort("右臂当前关节或 home 关节包含非有限数")
        error = float(np.max(np.abs(current - target)))
        tolerance = float(self.params.planned_start_tolerance_deg)
        if error <= tolerance:
            self.stage(
                "起点 home 检查",
                f"已在示教 home，最大关节偏差 {error:.2f}°，无需移动",
            )
            return False
        self.stage(
            "起点 home 归位",
            (
                f"当前不在示教 home（最大关节偏差 {error:.2f}°）；"
                "空爪预检已完成，使用 MoveIt 安全归位"
            ),
        )
        self._return_home()
        return True

    def _preflight_side_table_delivery(
        self, *, start: BodySnapshot | None = None
    ) -> BodySnapshot:
        """Verify that the already-captured body admission may be consumed.

        Calling a fresh body preflight here would be too late: this method is
        reached after arm initialization.  The immutable SHELF_READY snapshot
        must instead have been captured by the task before that initialization.
        """
        if self.delivery_safety is None or self.mobile_body is None:
            raise SafetyAbort("桌面送货模块未初始化")
        config = self.delivery_safety.side_table_delivery
        if config is None:
            raise SafetyAbort("桌面送货 profile 缺少 side_table_delivery")
        self._validate_side_table_profile_pair()
        snapshot = (
            start
            if start is not None
            else self.shelf_ready_body_snapshot
        )
        if not isinstance(snapshot, BodySnapshot):
            raise SafetyAbort(
                "桌面送货必须先完成并保存 SHELF_READY BodySnapshot"
            )
        if snapshot is not self.shelf_ready_body_snapshot:
            raise SafetyAbort("桌面送货拒绝替换已捕获的 SHELF_READY BodySnapshot")
        self._delivery_head_extrinsic = self.T_base_head_camera.copy()
        self.stage(
            "桌面送货只读预检",
            (
                f"底盘 {snapshot.chassis.control_mode}/{snapshot.chassis.robot_state}，"
                f"升降 {snapshot.lift.height_mm} mm；已使用 pre-arm SHELF_READY 快照"
            ),
        )
        return snapshot

    def _right_arm_at_delivery_home(self) -> bool:
        """Return whether the empty right arm is at the shared compact/home pose."""
        if self.robot is None or self.delivery_safety is None:
            return False
        home = np.asarray(self.delivery_safety.home_joints_deg, dtype=float)
        current = np.asarray(self.robot.joints_deg(), dtype=float)
        if (
            home.shape != (7,)
            or current.shape != (7,)
            or not np.all(np.isfinite(home))
            or not np.all(np.isfinite(current))
        ):
            return False
        return bool(
            np.max(np.abs(current - home))
            <= float(self.params.planned_start_tolerance_deg)
        )

    def _return_body_to_shelf_ready(
        self,
        *,
        start: BodySnapshot,
        authorization: ReturnAuthorization,
    ) -> BodySnapshot:
        """Return the stationary body only after task state authorizes it."""
        if self.mobile_body is None or self.delivery_safety is None:
            raise SafetyAbort("桌面送货 body 模块未初始化")
        config = self.delivery_safety.side_table_delivery
        if config is None:
            raise SafetyAbort("桌面送货 profile 缺少 side_table_delivery")
        if start is not self.shelf_ready_body_snapshot:
            raise SafetyAbort("返程必须使用本任务的 SHELF_READY BodySnapshot")
        restored = self.mobile_body.return_to_shelf_ready(
            config, start=start, authorization=authorization
        )
        if self.source_safety is None:
            raise SafetyAbort("返程后缺少源货架 profile，拒绝继续")
        # Only switch the arm fence back after physical pose/lift restoration
        # has succeeded.  No arm command is issued during this configuration
        # transition.
        self.safety = self.source_safety
        self.scene_boxes = self.safety.moveit_collision_boxes()
        self._base_pose_for_scene = np.eye(4)
        self.stage("SHELF_READY 已恢复", "底盘/升降已回到捕获快照；恢复源货架围栏")
        return restored

    def _set_held_bottle_guard(self, locked_target: Localization) -> None:
        """Provisional attach after gripper evidence and before any lift plan."""

        config = getattr(
            getattr(self, "delivery_safety", None),
            "side_table_delivery",
            None,
        )
        target = np.asarray(locked_target.point_base, dtype=float)
        current_tcp = np.asarray(self.robot.current_tcp(), dtype=float)
        if target.shape != (3,) or current_tcp.shape != (4, 4):
            raise SafetyAbort("携瓶碰撞包络缺少有效的实时目标/TCP")
        height = float(
            config.held_bottle_height_m
            if config is not None
            else self.params.held_bottle_height_m
        )
        diameter = float(
            config.held_bottle_diameter_m
            if config is not None
            else self.params.held_bottle_diameter_m
        )
        padding = float(
            config.held_bottle_guard_padding_m
            if config is not None
            else self.params.held_bottle_guard_padding_m
        )
        if config is not None:
            bottom_below_tcp = float(config.bottle_bottom_below_tcp_m)
        else:
            # The authored grasp point lies ``grasp_height_fraction`` down
            # from the bottle top.  Cover the entire lower remainder rather
            # than the historical 4 cm wrist-only stub.
            bottom_below_tcp = max(
                float(self.params.bottle_bottom_below_tcp_m),
                height * (1.0 - float(self.params.grasp_height_fraction)),
            )
        T_base_bottle = np.eye(4)
        T_base_bottle[:2, 3] = target[:2]
        T_base_bottle[2, 3] = (
            current_tcp[2, 3]
            + height / 2.0
            - bottom_below_tcp
        )
        # Use the same full calibrated link7→TCP chain as SDK TCP setup and
        # MoveIt target conversion. A pure +Z fallback is offline-only.
        T_base_link7 = current_tcp @ np.linalg.inv(self.T_link7_tcp)
        T_link7_bottle = np.linalg.inv(T_base_link7) @ T_base_bottle
        self.held_object_guard = {
            "size": [diameter + padding, diameter + padding, height + padding],
            "center": T_link7_bottle[:3, 3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(
                T_link7_bottle[:3, :3]
            ).as_quat().tolist(),
        }
        self.held_locked_target = locked_target
        (self.run_dir / "held_bottle_guard.json").write_text(
            json.dumps(self.held_object_guard, indent=2), encoding="utf-8"
        )

    def _scene_without_held_guard(
        self, voxels: Sequence[Sequence[float]]
    ) -> list[list[float]]:
        """Remove only cells intersecting the attached guard's measured box."""

        guard = getattr(self, "held_object_guard", None)
        if not guard:
            raise SafetyAbort("携瓶场景刷新缺少 provisional held guard")
        points = np.asarray(voxels, dtype=float)
        if points.size == 0:
            return []
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise SafetyAbort("携瓶场景体素必须是有限 Nx3")
        T_base_link7 = np.asarray(
            self.robot.current_tcp(), dtype=float
        ) @ np.linalg.inv(self.T_link7_tcp)
        local = (
            np.linalg.inv(T_base_link7)
            @ np.column_stack((points, np.ones(len(points)))).T
        ).T[:, :3]
        center = np.asarray(guard["center"], dtype=float)
        size = np.asarray(guard["size"], dtype=float)
        rotation = Rotation.from_quat(
            np.asarray(guard["quaternion_xyzw"], dtype=float)
        ).as_matrix()
        box_points = (rotation.T @ (local - center).T).T
        padding = self.params.scene_voxel_m / 2.0
        inside = np.all(np.abs(box_points) <= size / 2.0 + padding, axis=1)
        return points[~inside].tolist()

    def _detach_held_bottle_guard(self) -> None:
        """Detach only after confirmed release; failure keeps UNKNOWN guarded."""

        if getattr(self, "held_object_guard", None) is None:
            if getattr(getattr(self, "args", None), "task_mode", None):
                raise SafetyAbort(
                    "释放确认后缺少 held_bottle_guard，无法验证 detach"
                )
            # Legacy diagnostics predate held-object attachment.  They retain
            # their no-attachment behaviour; all supported task modes require
            # and verify the explicit REMOVE path.
            return
        if self.planner is None:
            raise SafetyAbort("释放确认后 MoveIt 不可用；保留 held guard")
        self.planner.detach_attached_object(
            name="held_bottle_guard",
            object_id="held_bottle_guard",
            planning_frame=self.safety.moveit_frame,
        )
        self.held_object_guard = None
        self.stage(
            "held object detach",
            "已发送显式 REMOVE，并读回 live attached IDs 确认 guard 消失",
        )

    def _refresh_held_scene(
        self, lifted_target: Localization | None
    ) -> Localization:
        """Refresh environment depth without demanding a visible held bottle."""

        locked = getattr(self, "held_locked_target", None)
        if locked is None:
            raise SafetyAbort("携瓶场景刷新缺少抓前锁定目标")
        pre_environment = self._scene_without_locked_target(
            np.asarray(locked.point_base, dtype=float)
        )
        before_frame = getattr(
            self, "_head_scene_reference_frame", self.safety.frame
        )
        before_pose = np.asarray(
            getattr(self, "_head_scene_base_pose", np.eye(4)), dtype=float
        )
        if self.camera_name != "head":
            self._start_camera("head")
        K, _distortion = self.camera.get_camera_intrinsics()
        if K is None:
            raise SafetyAbort("携瓶场景刷新缺少头部相机内参")
        captured = time.monotonic()
        depth_frames = self._collect_fresh_depth_frames(
            self.params.scene_samples,
            label="携瓶环境深度刷新",
        )
        per_frame = []
        for depth in depth_frames:
            points = head_scene_points(
                depth,
                K,
                self.T_base_head_camera,
                self.params,
                min_depth_m=self.params.head_min_depth_m,
                max_depth_m=self.params.head_max_depth_m,
                bottom_crop=self.params.scene_image_bottom_crop,
            )
            voxels = voxelize_scene_points(
                points,
                self.params,
                center_base=np.asarray(locked.point_base, dtype=float),
            )
            per_frame.append(self._scene_without_held_guard(voxels))
        post_environment = union_scene_voxels(per_frame, self.params)
        after_pose = np.asarray(
            getattr(self, "_base_pose_for_scene", np.eye(4)), dtype=float
        )
        self.scene_voxels = conservative_scene_union(
            pre_environment,
            post_environment,
            self.params,
            before_frame=before_frame,
            after_frame=self.safety.frame,
            before_base_pose=before_pose,
            after_base_pose=after_pose,
        )
        self.head_scene_voxels = list(self.scene_voxels)
        self.head_scene_captured_monotonic = captured
        self.stage(
            "携瓶场景刷新",
            "头部仅刷新环境深度；抓前/抓后同一 base pose 取保守并集，"
            "不要求固定头部看见空中瓶子",
        )
        return lifted_target if lifted_target is not None else locked

    def _move_held_to_transport_posture(
        self, lifted_target: Localization
    ) -> Localization:
        measured = self._refresh_held_scene(lifted_target)
        config = self.delivery_safety.side_table_delivery
        joints = list(config.transport_joints_deg)
        flange = self.robot.controller_flange_from_joints(joints)
        plan = self._plan_flange(
            "moveit_held_transport", flange, goal_joints=joints
        )
        self._execute_plan("携瓶移动到升降/旋转运输姿态", plan)
        self.stage(
            "携瓶运输姿态到位",
            "右臂已收进现场示教包络，瓶体附着包络仍参与碰撞检查",
        )
        return measured

    def _capture_output_table_scene(
        self, *, require_place_candidate: bool = True
    ) -> OutputTableObservation:
        """Rebuild table/world geometry in the *post-turn live base frame*."""
        if self.camera_name != "head":
            self._start_camera("head")
        reference = getattr(self, "_delivery_head_extrinsic", None)
        if reference is None or not np.allclose(
            reference, self.T_base_head_camera, atol=1e-12, rtol=0.0
        ):
            raise SafetyAbort("头部相机外参在升降前后被改写；拒绝重复补偿")
        config = self.delivery_safety.side_table_delivery
        K, _ = self.camera.get_camera_intrinsics()
        if K is None:
            raise SafetyAbort("转向后头部相机内参不可用")
        captured = time.monotonic()
        depth_frames = self._collect_fresh_depth_frames(
            self.params.scene_samples, label="转向后输出桌面场景"
        )
        point_frames = [
            head_scene_points(
                depth,
                K,
                self.T_base_head_camera,
                self.params,
                min_depth_m=self.params.head_min_depth_m,
                max_depth_m=self.params.head_max_depth_m,
                bottom_crop=self.params.scene_image_bottom_crop,
            )
            for depth in depth_frames
        ]
        observation = observe_output_table(
            point_frames,
            config,
            require_candidates=require_place_candidate,
        )
        roi_center = (
            np.asarray(config.table_roi_min)
            + np.asarray(config.table_roi_max)
        ) / 2.0
        per_frame_voxels = [
            voxelize_scene_points(
                points,
                self.params,
                center_base=roi_center,
            )
            for points in point_frames
        ]
        self.scene_voxels = union_scene_voxels(
            per_frame_voxels, self.params
        )
        # Replace any authored output-table box with the live measured top.
        dynamic_table = FenceBox(
            id="output_table_live",
            minimum=(
                float(config.table_roi_min[0]),
                float(config.table_roi_min[1]),
                float(self.delivery_safety.tcp_workspace.minimum[2]),
            ),
            maximum=(
                float(config.table_roi_max[0]),
                float(config.table_roi_max[1]),
                float(observation.table_height_m),
            ),
        )
        keepouts = tuple(
            item
            for item in self.delivery_safety.keepout_boxes
            if item.id != "output_table_live"
        ) + (dynamic_table,)
        self.safety = replace(self.delivery_safety, keepout_boxes=keepouts)
        self.scene_boxes = self.safety.moveit_collision_boxes()
        # The output tabletop is an intentional contact surface.  Keep its
        # exact measured solid in MoveIt instead of the normal extra planner
        # padding used for never-touch fixtures; the controlled lowering leg
        # separately enforces the bottle-bottom gap and removes only the
        # tabletop's surface voxels.  Every other keepout remains padded.
        exact = dynamic_table.moveit_box()
        exact["center"] = self.safety.point_to_moveit(
            exact["center"]
        ).tolist()
        self.scene_boxes = [
            item
            for item in self.scene_boxes
            if item["id"] != "fence_output_table_live"
        ] + [exact]
        self.head_scene_captured_monotonic = captured
        observation.write(self.run_dir / "output_table_observation.json")
        self.stage(
            "转向后桌面场景重建",
            (
                f"桌面 z={observation.table_height_m:.3f} m，"
                f"{len(observation.candidates)} 个实时净空候选，"
                f"{len(self.scene_voxels)} 个动态体素；头部外参未加升降补偿"
            ),
        )
        return observation

    def _output_contact_scene_voxels(
        self, observation: OutputTableObservation
    ) -> list[list[float]]:
        """Remove only the measured tabletop sheet for controlled placement.

        The exact solid ``output_table_live`` remains in MoveIt.  This avoids
        treating 6.5 cm RGB-D voxels centred on the contact plane as a thick
        floating obstacle while retaining every above-table obstacle voxel.
        """
        config = self.delivery_safety.side_table_delivery
        points = np.asarray(self.scene_voxels, dtype=float)
        if points.size == 0:
            return []
        lower = np.asarray(config.table_roi_min, dtype=float)
        upper = np.asarray(config.table_roi_max, dtype=float)
        is_table_sheet = (
            (points[:, 0] >= lower[0])
            & (points[:, 0] <= upper[0])
            & (points[:, 1] >= lower[1])
            & (points[:, 1] <= upper[1])
            & (
                np.abs(points[:, 2] - observation.table_height_m)
                <= self.params.scene_voxel_m
            )
        )
        return points[~is_table_sheet].tolist()

    def _output_candidate_poses(self, candidate) -> tuple[np.ndarray, np.ndarray]:
        config = self.delivery_safety.side_table_delivery
        current_tcp = self.robot.current_tcp()
        final_tcp = np.asarray(current_tcp, dtype=float).copy()
        final_tcp[0, 3], final_tcp[1, 3] = candidate.xy_base
        final_tcp[2, 3] = (
            candidate.table_height_m
            + float(config.bottle_bottom_below_tcp_m)
            + float(config.held_bottle_guard_padding_m) / 2.0
            + 0.005
        )
        preplace_tcp = final_tcp.copy()
        preplace_tcp[2, 3] += float(config.preplace_clearance_m)
        self.safety.assert_tcp_point(
            final_tcp[:3, 3], label="实时桌面放置 TCP"
        )
        self.safety.assert_tcp_point(
            preplace_tcp[:3, 3], label="实时桌面预放置 TCP"
        )
        return preplace_tcp, final_tcp

    def _plan_to_output_table(self, observation: OutputTableObservation):
        targets: list[PlanTarget] = []
        final_by_label: dict[str, np.ndarray] = {}
        for index, candidate in enumerate(observation.candidates, 1):
            try:
                preplace_tcp, final_tcp = self._output_candidate_poses(candidate)
                flange = preplace_tcp @ np.linalg.inv(self.T_flange_tcp)
                joints = self.robot.solve_flange_ik(flange, self.params)
            except SafetyAbort as exc:
                LOG.debug("桌面候选 %d 端点拒绝: %s", index, exc)
                continue
            label = f"实时桌面候选 {index}"
            final_by_label[label] = final_tcp
            targets.append(
                PlanTarget(
                    label=label,
                    flange=flange,
                    goal_joints=tuple(joints),
                    score=float(candidate.score),
                    goal_constraint="joints",
                )
            )
        if not targets:
            raise SafetyAbort("所有实时桌面净空候选都不可达或越出电子围栏")

        def validate_lowering(target: PlanTarget, trajectory: dict) -> None:
            endpoint = trajectory.get("points_deg", [])[-1]
            start_tcp = self.robot.tcp_from_joints(endpoint)
            final_tcp = final_by_label[target.label]
            poses = interpolate_poses(
                matrix_pose(start_tcp),
                matrix_pose(final_tcp),
                self.params.segment_m,
            )
            joints = self.robot.plan_ik(
                poses,
                self.params,
                seed_joints_deg=endpoint,
            )
            self._validate_local_joint_path(
                name="output_lowering_precheck",
                joints=joints,
                target_base=None,
                start_joints_deg=endpoint,
                scene_points_override=self._output_contact_scene_voxels(
                    observation
                ),
            )

        verified = self._verified_plan_targets(
            "moveit_output_table",
            targets,
            continuation_validator=validate_lowering,
        )
        return verified.trajectory, final_by_label[verified.target.label]

    def _dispense_to_side_table(
        self, lifted_target: Localization, *, start: BodySnapshot | None = None
    ) -> None:
        """Complete held-object transfer through release and safe retreat."""
        if start is None or start is not self.shelf_ready_body_snapshot:
            raise SafetyAbort("携瓶送桌必须使用 pre-arm SHELF_READY BodySnapshot")
        self._move_held_to_transport_posture(lifted_target)
        config = self.delivery_safety.side_table_delivery
        self.stage(
            "身体升降",
            f"目标 {config.body_lift_height_mm} mm；头/双臂随平台整体运动",
        )
        body = self.mobile_body.position_for_delivery(config, start=start)
        self._base_pose_for_scene = np.eye(4)
        self._base_pose_for_scene[:3, :3] = Rotation.from_euler(
            "z", float(body.chassis.yaw_rad)
        ).as_matrix()
        self._base_pose_for_scene[0, 3] = float(body.chassis.x_m)
        self._base_pose_for_scene[1, 3] = float(body.chassis.y_m)
        self.stage(
            "底盘原地旋转完成",
            (
                f"实时 yaw={math.degrees(body.chassis.yaw_rad):.1f}°，"
                f"x/y=({body.chassis.x_m:.3f},{body.chassis.y_m:.3f})；无平移指令"
            ),
        )
        observation = self._capture_output_table_scene()
        self.local_contact_target_base = None
        plan, final_tcp = self._plan_to_output_table(observation)

        refreshed = self._capture_output_table_scene()
        placement_still_valid(
            planned_table_height_m=observation.table_height_m,
            planned_xy_base=final_tcp[:2, 3],
            refreshed=refreshed,
            height_tolerance_m=float(config.refresh_height_tolerance_m),
            xy_tolerance_m=float(config.refresh_xy_tolerance_m),
        )
        # Revalidate the exact selected global path against the second capture.
        self._refresh_exact_plan_in_current_scene(
            name="moveit_output_table", plan=plan
        )
        self._execute_plan("携瓶移动到右侧桌面预放置位", plan)

        def build_lower_path() -> list[list[float]]:
            return interpolate_poses(
                matrix_pose(self.robot.current_tcp()),
                matrix_pose(final_tcp),
                self.params.segment_m,
            )

        lower_path = self._plan_local_leg(
            "桌面放低", build_lower_path, self.params
        )
        lower_joints = self.robot.plan_ik(lower_path, self.params)
        self._validate_local_joint_path(
            name="output_lowering_execute",
            joints=lower_joints,
            target_base=None,
            scene_points_override=self._output_contact_scene_voxels(
                refreshed
            ),
        )
        self.stage("桌面放低", "按实时桌高下降到瓶底小间隙")
        for pose in lower_path:
            self.robot.move_linear(pose, self.params.final_speed)

        expected_release_point = np.asarray(final_tcp[:3, 3], dtype=float)
        expected_release_point[2] = (
            observation.table_height_m
            + float(config.bottle_bottom_below_tcp_m)
        )
        self.stage("松开夹爪", "瓶底已贴近实测桌面")
        self.robot.open_gripper(self.params)

        def build_retreat_path() -> list[list[float]]:
            tcp = self.robot.current_tcp()
            retreat = tcp.copy()
            retreat[:3, 3] -= tcp[:3, 2] * float(config.retreat_standoff_m)
            self.safety.assert_tcp_point(
                retreat[:3, 3], label="桌面释放后退开点"
            )
            return interpolate_poses(
                matrix_pose(tcp), matrix_pose(retreat), self.params.segment_m
            )

        retreat_path = self._plan_local_leg(
            "桌面释放后退开", build_retreat_path, self.params
        )
        retreat_joints = self.robot.plan_ik(retreat_path, self.params)
        self._validate_local_joint_path(
            name="output_retreat_execute",
            joints=retreat_joints,
            target_base=None,
            scene_points_override=self._output_contact_scene_voxels(
                refreshed
            ),
        )
        for pose in retreat_path:
            self.robot.move_linear(pose, self.params.travel_speed)
        self._confirm_point_released(
            expected_release_point,
            label="右侧桌面释放三维确认",
            stage_name="右侧桌面释放确认",
            error_point_description="动态放置点",
            stage_point_description="动态放置点",
        )
        self._detach_held_bottle_guard()
        self.robot.close_empty_gripper(self.params)
        self.stage("右侧桌面放置完成", "瓶子已释放、视觉确认并安全退开")

    def _refresh_exact_plan_in_current_scene(self, *, name: str, plan: dict):
        if self.planner is None or self.left_robot is None:
            raise SafetyAbort("刷新送桌路径时 MoveIt/左臂状态不可用")
        start = np.asarray(plan.get("start_joints_deg"), dtype=float)
        actual = np.asarray(self.robot.joints_deg(), dtype=float)
        if start.shape != (7,) or np.max(np.abs(start - actual)) > self.params.planned_start_tolerance_deg:
            raise SafetyAbort("桌面刷新后右臂已偏离规划起点")
        left = self.left_robot.joints_deg()
        dense = interpolate_joint_path(
            actual,
            plan.get("points_deg") or [],
            self.params.planned_joint_step_deg,
        )
        self.robot.validate_planned_joints(
            plan.get("points_deg") or [],
            self.params.planned_joint_step_deg,
            self.safety,
            start_joints_deg=actual,
        )
        self.planner.validate_exact_path(
            name=f"{name}_fresh_output_scene",
            start_left_joints_deg=left,
            points_deg=dense,
            obstacles=self._moveit_scene_obstacles(self.scene_voxels),
            boxes=self.scene_boxes,
            planning_frame=self.safety.moveit_frame,
            tool_guard={
                "xy": self.params.tool_guard_xy_m,
                "length": self.params.tool_guard_length_m,
                "center_z": self.params.tool_guard_center_z_m,
            },
            held_object=self.held_object_guard,
            voxel_size=self.params.scene_voxel_m,
        )
        plan["start_joints_deg"] = actual.tolist()
        plan["start_left_joints_deg"] = list(map(float, left))
        plan["scene_captured_monotonic"] = float(
            self.head_scene_captured_monotonic
        )

    def _deliver_to_output(self):
        """把瓶子送到出货口：转移→松开→(可选)视觉确认→退开→空载收拢。

        转移段复用跟 `_return_home` 完全同一条 SafeMotionPlanner 规划链路，
        风险等级一致，不是另开一条低标准旁路。出货口坐标目前是占位关节角
        （`output_joints_deg`），现场量出真实取货口位置前不应标记
        `verified_for_execution=True`。

        释放确认默认只信夹爪反馈，不伪造视觉证据：`_confirm_released_target`
        （放回原位场景）假设瓶子还在头部相机能看到的原锁定点，但出货口大概率
        不在头部相机视野内。只有 profile 明确给出
        `output_visible_to_head_camera=True` 和 `output_point_base` 时才做
        跟放回原位同一套三维确认；否则如实记录"这里没有视觉证据"，不悄悄
        套用桌面 demo 那套确认逻辑却假装同等可信。
        """
        output = self.safety.output_joints_deg
        if not output:
            raise SafetyAbort(
                f"profile {self.safety.name} 未配置 output_joints_deg，"
                "无法送到出货口"
            )
        target_flange = self.robot.controller_flange_from_joints(list(output))
        plan = self._plan_flange(
            "moveit_deliver_output", target_flange, goal_joints=list(output)
        )
        self._execute_plan("送到出货口", plan)

        self.stage("松开夹爪")
        self.robot.open_gripper(self.params)

        if self.safety.output_visible_to_head_camera:
            if self.safety.output_point_base is None:
                raise SafetyAbort(
                    f"profile {self.safety.name} 声明 "
                    "output_visible_to_head_camera 但未配置 "
                    "output_point_base，无法做视觉释放确认"
                )
            self._confirm_point_released(
                self.safety.output_point_base,
                label="出货口三维确认",
                stage_name="出货口视觉确认",
                error_point_description="出货口锁定点",
                stage_point_description="出货口锁定点",
            )
        else:
            self.stage(
                "出货口释放（仅夹爪反馈）",
                "出货口不在头部相机视野内，仅凭夹爪张开反馈判定已释放，"
                "无视觉证据",
            )
        self._detach_held_bottle_guard()

        def build_retreat_path() -> list[list[float]]:
            tcp = self.robot.current_tcp()
            axis = tcp[:3, 2]
            retreat = tcp.copy()
            retreat[:3, 3] -= axis * self.params.retreat_standoff_m
            self.safety.assert_tcp_point(retreat[:3, 3], label="出货口退开点")
            return interpolate_poses(
                matrix_pose(tcp), matrix_pose(retreat), self.params.segment_m
            )

        retreat_path = self._plan_local_leg(
            "退开", build_retreat_path, self.params
        )
        self.stage(
            "退开",
            f"沿接近轴反向 {self.params.retreat_standoff_m * 100:.0f} cm",
        )
        for pose in retreat_path:
            self.robot.move_linear(pose, self.params.travel_speed)

        self.stage("收拢夹爪", "手臂已退开，空载闭合夹爪")
        self.robot.close_empty_gripper(self.params)
        self.stage("送货完成", "瓶子已送到出货口，手臂已退开，夹爪已收拢")

    def _preflight(self):
        """真机运动前自检：机械臂在线、无错误码、夹爪使能+收拢。plan-only 跳过。"""
        if not self.args.execute or self._is_read_only_vision_check():
            return
        recover = getattr(self.robot, "recover_transient_joint_frame_loss", None)
        recovered = recover() if callable(recover) else []
        if recovered:
            self.stage(
                "瞬态关节错误已恢复",
                f"已清除并连续复核通过: {','.join(f'J{joint}' for joint in recovered)}",
            )
        health = self.robot.assert_arm_healthy()
        self.robot.current_tcp()  # 内部校验 arm_err/sys_err，异常即抛
        controller_fence = self.robot.controller_fence_status()
        fence_state = controller_fence["state"]
        enabled = bool(fence_state.get("enable_state", False))
        self.stage(
            "控制器原生围栏检查",
            (
                f"enable={enabled}；current={controller_fence['current']}；"
                f"saved={controller_fence['saved']}"
            ),
        )
        if enabled:
            raise SafetyAbort(
                "控制器原生电子围栏仍处于启用状态；本程序不会擅自删除或"
                "关闭硬件安全配置。先核对/停用旧围栏后再运动: "
                f"{controller_fence}"
            )
        state = self.robot.gripper_state()
        self.stage(
            "运动前自检",
            (
                "控制器及 7 个关节无错误码且已使能；"
                f"夹爪 enable={state.get('enable_state')}；"
                f"controller={health['controller']}"
            ),
        )
        if getattr(self.args, "stop_after_observation", False):
            # A stop-after run is an observation transaction, not a grasp
            # rehearsal.  In particular, do not use the usual empty-gripper
            # pre-transit close: callers rely on this gate to prove that no
            # gripper-close command was sent before its EMPTY terminal state.
            self.stage(
                "观察后停止夹爪保护",
                "stop-after-observation：保留当前夹爪状态，不下发闭夹命令",
            )
        elif not getattr(self.args, "finish_from_current", False):
            self._close_gripper_if_open(state)

    def _close_gripper_if_open(self, state: dict):
        """转移开始前把张开的夹爪收拢到空载基线。

        张开的手指是比 tool_guard 固定防撞盒更宽、更不可预测的碰撞形状，
        闭合是已知、更小的包络，也不会在长距离转移途中勾挂到东西。用
        close_empty_gripper（无抓取判定的收拢语义）而不是 close_gripper——
        这里不是在抓取，强行套用抓取判定只会把"本来就是空的"误判成失败。
        只有 --finish-from-current（假设夹爪已抓着水瓶）跳过这一步，由
        `_preflight` 的调用方保证。
        """
        pos = int(state["pos"][0])
        if pos <= self.params.gripper_pretransit_open_threshold:
            return
        self.stage(
            "夹爪预备闭合",
            f"运动前检测到夹爪未闭合 (pos={pos})，先收拢到空载基线再继续",
        )
        self.robot.close_empty_gripper(self.params)

    def _restore_teleop(self):
        """best-effort 恢复官方遥操（--restore-teleop）。找不到脚本就只打印提示。"""
        import subprocess

        script = os.environ.get(
            "UPSTART_ALL", "/home/rm/rmc_aida_l_atom/scripts/upstart_all.sh"
        )
        if not os.path.exists(script):
            LOG.warning(
                "未找到 %s，无法自动恢复遥操；请手动运行官方 upstart_all.sh", script
            )
            return
        self.stage("恢复遥操", f"运行 {script}")
        subprocess.Popen(
            f"bash '{script}' > /home/rm/upstart_all_from_demo.log 2>&1 &",
            shell=True,
        )

    def close(self):
        self.stop_event.set()
        summary = {
            "stage": self.state.status(),
            "plan_only": bool(self.args.plan_only),
            "execute": bool(self.args.execute),
        }
        if self.dashboard:
            self.dashboard.close()
        if self.mobile_body:
            # Body cleanup is intentionally zero-speed only.  It must happen
            # even when arm/planner initialization failed partway through a
            # dispense task, and never attempts an automatic return turn.
            try:
                self.mobile_body.close()
            except Exception:
                LOG.exception("底盘零速度收尾失败；保留原始任务结果")
        if self.preview:
            self.preview.stop()
            if self.preview.is_alive():
                self.preview.join(timeout=2)
        if self.planner:
            self.planner.close()
        if self.robot:
            if self.robot.take_control:
                try:
                    summary["final_tcp"] = matrix_pose(self.robot.current_tcp())
                except Exception as exc:
                    summary["final_tcp_error"] = str(exc)
                self.robot.hold()
            self.robot.close()
        if self.left_robot:
            self.left_robot.close()
        (self.run_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if self.camera:
            self.camera.stop()
            if self.camera.is_alive():
                self.camera.join(timeout=3)
        logging.getLogger().removeHandler(self.run_log_handler)
        self.run_log_handler.close()
