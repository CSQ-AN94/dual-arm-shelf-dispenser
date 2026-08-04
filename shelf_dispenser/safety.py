"""Configuration-driven electronic fence checks for real-arm planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .core import SafetyAbort
from .grasp_orientation import (
    MEASURED,
    NOMINAL_FUNCTIONALLY_VALIDATED,
    NOMINAL_UNVALIDATED,
    GraspFrameSpec,
    ToolMountCalibration,
    authored_tcp_rotation,
    validate_rigid_transform,
)


class FenceViolation(SafetyAbort):
    """Structured electronic-fence rejection suitable for replanning."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        label: str,
        point: Sequence[float],
        object_id: str | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.label = label
        self.point = tuple(map(float, point))
        self.object_id = object_id


_MAX_TOOL_MOUNT_POSITION_RESIDUAL_M = 0.005
_MAX_TOOL_MOUNT_ORIENTATION_RESIDUAL_DEG = 1.0


@dataclass(frozen=True)
class FenceBox:
    id: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    @classmethod
    def from_dict(cls, data: dict, *, prefix: str) -> "FenceBox":
        object_id = str(data.get("id", prefix))
        minimum = np.asarray(data.get("min"), dtype=float)
        maximum = np.asarray(data.get("max"), dtype=float)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise SafetyAbort(f"电子围栏 {object_id} 的 min/max 必须各有 3 个数")
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise SafetyAbort(f"电子围栏 {object_id} 包含非有限数值")
        if np.any(maximum <= minimum):
            raise SafetyAbort(f"电子围栏 {object_id} 的 max 必须大于 min")
        return cls(
            id=object_id,
            minimum=tuple(map(float, minimum)),
            maximum=tuple(map(float, maximum)),
        )

    def contains(self, point: Sequence[float], margin_m: float = 0.0) -> bool:
        point = np.asarray(point, dtype=float)
        lower = np.asarray(self.minimum) + margin_m
        upper = np.asarray(self.maximum) - margin_m
        return bool(np.all(point >= lower) and np.all(point <= upper))

    def contains_expanded(
        self, point: Sequence[float], margin_m: float = 0.0
    ) -> bool:
        point = np.asarray(point, dtype=float)
        lower = np.asarray(self.minimum) - margin_m
        upper = np.asarray(self.maximum) + margin_m
        return bool(np.all(point >= lower) and np.all(point <= upper))

    def moveit_box(self) -> dict:
        minimum = np.asarray(self.minimum, dtype=float)
        maximum = np.asarray(self.maximum, dtype=float)
        return {
            "id": f"fence_{self.id}",
            "center": ((minimum + maximum) / 2).tolist(),
            "size": (maximum - minimum).tolist(),
        }


@dataclass(frozen=True)
class SafetyProfile:
    name: str
    description: str
    frame: str
    moveit_frame: str
    T_moveit_from_profile: np.ndarray
    verified_for_execution: bool
    clearance_m: float
    tcp_workspace: FenceBox
    allowed_tcp_zones: tuple[FenceBox, ...]
    keepout_boxes: tuple[FenceBox, ...]
    use_dynamic_rgbd: bool
    home_joints_deg: tuple[float, ...] | None
    # Repeatable, empty-handed shelf-pick admission state.  This is not the
    # post-pick carry/home posture: every new shelf grasp starts from this
    # measured dual-arm + lift state, while home remains a later task target.
    grasp_start_right_joints_deg: tuple[float, ...] | None = None
    grasp_start_left_joints_deg: tuple[float, ...] | None = None
    grasp_start_lift_height_mm: int | None = None
    # Optional open/high posture used to leave a low natural-hang start before
    # solving the target-dependent observation transfer.  This is distinct
    # from ``home_joints_deg``: home is where the task parks; staging is a
    # proven planning seed that avoids asking one global search to both unfold
    # a near-singular arm and arrive at the bottle-facing wrist pose.
    observation_staging_joints_deg: tuple[float, ...] | None = None
    # Real dispensing (as opposed to table_demo's place-back-in-place cycle):
    # where to carry a held bottle before releasing it. Same structural
    # contract as home_joints_deg — absent by default, _deliver_to_output
    # fails closed if a caller asks for it without this configured.
    output_joints_deg: tuple[float, ...] | None = None
    # Whether the output point is expected to be visible to the fixed head
    # camera, so _deliver_to_output knows whether a real 3-D release check is
    # possible or whether it must honestly fall back to gripper-feedback-only
    # evidence. False by default: assuming visibility that does not exist
    # would silently downgrade a safety check into a fabricated pass.
    output_visible_to_head_camera: bool = False
    output_point_base: tuple[float, float, float] | None = None
    side_table_delivery: "SideTableDeliveryConfig | None" = None
    # Optional task-geometry override.  A shelf can require a higher point on
    # the same bottle than the open-table demo because the hand envelope, not
    # just the TCP, must clear the shelf lip.  None preserves DemoParams.
    grasp_height_fraction: float | None = None
    # Widest target the installed fingers can span, measured on this robot.
    # None means "not measured yet", and the executor then performs no width
    # check: the dual_rm_75b_description figure of 0.065 m could not be
    # reconciled with grasps that demonstrably held, so it is not a safe
    # default to enforce.  Set it once the real stroke and bottle are on a
    # ruler, and the executor starts refusing targets that cannot fit.
    gripper_max_opening_m: float | None = None
    # The arm posture a person actually used to reach this shelf's grasp.  A
    # 7-DoF arm reaches one TCP pose from many configurations and the planner
    # picks among them at random -- replaying one scene eight times gave the
    # demonstrated posture four times and one 160-210 deg away twice.  They all
    # reach the grasp; only this one is known to work on the real shelf.
    demonstrated_grasp_right_joints_deg: tuple[float, ...] | None = None
    # Shelf-only authored physical grasp axes.  None deliberately preserves
    # the historical table/non-shelf behaviour of retaining the observation
    # wrist orientation.
    grasp_frame: GraspFrameSpec | None = None
    # The physical gripper can have a non-identity rotation relative to both
    # controller flange and MoveIt r_link7.  Shelf execution therefore uses
    # this explicit measured chain instead of inferring one from TCP-Z.
    tool_mount_calibration: ToolMountCalibration | None = None
    # The left tool is a separate record even though it is the same gripper
    # part.  A shared field would let the left arm inherit by default, which is
    # exactly what the right record's evidence_id forbids.
    left_tool_mount_calibration: ToolMountCalibration | None = None
    # Poses handed to the fence are named in whichever arm's base frame
    # produced them.  Every box here is authored in the right arm's, so a
    # left-arm view carries the rigid transform that names its points in the
    # same frame.  Converting the point is exact; rewriting the boxes is not,
    # because the axis-aligned hull of a rotated box is larger than the box --
    # which for an allowed zone hands out space the fence never granted.
    tcp_frame_transform: np.ndarray | None = None

    def in_fence_frame(self, point: Sequence[float]) -> np.ndarray:
        """Name a point in the frame the fence boxes are authored in."""
        point = np.asarray(point, dtype=float)
        if self.tcp_frame_transform is None:
            return point
        matrix = np.asarray(self.tcp_frame_transform, dtype=float)
        return matrix[:3, :3] @ point + matrix[:3, 3]

    def assert_grasp_start(
        self,
        *,
        right_joints_deg: Sequence[float],
        left_joints_deg: Sequence[float],
        lift_height_mm: int,
        lift_mode: int,
        joint_tolerance_deg: float,
        lift_tolerance_mm: int = 5,
    ) -> None:
        """Fail closed unless the live robot matches the taught grasp start."""
        expected_right = np.asarray(
            self.grasp_start_right_joints_deg, dtype=float
        )
        expected_left = np.asarray(
            self.grasp_start_left_joints_deg, dtype=float
        )
        right = np.asarray(right_joints_deg, dtype=float)
        left = np.asarray(left_joints_deg, dtype=float)
        if (
            expected_right.shape != (7,)
            or expected_left.shape != (7,)
            or right.shape != (7,)
            or left.shape != (7,)
            or not np.all(np.isfinite(expected_right))
            or not np.all(np.isfinite(expected_left))
            or not np.all(np.isfinite(right))
            or not np.all(np.isfinite(left))
            or self.grasp_start_lift_height_mm is None
        ):
            raise SafetyAbort("抓取初始位双臂/升降配置或实时反馈无效")
        if (
            not np.isfinite(joint_tolerance_deg)
            or joint_tolerance_deg <= 0.0
            or isinstance(lift_tolerance_mm, bool)
            or int(lift_tolerance_mm) < 0
        ):
            raise SafetyAbort("抓取初始位关节/升降容差无效")
        right_error = float(np.max(np.abs(right - expected_right)))
        left_error = float(np.max(np.abs(left - expected_left)))
        lift_error = abs(
            int(lift_height_mm) - int(self.grasp_start_lift_height_mm)
        )
        if int(lift_mode) != 0:
            raise SafetyAbort(f"抓取初始位升降仍在运动: mode={lift_mode}")
        if right_error > float(joint_tolerance_deg):
            raise SafetyAbort(
                "右臂未回到示教抓取初始位: "
                f"最大关节差={right_error:.2f}°，上限={joint_tolerance_deg:.2f}°"
            )
        if left_error > float(joint_tolerance_deg):
            raise SafetyAbort(
                "左臂未回到示教抓取初始位，禁止自动移动左臂: "
                f"最大关节差={left_error:.2f}°，上限={joint_tolerance_deg:.2f}°"
            )
        if lift_error > int(lift_tolerance_mm):
            raise SafetyAbort(
                "升降未回到示教抓取初始高度: "
                f"actual={int(lift_height_mm)} mm, "
                f"expected={self.grasp_start_lift_height_mm} mm, "
                f"上限={int(lift_tolerance_mm)} mm"
            )

    def assert_tcp_point(
        self,
        point: Sequence[float],
        *,
        label: str,
        margin_m: float | None = None,
    ):
        margin = self.clearance_m if margin_m is None else margin_m
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise SafetyAbort(f"{label} 的 TCP 坐标无效: {point.tolist()}")
        point = self.in_fence_frame(point)
        if not self.tcp_workspace.contains(point, margin):
            raise FenceViolation(
                f"{label} 越出总工作空间: {np.round(point, 4).tolist()}",
                kind="workspace",
                label=label,
                point=point,
                object_id=self.tcp_workspace.id,
            )
        for obstacle in self.keepout_boxes:
            if obstacle.contains_expanded(point, margin):
                raise FenceViolation(
                    f"{label} 进入禁入区 {obstacle.id}: "
                    f"{np.round(point, 4).tolist()}",
                    kind="keepout",
                    label=label,
                    point=point,
                    object_id=obstacle.id,
                )
        # Allowed zones are authored as already-safe corridors. Do not shrink
        # each box independently: doing so creates artificial gaps where two
        # valid transit volumes overlap. Clearance is still enforced against
        # the outer workspace and every expanded keepout object above.
        if not any(zone.contains(point, 0.0) for zone in self.allowed_tcp_zones):
            raise FenceViolation(
                f"{label} 不在任何允许区: {np.round(point, 4).tolist()}",
                kind="allowed_zone",
                label=label,
                point=point,
            )

    def assert_tcp_path(self, points: Iterable[Sequence[float]]) -> int:
        count = 0
        for count, point in enumerate(points, 1):
            self.assert_tcp_point(point, label=f"轨迹 TCP 点 {count}")
        if count == 0:
            raise SafetyAbort("电子围栏检查收到空轨迹")
        return count

    def moveit_collision_boxes(self) -> list[dict]:
        # MoveIt 只做几何碰撞，不知道围栏检查还要求 clearance_m 的 TCP 余量；
        # 如果给它精确盒子，它会规划出"贴着盒面飞"的路径，随后被离线围栏
        # 复核否决。这里把四个水平侧面各向外垫，顶面也垫同样距离，让规划
        # 阶段就留足余量；底面不影响桌面上方的规划。
        #
        # 2026-07-17 真机 observe 实测：clearance_m+1cm 的旧余量不够——MoveIt
        # 按其内部路径采样分辨率认为"没碰垫大的盒子"，但独立围栏用更密的
        # 插值复核发现实际路径已经比垫大后的盒子边界还深入 1~1.7cm（8个候选、
        # 16次尝试全部在这个narrow band里被拒）。根因是 OMPL 边碰撞检测的
        # 离散化盲区，已在 moveit_headless.py 用
        # longest_valid_segment_fraction 0.01->0.0025（4倍更密）从源头收紧，
        # 采样间隔按此比例线性缩小，预期把 1~1.7cm 的偏差压到约 0.25~0.4cm。
        #
        # 2026-07-18：把这里的余量从 +5cm 回调到 +2cm——clearance_m(2.5cm)+2cm
        # =4.5cm 仍比旧实测的最大偏差 1.7cm 宽裕得多，对采样密度修复后的
        # 预期偏差（~0.4cm）留有约10倍安全系数。但这个具体数值组合
        # （更密的lvsf + 更小的padding）还没有真机验证过，不能只信这个
        # 推算——下次连机器人必须先跑 observe/plan 多轮确认没有回到
        # narrow-band拒绝循环，再信任这个余量。
        # 2026-07-31：垂直方向的 padding 已移除，只保留四个水平侧面。
        # 原来的写法只把盒子往**上**长（center z += padding/2, size z +=
        # padding），逐个盒子看它其实没在任何地方起保护作用：
        #   - shelf_top 是头顶的天花板，往上长是背离机器人，无用；
        #   - shelf_back 是竖直背板，垂直方向无关；
        #   - shelf_bottom 是瓶子站着、手必须伸到其上方的支撑面——往上长
        #     直接把可抓取的那条带子吃掉。
        # 实测数据（直接量 r_hand.STL 网格）：手掌绕工具轴严重不对称，一侧
        # 伸出 97mm，另一侧 34mm；roll 0/roll 180 的区别就是哪一侧朝下。
        # 对 2026-07-30 现场目标（TCP z=-0.0547，shelf_bottom 实测顶面
        # -0.1830）：真实净空 128mm 装得下 97mm，而 +5cm 膨胀后只剩 78mm——
        # 78 < 97 < 128。这就是当时 "r_hand 与 fence_shelf_bottom 碰撞"
        # 拒绝 roll 180 的唯一原因，另一侧 34.3mm 也只剩 0.3mm 余量。
        # 水平 padding 的原始理由（2026-07-17 OMPL 离散化盲区让路径切进
        # keepout 1~1.7cm）保留；那次的根因已另在 moveit_headless.py 用
        # longest_valid_segment_fraction 0.01->0.0025 从源头收紧。TCP 自身
        # 的 clearance_m 余量仍由离线围栏和稠密 FK 审计独立把关。
        padding = self.clearance_m + 0.02
        result = []
        for box in self.keepout_boxes:
            item = box.moveit_box()
            item["size"][0] += 2 * padding
            item["size"][1] += 2 * padding
            item["center"] = self.point_to_moveit(item["center"]).tolist()
            result.append(item)
        return result

    def moveit_workspace(self) -> dict:
        minimum = self.point_to_moveit(self.tcp_workspace.minimum)
        maximum = self.point_to_moveit(self.tcp_workspace.maximum)
        return {
            "min": np.minimum(minimum, maximum).tolist(),
            "max": np.maximum(minimum, maximum).tolist(),
        }

    def point_to_moveit(self, point: Sequence[float]) -> np.ndarray:
        return (self.T_moveit_from_profile @ np.r_[point, 1.0])[:3]

    def points_to_moveit(
        self, points: Iterable[Sequence[float]]
    ) -> list[list[float]]:
        return [self.point_to_moveit(point).tolist() for point in points]

    def moveit_obstacles_outside_fences(
        self,
        points: Iterable[Sequence[float]],
        collision_boxes: Sequence[dict],
    ) -> list[list[float]]:
        """Convert RGB-D voxels and remove centres already fenced explicitly.

        The scene grid stores a 6.5 cm voxel at the centre of each occupied
        cell.  A shelf-plane sample near a cell boundary can therefore make
        that cube protrude centimetres above the separately fitted, padded
        shelf fence.  Keeping both representations double-inflates the same
        rigid surface.  A voxel is redundant only when its *centre* is inside
        an explicit ``fence_*`` box; every centre outside those conservative
        boxes remains a dynamic obstacle.
        """
        converted = np.asarray(self.points_to_moveit(points), dtype=float)
        if converted.size == 0:
            return []
        if (
            converted.ndim != 2
            or converted.shape[1] != 3
            or not np.all(np.isfinite(converted))
        ):
            raise SafetyAbort("MoveIt 动态障碍点必须是有限的 Nx3 数组")
        redundant = np.zeros(len(converted), dtype=bool)
        for item in collision_boxes:
            if not str(item.get("id", "")).startswith("fence_"):
                continue
            center = np.asarray(item.get("center"), dtype=float)
            size = np.asarray(item.get("size"), dtype=float)
            if (
                center.shape != (3,)
                or size.shape != (3,)
                or not np.all(np.isfinite(center))
                or not np.all(np.isfinite(size))
                or np.any(size <= 0)
            ):
                raise SafetyAbort("MoveIt fence 碰撞盒必须有有限正尺寸")
            half = size / 2.0
            redundant |= np.all(
                (converted >= center - half)
                & (converted <= center + half),
                axis=1,
            )
        return converted[~redundant].tolist()

    def pose_to_moveit(self, pose: np.ndarray) -> np.ndarray:
        return self.T_moveit_from_profile @ np.asarray(pose, dtype=float)

    def replan_exclusion_box(
        self,
        violation: FenceViolation,
        *,
        object_id: str,
        size_m: float,
    ) -> dict:
        """Turn an independently rejected TCP point into MoveIt feedback."""
        center = self.point_to_moveit(violation.point)
        return {
            "id": str(object_id),
            "center": center.tolist(),
            "size": [float(size_m)] * 3,
        }


@dataclass(frozen=True)
class ShelfReadyConfig:
    """Measured body pose required before a shelf-to-side-table task starts.

    The values are deliberately expressed in the chassis odometry frame, not
    as an assumed relative turn.  ``MobileBodyCoordinator`` compares a fresh
    :class:`BodySnapshot` against this record before either arm is allowed to
    move.
    """

    x_m: float
    y_m: float
    yaw_deg: float
    lift_height_mm: int
    xy_tolerance_m: float
    yaw_tolerance_deg: float
    lift_tolerance_mm: int


@dataclass(frozen=True)
class RotationSweepClearance:
    """Measured clearances for both signed 90-degree chassis sweeps."""

    positive_clearance_m: float
    negative_clearance_m: float
    positive_verified: bool
    negative_verified: bool


@dataclass(frozen=True)
class SideTableDeliveryConfig:
    """Measured envelope and motion limits for the post-grasp side table."""

    transport_joints_deg: tuple[float, ...]
    transport_pose_verified: bool
    shelf_ready: ShelfReadyConfig
    shelf_ready_verified: bool
    # This duplication is intentional and validated below.  It makes the
    # lift transition auditable while making it impossible for an author to
    # accidentally declare one source height for SHELF_READY and another for
    # the subsequent lift command.
    source_lift_height_mm: int
    target_lift_height_mm: int
    target_lift_tolerance_mm: int
    lift_transition_verified: bool
    body_lift_speed: int
    body_rotation_yaw_deg: float
    max_angular_speed_radps: float
    rotation_tolerance_deg: float
    rotation_timeout_s: float
    max_base_translation_m: float
    rotation_sweep: RotationSweepClearance
    table_roi_min: tuple[float, float, float]
    table_roi_max: tuple[float, float, float]
    table_roi_verified: bool
    workspace_verified: bool
    keepouts_verified: bool
    bottle_bottom_below_tcp_m: float
    held_bottle_height_m: float
    held_bottle_diameter_m: float
    held_bottle_guard_padding_m: float
    bottle_tcp_verified: bool
    preplace_clearance_m: float
    retreat_standoff_m: float
    table_height_bin_m: float = 0.01
    table_inlier_band_m: float = 0.012
    table_min_inliers: int = 50
    table_frame_agreement_m: float = 0.012
    table_edge_margin_m: float = 0.10
    table_support_radius_m: float = 0.06
    table_min_patch_points: int = 4
    place_clearance_radius_m: float = 0.10
    place_grid_m: float = 0.04
    obstacle_min_height_m: float = 0.025
    obstacle_max_height_m: float = 0.45
    max_place_candidates: int = 8
    refresh_height_tolerance_m: float = 0.012
    refresh_xy_tolerance_m: float = 0.04

    @property
    def body_lift_height_mm(self) -> int:
        """Compatibility alias while callers migrate to the explicit target."""
        return self.target_lift_height_mm


def _finite_vector(raw, size: int, *, label: str) -> tuple[float, ...]:
    try:
        value = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SafetyAbort(f"{label} 必须是 {size} 个有限数") from exc
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise SafetyAbort(f"{label} 必须是 {size} 个有限数")
    return tuple(map(float, value))


def _required_mapping(data: dict, key: str, *, label: str) -> dict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise SafetyAbort(f"{label}.{key} 必须是对象且不得为空")
    return value


def _required_value(data: dict, key: str, *, label: str):
    if key not in data or data[key] is None:
        raise SafetyAbort(f"{label}.{key} 是执行前必须现场量测的字段")
    return data[key]


def _finite_float(raw, *, label: str) -> float:
    if isinstance(raw, bool):
        raise SafetyAbort(f"{label} 必须是有限数")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SafetyAbort(f"{label} 必须是有限数") from exc
    if not np.isfinite(value):
        raise SafetyAbort(f"{label} 必须是有限数")
    return value


def _finite_int(raw, *, label: str) -> int:
    value = _finite_float(raw, label=label)
    if not value.is_integer():
        raise SafetyAbort(f"{label} 必须是整数")
    return int(value)


def _strict_bool(raw, *, label: str) -> bool:
    if not isinstance(raw, bool):
        raise SafetyAbort(f"{label} 必须是 true 或 false")
    return raw


def _profile_payload_sha256(raw: dict) -> str:
    """Return the manifest-compatible digest of one parsed JSON profile."""
    try:
        canonical = json.dumps(
            raw,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SafetyAbort("电子围栏 profile 不能规范化为可审计 JSON") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_tool_mount_calibration(
    raw: dict,
    *,
    profile_name: str,
    key: str = "tool_mount_calibration",
) -> ToolMountCalibration | None:
    """Parse an auditable link7/controller/TCP calibration chain.

    An unverified record is deliberately representable for offline review,
    but it has no usable matrices.  A record marked verified must contain two
    full proper transforms and provenance/residuals; a scalar Z offset is not
    an acceptable substitute for a tool-installation rotation.
    """
    data = raw.get(key)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise SafetyAbort(
            f"电子围栏 profile {profile_name} 的 {key} 必须是对象"
        )
    label = f"profile {profile_name}.{key}"
    verified = _strict_bool(data.get("verified"), label=f"{label}.verified")
    if not verified:
        return ToolMountCalibration(verified=False)

    provenance = data.get("provenance", MEASURED)
    if provenance not in (
        MEASURED,
        NOMINAL_FUNCTIONALLY_VALIDATED,
        NOMINAL_UNVALIDATED,
    ):
        raise SafetyAbort(
            f"{label}.provenance 必须是 {MEASURED}、"
            f"{NOMINAL_FUNCTIONALLY_VALIDATED} 或 {NOMINAL_UNVALIDATED}"
        )
    evidence_id = data.get("evidence_id")
    measured_at_utc = data.get("measured_at_utc")
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise SafetyAbort(f"{label}.evidence_id 必须是非空实测证据标识")
    if not isinstance(measured_at_utc, str) or not measured_at_utc.strip():
        raise SafetyAbort(f"{label}.measured_at_utc 必须是非空 UTC 时间")
    try:
        link7_to_flange = validate_rigid_transform(
            _required_value(
                data, "T_link7_controller_flange", label=label
            ),
            label=f"{label}.T_link7_controller_flange",
        )
        flange_to_tcp = validate_rigid_transform(
            _required_value(
                data, "T_controller_flange_tcp", label=label
            ),
            label=f"{label}.T_controller_flange_tcp",
        )
    except SafetyAbort:
        raise
    if provenance in (NOMINAL_FUNCTIONALLY_VALIDATED, NOMINAL_UNVALIDATED):
        # No metrology happened, so there is nothing to report a residual for.
        # Refuse a record that claims one anyway: that is exactly how the
        # nominal +Z fallback came to look like a 0.042 mm measurement.
        for key in ("max_position_residual_m", "max_orientation_residual_deg"):
            if data.get(key) is not None:
                raise SafetyAbort(
                    f"{label}.{key} 不能与 provenance={provenance} 并存："
                    "没有实测就没有残差"
                )
        return ToolMountCalibration(
            verified=True,
            provenance=provenance,
            evidence_id=evidence_id.strip(),
            measured_at_utc=measured_at_utc.strip(),
            T_link7_controller_flange=link7_to_flange,
            T_controller_flange_tcp=flange_to_tcp,
        )

    max_position_residual_m = _finite_float(
        _required_value(data, "max_position_residual_m", label=label),
        label=f"{label}.max_position_residual_m",
    )
    max_orientation_residual_deg = _finite_float(
        _required_value(data, "max_orientation_residual_deg", label=label),
        label=f"{label}.max_orientation_residual_deg",
    )
    if max_position_residual_m < 0.0 or max_orientation_residual_deg < 0.0:
        raise SafetyAbort(f"{label} 的残差不得为负")
    if (
        max_position_residual_m > _MAX_TOOL_MOUNT_POSITION_RESIDUAL_M
        or max_orientation_residual_deg
        > _MAX_TOOL_MOUNT_ORIENTATION_RESIDUAL_DEG
    ):
        raise SafetyAbort(
            f"{label} 的实测残差超出工具安装准入上限 "
            f"({_MAX_TOOL_MOUNT_POSITION_RESIDUAL_M * 1000:.1f} mm, "
            f"{_MAX_TOOL_MOUNT_ORIENTATION_RESIDUAL_DEG:.1f}°)"
        )
    return ToolMountCalibration(
        verified=True,
        provenance=provenance,
        evidence_id=evidence_id.strip(),
        measured_at_utc=measured_at_utc.strip(),
        max_position_residual_m=max_position_residual_m,
        max_orientation_residual_deg=max_orientation_residual_deg,
        T_link7_controller_flange=link7_to_flange,
        T_controller_flange_tcp=flange_to_tcp,
    )


def _load_side_table_delivery(
    raw: dict,
    *,
    profile_name: str,
    require_verified: bool,
    profile_clearance_m: float,
):
    data = raw.get("side_table_delivery")
    if data is None:
        return None
    if not isinstance(data, dict):
        raise SafetyAbort(
            f"profile {profile_name} 的 side_table_delivery 必须是对象"
        )
    label = f"profile {profile_name}.side_table_delivery"
    required = (
        "transport_joints_deg",
        "transport_pose_verified",
        "shelf_ready",
        "shelf_ready_verified",
        "source_lift_height_mm",
        "target_lift_height_mm",
        "target_lift_tolerance_mm",
        "lift_transition_verified",
        "body_lift_speed",
        "body_rotation_yaw_deg",
        "max_angular_speed_radps",
        "rotation_tolerance_deg",
        "rotation_timeout_s",
        "max_base_translation_m",
        "rotation_sweep",
        "table_roi",
        "table_roi_verified",
        "workspace_verified",
        "keepouts_verified",
        "bottle_bottom_below_tcp_m",
        "held_bottle_height_m",
        "held_bottle_diameter_m",
        "held_bottle_guard_padding_m",
        "bottle_tcp_verified",
        "preplace_clearance_m",
        "retreat_standoff_m",
        "table_height_bin_m",
        "table_inlier_band_m",
        "table_min_inliers",
        "table_frame_agreement_m",
        "table_edge_margin_m",
        "table_support_radius_m",
        "table_min_patch_points",
        "place_clearance_radius_m",
        "place_grid_m",
        "obstacle_min_height_m",
        "obstacle_max_height_m",
        "max_place_candidates",
        "refresh_height_tolerance_m",
        "refresh_xy_tolerance_m",
    )
    missing = [key for key in required if key not in data or data[key] is None]
    if missing:
        raise SafetyAbort(
            f"profile {profile_name} 的桌面送货参数未完成: {missing}"
        )
    transport = _finite_vector(
        _required_value(data, "transport_joints_deg", label=label),
        7,
        label=f"profile {profile_name}.transport_joints_deg",
    )
    shelf_raw = _required_mapping(data, "shelf_ready", label=label)
    shelf_label = f"{label}.shelf_ready"
    shelf_required = (
        "x_m",
        "y_m",
        "yaw_deg",
        "lift_height_mm",
        "xy_tolerance_m",
        "yaw_tolerance_deg",
        "lift_tolerance_mm",
    )
    shelf_missing = [
        key
        for key in shelf_required
        if key not in shelf_raw or shelf_raw[key] is None
    ]
    if shelf_missing:
        raise SafetyAbort(f"{shelf_label} 未完成: {shelf_missing}")
    shelf_ready = ShelfReadyConfig(
        x_m=_finite_float(
            _required_value(shelf_raw, "x_m", label=shelf_label),
            label=f"{shelf_label}.x_m",
        ),
        y_m=_finite_float(
            _required_value(shelf_raw, "y_m", label=shelf_label),
            label=f"{shelf_label}.y_m",
        ),
        yaw_deg=_finite_float(
            _required_value(shelf_raw, "yaw_deg", label=shelf_label),
            label=f"{shelf_label}.yaw_deg",
        ),
        lift_height_mm=_finite_int(
            _required_value(shelf_raw, "lift_height_mm", label=shelf_label),
            label=f"{shelf_label}.lift_height_mm",
        ),
        xy_tolerance_m=_finite_float(
            _required_value(shelf_raw, "xy_tolerance_m", label=shelf_label),
            label=f"{shelf_label}.xy_tolerance_m",
        ),
        yaw_tolerance_deg=_finite_float(
            _required_value(
                shelf_raw, "yaw_tolerance_deg", label=shelf_label
            ),
            label=f"{shelf_label}.yaw_tolerance_deg",
        ),
        lift_tolerance_mm=_finite_int(
            _required_value(
                shelf_raw, "lift_tolerance_mm", label=shelf_label
            ),
            label=f"{shelf_label}.lift_tolerance_mm",
        ),
    )
    roi = _required_mapping(data, "table_roi", label=label)
    roi_min = _finite_vector(
        _required_value(roi, "min", label=f"{label}.table_roi"),
        3,
        label="table_roi.min",
    )
    roi_max = _finite_vector(
        _required_value(roi, "max", label=f"{label}.table_roi"),
        3,
        label="table_roi.max",
    )
    if np.any(np.asarray(roi_max) <= np.asarray(roi_min)):
        raise SafetyAbort("table_roi.max 必须逐轴大于 min")
    sweep_raw = _required_mapping(data, "rotation_sweep", label=label)

    def load_sweep_direction(direction: str) -> tuple[float, bool]:
        direction_raw = _required_mapping(
            sweep_raw, direction, label=f"{label}.rotation_sweep"
        )
        direction_label = f"{label}.rotation_sweep.{direction}"
        clearance = _finite_float(
            _required_value(
                direction_raw, "clearance_m", label=direction_label
            ),
            label=f"{direction_label}.clearance_m",
        )
        verified = _strict_bool(
            _required_value(direction_raw, "verified", label=direction_label),
            label=f"{direction_label}.verified",
        )
        return clearance, verified

    positive_clearance_m, positive_verified = load_sweep_direction("positive")
    negative_clearance_m, negative_verified = load_sweep_direction("negative")
    rotation_sweep = RotationSweepClearance(
        positive_clearance_m=positive_clearance_m,
        negative_clearance_m=negative_clearance_m,
        positive_verified=positive_verified,
        negative_verified=negative_verified,
    )
    config = SideTableDeliveryConfig(
        transport_joints_deg=transport,
        transport_pose_verified=_strict_bool(
            _required_value(data, "transport_pose_verified", label=label),
            label=f"{label}.transport_pose_verified",
        ),
        shelf_ready=shelf_ready,
        shelf_ready_verified=_strict_bool(
            _required_value(data, "shelf_ready_verified", label=label),
            label=f"{label}.shelf_ready_verified",
        ),
        source_lift_height_mm=_finite_int(
            _required_value(data, "source_lift_height_mm", label=label),
            label=f"{label}.source_lift_height_mm",
        ),
        target_lift_height_mm=_finite_int(
            _required_value(data, "target_lift_height_mm", label=label),
            label=f"{label}.target_lift_height_mm",
        ),
        target_lift_tolerance_mm=_finite_int(
            _required_value(data, "target_lift_tolerance_mm", label=label),
            label=f"{label}.target_lift_tolerance_mm",
        ),
        lift_transition_verified=_strict_bool(
            _required_value(data, "lift_transition_verified", label=label),
            label=f"{label}.lift_transition_verified",
        ),
        body_lift_speed=_finite_int(
            _required_value(data, "body_lift_speed", label=label),
            label=f"{label}.body_lift_speed",
        ),
        body_rotation_yaw_deg=_finite_float(
            _required_value(data, "body_rotation_yaw_deg", label=label),
            label=f"{label}.body_rotation_yaw_deg",
        ),
        max_angular_speed_radps=_finite_float(
            _required_value(data, "max_angular_speed_radps", label=label),
            label=f"{label}.max_angular_speed_radps",
        ),
        rotation_tolerance_deg=_finite_float(
            _required_value(data, "rotation_tolerance_deg", label=label),
            label=f"{label}.rotation_tolerance_deg",
        ),
        rotation_timeout_s=_finite_float(
            _required_value(data, "rotation_timeout_s", label=label),
            label=f"{label}.rotation_timeout_s",
        ),
        max_base_translation_m=_finite_float(
            _required_value(data, "max_base_translation_m", label=label),
            label=f"{label}.max_base_translation_m",
        ),
        rotation_sweep=rotation_sweep,
        table_roi_min=roi_min,
        table_roi_max=roi_max,
        table_roi_verified=_strict_bool(
            _required_value(data, "table_roi_verified", label=label),
            label=f"{label}.table_roi_verified",
        ),
        workspace_verified=_strict_bool(
            _required_value(data, "workspace_verified", label=label),
            label=f"{label}.workspace_verified",
        ),
        keepouts_verified=_strict_bool(
            _required_value(data, "keepouts_verified", label=label),
            label=f"{label}.keepouts_verified",
        ),
        bottle_bottom_below_tcp_m=_finite_float(
            _required_value(data, "bottle_bottom_below_tcp_m", label=label),
            label=f"{label}.bottle_bottom_below_tcp_m",
        ),
        held_bottle_height_m=_finite_float(
            _required_value(data, "held_bottle_height_m", label=label),
            label=f"{label}.held_bottle_height_m",
        ),
        held_bottle_diameter_m=_finite_float(
            _required_value(data, "held_bottle_diameter_m", label=label),
            label=f"{label}.held_bottle_diameter_m",
        ),
        held_bottle_guard_padding_m=_finite_float(
            _required_value(
                data, "held_bottle_guard_padding_m", label=label
            ),
            label=f"{label}.held_bottle_guard_padding_m",
        ),
        bottle_tcp_verified=_strict_bool(
            _required_value(data, "bottle_tcp_verified", label=label),
            label=f"{label}.bottle_tcp_verified",
        ),
        preplace_clearance_m=_finite_float(
            _required_value(data, "preplace_clearance_m", label=label),
            label=f"{label}.preplace_clearance_m",
        ),
        retreat_standoff_m=_finite_float(
            _required_value(data, "retreat_standoff_m", label=label),
            label=f"{label}.retreat_standoff_m",
        ),
        table_height_bin_m=_finite_float(
            _required_value(data, "table_height_bin_m", label=label),
            label=f"{label}.table_height_bin_m",
        ),
        table_inlier_band_m=_finite_float(
            _required_value(data, "table_inlier_band_m", label=label),
            label=f"{label}.table_inlier_band_m",
        ),
        table_min_inliers=_finite_int(
            _required_value(data, "table_min_inliers", label=label),
            label=f"{label}.table_min_inliers",
        ),
        table_frame_agreement_m=_finite_float(
            _required_value(data, "table_frame_agreement_m", label=label),
            label=f"{label}.table_frame_agreement_m",
        ),
        table_edge_margin_m=_finite_float(
            _required_value(data, "table_edge_margin_m", label=label),
            label=f"{label}.table_edge_margin_m",
        ),
        table_support_radius_m=_finite_float(
            _required_value(data, "table_support_radius_m", label=label),
            label=f"{label}.table_support_radius_m",
        ),
        table_min_patch_points=_finite_int(
            _required_value(data, "table_min_patch_points", label=label),
            label=f"{label}.table_min_patch_points",
        ),
        place_clearance_radius_m=_finite_float(
            _required_value(data, "place_clearance_radius_m", label=label),
            label=f"{label}.place_clearance_radius_m",
        ),
        place_grid_m=_finite_float(
            _required_value(data, "place_grid_m", label=label),
            label=f"{label}.place_grid_m",
        ),
        obstacle_min_height_m=_finite_float(
            _required_value(data, "obstacle_min_height_m", label=label),
            label=f"{label}.obstacle_min_height_m",
        ),
        obstacle_max_height_m=_finite_float(
            _required_value(data, "obstacle_max_height_m", label=label),
            label=f"{label}.obstacle_max_height_m",
        ),
        max_place_candidates=_finite_int(
            _required_value(data, "max_place_candidates", label=label),
            label=f"{label}.max_place_candidates",
        ),
        refresh_height_tolerance_m=_finite_float(
            _required_value(data, "refresh_height_tolerance_m", label=label),
            label=f"{label}.refresh_height_tolerance_m",
        ),
        refresh_xy_tolerance_m=_finite_float(
            _required_value(data, "refresh_xy_tolerance_m", label=label),
            label=f"{label}.refresh_xy_tolerance_m",
        ),
    )
    if not (0 <= config.shelf_ready.lift_height_mm <= 2600):
        raise SafetyAbort("shelf_ready.lift_height_mm 必须在 0..2600")
    if not (0.005 <= config.shelf_ready.xy_tolerance_m <= 0.20):
        raise SafetyAbort("shelf_ready.xy_tolerance_m 必须在 5..200 mm")
    if not (0.5 <= config.shelf_ready.yaw_tolerance_deg <= 5.0):
        raise SafetyAbort("shelf_ready.yaw_tolerance_deg 必须在 0.5..5.0")
    if not (1 <= config.shelf_ready.lift_tolerance_mm <= 30):
        raise SafetyAbort("shelf_ready.lift_tolerance_mm 必须在 1..30 mm")
    if config.source_lift_height_mm != config.shelf_ready.lift_height_mm:
        raise SafetyAbort(
            "source_lift_height_mm 必须与 shelf_ready.lift_height_mm 相同"
        )
    if not (0 <= config.target_lift_height_mm <= 2600):
        raise SafetyAbort("target_lift_height_mm 必须在 0..2600")
    if not (1 <= config.target_lift_tolerance_mm <= 30):
        raise SafetyAbort("target_lift_tolerance_mm 必须在 1..30 mm")
    if not (1 <= config.body_lift_speed <= 30):
        raise SafetyAbort("body_lift_speed 必须在 1..30")
    if not (80.0 <= abs(config.body_rotation_yaw_deg) <= 100.0):
        raise SafetyAbort("body_rotation_yaw_deg 必须是约 90°")
    if not (0.03 <= config.max_angular_speed_radps <= 0.20):
        raise SafetyAbort("max_angular_speed_radps 必须在 0.03..0.20")
    if not (0.5 <= config.rotation_tolerance_deg <= 5.0):
        raise SafetyAbort("rotation_tolerance_deg 必须在 0.5..5.0")
    if not (5.0 <= config.rotation_timeout_s <= 60.0):
        raise SafetyAbort("rotation_timeout_s 必须在 5..60")
    if not (0.005 <= config.max_base_translation_m <= 0.08):
        raise SafetyAbort("max_base_translation_m 必须在 5..80 mm")
    if not (
        np.isfinite(profile_clearance_m) and profile_clearance_m > 0.0
    ):
        raise SafetyAbort("side-table profile clearance_m 必须是正的有限数")
    if (
        config.rotation_sweep.positive_clearance_m < profile_clearance_m
        or config.rotation_sweep.negative_clearance_m < profile_clearance_m
    ):
        raise SafetyAbort("正反向旋转扫掠净空不得小于 profile clearance_m")
    if not (0.03 <= config.bottle_bottom_below_tcp_m <= 0.40):
        raise SafetyAbort("bottle_bottom_below_tcp_m 超出 3..40 cm 合理范围")
    if not (0.08 <= config.held_bottle_height_m <= 0.50):
        raise SafetyAbort("held_bottle_height_m 超出 8..50 cm 合理范围")
    if not (0.03 <= config.held_bottle_diameter_m <= 0.20):
        raise SafetyAbort("held_bottle_diameter_m 超出 3..20 cm 合理范围")
    if not (0.0 <= config.held_bottle_guard_padding_m <= 0.06):
        raise SafetyAbort("held_bottle_guard_padding_m 必须在 0..6 cm")
    if not (0.05 <= config.preplace_clearance_m <= 0.40):
        raise SafetyAbort("preplace_clearance_m 必须在 5..40 cm")
    if not (0.05 <= config.retreat_standoff_m <= 0.40):
        raise SafetyAbort("retreat_standoff_m 必须在 5..40 cm")
    if not (0.002 <= config.table_height_bin_m <= 0.03):
        raise SafetyAbort("table_height_bin_m 必须在 2..30 mm")
    if not (0.003 <= config.table_inlier_band_m <= 0.04):
        raise SafetyAbort("table_inlier_band_m 必须在 3..40 mm")
    if not (0.003 <= config.table_frame_agreement_m <= 0.03):
        raise SafetyAbort("table_frame_agreement_m 必须在 3..30 mm")
    if not (0.03 <= config.table_edge_margin_m <= 0.30):
        raise SafetyAbort("table_edge_margin_m 必须在 3..30 cm")
    table_extent = np.asarray(config.table_roi_max) - np.asarray(
        config.table_roi_min
    )
    if np.any(table_extent[:2] <= 2.0 * config.table_edge_margin_m):
        raise SafetyAbort("桌面 ROI 扣除双侧边缘余量后必须仍有空间")
    if not (0.02 <= config.table_support_radius_m <= 0.20):
        raise SafetyAbort("table_support_radius_m 必须在 2..20 cm")
    if not (0.04 <= config.place_clearance_radius_m <= 0.35):
        raise SafetyAbort("place_clearance_radius_m 必须在 4..35 cm")
    if config.place_clearance_radius_m < (
        config.held_bottle_diameter_m / 2.0
        + config.held_bottle_guard_padding_m
    ):
        raise SafetyAbort("放置净空半径小于瓶体附着包络半径")
    if not (0.01 <= config.place_grid_m <= 0.10):
        raise SafetyAbort("place_grid_m 必须在 1..10 cm")
    if not (
        0.0 <= config.obstacle_min_height_m
        < config.obstacle_max_height_m
        <= 1.0
    ):
        raise SafetyAbort("桌面障碍物高度带无效")
    if not (0.003 <= config.refresh_height_tolerance_m <= 0.03):
        raise SafetyAbort("refresh_height_tolerance_m 必须在 3..30 mm")
    if not (0.005 <= config.refresh_xy_tolerance_m <= 0.08):
        raise SafetyAbort("refresh_xy_tolerance_m 必须在 5..80 mm")
    if (
        config.table_min_inliers < 20
        or config.table_min_patch_points < 3
        or not (1 <= config.max_place_candidates <= 32)
    ):
        raise SafetyAbort("输出桌面采样门槛无效")
    if require_verified:
        unverified = [
            label
            for label, verified in (
                ("transport_pose", config.transport_pose_verified),
                ("shelf_ready", config.shelf_ready_verified),
                ("lift_transition", config.lift_transition_verified),
                ("table_roi", config.table_roi_verified),
                ("workspace", config.workspace_verified),
                ("keepouts", config.keepouts_verified),
                ("bottle_tcp", config.bottle_tcp_verified),
                ("rotation_sweep.positive", config.rotation_sweep.positive_verified),
                ("rotation_sweep.negative", config.rotation_sweep.negative_verified),
            )
            if not verified
        ]
        if unverified:
            raise SafetyAbort(
                "side_table_delivery 尚未完成现场确认: "
                + ", ".join(unverified)
            )
    return config


def load_safety_profile(
    path: str | Path,
    profile_name: str,
    *,
    require_verified: bool,
    expected_profile_sha256: str | None = None,
) -> SafetyProfile:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SafetyAbort(f"电子围栏配置不存在: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyAbort(f"电子围栏配置无法读取: {path}: {exc}") from exc
    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        raise SafetyAbort(f"电子围栏 profile 不存在: {profile_name}")
    raw = profiles[profile_name]
    if not isinstance(raw, dict):
        raise SafetyAbort(f"电子围栏 profile 必须是对象: {profile_name}")
    if expected_profile_sha256 is not None:
        if (
            not isinstance(expected_profile_sha256, str)
            or not expected_profile_sha256
        ):
            raise SafetyAbort("run manifest 的 profile digest 无效")
        actual_profile_sha256 = _profile_payload_sha256(raw)
        if actual_profile_sha256 != expected_profile_sha256:
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 与 run manifest 记录不一致，"
                "拒绝加载不可复现的执行配置"
            )
    enabled = _strict_bool(
        raw.get("enabled"), label=f"电子围栏 profile {profile_name}.enabled"
    )
    if not enabled:
        raise SafetyAbort(f"电子围栏 profile 尚未启用: {profile_name}")
    verified = _strict_bool(
        raw.get("verified_for_execution"),
        label=f"电子围栏 profile {profile_name}.verified_for_execution",
    )
    if require_verified and not verified:
        raise SafetyAbort(
            f"电子围栏 profile {profile_name} 尚未现场测量确认，禁止真机执行"
        )
    frame = str(raw.get("frame", ""))
    if frame != "right_controller_base":
        raise SafetyAbort(
            f"电子围栏坐标系必须是 right_controller_base，当前为 {frame!r}"
        )
    moveit_frame = str(raw.get("moveit_frame", ""))
    if moveit_frame != "platform_base_link":
        raise SafetyAbort(
            f"MoveIt 围栏坐标系必须是 platform_base_link，当前为 {moveit_frame!r}"
        )
    transform = np.asarray(raw.get("T_moveit_from_profile"), dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise SafetyAbort("T_moveit_from_profile 必须是有效 4x4 矩阵")
    # 允许各轴 ±1 的对角旋转（每个轴仍映射到自身，只翻符号）：盒子的
    # size 语义与 min/max 重排在这种变换下保持成立。实测 right_controller_base
    # 相对 platform_base_link 是 yaw 180°（diag(-1,-1,1)），纯平移是错的。
    rotation = transform[:3, :3]
    if not np.allclose(np.abs(rotation), np.eye(3), atol=1e-6):
        raise SafetyAbort(
            "当前长方体围栏只支持与 platform_base_link 轴对齐"
            "（各轴仅允许±翻转）的坐标变换"
        )
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise SafetyAbort(
            "T_moveit_from_profile 旋转部分行列式必须为 +1（不允许镜像反射）"
        )
    clearance_m = _finite_float(
        _required_value(raw, "clearance_m", label=f"profile {profile_name}"),
        label=f"profile {profile_name}.clearance_m",
    )
    # A generic no-environment profile intentionally uses zero clearance.
    # Side-table delivery applies its stricter positive-clearance requirement
    # in `_load_side_table_delivery` because it carries a bottle through a
    # measured rotation envelope.
    if clearance_m < 0.0:
        raise SafetyAbort("clearance_m 必须是非负的有限数")
    workspace = FenceBox.from_dict(
        raw.get("tcp_workspace", {}), prefix="tcp_workspace"
    )
    zones = tuple(
        FenceBox.from_dict(item, prefix=f"allowed_{index}")
        for index, item in enumerate(raw.get("allowed_tcp_zones", []))
    )
    if not zones:
        raise SafetyAbort(f"电子围栏 profile {profile_name} 没有允许区")
    keepouts = tuple(
        FenceBox.from_dict(item, prefix=f"keepout_{index}")
        for index, item in enumerate(raw.get("keepout_boxes", []))
    )
    home_raw = raw.get("home_joints_deg")
    home_joints_deg = None
    if home_raw is not None:
        home = np.asarray(home_raw, dtype=float)
        if home.shape != (7,) or not np.all(np.isfinite(home)):
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的 home_joints_deg 必须是 7 个有限数"
            )
        home_joints_deg = tuple(map(float, home))
    grasp_start_keys = (
        "grasp_start_right_joints_deg",
        "grasp_start_left_joints_deg",
        "grasp_start_lift_height_mm",
    )
    grasp_start_values = [raw.get(key) for key in grasp_start_keys]
    if any(value is not None for value in grasp_start_values) and not all(
        value is not None for value in grasp_start_values
    ):
        raise SafetyAbort(
            f"电子围栏 profile {profile_name} 的抓取初始位必须同时配置双臂关节和升降高度"
        )
    if profile_name == "shelf_template" and not all(
        value is not None for value in grasp_start_values
    ):
        raise SafetyAbort("shelf_template 必须配置完整的双臂/升降抓取初始位")
    grasp_start_right_joints_deg = None
    grasp_start_left_joints_deg = None
    grasp_start_lift_height_mm = None
    if all(value is not None for value in grasp_start_values):
        grasp_start_right = np.asarray(grasp_start_values[0], dtype=float)
        grasp_start_left = np.asarray(grasp_start_values[1], dtype=float)
        if (
            grasp_start_right.shape != (7,)
            or grasp_start_left.shape != (7,)
            or not np.all(np.isfinite(grasp_start_right))
            or not np.all(np.isfinite(grasp_start_left))
        ):
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的抓取初始位双臂关节必须各为 7 个有限数"
            )
        lift_height = grasp_start_values[2]
        if (
            isinstance(lift_height, bool)
            or not isinstance(lift_height, int)
            or not 0 <= lift_height <= 2600
        ):
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的 "
                "grasp_start_lift_height_mm 必须是 0..2600 的整数"
            )
        grasp_start_right_joints_deg = tuple(map(float, grasp_start_right))
        grasp_start_left_joints_deg = tuple(map(float, grasp_start_left))
        grasp_start_lift_height_mm = int(lift_height)
    staging_raw = raw.get("observation_staging_joints_deg")
    observation_staging_joints_deg = None
    if staging_raw is not None:
        staging = np.asarray(staging_raw, dtype=float)
        if staging.shape != (7,) or not np.all(np.isfinite(staging)):
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的 "
                "observation_staging_joints_deg 必须是 7 个有限数"
            )
        observation_staging_joints_deg = tuple(map(float, staging))
    output_raw = raw.get("output_joints_deg")
    output_joints_deg = None
    if output_raw is not None:
        output = np.asarray(output_raw, dtype=float)
        if output.shape != (7,) or not np.all(np.isfinite(output)):
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的 output_joints_deg 必须是 7 个有限数"
            )
        output_joints_deg = tuple(map(float, output))
    output_point_raw = raw.get("output_point_base")
    output_point_base = None
    if output_point_raw is not None:
        output_point = np.asarray(output_point_raw, dtype=float)
        if output_point.shape != (3,) or not np.all(np.isfinite(output_point)):
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的 output_point_base 必须是 3 个有限数"
            )
        output_point_base = tuple(map(float, output_point))
    grasp_height_raw = raw.get("grasp_height_fraction")
    grasp_height_fraction = None
    if grasp_height_raw is not None:
        try:
            grasp_height_fraction = float(grasp_height_raw)
        except (TypeError, ValueError) as exc:
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的 "
                "grasp_height_fraction 必须是 0..1 之间的有限数"
            ) from exc
        if not (
            np.isfinite(grasp_height_fraction)
            and 0.0 < grasp_height_fraction < 1.0
        ):
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的 "
                "grasp_height_fraction 必须是 0..1 之间的有限数"
            )
    demonstrated_raw = raw.get("demonstrated_grasp_right_joints_deg")
    demonstrated_grasp_right_joints_deg = None
    if demonstrated_raw is not None:
        values = tuple(
            _finite_float(
                item,
                label=f"电子围栏 profile {profile_name} 的 "
                "demonstrated_grasp_right_joints_deg",
            )
            for item in demonstrated_raw
        )
        if len(values) != 7:
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的 "
                "demonstrated_grasp_right_joints_deg 必须是七个有限数"
            )
        demonstrated_grasp_right_joints_deg = values
    gripper_max_opening_raw = raw.get("gripper_max_opening_m")
    gripper_max_opening_m = None
    if gripper_max_opening_raw is not None:
        gripper_max_opening_m = _finite_float(
            gripper_max_opening_raw,
            label=f"电子围栏 profile {profile_name} 的 gripper_max_opening_m",
        )
        if not 0.0 < gripper_max_opening_m <= 0.5:
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的 "
                "gripper_max_opening_m 必须在 0..0.5 m"
            )
    grasp_frame = None
    grasp_frame_raw = raw.get("grasp_frame")
    if grasp_frame_raw is not None:
        if not isinstance(grasp_frame_raw, dict):
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的 grasp_frame 必须是对象"
            )
        try:
            grasp_frame = GraspFrameSpec(
                opening_normal_base=tuple(
                    grasp_frame_raw["opening_normal_base"]
                ),
                finger_axis_base=tuple(grasp_frame_raw["finger_axis_base"]),
                palm_vertical_base=tuple(
                    grasp_frame_raw["palm_vertical_base"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的 grasp_frame 缺少轴语义"
            ) from exc
        # Validate once while loading the profile, before any hardware motion.
        authored_tcp_rotation(grasp_frame)
    tool_mount_calibration = _load_tool_mount_calibration(
        raw,
        profile_name=profile_name,
    )
    left_tool_mount_calibration = _load_tool_mount_calibration(
        raw,
        profile_name=profile_name,
        key="left_tool_mount_calibration",
    )
    # This is a property of the installed right-hand tool, not of one
    # particular grasp authoring mode.  Requiring it for every real execution
    # profile prevents a future profile from silently deleting grasp_frame and
    # falling back to the historical identity/+Z installation assumption.
    if require_verified:
        if (
            tool_mount_calibration is None
            or not tool_mount_calibration.verified
        ):
            raise SafetyAbort(
                f"电子围栏 profile {profile_name} 的右臂执行需要已实测的 "
                "tool_mount_calibration；禁止猜测 link7/夹爪安装旋转"
            )
    side_table_delivery = _load_side_table_delivery(
        raw,
        profile_name=profile_name,
        require_verified=require_verified,
        profile_clearance_m=clearance_m,
    )
    if side_table_delivery is not None and not keepouts:
        raise SafetyAbort("side_table_delivery 必须至少配置一个实测 keepout_boxes")
    profile = SafetyProfile(
        name=profile_name,
        description=str(raw.get("description", "")),
        frame=frame,
        moveit_frame=moveit_frame,
        T_moveit_from_profile=transform,
        verified_for_execution=verified,
        clearance_m=clearance_m,
        tcp_workspace=workspace,
        allowed_tcp_zones=zones,
        keepout_boxes=keepouts,
        use_dynamic_rgbd=bool(raw.get("use_dynamic_rgbd", True)),
        home_joints_deg=home_joints_deg,
        grasp_start_right_joints_deg=grasp_start_right_joints_deg,
        grasp_start_left_joints_deg=grasp_start_left_joints_deg,
        grasp_start_lift_height_mm=grasp_start_lift_height_mm,
        observation_staging_joints_deg=observation_staging_joints_deg,
        output_joints_deg=output_joints_deg,
        output_visible_to_head_camera=bool(
            raw.get("output_visible_to_head_camera", False)
        ),
        output_point_base=output_point_base,
        side_table_delivery=side_table_delivery,
        grasp_height_fraction=grasp_height_fraction,
        gripper_max_opening_m=gripper_max_opening_m,
        demonstrated_grasp_right_joints_deg=demonstrated_grasp_right_joints_deg,
        grasp_frame=grasp_frame,
        tool_mount_calibration=tool_mount_calibration,
        left_tool_mount_calibration=left_tool_mount_calibration,
    )
    # Validate that each allowed zone is itself inside the global workspace.
    for zone in profile.allowed_tcp_zones:
        if not (
            profile.tcp_workspace.contains(zone.minimum, 0.0)
            and profile.tcp_workspace.contains(zone.maximum, 0.0)
        ):
            raise SafetyAbort(
                f"允许区 {zone.id} 超出总工作空间 {profile.tcp_workspace.id}"
            )
    return profile
