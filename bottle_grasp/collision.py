"""Wrist point-cloud safety gate for the straight gripper approach corridor."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .core import DemoParams, SafetyAbort


def classify_moveit_collision_probe(
    *, baseline_valid: bool, boxed_valid: bool, cleared_valid: bool
) -> tuple[str, str]:
    """Classify the three-state MoveIt world-collision probe accurately."""
    if not baseline_valid:
        return (
            "baseline_invalid",
            "无障碍基线姿态已经碰撞，探针姿态或残留场景无效",
        )
    if boxed_valid:
        return (
            "collision_missed",
            "巨型障碍盒中的姿态仍被判有效，世界碰撞检测未生效",
        )
    if not cleared_valid:
        return (
            "cleanup_failed",
            "撤除巨型障碍盒后没有恢复，规划场景清理或同步失败",
        )
    return "healthy", "基线、碰撞拒绝和场景恢复均符合预期"


def check_approach_corridor(
    *,
    camera: Any,
    robot: Any,
    target_box: Sequence[int] | None,
    target_base: np.ndarray,
    T_flange_camera: np.ndarray,
    params: DemoParams,
    corridor_waypoints_base: Sequence[Sequence[float]] | None = None,
) -> int:
    """Return blocker count or abort if the commanded TCP corridor is occupied.

    The target's own occupied cylinder is always removed.  A current wrist box
    narrows that removal further to ``box ∩ cylinder``; head-only confirmation
    uses the cylinder alone.  Neither case creates a broad clearance hole.
    When waypoints are supplied, check the actual piecewise-linear local path
    instead of the obsolete direct chord from the observation pose to target.
    """
    _, depth = camera.get_latest_frames()
    K, _ = camera.get_camera_intrinsics()
    if depth is None or K is None:
        raise SafetyAbort("通道检查缺少深度")

    stride = 3
    vv, uu = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    zz = depth[vv, uu]
    valid = (
        np.isfinite(zz)
        & (zz > params.min_depth_m)
        & (zz < params.max_depth_m)
    )
    z = zz[valid]
    u, v = uu[valid], vv[valid]
    camera_points = np.column_stack(
        (
            (u - K[0, 2]) * z / K[0, 0],
            (v - K[1, 2]) * z / K[1, 1],
            z,
            np.ones_like(z),
        )
    )
    T_base_camera = robot.current_flange() @ T_flange_camera
    points = (T_base_camera @ camera_points.T).T[:, :3]

    target = np.asarray(target_base, dtype=float)
    vertical_target_band = (
        (
            points[:, 2]
            >= target[2] - params.target_occupancy_below_grasp_m
        )
        & (
            points[:, 2]
            <= target[2] + params.target_occupancy_above_grasp_m
        )
    )
    if target_box is not None:
        x1, y1, x2, y2 = target_box
        in_current_box = (
            (u >= x1 - params.target_occupancy_box_pad_px)
            & (u <= x2 + params.target_occupancy_box_pad_px)
            & (v >= y1 - params.target_occupancy_box_pad_px)
            & (v <= y2 + params.target_occupancy_box_pad_px)
        )
        target_camera = (
            np.linalg.inv(T_base_camera) @ np.r_[target, 1.0]
        )[:3]
        if not np.all(np.isfinite(target_camera)):
            raise SafetyAbort("通道检查的目标相机坐标无效")
        # The detector silhouette plus a bounded optical-depth slab captures
        # both transparent bottle surfaces.  A whole-box mask would erase any
        # foreground object; the depth slab keeps such an object when it is
        # farther than one bottle radius plus sensor slack from the lock.
        target_depth_band = (
            np.abs(z - float(target_camera[2]))
            <= params.target_occupancy_radius_m
            + params.target_occupancy_depth_slack_m
        )
        target_samples = (
            in_current_box & target_depth_band & vertical_target_band
        )
    else:
        radial = np.linalg.norm(points[:, :2] - target[:2], axis=1)
        target_samples = (
            (radial <= params.target_occupancy_radius_m)
            & vertical_target_band
        )
    points = points[~target_samples]

    start = np.asarray(robot.current_tcp()[:3, 3], dtype=float)
    if corridor_waypoints_base is None:
        polyline = np.vstack((start, target))
        end_progress = 0.86
    else:
        waypoints = np.asarray(corridor_waypoints_base, dtype=float)
        if (
            waypoints.ndim != 2
            or waypoints.shape[1] != 3
            or not np.all(np.isfinite(waypoints))
        ):
            raise SafetyAbort("夹爪通道折线路径坐标无效")
        polyline = np.vstack((start, waypoints))
        # The supplied route ends at the pregrasp hover rather than at the
        # bottle contact point, so its tail remains ordinary free space.
        end_progress = 0.98

    segment_starts = polyline[:-1]
    vectors = np.diff(polyline, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    usable = lengths > 1e-9
    if not np.any(usable):
        return 0
    segment_starts = segment_starts[usable]
    vectors = vectors[usable]
    lengths = lengths[usable]
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    total_length = float(cumulative[-1])
    distance = np.full(points.shape[0], np.inf, dtype=float)
    progress = np.zeros(points.shape[0], dtype=float)
    for index, (segment_start, vector, length) in enumerate(
        zip(segment_starts, vectors, lengths)
    ):
        local = np.clip(
            ((points - segment_start) @ vector) / float(length * length),
            0.0,
            1.0,
        )
        candidate_distance = np.linalg.norm(
            points - (segment_start + local[:, None] * vector), axis=1
        )
        closer = candidate_distance < distance
        distance[closer] = candidate_distance[closer]
        progress[closer] = (
            cumulative[index] + local[closer] * length
        ) / total_length
    blockers = (
        (progress > 0.08)
        & (progress < end_progress)
        & (distance < params.corridor_radius_m)
    )
    count = int(np.count_nonzero(blockers))
    if count >= params.obstacle_min_points:
        raise SafetyAbort(f"夹爪前进通道被点云阻挡: {count} 点")
    return count
