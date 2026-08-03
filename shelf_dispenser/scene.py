"""RGB-D conversion into environment-independent MoveIt collision voxels."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .core import DemoParams, Localization, SafetyAbort


def _base_points(
    depth: np.ndarray,
    K: np.ndarray,
    T_base_camera: np.ndarray,
    params: DemoParams,
    stride: int = 6,
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
    bottom_crop: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bottom_crop is None:
        bottom_crop = params.scene_image_bottom_crop
    max_v = min(depth.shape[0], bottom_crop)
    vv, uu = np.mgrid[0:max_v:stride, 0 : depth.shape[1] : stride]
    zz = depth[vv, uu]
    min_depth_m = (
        params.head_min_depth_m if min_depth_m is None else min_depth_m
    )
    max_depth_m = (
        params.head_max_depth_m if max_depth_m is None else max_depth_m
    )
    valid = (
        np.isfinite(zz)
        & (zz >= min_depth_m)
        & (zz <= max_depth_m)
    )
    z = zz[valid]
    if z.size < 20:
        raise SafetyAbort("头部点云有效点不足")
    u, v = uu[valid], vv[valid]
    camera_points = np.column_stack(
        (
            (u - K[0, 2]) * z / K[0, 0],
            (v - K[1, 2]) * z / K[1, 1],
            z,
            np.ones_like(z),
        )
    )
    return (
        (T_base_camera @ camera_points.T).T[:, :3],
        u,
        v,
        camera_points[:, :3],
    )


def head_scene_points(
    depth: np.ndarray,
    K: np.ndarray,
    T_base_camera: np.ndarray,
    params: DemoParams,
    *,
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
    bottom_crop: int | None = None,
) -> np.ndarray:
    """Return the raw head point cloud in the right-arm base frame."""
    if depth is None or K is None:
        raise SafetyAbort("头部点云缺少深度或内参")
    points, _, _, _ = _base_points(
        depth,
        K,
        T_base_camera,
        params,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        bottom_crop=bottom_crop,
    )
    return points


def build_scene_voxels(
    depth: np.ndarray,
    K: np.ndarray,
    T_base_camera: np.ndarray,
    localization: Localization,
    params: DemoParams,
    *,
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
    bottom_crop: int | None = None,
    max_voxels: int | None = None,
) -> list[list[float]]:
    """Return occupied voxel centers in the right-arm base frame.

    The lower image strip is excluded because the robot's own arms and body
    occupy it in the fixed head-camera view. MoveIt handles robot self-collision
    from the URDF/SRDF; feeding those pixels back as world obstacles would make
    the start state falsely collide with itself.
    """
    if depth is None or K is None:
        raise SafetyAbort("头部点云缺少深度或内参")
    points, _, _, _ = _base_points(
        depth,
        K,
        T_base_camera,
        params,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        bottom_crop=bottom_crop,
    )
    target = np.asarray(localization.point_base, dtype=float)

    # Keep the local arm work volume, not the entire room.  Do not delete the
    # detector box or a target-centred sphere here.  The global leg stops at a
    # wrist observation pose and has no reason to contact the bottle; the old
    # 14 cm target hole could silently erase a real obstacle beside it.  Local
    # grasp contact is handled later by the wrist corridor and grasp recipe.
    relative = points - target
    valid_workspace = (
        (np.linalg.norm(relative[:, :2], axis=1) < 0.95)
        & (np.abs(relative[:, 2]) < 0.75)
    )
    points = points[valid_workspace]
    if points.size == 0:
        return []

    return voxelize_scene_points(points, params, max_voxels=max_voxels)


def _target_point_mask(
    points: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    camera_points: np.ndarray,
    localization: Localization,
    params: DemoParams,
) -> np.ndarray:
    target = np.asarray(localization.point_base, dtype=float)
    target_camera = np.asarray(localization.point_camera, dtype=float)
    box = np.asarray(localization.box, dtype=float)
    if (
        target.shape != (3,)
        or target_camera.shape != (3,)
        or box.shape != (4,)
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(target_camera))
        or not np.all(np.isfinite(box))
        or not np.isfinite(float(localization.depth_m))
    ):
        raise SafetyAbort("目标占据体素的定位结果无效")
    x1, y1, x2, y2 = box
    in_box = (
        (u >= x1 - params.target_occupancy_box_pad_px)
        & (u <= x2 + params.target_occupancy_box_pad_px)
        & (v >= y1 - params.target_occupancy_box_pad_px)
        & (v <= y2 + params.target_occupancy_box_pad_px)
    )
    foreground_tolerance = (
        params.target_occupancy_radius_m
        + params.target_occupancy_depth_slack_m
    )
    # The locked RGB-D sample lies on the visible front surface, not at the
    # cylinder centre. Cover the rear surface by one diameter while keeping
    # the foreground bound narrow so another object in front is retained.
    depth_delta = camera_points[:, 2] - float(localization.depth_m)
    in_depth = (
        (depth_delta >= -foreground_tolerance)
        & (
            depth_delta
            <= 2.0 * params.target_occupancy_radius_m
            + params.target_occupancy_depth_slack_m
        )
    )
    in_lateral = (
        np.abs(camera_points[:, 0] - target_camera[0])
        <= foreground_tolerance
    )
    # The detector box already bounds the bottle vertically. This matters for
    # shelf grasps, whose lock is 40% down the box: the lower bottle body can
    # extend far beyond the legacy 5.5 cm table-grasp band.
    return in_box & in_depth & in_lateral


def build_non_target_scene_voxels(
    depth: np.ndarray,
    K: np.ndarray,
    T_base_camera: np.ndarray,
    localization: Localization,
    params: DemoParams,
    *,
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
    bottom_crop: int | None = None,
    max_voxels: int | None = None,
) -> list[list[float]]:
    """Voxelize non-target points without deleting mixed target/obstacle cells."""
    if depth is None or K is None:
        raise SafetyAbort("非目标障碍场景缺少深度或内参")
    points, u, v, camera_points = _base_points(
        depth,
        K,
        T_base_camera,
        params,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        bottom_crop=bottom_crop,
    )
    target = np.asarray(localization.point_base, dtype=float)
    relative = points - target
    valid_workspace = (
        (np.linalg.norm(relative[:, :2], axis=1) < 0.95)
        & (np.abs(relative[:, 2]) < 0.75)
    )
    target_samples = _target_point_mask(
        points, u, v, camera_points, localization, params
    )
    points = points[valid_workspace & ~target_samples]
    if points.size == 0:
        return []
    return voxelize_scene_points(points, params, max_voxels=max_voxels)


def voxelize_scene_points(
    points: Sequence[Sequence[float]] | np.ndarray,
    params: DemoParams,
    *,
    center_base: Sequence[float] | None = None,
    horizontal_radius_m: float = 0.95,
    vertical_radius_m: float = 0.75,
    max_voxels: int | None = None,
) -> list[list[float]]:
    """Voxelize a generic live scene without inventing a target object.

    This is used after the chassis has turned toward the output table: there
    is no shelf target localization to reuse, so scene cropping is centred on
    the live table ROI rather than on a stale pre-turn bottle coordinate.
    """
    array = np.asarray(points, dtype=float)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or not np.all(np.isfinite(array))
    ):
        raise SafetyAbort("通用障碍场景点必须是有限 Nx3 数组")
    if center_base is not None:
        center = np.asarray(center_base, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise SafetyAbort("通用障碍场景中心必须是有限三维点")
        relative = array - center
        array = array[
            (np.linalg.norm(relative[:, :2], axis=1) < horizontal_radius_m)
            & (np.abs(relative[:, 2]) < vertical_radius_m)
        ]
    if array.size == 0:
        return []
    voxel = params.scene_voxel_m
    keys = np.floor(array / voxel).astype(np.int32)
    unique = np.unique(keys, axis=0)
    centers = (unique.astype(float) + 0.5) * voxel
    max_voxels = params.scene_max_voxels if max_voxels is None else max_voxels
    _assert_within_budget(len(centers), max_voxels)
    return centers.tolist()


def build_target_occupancy_voxels(
    depth: np.ndarray,
    K: np.ndarray,
    T_base_camera: np.ndarray,
    localization: Localization,
    params: DemoParams,
    *,
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
    bottom_crop: int | None = None,
) -> list[list[float]]:
    """Record scene cells supported by target-associated depth samples.

    The global observation transfer still receives the complete scene.  This
    second set is only a provenance tag used by contact-phase validation to
    remove the locked object itself. Association is bounded by the detector
    silhouette plus narrow optical depth and lateral slabs, so an overlapping
    foreground or same-depth neighbour remains.
    """
    if depth is None or K is None:
        raise SafetyAbort("目标占据体素缺少深度或内参")
    points, u, v, camera_points = _base_points(
        depth,
        K,
        T_base_camera,
        params,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        bottom_crop=bottom_crop,
    )
    target_points = points[
        _target_point_mask(
            points, u, v, camera_points, localization, params
        )
    ]
    if target_points.size == 0:
        return []
    keys = np.floor(target_points / params.scene_voxel_m).astype(np.int32)
    unique = np.unique(keys, axis=0)
    centers = (unique.astype(float) + 0.5) * params.scene_voxel_m
    return centers.tolist()


def _assert_within_budget(count: int, max_voxels: int) -> None:
    # Never manufacture free space to meet a performance budget.  The
    # previous "nearest to bottle" truncation could drop an obstacle near
    # the path start while retaining hundreds of table voxels.  The scene
    # must either be represented in full or the robot must not move.
    if count > max_voxels:
        raise SafetyAbort(
            "头部障碍体素超出安全场景预算，拒绝丢弃远处障碍: "
            f"{count} > {max_voxels}。清理视野或调大体素尺寸后重跑"
        )


def union_scene_voxels(
    voxel_lists: Sequence[Sequence[Sequence[float]]],
    params: DemoParams,
    *,
    max_voxels: int | None = None,
) -> list[list[float]]:
    """Merge per-frame occupancy into one conservative scene.

    Occupancy is combined by union, never by majority vote.  A surface that
    only registers in some frames (dark, specular, or grazing-angle
    geometry) is still a real obstacle; dropping it because it failed a
    per-voxel vote would delete obstacles the camera did see — the same
    "manufactured free space" failure the budget check above exists to
    prevent.  Erring toward too many obstacles can only cost a refused
    plan; erring toward too few costs a collision.

    Centres come from the same integer voxel grid in every frame, so equal
    cells produce bit-identical coordinates and set semantics are exact.
    """
    if not voxel_lists:
        raise SafetyAbort("障碍体素合并收到空的帧列表")
    merged: dict[tuple[float, float, float], list[float]] = {}
    for index, centers in enumerate(voxel_lists, 1):
        for center in centers:
            point = np.asarray(center, dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise SafetyAbort(f"第 {index} 帧障碍体素坐标无效")
            merged.setdefault(
                (float(point[0]), float(point[1]), float(point[2])),
                point.tolist(),
            )
    max_voxels = params.scene_max_voxels if max_voxels is None else max_voxels
    # The union is what the planner will actually see, so the budget applies
    # to it — not just to whichever single frame happened to be smallest.
    _assert_within_budget(len(merged), max_voxels)
    return [merged[key] for key in sorted(merged)]


def conservative_scene_union(
    before: Sequence[Sequence[float]],
    after: Sequence[Sequence[float]],
    params: DemoParams,
    *,
    before_frame: str,
    after_frame: str,
    before_base_pose: np.ndarray,
    after_base_pose: np.ndarray,
) -> list[list[float]]:
    """Union two snapshots only when they describe the same physical frame.

    A controller-base coordinate is body-relative.  After chassis motion the
    same numeric voxel means a different world location, so cross-pose merging
    would manufacture obstacles and free space simultaneously.
    """

    before_pose = np.asarray(before_base_pose, dtype=float)
    after_pose = np.asarray(after_base_pose, dtype=float)
    if before_frame != after_frame:
        raise SafetyAbort(
            f"禁止合并不同场景坐标系: {before_frame!r} != {after_frame!r}"
        )
    if (
        before_pose.shape != (4, 4)
        or after_pose.shape != (4, 4)
        or not np.all(np.isfinite(before_pose))
        or not np.all(np.isfinite(after_pose))
    ):
        raise SafetyAbort("场景并集缺少有效的底盘姿态")
    if not np.allclose(before_pose, after_pose, atol=1e-9, rtol=0.0):
        raise SafetyAbort("禁止直接合并跨底盘姿态点云")
    return union_scene_voxels([before, after], params)
