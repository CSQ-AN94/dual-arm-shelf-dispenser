"""Placement-only endpoint search for an already-held upright bottle."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .core import SafetyAbort
from .delivery_table import PlaceCandidate
from .relative_place import assert_planning_usable


@dataclass(frozen=True)
class DemonstratedPlace:
    """One taught endpoint at which the bottle demonstrably came to rest.

    This is a *prior for the search*, never a path to replay.  It supplies
    three references the run has no other honest source for:

    * a wrist orientation that has actually set a bottle down on this table,
      instead of the orientation the bottle happens to be carried in;
    * a joint configuration to seed the controller's IK from, instead of the
      transport tuck -- the controller's solver is seed-dependent and on
      2026-08-13 returned rc=1 for whole candidate patches when seeded from
      the tuck;
    * a support height measured by contact rather than by camera, which the
      fitted plane is checked against.

    The endpoint is expressed in the live base frame by the caller, which
    owns forward kinematics; this module stays pure.
    """

    joints_deg: tuple[float, ...]
    tcp: np.ndarray
    bottle_center: np.ndarray

    @property
    def support_z(self) -> float:
        """Height of the surface the taught bottle came to rest on."""
        return float(self.bottle_center[2])


def demonstrated_place_from_trajectory(
    demonstration: dict,
    *,
    tcp_from_joints,
    tcp_from_bottle,
    bottle_height_m: float,
    label: str = "桌面放置",
) -> DemonstratedPlace:
    """Read the taught endpoint and where its bottle came to rest.

    ``tcp_from_joints`` is the live arm's forward kinematics, so the endpoint
    lands in the same base frame as everything else this run measures.
    """
    samples = assert_planning_usable(demonstration, label=label)
    joints = np.asarray(samples[-1]["joints_deg"], dtype=float)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise SafetyAbort(f"{label} 示教终点关节值无效")
    tcp = _rigid_transform(tcp_from_joints(joints.tolist()), f"{label} 示教终点 TCP")
    bottle = tcp @ _rigid_transform(tcp_from_bottle, "TCP→瓶子")
    tilt_deg = math.degrees(
        math.acos(float(np.clip(bottle[2, 2], -1.0, 1.0)))
    )
    if tilt_deg > 10.0:
        raise SafetyAbort(
            f"{label} 示教终点瓶子倾斜 {tilt_deg:.1f}°，不能当作落瓶基准"
        )
    # The taught bottle rested on the surface, so its support height is the
    # centre less half the body -- the one place height in this pipeline that
    # owes nothing to the nominal tool-mount chain.
    support = bottle.copy()
    support[2, 3] = float(bottle[2, 3]) - float(bottle_height_m) / 2.0
    return DemonstratedPlace(
        joints_deg=tuple(map(float, joints)),
        tcp=tcp.copy(),
        bottle_center=support[:3, 3].copy(),
    )


@dataclass(frozen=True)
class FreeTablePlaceTarget:
    """One reachable pre-place endpoint and its short final descent."""

    label: str
    candidate: PlaceCandidate
    final_bottle_yaw_deg: float
    preplace_bottle_tilt_deg: float
    preplace_bottle_azimuth_deg: float
    preplace_tcp: np.ndarray
    place_tcp: np.ndarray
    flange: np.ndarray
    goal_joints: tuple[float, ...]


def _rigid_transform(value, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if (
        transform.shape != (4, 4)
        or not np.all(np.isfinite(transform))
        or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
        or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6
        )
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-6)
    ):
        raise SafetyAbort(f"{label}必须是有效刚体变换")
    return transform


def build_free_table_place_targets(
    *,
    candidates: Sequence[PlaceCandidate],
    current_tcp: np.ndarray,
    current_joints_deg: Sequence[float],
    tcp_from_bottle: np.ndarray,
    flange_from_tcp: np.ndarray,
    bottle_height_m: float,
    bottle_radius_m: float,
    release_gap_m: float,
    preplace_clearance_m: float,
    table_xy_min: Sequence[float],
    table_xy_max: Sequence[float],
    solve_flange_ik_candidates: Callable[
        [np.ndarray, Sequence[float]], Sequence[Sequence[float]]
    ],
    demonstration: DemonstratedPlace | None = None,
    demonstrated_support_tolerance_m: float = 0.030,
    yaw_step_deg: float = 30.0,
    preplace_tilt_degrees: Sequence[float] = (0.0, 30.0, 60.0, 90.0),
    preplace_azimuth_step_deg: float = 90.0,
    max_targets: int = 24,
    max_targets_per_patch: int = 4,
) -> tuple[FreeTablePlaceTarget, ...]:
    """Enumerate empty patches, full bottle yaw, and all bounded IK branches.

    The successful grasp contributes only ``tcp_from_bottle``.  Placement owns
    every generated endpoint after that seam.  The bottle is upright only at
    the final support point.  At the collision-free pre-place height it may
    lean in any sampled direction, so MoveIt is not forced toward the table in
    the grasp/transport wrist orientation.

    ``demonstration`` does not add or remove a single endpoint: the same
    patches, yaws, tilts and IK branches are enumerated either way.  It only
    changes the three arbitrary references the search would otherwise take
    from the carrying pose -- which orientation is tried first, which patch is
    tried first, and what the controller's seed-dependent IK is seeded with.
    A search whose first tries are near a configuration that has already put
    a bottle on this table finds a solution inside its budget far more often
    than one that starts from the transport tuck and works outward.
    """
    current_tcp = _rigid_transform(current_tcp, "当前 TCP")
    tcp_from_bottle = _rigid_transform(tcp_from_bottle, "TCP→瓶子")
    flange_from_tcp = _rigid_transform(flange_from_tcp, "法兰→TCP")
    current_joints = np.asarray(current_joints_deg, dtype=float)
    # Everything below reads its references from here, so an absent
    # demonstration degrades to exactly the previous behaviour.
    reference_rotation = current_tcp[:3, :3]
    seed_joints = current_joints
    if demonstration is not None:
        reference_rotation = _rigid_transform(
            demonstration.tcp, "示教终点 TCP"
        )[:3, :3]
        seed_joints = np.asarray(demonstration.joints_deg, dtype=float)
        if seed_joints.shape != (7,) or not np.all(np.isfinite(seed_joints)):
            raise SafetyAbort("示教终点关节种子无效")
    table_lower = np.asarray(table_xy_min, dtype=float)
    table_upper = np.asarray(table_xy_max, dtype=float)
    tilts = np.asarray(preplace_tilt_degrees, dtype=float)
    scalars = (
        bottle_height_m,
        bottle_radius_m,
        release_gap_m,
        preplace_clearance_m,
        yaw_step_deg,
        preplace_azimuth_step_deg,
    )
    if (
        current_joints.shape != (7,)
        or not np.all(np.isfinite(current_joints))
        or table_lower.shape != (2,)
        or table_upper.shape != (2,)
        or not np.all(np.isfinite(table_lower))
        or not np.all(np.isfinite(table_upper))
        or np.any(table_upper <= table_lower)
        or tilts.ndim != 1
        or not len(tilts)
        or not np.all(np.isfinite(tilts))
        or np.any(tilts < 0.0)
        or np.any(tilts > 90.0)
        or not np.all(np.isfinite(scalars))
        or bottle_height_m <= 0.0
        or bottle_radius_m <= 0.0
        or release_gap_m < 0.0
        or preplace_clearance_m <= 0.0
        or not 1.0 <= yaw_step_deg <= 180.0
        or not 1.0 <= preplace_azimuth_step_deg <= 180.0
        or max_targets < 1
        or max_targets_per_patch < 1
    ):
        raise SafetyAbort("自由桌面放置输入或搜索上限无效")
    if not candidates:
        raise SafetyAbort("桌面观测没有空位候选")

    for candidate in candidates:
        xy = np.asarray(candidate.xy_base, dtype=float)
        nearest_obstacle = float(candidate.nearest_obstacle_m)
        if (
            xy.shape != (2,)
            or not np.all(np.isfinite(xy))
            or not math.isfinite(float(candidate.table_height_m))
            or (not math.isfinite(nearest_obstacle) and nearest_obstacle != math.inf)
        ):
            raise SafetyAbort("桌面空位候选含无效坐标或高度")

    current_bottle = current_tcp @ tcp_from_bottle
    reference_xy = current_bottle[:2, 3]
    if demonstration is not None:
        taught_support = float(demonstration.support_z)
        table_height = float(candidates[0].table_height_m)
        support_error = abs(taught_support - table_height)
        # The taught bottle reached its support by contact; the candidates
        # reached theirs by camera.  A disagreement means the table or the
        # chassis is not where the demonstration left it, and neither height
        # is then trustworthy.  The tolerance is not a measured constant: on
        # 2026-08-13 the fitted plane's own scatter across the table was
        # about 16 mm and the taught endpoint agreed with it to 4 mm, so this
        # sits above the observed noise and far below a moved table.
        if support_error > float(demonstrated_support_tolerance_m):
            raise SafetyAbort(
                "示教落瓶高度与实时拟合桌面不一致，桌子或底盘可能已移动: "
                f"taught={taught_support:.4f}m, fitted={table_height:.4f}m, "
                f"error={support_error * 1000:.1f}mm"
            )
        reference_xy = np.asarray(demonstration.bottle_center[:2], dtype=float)
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: float(
            np.linalg.norm(
                np.asarray(candidate.xy_base, dtype=float) - reference_xy
            )
        ),
    )
    yaw_count = max(2, int(math.ceil(360.0 / yaw_step_deg)))
    yaws = [index * 360.0 / yaw_count for index in range(yaw_count)]
    azimuth_count = max(
        2, int(math.ceil(360.0 / preplace_azimuth_step_deg))
    )
    azimuths = [index * 360.0 / azimuth_count for index in range(azimuth_count)]
    orientations = []
    inverse_tcp_from_bottle = np.linalg.inv(tcp_from_bottle)
    for yaw in yaws:
        final_rotation = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        for tilt in tilts:
            lean_directions = [0.0] if abs(float(tilt)) < 1e-9 else azimuths
            for azimuth in lean_directions:
                preplace_rotation = (
                    Rotation.from_euler("z", azimuth, degrees=True).as_matrix()
                    @ Rotation.from_euler("y", tilt, degrees=True).as_matrix()
                    @ final_rotation
                )
                target_tcp_rotation = (
                    preplace_rotation @ inverse_tcp_from_bottle[:3, :3]
                )
                rotation_cost = float(
                    Rotation.from_matrix(
                        reference_rotation.T @ target_tcp_rotation
                    ).magnitude()
                )
                orientations.append(
                    (
                        rotation_cost,
                        float(yaw),
                        float(tilt),
                        float(azimuth),
                        final_rotation,
                        preplace_rotation,
                    )
                )
    orientations.sort(key=lambda item: item[0])

    targets: list[FreeTablePlaceTarget] = []
    rejected = 0
    for patch_index, candidate in enumerate(ordered_candidates, 1):
        patch_targets: list[tuple[tuple[float, float], FreeTablePlaceTarget]] = []
        for _, yaw, tilt, azimuth, final_rotation, preplace_rotation in orientations:
            final_bottle = np.eye(4)
            final_bottle[:3, :3] = final_rotation
            final_bottle[:3, 3] = (
                float(candidate.xy_base[0]),
                float(candidate.xy_base[1]),
                float(candidate.table_height_m)
                + bottle_height_m / 2.0
                + release_gap_m,
            )
            place_tcp = final_bottle @ inverse_tcp_from_bottle

            axis = preplace_rotation[:, 2]
            radial = np.sqrt(np.maximum(0.0, 1.0 - axis * axis))
            half_extent = bottle_height_m / 2.0 * np.abs(axis) + bottle_radius_m * radial
            center_xy = np.asarray(candidate.xy_base, dtype=float)
            if np.any(center_xy - half_extent[:2] < table_lower) or np.any(
                center_xy + half_extent[:2] > table_upper
            ):
                rejected += 1
                continue
            horizontal_radius = (
                bottle_height_m / 2.0 * float(np.linalg.norm(axis[:2]))
                + bottle_radius_m
            )
            if (
                math.isfinite(float(candidate.nearest_obstacle_m))
                and float(candidate.nearest_obstacle_m) < horizontal_radius
            ):
                rejected += 1
                continue
            preplace_bottle = np.eye(4)
            preplace_bottle[:3, :3] = preplace_rotation
            preplace_bottle[:2, 3] = center_xy
            preplace_bottle[2, 3] = (
                float(candidate.table_height_m)
                + float(half_extent[2])
                + preplace_clearance_m
            )
            preplace_tcp = preplace_bottle @ inverse_tcp_from_bottle
            flange = preplace_tcp @ np.linalg.inv(flange_from_tcp)
            try:
                solutions = solve_flange_ik_candidates(flange, seed_joints)
            except SafetyAbort:
                rejected += 1
                continue
            for branch_index, values in enumerate(solutions, 1):
                joints = np.asarray(values, dtype=float)
                if joints.shape != (7,) or not np.all(np.isfinite(joints)):
                    rejected += 1
                    continue
                delta = np.abs(joints - current_joints)
                label = (
                    f"自由桌面空位{patch_index} "
                    f"落瓶yaw={yaw:.0f}° "
                    f"预放tilt={tilt:.0f}°/az={azimuth:.0f}° IK{branch_index}"
                )
                target = FreeTablePlaceTarget(
                    label=label,
                    candidate=candidate,
                    final_bottle_yaw_deg=float(yaw),
                    preplace_bottle_tilt_deg=float(tilt),
                    preplace_bottle_azimuth_deg=float(azimuth),
                    preplace_tcp=preplace_tcp.copy(),
                    place_tcp=place_tcp.copy(),
                    flange=flange.copy(),
                    goal_joints=tuple(map(float, joints)),
                )
                patch_targets.append(
                    ((float(np.max(delta)), float(np.sum(delta))), target)
                )
        patch_targets.sort(key=lambda item: item[0])
        targets.extend(
            target for _, target in patch_targets[:max_targets_per_patch]
        )
        if len(targets) >= max_targets:
            break

    if not targets:
        raise SafetyAbort(
            "桌面空位的全周向姿态和多分支 IK 均不可达"
            + (f"（拒绝 {rejected} 组）" if rejected else "")
        )
    return tuple(targets[:max_targets])
