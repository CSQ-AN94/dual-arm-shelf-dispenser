"""Verified global motion planning with bounded, feedback-driven replanning."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .core import DemoParams, SafetyAbort, interpolate_joint_path
from .safety import FenceViolation, SafetyProfile


@dataclass(frozen=True)
class PlanTarget:
    """One reachable controller-flange goal offered to the safe planner."""

    label: str
    flange: np.ndarray
    goal_joints: tuple[float, ...]
    score: float = 0.0
    goal_constraint: str = "pose"


@dataclass(frozen=True)
class VerifiedPlan:
    """A trajectory accepted by MoveIt and the independent electronic fence."""

    trajectory: dict
    target: PlanTarget
    checked_tcp_points: int
    attempts: int
    rejections: tuple[str, ...]
    covered_candidates: int
    total_candidates: int
    planners_tried: tuple[str, ...]


class SafeMotionPlanner:
    """Find and independently verify a global arm trajectory.

    The interface deliberately exposes one operation. MoveIt sampling,
    candidate fallback, fence feedback, duplicate suppression and failure
    aggregation remain implementation details behind this seam.
    """

    def __init__(
        self,
        *,
        moveit,
        robot,
        left_robot,
        safety: SafetyProfile,
        params: DemoParams,
        report: Optional[Callable[[str, str], None]] = None,
        held_object: Optional[dict] = None,
        link7_to_controller_flange: np.ndarray | None = None,
        planning_group: str = "right_arm",
        joint_signs: Sequence[int] | None = None,
    ):
        if planning_group not in {"right_arm", "left_arm"}:
            raise SafetyAbort(f"未知规划组: {planning_group!r}")
        self.planning_group = planning_group
        self.joint_signs = joint_signs
        self.moveit = moveit
        self.robot = robot
        self.left_robot = left_robot
        self.safety = safety
        self.params = params
        self.report = report or (lambda _name, _message: None)
        self.held_object = held_object
        if link7_to_controller_flange is None:
            legacy = np.eye(4)
            legacy[2, 3] = self.params.moveit_link7_to_controller_flange_m
            link7_to_controller_flange = legacy
        transform = np.asarray(link7_to_controller_flange, dtype=float)
        if (
            transform.shape != (4, 4)
            or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
            or not np.allclose(
                transform[:3, :3].T @ transform[:3, :3],
                np.eye(3),
                atol=1e-6,
            )
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-6)
        ):
            raise SafetyAbort("MoveIt link7→控制器法兰变换必须是有效刚体变换")
        self.T_link7_controller_flange = transform.copy()

    @staticmethod
    def _trajectory_fingerprint(points: Sequence[Sequence[float]]) -> tuple:
        return tuple(
            tuple(round(float(value), 3) for value in point)
            for point in points
        )

    def _target_link7_in_moveit(self, target: PlanTarget) -> np.ndarray:
        target_link7 = target.flange @ np.linalg.inv(
            self.T_link7_controller_flange
        )
        return self.safety.pose_to_moveit(target_link7)

    def _assert_endpoint_matches_controller_model(
        self, target: PlanTarget, trajectory: dict
    ) -> None:
        """Refuse a plan whose endpoint means a different pose to the SDK.

        MoveIt and the RealMan SDK are independent kinematic implementations.
        The 2026-07-18 trace showed why passing a joint goal while separately
        logging a pose goal is unsafe: the pose was silently ignored.  Every
        accepted endpoint now has to agree in the execution model as well.
        """
        points = trajectory.get("points_deg", [])
        if not points:
            raise SafetyAbort("MoveIt 返回空轨迹，无法复核端点")
        actual = self.robot.controller_flange_from_joints(points[-1])
        expected = np.asarray(target.flange, dtype=float)
        if (
            np.asarray(actual).shape != (4, 4)
            or expected.shape != (4, 4)
            or not np.all(np.isfinite(actual))
            or not np.all(np.isfinite(expected))
        ):
            raise SafetyAbort("MoveIt/RealMan 端点 FK 含非有限数或形状无效")
        position_error = float(
            np.linalg.norm(actual[:3, 3] - expected[:3, 3])
        )
        relative = expected[:3, :3].T @ actual[:3, :3]
        cosine = float((np.trace(relative) - 1.0) / 2.0)
        orientation_error = float(
            np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        )
        if (
            not np.isfinite(position_error)
            or not np.isfinite(orientation_error)
            or position_error
            > self.params.moveit_endpoint_position_tolerance_m
            or orientation_error
            > self.params.moveit_endpoint_orientation_tolerance_deg
        ):
            raise SafetyAbort(
                "MoveIt/RealMan 端点运动学不一致，禁止执行: "
                f"位置差={position_error * 1000:.1f} mm "
                f"(上限 {self.params.moveit_endpoint_position_tolerance_m * 1000:.0f} mm)，"
                f"姿态差={orientation_error:.1f}° "
                f"(上限 {self.params.moveit_endpoint_orientation_tolerance_deg:.1f}°)"
            )

    @staticmethod
    def _fk_matrix(payload: dict) -> np.ndarray:
        position = np.asarray(payload.get("position"), dtype=float)
        quaternion = np.asarray(
            payload.get("quaternion_xyzw"), dtype=float
        )
        if (
            position.shape != (3,)
            or quaternion.shape != (4,)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(quaternion))
            or float(np.linalg.norm(quaternion)) < 1e-9
        ):
            raise SafetyAbort("MoveIt FK 返回非有限或无效位姿")
        transform = np.eye(4)
        transform[:3, 3] = position
        transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        return transform

    def _assert_runtime_fk_contract(
        self,
        start_joints: Sequence[float],
        trajectory: dict,
    ) -> None:
        """Compare MoveIt and SDK FK at the exact same joint states."""
        if not getattr(self.moveit, "enforces_model_contract", False):
            return

        checks = (
            ("起点", start_joints, trajectory.get("start_link7_fk")),
            (
                "端点",
                trajectory.get("points_deg", [])[-1],
                trajectory.get("endpoint_link7_fk"),
            ),
        )
        for label, joints, moveit_payload in checks:
            if not moveit_payload:
                raise SafetyAbort(
                    f"MoveIt 未返回{label} r_link7 FK，禁止执行"
                )
            controller_flange = self.robot.controller_flange_from_joints(
                joints
            )
            if (
                np.asarray(controller_flange).shape != (4, 4)
                or not np.all(np.isfinite(controller_flange))
            ):
                raise SafetyAbort(f"SDK {label} FK 含非有限数或形状无效")
            sdk_link7 = controller_flange @ np.linalg.inv(
                self.T_link7_controller_flange
            )
            sdk_in_moveit = self.safety.pose_to_moveit(sdk_link7)
            moveit_link7 = self._fk_matrix(moveit_payload)
            position_error = float(
                np.linalg.norm(
                    sdk_in_moveit[:3, 3] - moveit_link7[:3, 3]
                )
            )
            relative = sdk_in_moveit[:3, :3].T @ moveit_link7[:3, :3]
            cosine = float((np.trace(relative) - 1.0) / 2.0)
            orientation_error = float(
                np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            )
            if (
                not np.isfinite(position_error)
                or not np.isfinite(orientation_error)
                or position_error
                > self.params.moveit_endpoint_position_tolerance_m
                or orientation_error
                > self.params.moveit_endpoint_orientation_tolerance_deg
            ):
                raise SafetyAbort(
                    f"{label} MoveIt/SDK 同状态 FK 不一致，禁止执行: "
                    f"位置差={position_error * 1000:.1f} mm，"
                    f"姿态差={orientation_error:.1f}°"
                )

    def plan(
        self,
        *,
        name: str,
        targets: Sequence[PlanTarget],
        obstacle_points: Sequence[Sequence[float]],
        collision_boxes: Sequence[dict],
        start_right_joints_deg: Optional[Sequence[float]] = None,
        continuation_validator: Optional[
            Callable[[PlanTarget, dict], None]
        ] = None,
        trajectory_validator: Optional[
            Callable[[PlanTarget, dict], None]
        ] = None,
        enforce_endpoint_vertical_floor: bool = False,
    ) -> VerifiedPlan:
        candidates = list(targets)
        if not candidates:
            raise SafetyAbort(f"{name} 没有可规划的目标候选")
        for target in candidates:
            flange = np.asarray(target.flange, dtype=float)
            goal = np.asarray(target.goal_joints, dtype=float)
            if (
                flange.shape != (4, 4)
                or not np.all(np.isfinite(flange))
                or goal.shape != (7,)
                or not np.all(np.isfinite(goal))
                or not np.isfinite(float(target.score))
            ):
                raise SafetyAbort(f"规划目标 {target.label} 含非有限数或形状无效")
        # Candidate order is a caller-owned contract.  Observation planning
        # grades continuation clearance before transfer cost; sorting by the
        # raw score here used to silently undo that safety ranking.
        ranked = candidates[: self.params.global_plan_max_candidates]

        # A supplied start is used only for chained, no-motion rehearsal (for
        # example staging -> observation).  Executable plans retain this exact
        # start in their credential, and _execute_plan still rejects them if
        # the physical arm is elsewhere.
        start_right_raw = (
            self.robot.joints_deg()
            if start_right_joints_deg is None
            else start_right_joints_deg
        )
        start_right_array = np.asarray(start_right_raw, dtype=float)
        if (
            start_right_array.shape != (7,)
            or not np.all(np.isfinite(start_right_array))
        ):
            raise SafetyAbort("全局规划右臂起点必须是 7 个有限关节角")
        start_right = list(map(float, start_right_array))
        start_left = self.left_robot.joints_deg()
        base_boxes = list(collision_boxes)
        obstacle_filter = getattr(
            self.safety, "moveit_obstacles_outside_fences", None
        )
        moveit_obstacles = (
            obstacle_filter(obstacle_points, base_boxes)
            if callable(obstacle_filter)
            else self.safety.points_to_moveit(obstacle_points)
        )
        seen_trajectories: set[tuple] = set()
        rejections: list[str] = []
        attempts = 0
        covered_candidate_indices: set[int] = set()
        planners_tried: list[str] = []
        planner_ids = tuple(self.params.moveit_planner_ids)
        if not planner_ids:
            raise SafetyAbort("未配置任何 MoveIt 路线搜索器")
        deadline = time.monotonic() + self.params.global_plan_search_budget_s
        budget_exhausted = False

        feedback_boxes: list[dict] = []
        feedback_box_keys: set[tuple] = set()
        eliminated_candidates: set[int] = set()
        for route_index in range(
            1, self.params.global_plan_attempts_per_candidate + 1
        ):
            planner_id = planner_ids[(route_index - 1) % len(planner_ids)]
            for candidate_index, target in enumerate(ranked, 1):
                if candidate_index in eliminated_candidates:
                    continue
                target_moveit = self._target_link7_in_moveit(target)
                minimum_link7_z = None
                if enforce_endpoint_vertical_floor:
                    start_flange = self.robot.controller_flange_from_joints(
                        start_right
                    )
                    start_link7 = start_flange @ np.linalg.inv(
                        self.T_link7_controller_flange
                    )
                    start_link7_moveit = self.safety.pose_to_moveit(
                        start_link7
                    )
                    minimum_link7_z = float(
                        min(
                            start_link7_moveit[2, 3],
                            target_moveit[2, 3],
                        )
                        - self.params.observation_vertical_undershoot_tolerance_m
                    )
                remaining_budget_s = deadline - time.monotonic()
                if remaining_budget_s <= 0:
                    budget_exhausted = True
                    break
                remaining_candidates = sum(
                    index not in eliminated_candidates
                    for index in range(candidate_index, len(ranked) + 1)
                )
                allowed_planning_time_s = min(
                    self.params.moveit_allowed_planning_time_s,
                    remaining_budget_s / max(1, remaining_candidates),
                )
                attempts += 1
                covered_candidate_indices.add(candidate_index)
                if planner_id not in planners_tried:
                    planners_tried.append(planner_id)
                attempt_label = (
                    f"{name}_c{candidate_index:02d}_r{route_index:02d}"
                )
                self.report(
                    "安全规划尝试",
                    (
                        f"{name}: {target.label}，路线 {route_index}，"
                        f"planner={planner_id}；"
                        f"覆盖={len(covered_candidate_indices)}/{len(ranked)}；"
                        f"时间片={allowed_planning_time_s:.2f}s"
                    ),
                )
                try:
                    trajectory = self.moveit.plan(
                        name=attempt_label,
                        planning_group=self.planning_group,
                        joint_signs=self.joint_signs,
                        start_joints_deg=start_right,
                        start_left_joints_deg=start_left,
                        goal_joints_deg=target.goal_joints,
                        target_flange=target_moveit,
                        goal_constraint=target.goal_constraint,
                        planner_id=planner_id,
                        allowed_planning_time_s=allowed_planning_time_s,
                        num_planning_attempts=(
                            self.params.moveit_num_planning_attempts
                        ),
                        obstacles=moveit_obstacles,
                        boxes=[*base_boxes, *feedback_boxes],
                        workspace=self.safety.moveit_workspace(),
                        planning_frame=self.safety.moveit_frame,
                        tool_guard={
                            "xy": self.params.tool_guard_xy_m,
                            "length": self.params.tool_guard_length_m,
                            "center_z": self.params.tool_guard_center_z_m,
                        },
                        held_object=self.held_object,
                        voxel_size=self.params.scene_voxel_m,
                        minimum_link7_z=minimum_link7_z,
                    )
                except SafetyAbort as exc:
                    reason = f"{target.label}/路线{route_index} 规划失败: {exc}"
                    rejections.append(reason)
                    self.report("规划未通过，自动换路", reason)
                    continue

                fingerprint = self._trajectory_fingerprint(
                    trajectory.get("points_deg", [])
                )
                if fingerprint in seen_trajectories:
                    reason = f"{target.label}/路线{route_index} 与已拒绝轨迹重复"
                    rejections.append(reason)
                    self.report("规划未通过，自动换规划器", reason)
                    continue
                seen_trajectories.add(fingerprint)

                try:
                    self._assert_runtime_fk_contract(
                        start_right, trajectory
                    )
                    self._assert_endpoint_matches_controller_model(
                        target, trajectory
                    )
                except SafetyAbort as exc:
                    reason = (
                        f"{target.label}/路线{route_index} 模型契约拒绝: {exc}"
                    )
                    rejections.append(reason)
                    self.report("规划模型不一致，自动换目标", reason)
                    eliminated_candidates.add(candidate_index)
                    continue

                if trajectory_validator is not None:
                    try:
                        trajectory_validator(target, trajectory)
                    except SafetyAbort as exc:
                        reason = (
                            f"{target.label}/路线{route_index} "
                            f"轨迹形状拒绝: {exc}"
                        )
                        rejections.append(reason)
                        self.report("轨迹形状不合要求，自动换路", reason)
                        # Shape is route-specific.  Keep the endpoint so a
                        # different planner/seed may approach it cleanly.
                        continue

                if continuation_validator is not None:
                    try:
                        continuation_validator(target, trajectory)
                    except SafetyAbort as exc:
                        reason = (
                            f"{target.label}/路线{route_index} "
                            f"后续抓放预演拒绝: {exc}"
                        )
                        rejections.append(reason)
                        self.report("观察终点后续不可行，自动换目标", reason)
                        eliminated_candidates.add(candidate_index)
                        continue

                dense_joint_points = interpolate_joint_path(
                    start_right,
                    trajectory["points_deg"],
                    self.params.planned_joint_step_deg,
                )
                checked = None
                fence_error: SafetyAbort | None = None
                moveit_error: SafetyAbort | None = None
                next_feedback_box: dict | None = None
                try:
                    checked = self.robot.validate_planned_joints(
                        trajectory["points_deg"],
                        self.params.planned_joint_step_deg,
                        self.safety,
                        start_joints_deg=start_right,
                    )
                except SafetyAbort as exc:
                    fence_error = exc
                    if isinstance(exc, FenceViolation):
                        next_feedback_box = self.safety.replan_exclusion_box(
                            exc,
                            object_id=f"replan_{attempts:02d}",
                            size_m=self.params.replan_exclusion_size_m,
                        )

                # Always run both independent validators.  The real trace's
                # early SDK-fence ``continue`` prevented a MoveIt artifact
                # from being written precisely on the disputed trajectories.
                try:
                    self.moveit.validate_exact_path(
                        name=f"{attempt_label}_postcheck",
                        planning_group=self.planning_group,
                        start_left_joints_deg=start_left,
                        points_deg=dense_joint_points,
                        obstacles=moveit_obstacles,
                        # Diagnose the exact scene that produced this path.
                        # The SDK-derived exclusion box belongs only to the
                        # *next* planning attempt; adding it here would force
                        # MoveIt to reject by construction and destroy the
                        # independence of this comparison.
                        boxes=[*base_boxes, *feedback_boxes],
                        planning_frame=self.safety.moveit_frame,
                        tool_guard={
                            "xy": self.params.tool_guard_xy_m,
                            "length": self.params.tool_guard_length_m,
                            "center_z": self.params.tool_guard_center_z_m,
                        },
                        held_object=self.held_object,
                        voxel_size=self.params.scene_voxel_m,
                    )
                except SafetyAbort as exc:
                    moveit_error = exc

                if next_feedback_box is not None:
                    feedback_key = tuple(
                        round(float(value), 4)
                        for field in ("center", "size")
                        for value in next_feedback_box.get(field, [])
                    )
                    if feedback_key not in feedback_box_keys:
                        feedback_box_keys.add(feedback_key)
                        feedback_boxes.append(next_feedback_box)

                if fence_error is not None or moveit_error is not None:
                    independent = []
                    independent.append(
                        "SDK围栏=通过"
                        if fence_error is None
                        else f"SDK围栏=拒绝({fence_error})"
                    )
                    independent.append(
                        "MoveIt密集复核=通过"
                        if moveit_error is None
                        else f"MoveIt密集复核=拒绝({moveit_error})"
                    )
                    reason = (
                        f"{target.label}/路线{route_index} 独立复核: "
                        + "；".join(independent)
                    )
                    rejections.append(reason)
                    self.report("轨迹独立复核拒绝，自动重规划", reason)
                    continue

                assert checked is not None

                self.report(
                    "安全轨迹确定",
                    (
                        f"{name}: {target.label}；第 {attempts} 次尝试；"
                        f"{checked} 个密集 TCP 点通过"
                    ),
                )
                executable_trajectory = dict(trajectory)
                executable_trajectory["start_joints_deg"] = list(start_right)
                executable_trajectory["start_left_joints_deg"] = list(
                    start_left
                )
                executable_trajectory["search_coverage"] = {
                    "attempted_candidates": len(covered_candidate_indices),
                    "total_candidates": len(ranked),
                    "planner_ids": list(planners_tried),
                    "attempts": attempts,
                }
                return VerifiedPlan(
                    trajectory=executable_trajectory,
                    target=target,
                    checked_tcp_points=checked,
                    attempts=attempts,
                    rejections=tuple(rejections),
                    covered_candidates=len(covered_candidate_indices),
                    total_candidates=len(ranked),
                    planners_tried=tuple(planners_tried),
                )

            if budget_exhausted:
                break

        tail = "；".join(rejections[-4:]) if rejections else "无详细拒绝原因"
        raise SafetyAbort(
            f"{name} 在 {attempts} 次安全规划后仍无可执行轨迹"
            + (
                f"（已用满 {self.params.global_plan_search_budget_s:.0f}s 搜索预算）"
                if budget_exhausted
                else "（全部候选/规划器组合已搜索）"
            )
            + "；"
            f"覆盖={len(covered_candidate_indices)}/{len(ranked)}；"
            f"planners={planners_tried}；最后拒绝: {tail}"
        )
