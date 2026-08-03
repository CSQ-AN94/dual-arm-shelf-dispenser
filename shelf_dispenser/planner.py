"""Subprocess bridge from the vision environment to ROS2 MoveIt."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .core import SafetyAbort
from .ros.scene_ids import RGBD_VOXELS_ID

LOG = logging.getLogger("bottle_demo")


class MoveItPlanner:
    enforces_model_contract = True

    def __init__(self, project_root: Path, run_dir: Path):
        self.project_root = Path(project_root)
        self.run_dir = Path(run_dir)
        self.process = None
        self.log_handle = None

    @property
    def ros_prefix(self) -> str:
        return (
            "source /opt/ros/humble/setup.bash && "
            "source /home/rm/ros2_ws/install/setup.bash && "
        )

    def start(self):
        launch_script = self.project_root / "shelf_dispenser" / "ros" / "headless.py"
        self.log_handle = open(self.run_dir / "moveit.log", "w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                "bash",
                "-lc",
                self.ros_prefix + f"exec python3 '{launch_script}'",
            ],
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + 25
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise SafetyAbort("MoveIt2 启动失败，查看 moveit.log")
            result = subprocess.run(
                [
                    "bash",
                    "-lc",
                    self.ros_prefix
                    + "ros2 service list | grep -q '^/plan_kinematic_path$'",
                ],
                check=False,
            )
            if result.returncode == 0:
                LOG.info("MoveIt2 只规划服务已就绪")
                return
            time.sleep(1)
        raise SafetyAbort("等待 MoveIt2 规划服务超时")

    def _run_json_helper(
        self,
        *,
        operation: str,
        helper: Path,
        request_path: Path,
        output_path: Path,
        timeout_s: float,
    ) -> dict:
        """Run one ROS helper and turn transport/file failures into SafetyAbort."""
        try:
            output_path.unlink(missing_ok=True)
            result = subprocess.run(
                [
                    "bash",
                    "-lc",
                    self.ros_prefix
                    + f"python3 '{helper}' '{request_path}' '{output_path}'",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise SafetyAbort(
                f"{operation} 超时（上限 {timeout_s:.0f}s）"
            ) from exc
        except OSError as exc:
            raise SafetyAbort(f"{operation} 无法启动: {exc}") from exc

        if not output_path.exists():
            stderr = (result.stderr or "").strip()[-800:]
            stdout = (result.stdout or "").strip()[-800:]
            raise SafetyAbort(
                f"{operation} 没有返回结果文件: rc={result.returncode}; "
                f"stdout={stdout!r}; stderr={stderr!r}"
            )
        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SafetyAbort(f"{operation} 结果无法解析: {exc}") from exc

    @staticmethod
    def _normalize_planned_arm_trajectory(
        plan: dict, planning_group: str = "right_arm"
    ) -> None:
        """Interpret trajectory columns by name, never by message order.

        The names belong to whichever arm was planned; hardcoding the right
        arm's rejected every valid left-arm trajectory.
        """
        prefix = "r" if planning_group == "right_arm" else "l"
        expected = [f"{prefix}_joint{i}" for i in range(1, 8)]
        names = list(plan.get("joint_names") or [])
        if (
            len(names) != len(expected)
            or len(set(names)) != len(names)
            or set(names) != set(expected)
        ):
            raise SafetyAbort(
                "MoveIt 右臂轨迹关节集合/顺序契约无效: " + repr(names)
            )
        source_index = {name: index for index, name in enumerate(names)}
        normalized = []
        for point_index, raw in enumerate(plan.get("points_deg") or [], 1):
            point = np.asarray(raw, dtype=float)
            if point.shape != (len(names),) or not np.all(np.isfinite(point)):
                raise SafetyAbort(
                    f"MoveIt 轨迹点 {point_index} 维度或数值无效"
                )
            normalized.append(
                [float(point[source_index[name]]) for name in expected]
            )
        if not normalized:
            raise SafetyAbort("MoveIt 返回空轨迹")
        plan["joint_names"] = expected
        plan["points_deg"] = normalized

    @staticmethod
    def _assert_live_scene_contract(
        result: dict,
        *,
        obstacles: Sequence[Sequence[float]],
        boxes: Sequence[dict],
        tool_guard: dict,
        held_object: dict | None = None,
    ) -> None:
        attached_ids = set(result.get("attached_object_ids") or [])
        if tool_guard and "bottle_tool_guard" not in attached_ids:
            raise SafetyAbort(
                "MoveIt live PlanningScene 未发现 bottle_tool_guard，禁止执行"
            )
        if held_object and "held_bottle_guard" not in attached_ids:
            raise SafetyAbort(
                "MoveIt live PlanningScene 未发现 held_bottle_guard，禁止携瓶运动"
            )
        expected_world_ids = {str(item["id"]) for item in boxes}
        if obstacles:
            expected_world_ids.add(RGBD_VOXELS_ID)
        live_world_ids = set(result.get("world_collision_ids") or [])
        missing = sorted(expected_world_ids - live_world_ids)
        if missing:
            raise SafetyAbort(
                "MoveIt live PlanningScene 缺少请求的碰撞几何，禁止执行: "
                + ", ".join(missing[:8])
            )

    def plan(
        self,
        *,
        name: str,
        start_joints_deg: Sequence[float],
        start_left_joints_deg: Sequence[float] | None,
        goal_joints_deg: Sequence[float],
        target_flange: np.ndarray,
        obstacles: Sequence[Sequence[float]],
        boxes: Sequence[dict],
        workspace: dict,
        planning_frame: str,
        tool_guard: dict,
        held_object: dict | None = None,
        voxel_size: float,
        goal_constraint: str = "pose",
        planner_id: str = "RRTConnectkConfigDefault",
        allowed_planning_time_s: float = 6.0,
        num_planning_attempts: int = 12,
        minimum_link7_z: float | None = None,
        planning_group: str = "right_arm",
    ) -> dict:
        if goal_constraint not in {"pose", "joints"}:
            raise SafetyAbort(
                f"未知 MoveIt 目标约束类型: {goal_constraint!r}"
            )
        if (
            not planner_id
            or not np.isfinite(allowed_planning_time_s)
            or allowed_planning_time_s <= 0
            or int(num_planning_attempts) < 1
        ):
            raise SafetyAbort("MoveIt 搜索策略或时间/尝试预算无效")
        if minimum_link7_z is not None and not np.isfinite(minimum_link7_z):
            raise SafetyAbort("MoveIt r_link7 最低高度约束无效")
        request_path = self.run_dir / f"{name}_request.json"
        output_path = self.run_dir / f"{name}_plan.json"
        if planning_group not in {"right_arm", "left_arm"}:
            raise SafetyAbort(f"未知规划组: {planning_group!r}")
        request_payload = {
            "planning_group": planning_group,
            "start_joints_deg": list(map(float, start_joints_deg)),
            # Named for the arm that is not being planned, whichever that
            # is; the old key is kept so an unchanged caller still works.
            "start_other_joints_deg": (
                None
                if start_left_joints_deg is None
                else list(map(float, start_left_joints_deg))
            ),
            "goal_joints_deg": list(map(float, goal_joints_deg)),
            "target_flange": np.asarray(target_flange, dtype=float).tolist(),
            "goal_constraint": goal_constraint,
            "planner_id": str(planner_id),
            "allowed_planning_time_s": float(allowed_planning_time_s),
            "num_planning_attempts": int(num_planning_attempts),
            "obstacles": [list(map(float, item)) for item in obstacles],
            "boxes": list(boxes),
            "workspace": workspace,
            "planning_frame": planning_frame,
            "tool_guard": tool_guard,
            "held_object": held_object,
            "voxel_size": float(voxel_size),
        }
        if minimum_link7_z is not None:
            request_payload["minimum_link7_z"] = float(minimum_link7_z)
        request_path.write_text(
            json.dumps(
                request_payload,
                indent=2,
            ),
            encoding="utf-8",
        )
        helper = self.project_root / "shelf_dispenser" / "ros" / "plan_once.py"
        plan = self._run_json_helper(
            operation=f"MoveIt2 规划 {name}",
            helper=helper,
            request_path=request_path,
            output_path=output_path,
            timeout_s=40,
        )
        self._assert_live_scene_contract(
            plan,
            obstacles=obstacles,
            boxes=boxes,
            tool_guard=tool_guard,
            held_object=held_object,
        )
        if not plan.get("success"):
            contacts = plan.get("goal_state_contacts") or []
            detail = (
                f"; 候选关节终态碰撞={contacts[:3]}"
                if contacts
                else ""
            )
            raise SafetyAbort(
                "MoveIt2 规划失败: "
                f"error={plan.get('error_code')}; "
                f"goal_constraint={plan.get('goal_constraint', goal_constraint)}"
                f"{detail}"
            )
        self._normalize_planned_arm_trajectory(plan, planning_group)
        required_fk = ("start_link7_fk", "endpoint_link7_fk")
        missing_fk = [name for name in required_fk if not plan.get(name)]
        if missing_fk:
            raise SafetyAbort(
                "MoveIt2 未返回运行时 FK 对齐证据，禁止执行: "
                + ", ".join(missing_fk)
            )
        LOG.info(
            "MoveIt2 规划 %s 成功: %d 点, %.3fs",
            name,
            len(plan["points_deg"]),
            plan["planning_time"],
        )
        return plan

    def validate_exact_path(
        self,
        *,
        name: str,
        start_left_joints_deg: Sequence[float],
        points_deg: Sequence[Sequence[float]],
        obstacles: Sequence[Sequence[float]],
        boxes: Sequence[dict],
        planning_frame: str,
        tool_guard: dict,
        held_object: dict | None = None,
        voxel_size: float,
    ) -> dict:
        request_path = self.run_dir / f"{name}_validation_request.json"
        output_path = self.run_dir / f"{name}_validation.json"
        request_path.write_text(
            json.dumps(
                {
                    "start_left_joints_deg": list(
                        map(float, start_left_joints_deg)
                    ),
                    "points_deg": [
                        list(map(float, item)) for item in points_deg
                    ],
                    "obstacles": [
                        list(map(float, item)) for item in obstacles
                    ],
                    "boxes": list(boxes),
                    "planning_frame": planning_frame,
                    "tool_guard": tool_guard,
                    "held_object": held_object,
                    "voxel_size": float(voxel_size),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        helper = self.project_root / "shelf_dispenser" / "ros" / "validate_path.py"
        validation = self._run_json_helper(
            operation=f"MoveIt2 轨迹复核 {name}",
            helper=helper,
            request_path=request_path,
            output_path=output_path,
            timeout_s=90,
        )
        self._assert_live_scene_contract(
            validation,
            obstacles=obstacles,
            boxes=boxes,
            tool_guard=tool_guard,
            held_object=held_object,
        )
        if not validation.get("success"):
            raise SafetyAbort(
                "轨迹碰撞检查失败: "
                f"{validation.get('invalid', [])[:1]}"
            )
        LOG.info(
            "轨迹 MoveIt 碰撞复核通过: %d 个状态",
            validation["checked_states"],
        )
        return validation

    def detach_attached_object(
        self,
        *,
        name: str,
        object_id: str,
        planning_frame: str,
    ) -> dict:
        """Send an explicit REMOVE and verify live attached IDs afterwards."""

        request_path = self.run_dir / f"{name}_detach_request.json"
        output_path = self.run_dir / f"{name}_detach.json"
        request_path.write_text(
            json.dumps(
                {
                    "object_id": str(object_id),
                    "link_name": "r_link7",
                    "planning_frame": str(planning_frame),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        helper = self.project_root / "shelf_dispenser" / "ros" / "detach_object.py"
        result = self._run_json_helper(
            operation=f"MoveIt2 显式 detach {object_id}",
            helper=helper,
            request_path=request_path,
            output_path=output_path,
            timeout_s=40,
        )
        attached_ids = set(result.get("attached_object_ids") or [])
        if not result.get("success") or object_id in attached_ids:
            raise SafetyAbort(
                "MoveIt live PlanningScene 仍含 attached object "
                f"{object_id}，保留 guard 并拒绝后续运动"
            )
        return result

    def close(self):
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
        if self.log_handle:
            self.log_handle.close()
