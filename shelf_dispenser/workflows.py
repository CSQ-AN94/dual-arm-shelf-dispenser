"""The supported real-robot bottle pick/place workflows.

The public task interface is deliberately smaller than the legacy demo CLI:
one operation, three explicit starting conditions, and one shared grasp/place tail.
Historical run artefacts are evidence only; neither workflow resumes from
saved localizations or trajectories.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .core import SafetyAbort
from .environment_guard import LeftArmStabilityGuard
from .mobile_body import ReturnAuthorization


class StartMode(str, Enum):
    """The only supported, physically verified task entry conditions."""

    FROM_PREGRASP = "from-pregrasp"
    FROM_OBSERVATION = "from-observation"
    FROM_START = "from-start"


class DeliverMode(str, Enum):
    """How the shared pick/place tail disposes of a held bottle.

    PLACE_BACK is table_demo's existing behaviour (put it back at the locked
    pick point) and stays the default so no existing caller changes
    behaviour by omission. DISPENSE carries it to the profile's configured
    output point instead — real vending delivery, not a verify-and-replace
    cycle.
    """

    PLACE_BACK = "place_back"
    DISPENSE = "dispense"


class TaskPhase(str, Enum):
    START = "start"
    SHELF_READY = "shelf_ready"
    PREFLIGHT = "preflight"
    HEAD_LOCK = "head_lock"
    SCENE_SYNC = "scene_sync"
    START_HOME_CHECK = "start_home_check"
    MOVE_TO_OBSERVATION_STAGING = "move_to_observation_staging"
    MOVE_TO_OBSERVATION = "move_to_observation"
    WRIST_LOCK = "wrist_lock"
    CONFIRM_BEFORE_GRASP = "confirm_before_grasp"
    GRASP_AND_LIFT = "grasp_and_lift"
    GRASP_VERIFIED = "grasp_verified"
    BODY_POSITIONING = "body_positioning"
    BODY_RETURN = "body_return"
    SHELF_RESTORED = "shelf_restored"
    OUTPUT_SCENE_SYNC = "output_scene_sync"
    PLACE_AND_RETREAT = "place_and_retreat"
    RELEASE_VERIFIED = "release_verified"
    RETURN_HOME = "return_home"
    DONE = "done"
    ABORTED = "aborted"


class ObjectState(str, Enum):
    EMPTY = "empty"
    UNKNOWN = "unknown"
    HELD = "held"


class RunStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    SAFE_ABORT = "safe_abort"
    FAULT = "fault"


@dataclass(frozen=True)
class RunResult:
    run_id: str
    mode: str
    status: str
    phase: str
    object_state: str
    evidence_dir: str
    error: str | None = None


class BottlePickPlaceTask:
    """Run one fresh, bounded pick/place transaction.

    ``demo`` is the hardware composition root.  This module owns legal task
    ordering and object-state semantics; camera, robot and planner details stay
    behind the existing adapters used by ``RunOrchestrator``.
    """

    def __init__(self, demo: Any):
        self.demo = demo
        self.run_id = str(uuid4())
        self.phase = TaskPhase.START
        self.object_state = ObjectState.EMPTY
        self.status = RunStatus.RUNNING
        self.error: str | None = None
        self._sequence = 0

    @property
    def _run_dir(self) -> Path:
        return Path(self.demo.run_dir)

    def _snapshot(self) -> RunResult:
        return RunResult(
            run_id=self.run_id,
            mode=str(self.demo.args.task_mode),
            status=self.status.value,
            phase=self.phase.value,
            object_state=self.object_state.value,
            evidence_dir=str(self._run_dir),
            error=self.error,
        )

    def _write_json_atomic(self, name: str, payload: dict) -> None:
        path = self._run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _record(self, phase: TaskPhase, message: str) -> None:
        self.phase = phase
        self._sequence += 1
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": self._sequence,
            "run_id": self.run_id,
            "mode": str(self.demo.args.task_mode),
            "phase": phase.value,
            "object_state": self.object_state.value,
            "message": message,
        }
        journal = self._run_dir / "task_journal.jsonl"
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.demo.stage(f"任务/{phase.value}", message)

    def _finish(self) -> RunResult:
        result = self._snapshot()
        self._write_json_atomic("task_result.json", asdict(result))
        return result

    def run(
        self,
        mode: StartMode,
        deliver_mode: DeliverMode = DeliverMode.PLACE_BACK,
    ) -> RunResult:
        """Execute the selected prefix followed by the shared pick/place tail.

        `deliver_mode` only changes how the held bottle is disposed of after
        a verified grasp+lift; it is not a third entry point and does not
        change the legal `mode` values or their ordering.
        """

        try:
            deliver_mode = DeliverMode(deliver_mode)
        except (TypeError, ValueError) as exc:
            raise SafetyAbort(f"未知送货模式: {deliver_mode!r}") from exc
        if deliver_mode is DeliverMode.DISPENSE:
            # This project delivers to the shelf's lower layer, not to a side
            # table.  The side-table code stays -- placing on a table and
            # placing in a bin share most of their machinery, and it will be
            # wanted again -- but the entry is closed so nothing reaches it by
            # accident.  Reopening it means deciding what the two flows share
            # first, starting with home_joints_deg, which they should not.
            raise SafetyAbort(
                "桌面送货入口已关闭：本项目出货放在下层货架。"
                "侧桌代码保留但不接线，重开前先决定两条流程该共用什么"
            )
        if StartMode(self.demo.args.task_mode) is not mode:
            raise SafetyAbort(
                "任务模式在解析后发生变化，拒绝启动不一致的实机流程"
            )
        declared_dispense = getattr(self.demo.args, "dispense", None)
        if (
            declared_dispense is not None
            and bool(declared_dispense) != (deliver_mode is DeliverMode.DISPENSE)
        ):
            raise SafetyAbort(
                "任务参数 --dispense 与 DeliverMode 不一致，拒绝绕过 "
                "SHELF_READY/body-return 门禁"
            )
        if (
            mode is StartMode.FROM_PREGRASP
            and getattr(self.demo.args, "stop_after_observation", False)
        ):
            raise SafetyAbort(
                "stop-after-observation 只支持 from-start 或 from-observation；"
                "from-pregrasp 已越过观察位"
            )
        environment_guard: LeftArmStabilityGuard | None = None
        shelf_ready_start = None
        try:
            if deliver_mode is DeliverMode.DISPENSE:
                # This must precede initialize(): a controlled RobotSession
                # can stop teleop, select a tool frame, power the gripper and
                # issue arm-adjacent setup.  SHELF_READY is body-only.
                self._record(
                    TaskPhase.SHELF_READY,
                    "在任何 arm/gripper 命令前捕获并验证底盘/升降 SHELF_READY",
                )
                shelf_ready_start = self.demo._capture_shelf_ready_for_dispense()
            self.demo.initialize()
            self._record(TaskPhase.PREFLIGHT, "执行硬件、夹爪和安全配置预检")
            self.demo._preflight()
            if deliver_mode is DeliverMode.DISPENSE:
                self.demo._preflight_side_table_delivery(start=shelf_ready_start)
            if self.demo.left_robot is None:
                raise SafetyAbort(
                    "完整实机任务必须读取并监控左臂碰撞快照"
                )
            environment_guard = LeftArmStabilityGuard(
                left_reader=self.demo.left_robot,
                stop_event=self.demo.stop_event,
                tolerance_deg=self.demo.params.planned_start_tolerance_deg,
            )
            self.demo.task_left_reference_joints_deg = (
                environment_guard.start()
            )

            self._record(TaskPhase.HEAD_LOCK, "本轮重新采集固定头部目标")
            head_target = self.demo._fresh_head_target()

            self._record(TaskPhase.SCENE_SYNC, "用本轮头部 RGB-D 重建场景与桌面")
            self.demo._build_head_scene(head_target)

            if mode is StartMode.FROM_START:
                self._record(
                    TaskPhase.START_HOME_CHECK,
                    "检查右臂是否在示教 home；非 home 时从本轮场景安全规划归位",
                )
                moved_to_home = self.demo._normalize_start_home()
                if moved_to_home:
                    # The arm has crossed the head camera's field and the old
                    # target/world snapshot is stale.  Reacquire both before
                    # planning any target-dependent observation transfer.
                    self._record(
                        TaskPhase.HEAD_LOCK,
                        "右臂归位后重新采集固定头部目标",
                    )
                    head_target = self.demo._fresh_head_target()
                    self._record(
                        TaskPhase.SCENE_SYNC,
                        "右臂归位后重建场景，再规划观察位",
                    )
                    self.demo._build_head_scene(head_target)

                staging_joints = getattr(
                    self.demo.safety,
                    "observation_staging_joints_deg",
                    None,
                )
                if staging_joints is not None:
                    self._record(
                        TaskPhase.MOVE_TO_OBSERVATION_STAGING,
                        "先规划并执行抬高展开准备位，解除自然下垂起点的奇异/内收负担",
                    )
                    staging_plan = self.demo._plan_observation_staging()
                    if staging_plan is not None:
                        self.demo._refresh_and_revalidate_plan(
                            name="moveit_observation_staging",
                            plan=staging_plan,
                            locked_target=head_target,
                        )
                        self.demo._execute_plan(
                            "抬高展开到观察准备位", staging_plan
                        )

                        # The arm has moved through the head camera's scene
                        # and planning may have taken tens of seconds.  Do not
                        # reuse either the target lock or world snapshot for
                        # the target-dependent observation transfer.
                        self._record(
                            TaskPhase.HEAD_LOCK,
                            "准备位到达后重新采集固定头部目标",
                        )
                        head_target = self.demo._fresh_head_target()
                        self._record(
                            TaskPhase.SCENE_SYNC,
                            "准备位到达后重建场景，再规划观察位",
                        )
                        self.demo._build_head_scene(head_target)

                self._record(
                    TaskPhase.MOVE_TO_OBSERVATION,
                    "规划、复核并执行到右腕观察位",
                )
                observation_plan = self.demo._plan_observation(
                    np.asarray(head_target.point_base, dtype=float)
                )
                # A complete candidate/planner search may legitimately use
                # most of its 120-second budget.  Never execute that answer
                # against the old camera snapshot: reacquire the world and
                # revalidate the exact chosen trajectory first.
                self.demo._refresh_and_revalidate_plan(
                    name="moveit_observation",
                    plan=observation_plan,
                    locked_target=head_target,
                )
                self.demo._execute_plan(
                    "避障移动到右腕观察位", observation_plan
                )

            self._record(
                TaskPhase.WRIST_LOCK,
                (
                    "验证当前预抓取悬停位并建立新鲜目标锁"
                    if mode is StartMode.FROM_PREGRASP
                    else (
                        "验证当前右腕观察位并建立新鲜目标锁"
                        if mode is StartMode.FROM_OBSERVATION
                        else "到位后建立新鲜右腕目标锁"
                    )
                ),
            )
            wrist_target = self.demo._fresh_wrist_target(head_target)
            if mode is StartMode.FROM_PREGRASP:
                self.demo._verify_wrist_pregrasp_start(wrist_target)
            else:
                self.demo._verify_wrist_observation_start(wrist_target)

            if getattr(self.demo.args, "stop_after_observation", False):
                environment_guard.close()
                environment_guard = None
                self.status = RunStatus.DONE
                self._record(
                    TaskPhase.DONE,
                    "已真实移动到观察位并完成右腕定位/抓放预演；按要求不抓取",
                )
                return self._finish()

            if getattr(self.demo.args, "confirm_before_grasp", False):
                # This is a real task-state transition, not the old legacy
                # workflow's ad-hoc stdin pause.  ObjectState stays EMPTY
                # until the operator releases this gate and a grasp command
                # is actually about to be issued.
                self._record(
                    TaskPhase.CONFIRM_BEFORE_GRASP,
                    "已完成观察位和右腕定位；等待操作者确认后才允许闭夹",
                )
                self.demo._wait_for_grasp_confirmation()

            # Entering a grasp command makes the physical object state
            # conservative UNKNOWN until gripper feedback and lift both pass.
            self.object_state = ObjectState.UNKNOWN
            self._record(
                TaskPhase.GRASP_AND_LIFT,
                "执行共享接近、夹取反馈判定和抬升",
            )
            lifted_target = (
                self.demo._grasp_and_lift_from_pregrasp(wrist_target)
                if mode is StartMode.FROM_PREGRASP
                else self.demo._grasp_and_lift(wrist_target)
            )
            self.object_state = ObjectState.HELD
            self._record(
                TaskPhase.GRASP_VERIFIED,
                "夹爪反馈通过且瓶子已完成抬升，物体状态记为 held",
            )

            # Release may happen before a later retreat/vision failure, so do
            # not claim HELD or EMPTY while this compound action is in flight.
            self.object_state = ObjectState.UNKNOWN
            if deliver_mode is DeliverMode.DISPENSE:
                self._record(
                    TaskPhase.BODY_POSITIONING,
                    "携瓶收进示教运输姿态，身体升降后底盘仅原地旋转约90°",
                )
                self.demo._dispense_to_side_table(
                    lifted_target, start=shelf_ready_start
                )
                self.object_state = ObjectState.EMPTY
                self._record(
                    TaskPhase.RELEASE_VERIFIED,
                    "瓶子已在实时点云选择的右侧桌面位置释放并完成三维确认",
                )
            else:
                self._record(
                    TaskPhase.PLACE_AND_RETREAT,
                    "放回锁定位置、松开夹爪、安全退开后由固定头部确认释放",
                )
                self.demo._place_back(wrist_target, lifted_target)
                self.object_state = ObjectState.EMPTY
                self._record(
                    TaskPhase.RELEASE_VERIFIED,
                    "夹爪打开、瓶子在锁定放置点得到视觉确认且机械臂已退开",
                )

            # A successful cycle must leave the right arm in the taught,
            # head-camera-clear home posture regardless of which verified
            # physical entry point was used.  Refresh the world after release
            # before planning this global leg; saved approach-time geometry is
            # stale after the bottle and arm have moved.
            self._record(
                TaskPhase.RETURN_HOME,
                "释放并退开后重新采集头部障碍场景，再用同一安全规划链返回右臂初始姿态",
            )
            if deliver_mode is DeliverMode.DISPENSE:
                self._record(
                    TaskPhase.OUTPUT_SCENE_SYNC,
                    "释放后再次重建右侧桌面场景，再规划返回无遮挡初始姿态",
                )
                self.demo._capture_output_table_scene(
                    require_place_candidate=False
                )
            else:
                self.demo._refresh_head_scene_for_global_motion(wrist_target)
            self.demo._return_home()

            if deliver_mode is DeliverMode.DISPENSE:
                # A body return is categorically unavailable while HELD or
                # UNKNOWN.  The arm has just completed its empty-profile home
                # path, then both arm guards are checked immediately before
                # handing this immutable authorization to the body controller.
                right_arm_compact_or_home = bool(
                    self.demo._right_arm_at_delivery_home()
                )
                environment_guard.check()
                authorization = ReturnAuthorization(
                    release_verified=True,
                    object_state=self.object_state.value,
                    right_arm_compact_or_home=right_arm_compact_or_home,
                    left_arm_stable=True,
                )
                self._record(
                    TaskPhase.BODY_RETURN,
                    "释放已验证、物体 empty 且双臂安全，授权底盘反向回到 SHELF_READY",
                )
                self.demo._return_body_to_shelf_ready(
                    start=shelf_ready_start,
                    authorization=authorization,
                )
                self._record(
                    TaskPhase.SHELF_RESTORED,
                    "底盘/升降恢复已验证；右臂保持 compact/home",
                )

            environment_guard.close()
            environment_guard = None

            self.status = RunStatus.DONE
            self._record(
                TaskPhase.DONE,
                "抓取、抬升、放置、释放确认、退开和返回右臂初始姿态均已完成；任务正常退出",
            )
            return self._finish()
        except Exception as exc:
            if shelf_ready_start is not None:
                body = getattr(self.demo, "mobile_body", None)
                if body is not None:
                    try:
                        body.close()
                    except Exception:
                        # The coordinator already tries repeated zero-speed
                        # stops.  Preserve the original failure.
                        pass
            if environment_guard is not None:
                try:
                    environment_guard.close()
                except Exception:
                    # Preserve the first physical/safety failure.  The guard
                    # already set stop_event if it was the source.
                    pass
            self.status = (
                RunStatus.SAFE_ABORT
                if isinstance(exc, SafetyAbort)
                else RunStatus.FAULT
            )
            self.error = f"{type(exc).__name__}: {exc}"
            try:
                self._record(TaskPhase.ABORTED, self.error)
                self._finish()
            except Exception:
                # Evidence I/O must never hide the original hardware/safety
                # failure that determines the process exit code.
                pass
            raise
