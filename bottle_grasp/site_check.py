"""On-robot, no-arm-motion site check: the rehearsal that predicts a real run.

本地 mock 测试只能防"逻辑被改坏"，对"上机能不能过"没有预测力；旧 plan
模式只覆盖链路前段（头部定位+规划），2026-07-18 真机两次失败（J2 贴限位、
MoveIt error=99999）都发生在 plan 测不到的段落。这个模块在真机上、不动
机械臂的前提下，用真实数据把整条链路逐环节体检一遍：

时钟 → MoveIt 碰撞三态探针 → 相机/SDK/MoveIt 栈初始化 → 右臂健康 →
夹爪反馈 → 头部深度流 → 真实 YOLO 定位 → 桌面拟合+场景预算 →
完整规划彩排（真实场景、真实 IK、FK 契约、围栏、MoveIt 密集后验，
产出一条真的可执行轨迹）→ 腕部相机流。

每项独立隔离：一项失败不阻断无关项，只把依赖它的项标 SKIP。全绿 = GO，
这条彩排轨迹就是 from-start 真跑时会执行的同款产物。
"""

from __future__ import annotations

import datetime
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .core import SafetyAbort

PASS = "pass"
FAIL = "fail"
SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    title: str
    status: str
    detail: str
    duration_s: float


@dataclass
class Check:
    name: str
    title: str
    fn: Callable[[dict], str]
    requires: tuple[str, ...] = field(default_factory=tuple)


class SiteCheckRunner:
    """Run ordered, dependency-aware checks with per-check isolation."""

    def __init__(self, checks: list[Check]):
        names = [check.name for check in checks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate check names: {names}")
        known: set[str] = set()
        for check in checks:
            missing = [
                requirement
                for requirement in check.requires
                if requirement not in known
            ]
            if missing:
                raise ValueError(
                    f"check {check.name} requires later/unknown: {missing}"
                )
            known.add(check.name)
        self.checks = checks

    def run(self, context: dict | None = None) -> list[CheckResult]:
        context = {} if context is None else context
        results: list[CheckResult] = []
        passed: set[str] = set()
        for check in self.checks:
            unmet = [
                requirement
                for requirement in check.requires
                if requirement not in passed
            ]
            started = time.monotonic()
            if unmet:
                results.append(
                    CheckResult(
                        name=check.name,
                        title=check.title,
                        status=SKIP,
                        detail=f"前置项未通过: {', '.join(unmet)}",
                        duration_s=0.0,
                    )
                )
                continue
            try:
                detail = check.fn(context)
                status = PASS
                passed.add(check.name)
            except SafetyAbort as exc:
                status, detail = FAIL, f"SafetyAbort: {exc}"
            except Exception as exc:  # noqa: BLE001 — 体检必须报告而不是崩溃
                status, detail = FAIL, f"{type(exc).__name__}: {exc}"
            results.append(
                CheckResult(
                    name=check.name,
                    title=check.title,
                    status=status,
                    detail=str(detail),
                    duration_s=round(time.monotonic() - started, 2),
                )
            )
        return results


def verdict(results: list[CheckResult]) -> str:
    """GO only when every check passed; SKIP means untested, not OK."""
    return (
        "GO"
        if results and all(result.status == PASS for result in results)
        else "NO-GO"
    )


def render_table(results: list[CheckResult]) -> str:
    lines = [
        f"{'状态':<6} {'耗时':>7}  {'检查项':<26} 详情",
        "-" * 78,
    ]
    marks = {PASS: "✅ 过", FAIL: "❌ 挂", SKIP: "⏭  跳"}
    for result in results:
        lines.append(
            f"{marks[result.status]:<5} {result.duration_s:>6.1f}s  "
            f"{result.title:<25} {result.detail}"
        )
    return "\n".join(lines)


def write_report(run_dir: Path, results: list[CheckResult]) -> Path:
    payload = {
        "verdict": verdict(results),
        "finished_at": datetime.datetime.now().isoformat(),
        "results": [asdict(result) for result in results],
    }
    path = Path(run_dir) / "site_check.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# The real check implementations.  Each takes/updates the shared context dict.
# ---------------------------------------------------------------------------


def build_site_checks(demo) -> list[Check]:
    """Assemble the on-robot check sequence around a plan-only BottleDemo."""

    def check_clock(_context: dict) -> str:
        year = datetime.datetime.now().year
        if year < 2026:
            raise SafetyAbort(
                f"系统时钟异常（{year} 年）——RTC/CMOS 电池问题复发，"
                "证据目录时间戳与日志排序都会失真，先手动对时"
            )
        return f"系统时间 {datetime.datetime.now():%Y-%m-%d %H:%M}"

    def check_model_assets(context: dict) -> str:
        # This is intentionally before *every* hardware-adjacent check.  A
        # missing/tampered exact-archive asset must not touch head telemetry,
        # MoveIt, a camera, an SDK, or the Demo initialization path.
        contract = demo._verify_detector_assets()
        context["model_asset_contract"] = contract
        optional = contract.get("optional_unavailable") or []
        optional_note = (
            f"；optional 未提供: {', '.join(optional)}" if optional else ""
        )
        return f"主模型资产 SHA/字节数已校验{optional_note}"

    def check_head_servo(_context: dict) -> str:
        from . import head_lock

        # A head-controller serial transaction can pause its UDP broadcast for
        # 2--4 seconds.  This check is read-only, so wait through the shared
        # documented gap instead of declaring a transient silence as a fault.
        current = head_lock.read_current_angle(
            patience=head_lock.BROADCAST_GAP_PATIENCE
        )
        if current is None:
            raise SafetyAbort("读不到头部舵机角度广播")
        if not head_lock.is_at_reference(current):
            raise SafetyAbort(
                f"头部不在标定基准角度: {current}，头部定位不可信；"
                "先运行头部回中再体检"
            )
        return f"头部在标定基准: {current}"

    def check_collision_probe(_context: dict) -> str:
        # 独立三态探针（无盒基线 valid → 巨盒 invalid → 撤盒恢复 valid）。
        # 它自带只规划 move_group，必须在 demo 的 MoveIt 栈启动之前跑完。
        script = demo.project_root / "bottle_grasp" / "moveit_collision_selftest.py"
        ros_prefix = (
            "source /opt/ros/humble/setup.bash && "
            "source /home/rm/ros2_ws/install/setup.bash && "
        )
        result = subprocess.run(
            ["bash", "-lc", ros_prefix + f"python3 '{script}'"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            tail = (result.stdout or "")[-400:] + (result.stderr or "")[-400:]
            raise SafetyAbort(f"MoveIt 碰撞三态探针失败: {tail}")
        return "基线 valid → 巨盒 invalid → 撤盒恢复，碰撞链健康"

    def check_init(_context: dict) -> str:
        demo.initialize()
        return (
            f"相机/YOLO/SDK/MoveIt 栈就绪；围栏 profile={demo.safety.name}"
        )

    def check_arm(_context: dict) -> str:
        # Match the real task preflight.  A released green drag button can
        # leave only 0xF000 latched even though all joints/controller are
        # healthy; RobotSession clears exactly that narrow case and demands
        # two clean rereads.  This changes no joint position.
        recovered = demo.robot.recover_transient_joint_frame_loss()
        demo.robot.assert_arm_healthy()
        joints = np.asarray(demo.robot.joints_deg(), dtype=float)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise SafetyAbort(f"关节读数无效: {joints.tolist()}")
        flange = demo.robot.current_flange()
        if not np.all(np.isfinite(flange)):
            raise SafetyAbort("FK 结果含非有限数")
        recovery_note = (
            f"已清除并双读复核 {','.join(f'J{joint}' for joint in recovered)} 的 0xF000；"
            if recovered
            else ""
        )
        return (
            f"{recovery_note}无错误码；J={np.round(joints, 1).tolist()}"
        )

    def check_gripper(_context: dict) -> str:
        state = demo.robot.gripper_state()
        return (
            f"pos={state['pos'][0]} dof_state={state['dof_state'][0]}，"
            "反馈链正常（只读，不动作）"
        )

    def check_head_stream(_context: dict) -> str:
        if demo.camera_name != "head":
            demo._start_camera("head")
        deadline = time.time() + 5
        while time.time() < deadline:
            if demo.camera.get_frame_timestamp() > time.time() - 1.0:
                break
            time.sleep(0.2)
        else:
            raise SafetyAbort("头部相机 5 秒内无新鲜帧")
        _color, depth = demo.camera.get_latest_frames()
        K, _ = demo.camera.get_camera_intrinsics()
        if depth is None or K is None:
            raise SafetyAbort("头部深度或内参缺失")
        valid = float(np.mean(np.isfinite(depth) & (depth > 0)))
        if valid < 0.30:
            raise SafetyAbort(f"头部深度有效像素只有 {valid:.0%}，画面异常")
        return f"帧新鲜；深度有效像素 {valid:.0%}"

    def check_head_localization(context: dict) -> str:
        target = demo._fresh_head_target()
        context["head_target"] = target
        return (
            f"7 帧共识，散布 {target.position_spread_m * 1000:.1f} mm，"
            f"conf={target.confidence:.2f}，"
            f"base={np.round(target.point_base, 3).tolist()}"
        )

    def check_scene_and_table(context: dict) -> str:
        demo._build_head_scene(context["head_target"])
        return (
            f"{len(demo.scene_voxels)} 体素（预算内）；"
            f"{len(demo.scene_boxes)} 个围栏盒已随实测桌面自适应"
        )

    def check_plan_rehearsal(context: dict) -> str:
        # 真实候选生成 + 分级抓取预演 + SafeMotionPlanner 全套契约
        #（FK 契约、live 场景契约、独立围栏、MoveIt 密集后验、终点续演）。
        # 配置准备位时先从真实当前姿态规划第一段，再把其终点作为第二段的
        # 显式起点。全程只生成/复核轨迹，不发送任何机械臂运动命令。
        target = context["head_target"]
        staging_plan = demo._plan_observation_staging()
        planning_start = None
        staging_points = 0
        if staging_plan is not None:
            staging_trajectory = staging_plan.get("points_deg") or []
            if not staging_trajectory:
                raise SafetyAbort("观察准备位规划返回空轨迹")
            planning_start = staging_trajectory[-1]
            staging_points = len(staging_trajectory)
        plan = demo._plan_observation(
            np.asarray(target.point_base, dtype=float),
            start_right_joints_deg=planning_start,
        )
        coverage = plan.get("search_coverage", {})
        return (
            (
                f"准备位轨迹 {staging_points} 点；"
                if staging_plan is not None
                else "无需准备位；"
            )
            + f"观察位轨迹 {len(plan.get('points_deg', []))} 点；"
            f"搜索覆盖 {coverage.get('attempted_candidates')}"
            f"/{coverage.get('total_candidates')} 候选，"
            f"planners={coverage.get('planner_ids')}"
        )

    def check_wrist_stream(_context: dict) -> str:
        demo._start_camera("right_wrist")
        deadline = time.time() + 5
        while time.time() < deadline:
            if demo.camera.get_frame_timestamp() > time.time() - 1.0:
                break
            time.sleep(0.2)
        else:
            raise SafetyAbort("右腕相机 5 秒内无新鲜帧")
        K, _ = demo.camera.get_camera_intrinsics()
        if K is None:
            raise SafetyAbort("右腕相机内参缺失")
        return "帧新鲜、内参可用（腕部检测在观察位才有意义，不在此强求）"

    return [
        Check("clock", "系统时钟", check_clock),
        Check(
            "model_assets",
            "YOLO 模型资产契约",
            check_model_assets,
            requires=("clock",),
        ),
        Check(
            "head_servo",
            "头部舵机基准位",
            check_head_servo,
            requires=("model_assets",),
        ),
        Check(
            "collision_probe",
            "MoveIt 碰撞三态探针",
            check_collision_probe,
            requires=("model_assets",),
        ),
        Check(
            "init_stack",
            "相机/SDK/MoveIt 栈初始化",
            check_init,
            requires=("model_assets", "collision_probe"),
        ),
        Check("arm", "右臂健康", check_arm, requires=("init_stack",)),
        Check("gripper", "夹爪反馈链", check_gripper, requires=("init_stack",)),
        Check(
            "head_stream",
            "头部 RGB-D 流",
            check_head_stream,
            requires=("init_stack",),
        ),
        Check(
            "head_localization",
            "真实头部定位",
            check_head_localization,
            requires=("head_stream", "head_servo"),
        ),
        Check(
            "scene_table",
            "桌面拟合+场景预算",
            check_scene_and_table,
            requires=("head_localization",),
        ),
        Check(
            "plan_rehearsal",
            "完整规划彩排",
            check_plan_rehearsal,
            requires=("scene_table", "arm"),
        ),
        Check(
            "wrist_stream",
            "右腕相机流",
            check_wrist_stream,
            requires=("init_stack",),
        ),
    ]
