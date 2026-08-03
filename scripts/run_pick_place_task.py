#!/usr/bin/env python3
"""Thin robot-side entrypoint for the right-wrist bottle grasp demo."""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser import console
from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.run_manifest import write_run_manifest
from utils.config import load_config

LOG = logging.getLogger("bottle_demo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Right wrist RGB-D bottle grasp demo"
    )
    parser.add_argument(
        "--task-mode",
        choices=("from-pregrasp", "from-observation", "from-start"),
        help=(
            "run one supported real-robot transaction: resume at the verified "
            "pregrasp hover, start at the right-wrist observation pose, or "
            "start with head localization"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the collision-checked plan on the real right arm",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="run head localization and MoveIt planning, but never move hardware",
    )
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
        help="configuration-driven electronic fence profiles",
    )
    parser.add_argument(
        "--safety-profile",
        default="table_demo",
        help="electronic fence profile name",
    )
    parser.add_argument(
        "--stop-after-observation",
        action="store_true",
        help="execute only through wrist observation and localization",
    )
    parser.add_argument(
        "--confirm-before-grasp",
        action="store_true",
        help=(
            "到达观察位、检测到水瓶后暂停等待终端 Enter 再继续抓取；"
            "同一进程内暂停，不重启相机/YOLO/MoveIt。与 --stop-after-observation"
            "互斥（后者直接不抓取退出）"
        ),
    )
    parser.add_argument(
        "--visual-servo",
        action="store_true",
        help=(
            "在已验证的预抓取悬停位启用低频腕部闭环：重观测、限幅平移、"
            "再观测；默认关闭，去掉此参数即一键回退原路径"
        ),
    )
    parser.add_argument(
        "--visual-servo-mode",
        choices=("off", "shadow", "active"),
        default=None,
        help=(
            "预抓取视觉闭环模式：off 保持原路径；shadow 只重观测和报告"
            "受限修正建议、不运动；active 执行已验证的限幅修正。"
            "未指定时兼容 --visual-servo（active）和原默认（off）"
        ),
    )
    parser.add_argument(
        "--commissioning-speed",
        type=int,
        default=None,
        help=(
            "调试/commissioning 速度上限（1-100%）；同时限制全局转移、"
            "局部转移和接触邻近段，未指定则保持 profile 默认速度"
        ),
    )
    parser.add_argument(
        "--visual-servo-max-corrections",
        type=int,
        default=None,
        help="视觉闭环最多修正次数（1-3；默认 2）",
    )
    parser.add_argument(
        "--visual-servo-step-mm",
        type=float,
        default=None,
        help="视觉闭环单步平移上限 mm（默认 8）",
    )
    parser.add_argument(
        "--visual-servo-total-mm",
        type=float,
        default=None,
        help="视觉闭环相对初始锁定的累计平移上限 mm（默认 15）",
    )
    parser.add_argument(
        "--visual-servo-convergence-mm",
        type=float,
        default=None,
        help="视觉闭环收敛阈值 mm（默认 4）",
    )
    parser.add_argument(
        "--place-back",
        action="store_true",
        help="抓取抬升后把瓶子放回桌面并退开（不加则保持抓着不动）",
    )
    parser.add_argument(
        "--return-home",
        action="store_true",
        help=(
            "放回后额外用 MoveIt 规划返回 profile 里的 home_joints_deg"
            "（跟去程一样只受电子围栏保护，需要该 profile 配置了 home_joints_deg）"
        ),
    )
    parser.add_argument(
        "--dispense",
        action="store_true",
        help=(
            "抓取抬升后：携瓶运输姿态→身体升降→底盘原地约90°→"
            "头部重建右侧桌面→动态选净空落点→放置→返回无遮挡初始姿态"
        ),
    )
    parser.add_argument(
        "--delivery-safety-profile",
        default=None,
        help=(
            "转向后的独立桌面电子围栏 profile；必须现场测量并配置 "
            "side_table_delivery，禁止复用转向前货架坐标"
        ),
    )
    parser.add_argument(
        "--chassis-rotate-helper",
        default="/home/rm/agv_debug_tools/grabber_rotate_relative",
        help="机器人端闭环原地旋转工具（只允许约90°、linear恒为0）",
    )
    parser.add_argument(
        "--chassis-diagnostic-path",
        default="/home/rm/agv_debug_tools/agv_diag",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--chassis-pose-query-path",
        default="/home/rm/agv_debug_tools/agv_pose_query",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--target-product",
        default=None,
        help=(
            "按商品类别选择要抓的目标（YOLO 类别名，可用逗号分隔多个别名）；"
            "不给则保持现状——detector 内置的通用瓶子类别"
        ),
    )
    parser.add_argument(
        "--restore-teleop",
        action="store_true",
        help="demo 结束（STOP/Ctrl+C 退出保持）后自动运行官方 upstart_all.sh 恢复遥操",
    )
    parser.add_argument(
        "--resume-at-wrist",
        action="store_true",
        help="keep the current right-arm pose and resume wrist visual grasping",
    )
    parser.add_argument(
        "--finish-from-current",
        action="store_true",
        help=(
            "跳过定位与抓取，假设夹爪已经抓着水瓶（上一轮运行遗留在原地），"
            "从当前姿态直接按 --place-back/--return-home 收尾"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "shelf_dispenser")
    )
    parser.add_argument("--observe-seconds", type=float, default=10.0)
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """Reject unsafe/unsupported CLI combinations before creating a demo."""
    if (
        not args.task_mode
        and os.environ.get("BOTTLE_GRASP_ALLOW_LEGACY") != "1"
    ):
        raise SystemExit(
            "legacy phase flags are disabled; use --execute --task-mode "
            "{from-pregrasp|from-observation|from-start}. Developers may set "
            "BOTTLE_GRASP_ALLOW_LEGACY=1 for isolated diagnostics only."
        )
    if args.execute and args.plan_only:
        raise SystemExit("--execute and --plan-only are mutually exclusive")
    if args.confirm_before_grasp and args.stop_after_observation:
        raise SystemExit(
            "--confirm-before-grasp and --stop-after-observation are "
            "mutually exclusive"
        )
    if args.stop_after_observation and args.task_mode == "from-pregrasp":
        raise SystemExit(
            "--stop-after-observation 只支持 from-start 或 from-observation；"
            "from-pregrasp 已越过观察位"
        )
    if (
        args.commissioning_speed is not None
        and not 1 <= args.commissioning_speed <= 100
    ):
        raise SystemExit("--commissioning-speed 必须是 1-100 的整数")
    if args.visual_servo_mode is None:
        args.visual_servo_mode = "active" if args.visual_servo else "off"
    elif args.visual_servo and args.visual_servo_mode != "active":
        raise SystemExit(
            "--visual-servo 与 --visual-servo-mode=off/shadow 不能同时使用"
        )
    # Keep the historical attribute truthful for custom callers that only
    # inspect ``visual_servo``.  The explicit mode remains authoritative.
    args.visual_servo = args.visual_servo_mode == "active"
    # These controls affect a motion path when active.  Validate their fully
    # resolved relationship here, before RunOrchestrator can create a camera,
    # controller, gripper, or planner.  Keeping a duplicate runtime check is
    # intentional defence in depth for non-CLI embedders.
    max_corrections = (
        2
        if args.visual_servo_max_corrections is None
        else args.visual_servo_max_corrections
    )
    step_mm = 8.0 if args.visual_servo_step_mm is None else args.visual_servo_step_mm
    total_mm = 15.0 if args.visual_servo_total_mm is None else args.visual_servo_total_mm
    convergence_mm = (
        4.0
        if args.visual_servo_convergence_mm is None
        else args.visual_servo_convergence_mm
    )
    try:
        tuning_mm = tuple(
            float(value) for value in (step_mm, total_mm, convergence_mm)
        )
    except (TypeError, ValueError):
        tuning_mm = ()
    if not (
        isinstance(max_corrections, int)
        and not isinstance(max_corrections, bool)
        and 1 <= max_corrections <= 3
        and len(tuning_mm) == 3
        and all(math.isfinite(value) for value in tuning_mm)
        and 0.0 < tuning_mm[2] <= tuning_mm[0] <= tuning_mm[1]
    ):
        raise SystemExit(
            "视觉闭环参数无效：要求 corrections=1..3 且 "
            "0 < convergence <= step <= total（单位 mm）"
        )
    if args.task_mode and not args.execute:
        raise SystemExit("--task-mode requires --execute")
    if args.dispense and not args.task_mode:
        raise SystemExit("--dispense requires --task-mode")
    if args.dispense and not args.delivery_safety_profile:
        raise SystemExit("--dispense requires --delivery-safety-profile")
    if args.delivery_safety_profile and not args.dispense:
        raise SystemExit("--delivery-safety-profile requires --dispense")
    if args.task_mode and any(
        (
            args.plan_only,
            args.place_back,
            args.return_home,
            args.resume_at_wrist,
            args.finish_from_current,
        )
    ):
        raise SystemExit(
            "--task-mode owns the complete workflow and cannot be combined "
            "with legacy phase flags"
        )
    return args


def main() -> int:
    args = validate_args(build_parser().parse_args())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[])
    demo = RunOrchestrator(args, load_config(args.config))
    # Evidence must exist before initialization opens a camera, controller,
    # MoveIt or gripper.  A provenance-write failure is therefore a safe
    # preflight failure rather than a partially attributable robot run.
    try:
        manifest_path = write_run_manifest(
            demo.run_dir,
            args=demo.args,
            config=demo.cfg,
            project_root=demo.project_root,
            params=demo.params,
        )
    except OSError as exc:
        demo.close()
        raise SystemExit(f"无法写入 run manifest，拒绝启动任务: {exc}") from exc
    LOG.info("run manifest 已写入: %s", manifest_path)
    # Full detail keeps going to latest.log and the per-run run.log; the
    # terminal gets phase structure, live progress and timing instead.
    demo.timeline = console.install(
        latest_log=Path(args.output_dir) / "latest.log"
    )

    def request_stop(signum=None, frame=None):
        LOG.warning(
            "收到停止请求：后台监控线程正请求控制器缓停；不自动后退。"
            "硬件急停仍是最终保护"
        )
        demo.stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        demo.run()
        return 0
    except SafetyAbort as exc:
        LOG.error("安全中止: %s", exc)
        demo.state.update(stage="安全中止", message=str(exc))
        return 2
    except Exception:
        LOG.exception("未处理异常，停止并保持")
        demo.state.update(stage="程序错误", message="查看日志")
        return 1
    finally:
        demo.close()
        # Where the wall clock actually went.  Printed on success and on
        # failure alike: a slow run and an aborted run are both worth
        # attributing to a phase.
        if demo.timeline is not None:
            summary = demo.timeline.render()
            if summary:
                print(summary)


if __name__ == "__main__":
    raise SystemExit(main())
