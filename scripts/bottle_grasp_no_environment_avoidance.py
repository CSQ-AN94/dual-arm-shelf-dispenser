#!/usr/bin/env python3
"""Historical real-robot entrypoint with environmental avoidance disabled.

This deliberately does not alter the normal bottle-grasp entrypoint.  It
removes world obstacles only; robot self-collision, left-arm collision, joint
limits, IK/path validity, tool geometry, and controller stop handling remain.
Its checked-in profile is currently execution-disabled until the shared
right-tool installation transform is independently measured.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import signal
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bottle_grasp import console
from bottle_grasp.core import SafetyAbort
from bottle_grasp.demo import BottleDemo
from bottle_grasp.run_manifest import write_run_manifest

LOG = logging.getLogger("bottle_demo_no_environment_avoidance")
PROFILE_PATH = ROOT / "bottle_grasp" / "no_environment_avoidance_profile.json"
PROFILE_NAME = "no_environment_avoidance"


class NoEnvironmentAvoidanceDemo(BottleDemo):
    """Bottle workflow with world-obstacle inputs deliberately empty."""

    def __init__(self, args, cfg):
        super().__init__(args, cfg)
        # Keep non-contact transfer/retreat at 15%; final approach, lift and
        # lower remain at the contact-adjacent 3% used by the shared workflow.
        self.params = dataclasses.replace(
            self.params,
            transit_speed=15,
            travel_speed=15,
            final_speed=min(self.params.final_speed, 3),
        )

    def initialize(self):
        super().initialize()
        if self.safety is None:
            raise SafetyAbort("无环境避障入口没有加载到专用 profile")
        if (
            self.safety.name != PROFILE_NAME
            or self.safety.use_dynamic_rgbd
            or self.safety.keepout_boxes
        ):
            raise SafetyAbort(
                "无环境避障入口只能使用专用空场景 profile，拒绝混用其他配置"
            )
        self.stage(
            "环境避障已关闭",
            "无桌面围栏、无 RGB-D 障碍体素、无右腕通道障碍拒绝；"
            "前后转移 15%，最终接近/升降 3%",
        )

    def collision_gate(
        self,
        target_box: Optional[Sequence[int]],
        target_base,
    ) -> None:
        # BottleDemo normally rejects a local approach using wrist depth.  This
        # override is the one environment check that the profile cannot turn
        # off.  Target localization/lock remains active.
        self.stage("右腕点云通道检查", "由无环境避障入口显式跳过")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bottle grasp with environmental collision checks disabled"
    )
    parser.add_argument(
        "--task-mode",
        required=True,
        choices=("from-observation", "from-start"),
    )
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument(
        "--stop-after-observation",
        action="store_true",
        help="execute the from-start transfer and stop after wrist verification",
    )
    parser.add_argument(
        "--acknowledge-no-environment-collision-check",
        action="store_true",
        required=True,
        help="explicit acknowledgement required by this unsafe entrypoint",
    )
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8879)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "bottle_grasp_no_avoidance"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # Keep configuration IO after argument parsing so --help and contract
    # tests do not require robot-side YAML dependencies.
    from utils.config import load_config

    # BottleDemo consumes these standard entrypoint attributes.
    args.safety_config = str(PROFILE_PATH)
    args.safety_profile = PROFILE_NAME
    args.plan_only = False
    args.confirm_before_grasp = False
    args.place_back = False
    args.return_home = False
    args.restore_teleop = False
    args.resume_at_wrist = False
    args.finish_from_current = False
    args.observe_seconds = 10.0

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[])
    demo = NoEnvironmentAvoidanceDemo(args, load_config(args.config))
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
        if demo.timeline is not None:
            summary = demo.timeline.render()
            if summary:
                print(summary)


if __name__ == "__main__":
    raise SystemExit(main())
