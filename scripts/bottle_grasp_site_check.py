#!/usr/bin/env python3
"""No-arm-motion on-robot site check for the bottle pick/place task.

在真机上把整条链路用真实数据逐环节体检（不动机械臂），给出逐项
PASS/FAIL 和最终 GO/NO-GO。这是上机前的标准动作：本地 mock 测试防的是
"逻辑改坏"，这个体检防的是"上机才发现相机/舵机/时钟/规划链有问题"。
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bottle_grasp import console
from bottle_grasp.demo import BottleDemo
from bottle_grasp.run_manifest import write_run_manifest
from bottle_grasp.site_check import (
    SiteCheckRunner,
    build_site_checks,
    render_table,
    verdict,
    write_report,
)
from utils.config import load_config

LOG = logging.getLogger("bottle_demo")


def build_args(cli) -> SimpleNamespace:
    """The exact plan-only argument shape BottleDemo expects; no motion."""
    return SimpleNamespace(
        task_mode=None,
        execute=False,
        plan_only=True,
        config=cli.config,
        safety_config=cli.safety_config,
        safety_profile=cli.safety_profile,
        stop_after_observation=False,
        confirm_before_grasp=False,
        place_back=False,
        return_home=False,
        restore_teleop=False,
        resume_at_wrist=False,
        finish_from_current=False,
        host="127.0.0.1",
        port=cli.port,
        output_dir=cli.output_dir,
        observe_seconds=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "bottle_grasp" / "safety_profiles.json"),
    )
    parser.add_argument("--safety-profile", default="table_demo")
    parser.add_argument("--port", type=int, default=8879)
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "bottle_grasp")
    )
    cli = parser.parse_args()

    Path(cli.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[])
    demo = BottleDemo(build_args(cli), load_config(cli.config))
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
        raise SystemExit(f"无法写入 run manifest，拒绝启动体检: {exc}") from exc
    LOG.info("run manifest 已写入: %s", manifest_path)
    demo.timeline = console.install(
        latest_log=Path(cli.output_dir) / "latest.log"
    )

    def request_stop(signum=None, frame=None):
        LOG.warning("收到停止请求，体检中止")
        demo.stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        results = SiteCheckRunner(build_site_checks(demo)).run()
    finally:
        demo.close()

    report_path = write_report(demo.run_dir, results)
    print()
    print(render_table(results))
    final = verdict(results)
    print(f"\n结论: {final}   （逐项证据: {report_path}）")
    if final == "GO":
        print(
            "体检全绿。可以运行 scripts/run_bottle_grasp.sh "
            "{from-observation|from-start}（低速、有人守急停）。"
        )
        return 0
    print("存在未通过/未测项，先解决 ❌ 的项再上真机任务。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
