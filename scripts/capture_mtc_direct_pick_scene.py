#!/usr/bin/env python3
"""Capture a fixed-head shelf scene and emit an MTC pick-only scenario.

This entry never connects either arm: it only captures the head camera and
emits a scenario for a later MTC plan-only run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser import head_lock
from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.run_manifest import write_run_manifest
from run_pick_place_task import build_args
from localization_to_mtc_scenario import build_scenario
from utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=(
            ROOT
            / "mtc_ws/src/grabber_mtc_planner/scenarios/shelf_transfer_fixture.yaml"
        ),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/mtc_direct_pick"))
    parser.add_argument("--scenario-out", type=Path)
    parser.add_argument("--port", type=int, default=8881)
    parser.add_argument(
        "--planning-arm",
        choices=("right_arm", "left_arm"),
        default="right_arm",
        help="选择 MTC 规划臂；left_arm 只生成不可执行的 plan-only 导出",
    )
    parser.add_argument(
        "--target-product",
        default=None,
        help=(
            "按商品类别选择要抓的目标（YOLO 类别名，可用逗号分隔多个别名）；"
            "不给则保持 detector 内置的通用瓶子类别。注意仓库现有模型只认通用"
            "瓶子，按 SKU 选格位需要先补类别标注的训练数据"
        ),
    )
    cli = parser.parse_args()
    cli.safety_profile = "shelf_template"

    Path(cli.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    demo_args = build_args(cli)
    # RunOrchestrator reads this at construction, so it has to land before the ctor.
    demo_args.target_product = cli.target_product
    # RunOrchestrator's plan_only initialization connects both arms for planning.
    # This entry only needs fixed-head perception, so keep it camera-only.
    demo_args.plan_only = False
    demo = RunOrchestrator(demo_args, load_config(cli.config))
    # Coarser cells can merge the target silhouette with nearby background and
    # then conservatively block attach. Keep this refinement direct-pick-only.
    demo.params = replace(
        demo.params,
        scene_voxel_m=0.025,
        scene_max_voxels=3000,
        # Live shelf frames put the transparent bottle's rear return up to
        # 57 mm behind the locked front surface; 58 mm remains below the
        # existing 75 mm foreground-obstacle regression.
        target_occupancy_depth_slack_m=0.02,
    )
    scenario_out = cli.scenario_out or demo.run_dir / "mtc_direct_pick.yaml"
    try:
        write_run_manifest(
            demo.run_dir,
            args=demo.args,
            config=demo.cfg,
            project_root=demo.project_root,
            params=demo.params,
        )
        demo.initialize()
        current_head = head_lock.read_current_angle()
        if current_head is None:
            current_head = head_lock.read_current_angle_direct()
        if not head_lock.is_at_reference(current_head):
            raise SafetyAbort(
                "固定头部不在已标定基准角，camera-only 入口拒绝自动转动: "
                f"current={current_head}, expected={head_lock.HEAD_REFERENCE}"
            )
        target = demo._fresh_head_target()
        demo._build_head_scene(target, full_frame=True)
        localization_path = demo.run_dir / "头部粗定位_localization.json"
        scene_path = demo.run_dir / "head_scene.json"
        scenario = build_scenario(
            localization_path,
            cli.template,
            Path(cli.safety_config),
            "shelf_template",
            max_age_s=demo.params.scene_max_age_s,
            allow_stale=False,
            scene_path=scene_path,
            pick_only=True,
            planning_arm_id=cli.planning_arm,
        )
        scenario_out.parent.mkdir(parents=True, exist_ok=True)
        scenario_out.write_text(
            yaml.safe_dump(scenario, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    finally:
        demo.close()

    print(
        f"已生成固定头部 MTC {cli.planning_arm} pick-only 场景: {scenario_out}"
    )
    print("未连接机械臂、未移动夹爪；下一步只能运行 MTC plan-only。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
