#!/usr/bin/env python3
"""On-site, no-arm-motion shelf-panel measurement (robot-side entrypoint).

Companion to `scripts/bottle_grasp_site_check.py`, but leaner: this tool
only needs the head RGB-D stream and calibration, so it never creates a
RobotSession or touches the arm/teleop — unlike site_check, which builds
the full stack to rehearse planning. Run this once per physical shelf
layout (or whenever the shelf/robot placement changes) to get a *draft*
`keepout_boxes` fragment for `bottle_grasp/safety_profiles.json`'s
`shelf_template`.

This script never writes `safety_profiles.json` itself. Measuring, editing
the profile, and marking it `verified_for_execution: true` are three
separate human-reviewed steps on purpose — see `bottle_grasp/SAFETY_PROFILES.md`.

Typical use: place a bottle (or any small marker) at the picking position
you care about, note its approximate coordinate in `right_controller_base`
(from a prior `from-start --plan-only` run's head localization log, or a
previous run of this same script), then run:

    python scripts/measure_shelf_geometry.py \\
        --target-base 0.0 0.55 -0.10 \\
        --faces shelf_bottom,shelf_back,shelf_left_panel,shelf_right_panel
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bottle_grasp.demo import BottleDemo
from bottle_grasp.shelf_model import FACE_SPECS
from bottle_grasp.shelf_survey import run_shelf_survey
from utils.config import load_config

LOG = logging.getLogger("bottle_demo")


def build_args(cli) -> SimpleNamespace:
    """Just enough for BottleDemo.__init__ + _ensure_head_reference +
    _start_camera("head"). Deliberately does not request a RobotSession —
    this tool has no business touching the arm or teleop."""
    return SimpleNamespace(
        task_mode=None,
        execute=cli.execute_head_correction,
        plan_only=False,
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
        target_product=None,
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
    parser.add_argument("--safety-profile", default="shelf_template")
    parser.add_argument(
        "--target-base",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help=(
            "近似目标坐标（right_controller_base 系，米），比如放在要测的格位"
            "的一个瓶子/标记物的大致位置"
        ),
    )
    parser.add_argument(
        "--faces",
        default=",".join(sorted(FACE_SPECS)),
        help=f"逗号分隔的货架面列表，可选: {', '.join(sorted(FACE_SPECS))}",
    )
    parser.add_argument(
        "--execute-head-correction",
        action="store_true",
        help=(
            "头部偏离标定基准角度时真的驱动舵机校正回去（只动头部舵机，"
            "不涉及机械臂）；不给则只警告不动"
        ),
    )
    parser.add_argument(
        "--min-gap-m",
        type=float,
        default=None,
        help=(
            "覆盖 shelf_fit_min_gap_m（默认0.02）。测 shelf_top 时如果目标"
            "本身（比如瓶子）比这个值高，会把目标自己的顶部误判成货架面——"
            "把这个调大到目标物体高度以上再重测"
        ),
    )
    parser.add_argument(
        "--max-gap-m",
        type=float,
        default=None,
        help=(
            "覆盖 shelf_fit_max_gap_m（默认0.35）。测左右侧板时如果格位实际"
            "宽度超过这个值的两倍，两侧板会搜不到真正的板子而是找到更近的"
            "别的东西——把这个调大到格位实际半宽以上再重测"
        ),
    )
    parser.add_argument("--port", type=int, default=8879)
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "bottle_grasp")
    )
    cli = parser.parse_args()

    faces = [item.strip() for item in cli.faces.split(",") if item.strip()]
    unknown = [face for face in faces if face not in FACE_SPECS]
    if unknown:
        raise SystemExit(
            f"未知的货架面: {unknown}；可选: {sorted(FACE_SPECS)}"
        )

    Path(cli.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[])
    demo = BottleDemo(build_args(cli), load_config(cli.config))

    try:
        demo._ensure_head_reference()
        demo._start_camera("head")
        results = run_shelf_survey(
            demo,
            faces,
            cli.target_base,
            min_gap_m=cli.min_gap_m,
            max_gap_m=cli.max_gap_m,
        )
    finally:
        demo.close()

    draft_path = demo.run_dir / "shelf_survey_draft.json"
    draft_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    for face in faces:
        fit = results[face]["fit"]
        box = results[face]["suggested_box"]
        print(f"{face}: 实测坐标={fit['plane_m']:.4f} 内点数={fit['inliers']}")
        print(f"  草稿 box: min={box['min']} max={box['max']}")
    print(
        f"\n完整草稿已写入 {draft_path}——这只是草稿，人工核对/修正后再手动合并"
        "进 safety_profiles.json 的 keepout_boxes，且不要自动把"
        "verified_for_execution 改成 true。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
