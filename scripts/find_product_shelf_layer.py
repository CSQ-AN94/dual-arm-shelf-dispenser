#!/usr/bin/env python3
"""Find which shelf layer holds a product, using the lift and the head camera.

Runs on the robot, before the dispense task starts.  Reports the selected lift
height on a marked stdout line, the same way ArmJointReader hands a subprocess
result back:

    SHELF_LAYER_LIFT_MM=250

Marked rather than bare, because stdout is not ours alone: CameraThread prints
its RealSense banner with plain print(), so a caller capturing all of stdout
gets the banner concatenated with the number.  Grep the marker:

    LIFT_MM=$(python3 scripts/find_product_shelf_layer.py --target-product P01 \
              | sed -n 's/^SHELF_LAYER_LIFT_MM=//p')

Why this is a separate entry point and not a step inside RunOrchestrator: the
dispense flow captures its immutable SHELF_READY snapshot *before*
``initialize()``, so that no arm or gripper command can precede it.  The
detector only exists after ``initialize()``.  Searching therefore cannot
happen inside that window without reordering a function whose ordering is the
safety property.  Moving the lift is body-only work -- the same thing
run_cross_layer_cycle.sh already does before its own pick -- so the search
belongs out here, ahead of the task.

Exit codes
  0   found; the selected lift height is on stdout
  3   the product is on none of the searched layers (an empty shelf, not a
      fault) -- distinct so a caller can tell "restock it" from "fix it"
  1/2 anything else: camera, depth, lift or profile failure.  Never treated
      as an empty layer, because a broken camera would otherwise send the
      platform hunting through every height and then blame the shelf.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shelf_dispenser.core import BottleDetectionLost, SafetyAbort  # noqa: E402
from shelf_dispenser.mobile_body import LiftSocketAdapter  # noqa: E402
from shelf_dispenser.orchestrator import RunOrchestrator  # noqa: E402
from shelf_dispenser.safety import load_safety_profile  # noqa: E402
from utils.config import load_config  # noqa: E402
from run_pick_place_task import build_parser as build_task_parser  # noqa: E402

LOG = logging.getLogger("find_product_shelf_layer")

# stdout is shared with CameraThread's plain print() banner, so the result
# needs a marker the caller can grep for.  Same contract as ArmJointReader's
# BOTTLE_JOINTS_JSON=.
RESULT_MARKER = "SHELF_LAYER_LIFT_MM="


def _search_heights(safety_config: str, delivery_profile: str) -> tuple[int, ...]:
    profile = load_safety_profile(
        safety_config, delivery_profile, require_verified=True
    )
    if profile.side_table_delivery is None:
        raise SafetyAbort(
            f"profile {delivery_profile} 缺少 side_table_delivery"
        )
    return tuple(profile.side_table_delivery.search_lift_heights_mm)


def _move_empty_lift(cfg, height_mm: int) -> None:
    lift = LiftSocketAdapter(cfg.connections.left_arm_ip, cfg.connections.arm_port)
    state = lift.state()
    if abs(int(state.height_mm) - height_mm) <= 5:
        LOG.info("升降已在 %d mm，跳过移动", height_mm)
        return
    LOG.info("空载升降 %d -> %d mm", int(state.height_mm), height_mm)
    after = lift.move_to(height_mm, speed=30)
    if abs(int(after.height_mm) - height_mm) > 5:
        raise SafetyAbort(
            f"升降未到位: target={height_mm} actual={after.height_mm}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument("--safety-profile", default="shelf_template")
    parser.add_argument("--delivery-safety-profile", default="side_table_template")
    parser.add_argument("--target-product", required=True)
    parser.add_argument("--port", type=int, default=8882)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "layer_search"))
    cli = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    heights = _search_heights(cli.safety_config, cli.delivery_safety_profile)
    LOG.info("将按顺序搜索升降高度: %s", list(heights))
    cfg = load_config(cli.config)

    for height in heights:
        _move_empty_lift(cfg, height)
        demo_args = build_task_parser().parse_args(
            [
                "--config", cli.config,
                "--safety-config", cli.safety_config,
                "--safety-profile", cli.safety_profile,
                "--port", str(cli.port),
                "--output-dir", cli.output_dir,
            ]
        )
        demo_args.target_product = cli.target_product
        # Camera-and-detector only: this entry never opens an arm session.
        demo_args.plan_only = False
        demo = RunOrchestrator(demo_args, cfg)
        try:
            demo.initialize()
            demo._fresh_head_target()
        except BottleDetectionLost as exc:
            LOG.info("升降 %d mm 这一层没有 %s: %s", height, cli.target_product, exc)
            continue
        finally:
            demo.close()
        LOG.info("在升降 %d mm 找到 %s", height, cli.target_product)
        print(f"{RESULT_MARKER}{height}", flush=True)
        return 0

    LOG.error(
        "搜索过的每一层都没有 %s: %s", cli.target_product, list(heights)
    )
    return 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        LOG.error("层间搜索中止: %s", exc)
        raise SystemExit(1)
