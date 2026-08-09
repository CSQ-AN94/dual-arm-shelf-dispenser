#!/usr/bin/env python3
"""Record a hand-taught grasp, and say how it differs from the computed one.

Thirteen consecutive planning attempts have failed at the same place: the
gripper cannot reach the grasp point the pipeline computes without the model
putting it inside the shelf's bottom panel.  Arguing about
``grasp_height_fraction`` from first principles has not settled it, and the
shelf's own geometry was wrong by 33 mm until an hour ago.

A demonstration replaces the argument.  Put the gripper where it would actually
close on the bottle, and this records where that is -- then compares it against
what the perception pipeline computed for the same bottle in the same run.  The
difference is the answer: if the taught grasp is higher up the bottle, the
fraction is wrong; if it is at the same height but further out, the approach
depth is; if it is at the same place and the planner still refuses, the gripper
does not fit and no parameter will fix that.

Nothing moves.  The arm is read, never commanded.

    python scripts/record_demonstrated_grasp.py
    python scripts/record_demonstrated_grasp.py --scenario /home/rm/cycle_claude_02/p1.yaml
    python scripts/record_demonstrated_grasp.py --write     # 写入 profile
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.arm import RobotSession
from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config

LOG = logging.getLogger("record_demonstrated_grasp")

# A pose read while the arm is still settling is worth nothing, and the taught
# left-arm pose that cost two days was recorded exactly that way.
MAX_DRIFT_DEG = 0.5
SAMPLES = 7
SAMPLE_GAP_S = 1.5


def read_settled(robot) -> list[float]:
    readings = []
    for index in range(SAMPLES):
        readings.append(list(robot.joints_deg()))
        if index < SAMPLES - 1:
            time.sleep(SAMPLE_GAP_S)
    values = np.asarray(readings, dtype=float)
    drift = float(np.max(values.max(axis=0) - values.min(axis=0)))
    if drift > MAX_DRIFT_DEG:
        raise SafetyAbort(
            f"右臂还在动（{SAMPLES} 次读数波动 {drift:.2f}° > {MAX_DRIFT_DEG}°）；"
            "松手等它停稳再记录"
        )
    LOG.info("%d 次读数波动 %.3f°，已稳定", SAMPLES, drift)
    return [float(v) for v in values.mean(axis=0)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        help="A pick scenario YAML from the same bottle, to compare against",
    )
    parser.add_argument("--write", action="store_true")
    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    link7_to_flange, flange_to_tcp = (
        profile.tool_mount_calibration.require_transforms()
    )
    robot = RobotSession(
        cfg.connections.right_arm_ip,
        cfg.connections.arm_port,
        threading.Event(),
        params.tcp_z_m,
        params.moveit_link7_to_controller_flange_m,
        take_control=False,
        tcp_transform=flange_to_tcp,
        link7_to_controller_flange=link7_to_flange,
    )
    try:
        joints = read_settled(robot)
        tcp = np.asarray(robot.tcp_from_joints(joints), dtype=float)
    finally:
        robot.close()

    position = tcp[:3, 3]
    print("\n示教抓取位姿")
    print("  关节 " + " ".join(f"{v:8.3f}" for v in joints))
    print(f"  TCP（围栏系） {np.round(position, 4).tolist()}")

    shelf_bottom = next(
        (box for box in profile.keepout_boxes if box.id == "shelf_bottom"), None
    )
    if shelf_bottom is not None:
        print(
            f"  离底板顶面 {position[2] - shelf_bottom.maximum[2]:.4f} m"
            f"（底板 z={shelf_bottom.maximum[2]}）"
        )

    if cli.scenario and cli.scenario.exists():
        scenario = yaml.safe_load(cli.scenario.read_text(encoding="utf-8"))
        computed_moveit = np.asarray(
            scenario["source_grasp_candidates"][0]["pose"]["xyz"], dtype=float
        )
        bridge = np.asarray(profile.T_moveit_from_profile, dtype=float)
        computed = (
            np.linalg.inv(bridge) @ np.append(computed_moveit, 1.0)
        )[:3]
        delta = position - computed
        print("\n对比流水线算出来的抓取点")
        print(f"  算出来的 TCP（围栏系） {np.round(computed, 4).tolist()}")
        print(f"  示教 − 计算            {np.round(delta, 4).tolist()} m")
        print(
            f"    横向 {delta[0] * 1000:+7.1f} mm   "
            f"进深 {delta[1] * 1000:+7.1f} mm   "
            f"高度 {delta[2] * 1000:+7.1f} mm"
        )
        bottle = scenario.get("bottle") or {}
        height = bottle.get("height_m")
        if height and shelf_bottom is not None:
            base = shelf_bottom.maximum[2]
            taught_fraction = (position[2] - base) / float(height)
            computed_fraction = (computed[2] - base) / float(height)
            print(
                f"\n  瓶高 {height} m，从瓶底往上算："
                f"示教 {taught_fraction:.0%}，流水线 {computed_fraction:.0%}"
            )
            print(
                "  （grasp_height_fraction 是从瓶顶往下量的，"
                f"所以对应值约 {1 - taught_fraction:.2f}）"
            )

    if not cli.write:
        print("\n未写入。加 --write 把关节角存进 profile 的 "
              "demonstrated_grasp_right_joints_deg。")
        return 0

    path = Path(cli.safety_config)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["profiles"]["shelf_template"]["demonstrated_grasp_right_joints_deg"] = [
        round(v, 6) for v in joints
    ]
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n已写入 {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
