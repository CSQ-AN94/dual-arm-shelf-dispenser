#!/usr/bin/env python3
"""Drive the arm along a hand-taught trajectory so a person can watch it.

The demonstrations are evidence, not plans: they were recorded by dragging the
arm, never validated, and 2026-08-06 measured that the current collision model
flags 25 of 36 states on one of them as colliding while the operator watched
that same path touch nothing.  Replaying one is therefore an operator decision,
taken with a hand on the e-stop, and this script exists to make that decision
as informed as it can be rather than to make it routine.

What it checks before moving, in this order:

  * every sampled TCP against the live electronic fence, using the same
    ``assert_tcp_point`` call the recorder used -- so a fence that has changed
    since the recording is caught here, not by the arm;
  * the live arm against the demonstration's own first sample, because a
    trajectory replayed from somewhere else is a different trajectory;
  * the waypoint count against the controller's connected-command queue.

What it does NOT check: the MoveIt collision scene.  That is the honest gap.
The fence knows the shelf panels and the workspace box; it does not know a
bottle somebody moved into the path since the demonstration was recorded.

Defaults to a dry run.  ``--execute`` moves the arm; ``--speed`` defaults low
and ``--stop-at-fraction`` can replay only the opening part of a path.

    python3 scripts/replay_demonstrated_trajectory.py \\
        outputs/demonstrated_trajectories/right_lower_place_retry_20260806.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.live_arm import open_live_arm
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config

# Mirrors the recorder's own idea of "not moving yet".
STATIC_STEP_DEG = 0.05
# EXECUTION_CONTROLLER_MAX_COMMANDS in the MTC planner, less its own margin.
CONTROLLER_QUEUE_LIMIT = 29
# CONNECTED_TRAJECTORY_MAX_STEP_DEG in arm.py; kept a hair under it.
SEGMENT_LIMIT_DEG = 14.0


def trim_static_ends(samples: list[dict]) -> tuple[int, int]:
    joints = [s["joints_deg"] for s in samples]
    low = 0
    while low + 1 < len(joints) and _max_delta(joints[low], joints[low + 1]) < STATIC_STEP_DEG:
        low += 1
    high = len(joints) - 1
    while high > low + 1 and _max_delta(joints[high], joints[high - 1]) < STATIC_STEP_DEG:
        high -= 1
    return low, high


def _max_delta(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def simplify(path: list[list[float]], tolerance_deg: float) -> list[int]:
    """Douglas-Peucker in joint space: the fewest corners within tolerance."""
    keep = {0, len(path) - 1}
    stack = [(0, len(path) - 1)]
    while stack:
        low, high = stack.pop()
        if high <= low + 1:
            continue
        direction = [path[high][j] - path[low][j] for j in range(7)]
        squared = sum(value * value for value in direction)
        worst, worst_index = -1.0, -1
        for middle in range(low + 1, high):
            if squared <= 1e-12:
                error = _max_delta(path[middle], path[low])
            else:
                fraction = sum(
                    (path[middle][j] - path[low][j]) * direction[j] for j in range(7)
                ) / squared
                fraction = min(1.0, max(0.0, fraction))
                error = max(
                    abs(path[middle][j] - (path[low][j] + fraction * direction[j]))
                    for j in range(7)
                )
            if error > worst:
                worst, worst_index = error, middle
        if worst > tolerance_deg:
            keep.add(worst_index)
            stack.append((low, worst_index))
            stack.append((worst_index, high))
    return sorted(keep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("demonstration", type=Path)
    parser.add_argument("--arm", default="right_arm")
    parser.add_argument("--profile", default="shelf_template")
    parser.add_argument("--tolerance-deg", type=float, default=1.0)
    parser.add_argument("--speed", type=int, default=15)
    parser.add_argument("--max-step-deg", type=float, default=2.0)
    parser.add_argument(
        "--stop-at-fraction",
        type=float,
        default=1.0,
        help="只回放前一段，1.0 是全程",
    )
    parser.add_argument("--start-tolerance-deg", type=float, default=2.0)
    parser.add_argument(
        "--reverse",
        action="store_true",
        help=(
            "沿示教路径倒回起点。退出货架用这个，而不是重新规划一条："
            "这条路刚刚在同一个场景里物理走通过"
        ),
    )
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument("--execute", action="store_true")
    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if not 0.0 < cli.stop_at_fraction <= 1.0:
        raise SafetyAbort("--stop-at-fraction 必须在 (0, 1]")
    if not 1 <= cli.speed <= 100:
        raise SafetyAbort("--speed 必须在 1..100")

    record = json.loads(cli.demonstration.read_text(encoding="utf-8"))
    samples = record["samples"]
    low, high = trim_static_ends(samples)
    high = low + max(1, int(round((high - low) * cli.stop_at_fraction)))
    segment = samples[low : high + 1]
    path = [list(map(float, s["joints_deg"])) for s in segment]
    if cli.reverse:
        path.reverse()
    print(f"  方向: {'倒回起点' if cli.reverse else '正向'}")
    print(
        f"示教 {record.get('label')} / {record.get('arm_id')}，"
        f"升降 {record.get('lift_height_mm')} mm，"
        f"原始 {len(samples)} 点，有效段 {low}..{high}（{len(path)} 点）"
    )
    print(f"  录制时的围栏违规: {len(record.get('fence_violations') or [])}")

    corners = simplify(path, cli.tolerance_deg)
    simplified = [path[i] for i in corners]
    # Douglas-Peucker bounds how far the polyline strays, not how long a
    # segment gets, and the controller refuses a commanded step over 15 deg on
    # any joint.  Subdividing on the straight line between two corners adds
    # queue slots and no deviation.
    waypoints = [simplified[0]]
    for point in simplified[1:]:
        previous = waypoints[-1]
        steps = max(1, math.ceil(_max_delta(previous, point) / SEGMENT_LIMIT_DEG - 1e-9))
        for step in range(1, steps + 1):
            fraction = step / steps
            waypoints.append(
                [previous[j] + fraction * (point[j] - previous[j]) for j in range(7)]
            )
    print(
        f"  按 {cli.tolerance_deg:g}° 简化: {len(simplified)} 个拐点，"
        f"拆分 15° 单段上限后 {len(waypoints)} 个路点"
        f"（控制器队列上限 {CONTROLLER_QUEUE_LIMIT}）"
    )
    if len(waypoints) > CONTROLLER_QUEUE_LIMIT:
        raise SafetyAbort(
            f"简化后仍有 {len(waypoints)} 个路点，超出控制器连续指令队列；"
            "调大 --tolerance-deg 或用 --stop-at-fraction 分段回放"
        )

    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, cli.profile, require_verified=False
    )
    robot, view = open_live_arm(
        cfg, params, profile, cli.arm, take_control=cli.execute
    )
    try:
        # Re-check the fence now, against today's profile, on every sample --
        # not only the corners the arm will be commanded through.
        violations = []
        for index, joints in enumerate(path):
            tcp = np.asarray(robot.tcp_from_joints(joints), dtype=float)
            try:
                view.assert_tcp_point(tcp[:3, 3], label="示教回放 TCP")
            except SafetyAbort as exc:
                violations.append((index, str(exc)))
        if violations:
            print(f"  围栏复核: {len(violations)} 个点越界，首个 {violations[0]}")
            raise SafetyAbort(
                f"示教轨迹在当前围栏下有 {len(violations)} 个越界点，拒绝回放"
            )
        print(f"  围栏复核: {len(path)} 个点全部在界内")

        live = np.asarray(robot.joints_deg(), dtype=float)
        start_error = float(np.max(np.abs(live - np.asarray(path[0], dtype=float))))
        print(f"  实机当前   {[round(v, 2) for v in live.tolist()]}")
        print(f"  示教起点   {[round(v, 2) for v in path[0]]}")
        print(f"  起点最大关节差 {start_error:.2f}°（上限 {cli.start_tolerance_deg:g}°）")
        if start_error > cli.start_tolerance_deg:
            raise SafetyAbort(
                "实机不在示教起点，拒绝回放：先把臂归到该位姿，"
                "或用别的示教"
            )

        if not cli.execute:
            print("\n干跑：未下发任何运动。加 --execute 才会移动。")
            return 0

        print(f"\n开始回放，速度 {cli.speed}%，人守急停。")
        robot.execute_planned_joints(
            waypoints,
            cli.speed,
            cli.max_step_deg,
            expected_start_joints_deg=path[0],
            start_tolerance_deg=cli.start_tolerance_deg,
        )
        final = np.asarray(robot.joints_deg(), dtype=float)
        print(f"  回放结束，终点最大偏差 {float(np.max(np.abs(final - np.asarray(path[-1])))):.2f}°")
    finally:
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
