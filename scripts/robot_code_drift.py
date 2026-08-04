#!/usr/bin/env python3
"""Compare the robot's deployed copy against this working tree, file by file.

There are twenty-odd `Grabber__*` directories on the robot and no git metadata
in any of them, so "which code is actually running" has never been answerable
by looking.  Every wrong answer so far has looked like a robot problem: a fence
that was already widened here still rejecting there, an arm driven to a pose
this tree stopped using.

This hashes the files that decide what the arm does and says which ones differ.
``--push`` copies the differing ones over and re-verifies; without it nothing
is written anywhere.

    python scripts/robot_code_drift.py
    python scripts/robot_code_drift.py --push
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "rm@192.168.3.68"
DEFAULT_REMOTE = "/home/rm/dual-arm-shelf-dispenser"

# Everything that changes where the arm goes or whether it is allowed to.
# Deliberately not the whole tree: a drifting README is not worth an alarm that
# people learn to ignore.
TRACKED = [
    "shelf_dispenser/core.py",
    "shelf_dispenser/orchestrator.py",
    "shelf_dispenser/grasp_orientation.py",
    "shelf_dispenser/mtc_execution.py",
    "shelf_dispenser/mtc_pick_contract.py",
    "shelf_dispenser/planner.py",
    "shelf_dispenser/ros/plan_once.py",
    "shelf_dispenser/ros/scene_helpers.py",
    "shelf_dispenser/ros/validate_path.py",
    "shelf_dispenser/arm.py",
    "shelf_dispenser/safe_planner.py",
    "shelf_dispenser/safety.py",
    "shelf_dispenser/safety_profiles.json",
    "shelf_dispenser/scene.py",
    "scripts/calibrate_mtc_gripper.py",
    "scripts/capture_empty_shelf_places.py",
    "scripts/capture_mtc_direct_pick_scene.py",
    "scripts/empty_shelf_places_to_mtc_scenario.py",
    "scripts/execute_mtc_lift_transfer.py",
    "scripts/execute_mtc_trajectory.py",
    "scripts/localization_to_mtc_scenario.py",
    "shelf_dispenser/arm_worker.py",
    "shelf_dispenser/left_arm.py",
    "scripts/normalize_left_arm.py",
    "scripts/normalize_to_grasp_start.py",
    "scripts/measure_left_arm_bridge.py",
    "scripts/solve_left_arm_model.py",
    "scripts/run_cross_layer_cycle.sh",
]


def local_digests() -> dict[str, str]:
    out = {}
    for rel in TRACKED:
        path = ROOT / rel
        if not path.is_file():
            raise SystemExit(f"本地缺文件: {rel}")
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def remote_digests(host: str, remote: str) -> dict[str, str]:
    listing = " ".join(f"{remote}/{rel}" for rel in TRACKED)
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=8", host, f"sha256sum {listing} 2>&1"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 and not result.stdout.strip():
        raise SystemExit(f"连不上机器人: {result.stderr.strip()}")
    out = {}
    for line in result.stdout.splitlines():
        if ": No such file" in line or "无法" in line:
            missing = line.split(":")[0].split("/")[-1]
            out[missing] = "缺失"
            continue
        parts = line.split()
        if len(parts) == 2:
            out[parts[1][len(remote) + 1 :]] = parts[0]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument(
        "--push",
        action="store_true",
        help="Copy the differing files over and re-verify",
    )
    cli = parser.parse_args()

    local = local_digests()
    remote = remote_digests(cli.host, cli.remote)
    drift = [
        rel
        for rel in TRACKED
        if remote.get(rel, "缺失") != local[rel]
    ]

    print(f"本地 {ROOT}")
    print(f"机器人 {cli.host}:{cli.remote}\n")
    if not drift:
        print(f"✓ {len(TRACKED)} 个关键文件全部一致")
        return 0
    print(f"✗ {len(drift)}/{len(TRACKED)} 个文件不一致：")
    for rel in drift:
        state = "机器人上没有" if remote.get(rel) == "缺失" else "内容不同"
        print(f"    {rel:52s} {state}")
    if not cli.push:
        print("\n加 --push 同步这些文件。未同步前跑真机 = 跑的不是你看的代码。")
        return 1

    print("\n同步中……")
    for rel in drift:
        target = f"{cli.host}:{cli.remote}/{rel}"
        result = subprocess.run(
            ["scp", "-q", str(ROOT / rel), target], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise SystemExit(f"{rel} 同步失败: {result.stderr.strip()}")
    after = remote_digests(cli.host, cli.remote)
    still = [rel for rel in TRACKED if after.get(rel, "缺失") != local[rel]]
    if still:
        print(f"✗ 同步后仍不一致: {still}")
        return 1
    print(f"✓ {len(drift)} 个文件已同步并复验一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
