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
#
# But an entry point's imports are not optional extras -- they are the entry
# point.  This list was hand-kept and had fallen 18 modules behind, so the tool
# printed "✓ 已同步" while perception.py on the robot still carried the alias
# set from before the trained model existed: no p01..p06, and no case folding
# on the allowed set.  Pushing the new six-class model against that copy would
# have matched nothing at all, and the cycle would have failed six captures in
# a row on localization, looking nothing like a sync problem.
#
# test_robot_drift_tracks_every_reachable_runtime_module recomputes the import
# closure and fails when this list falls behind again.  Non-Python assets still
# have to be added by hand; nothing can derive those.
TRACKED = [
    "config.yaml",
    "shelf_dispenser/model_assets.lock.json",
    "intelligence/yolo_models/mixed_shelf_yolo26s_all51.pt",
    "shelf_dispenser/core.py",
    "shelf_dispenser/orchestrator.py",
    "shelf_dispenser/grasp_orientation.py",
    "shelf_dispenser/mtc_execution.py",
    "shelf_dispenser/mtc_pick_contract.py",
    "mtc_ws/src/grabber_mtc_planner/src/plan_shelf_transfer.cpp",
    "mtc_ws/src/grabber_mtc_planner/src/scenario.cpp",
    "mtc_ws/src/grabber_mtc_planner/src/scenario.hpp",
    "mtc_ws/src/grabber_mtc_planner/scenarios/shelf_transfer_fixture.yaml",
    "mtc_ws/src/grabber_mtc_planner/launch/plan_shelf_transfer_experimental.launch.py",
    "mtc_ws/src/grabber_mtc_planner/package.xml",
    "mtc_ws/src/grabber_robot_state_bridge/grabber_robot_state_bridge/robot_description.py",
    "mtc_ws/src/grabber_robot_state_bridge/launch/live_state_plan_only.launch.py",
    "shelf_dispenser/planner.py",
    "shelf_dispenser/ros/plan_once.py",
    "shelf_dispenser/ros/scene_helpers.py",
    "shelf_dispenser/ros/validate_path.py",
    "shelf_dispenser/arm.py",
    "shelf_dispenser/safe_planner.py",
    "shelf_dispenser/safety.py",
    "shelf_dispenser/safety_profiles.json",
    "shelf_dispenser/scene.py",
    "shelf_dispenser/shelf_model.py",
    "shelf_dispenser/grasp_ledger.py",
    "shelf_dispenser/inventory.py",
    "shelf_dispenser/relative_place.py",
    "utils/items_info.py",
    "scripts/calibrate_mtc_gripper.py",
    "scripts/capture_empty_shelf_places.py",
    "scripts/capture_mtc_direct_pick_scene.py",
    "scripts/_patch_after_raise.py",
    "scripts/empty_shelf_places_to_mtc_scenario.py",
    "scripts/execute_mtc_lift_transfer.py",
    "scripts/execute_mtc_trajectory.py",
    "scripts/localization_to_mtc_scenario.py",
    "scripts/diagnose_gripper_voxel_contact.py",
    "shelf_dispenser/arm_worker.py",
    "shelf_dispenser/left_arm.py",
    "scripts/normalize_left_arm.py",
    "scripts/normalize_right_arm.py",
    "scripts/shelf_row_template_tool.py",
    "scripts/apply_demonstrated_grasp_to_scenario.py",
    "scripts/gripper_grasp_test.py",
    "scripts/replay_demonstrated_trajectory.py",
    "shelf_dispenser/live_arm.py",
    "scripts/normalize_to_grasp_start.py",
    "scripts/measure_left_arm_bridge.py",
    "scripts/solve_left_arm_model.py",
    "scripts/run_cross_layer_cycle.sh",
    "outputs/row_templates.json",
    "sensors/camera_thread.py",
    "shelf_dispenser/camera_access.py",
    "shelf_dispenser/collision.py",
    "shelf_dispenser/console.py",
    "shelf_dispenser/dashboard.py",
    "shelf_dispenser/delivery_table.py",
    "shelf_dispenser/environment_guard.py",
    "shelf_dispenser/head_lock.py",
    "shelf_dispenser/lift_evidence.py",
    "shelf_dispenser/mobile_body.py",
    "shelf_dispenser/model_assets.py",
    "shelf_dispenser/perception.py",
    "shelf_dispenser/ros/scene_ids.py",
    "shelf_dispenser/run_manifest.py",
    "shelf_dispenser/table_model.py",
    "shelf_dispenser/target_guard.py",
    "shelf_dispenser/workflows.py",
    "utils/config.py",
    "outputs/demonstrated_trajectories/place_A1_left_20260808.json",
    "outputs/demonstrated_trajectories/place_A2_left_20260808.json",
    "outputs/demonstrated_trajectories/place_A3_right_20260808.json",
    "outputs/demonstrated_trajectories/place_A4_right_20260808.json",
    "outputs/demonstrated_trajectories/place_B1_left_20260808.json",
    "outputs/demonstrated_trajectories/place_B2_left_20260808.json",
    "outputs/demonstrated_trajectories/place_B3_right_20260808.json",
    "outputs/demonstrated_trajectories/place_B4_right_20260808.json",
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
    remote_hashes = remote_digests(cli.host, cli.remote)
    drift = [
        rel
        for rel in TRACKED
        if remote_hashes.get(rel, "缺失") != local[rel]
    ]

    print(f"本地 {ROOT}")
    print(f"机器人 {cli.host}:{cli.remote}\n")
    if not drift:
        print(f"✓ {len(TRACKED)} 个关键文件全部一致")
        return 0
    print(f"✗ {len(drift)}/{len(TRACKED)} 个文件不一致：")
    for rel in drift:
        state = (
            "机器人上没有"
            if remote_hashes.get(rel) == "缺失"
            else "内容不同"
        )
        print(f"    {rel:52s} {state}")
    if not cli.push:
        print("\n加 --push 同步这些文件。未同步前跑真机 = 跑的不是你看的代码。")
        return 1

    print("\n同步中……")
    for rel in drift:
        parent = str(Path(cli.remote, rel).parent)
        mkdir = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=8", cli.host, "mkdir", "-p", parent],
            capture_output=True,
            text=True,
        )
        if mkdir.returncode != 0:
            raise SystemExit(f"{rel} 远端目录创建失败: {mkdir.stderr.strip()}")
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
