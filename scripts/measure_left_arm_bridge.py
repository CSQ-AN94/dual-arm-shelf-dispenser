#!/usr/bin/env python3
"""Measure the left arm's bridge from its controller base frame to MoveIt.

The profile carries one for the right arm.  Using it for the left arm put
MoveIt ``l_link7`` and the SDK's left flange 127.2 mm and 179.9 degrees apart
at the same joint state on 2026-08-04, which is why left-arm planning is shut.

The two sides are both forward kinematics, so this needs no motion at all:

    Bridge @ sdk_link7(q) == moveit_l_link7(q)

evaluated at many q.  Each q gives one estimate; a constant bridge means they
all agree, and the spread between them is the honest error bar.  A spread that
does not close is itself the finding -- it would mean the two kinematics
disagree about something other than a fixed frame, most likely joint sign or
order, and no single transform would fix that.

    python scripts/measure_left_arm_bridge.py              # 只测量并报告

Use ``solve_left_arm_model.py --write`` to solve and persist the complete
bridge/tool model.  A bridge-only result is diagnostic evidence, not a model
that may be written into the execution profile.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.left_arm import open_left_arm
from shelf_dispenser.planner import MoveItPlanner
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config

LOG = logging.getLogger("measure_left_arm_bridge")

# Admission limits for the result.  Deliberately tighter than the runtime FK
# contract: a bridge that only just passes the gate it feeds is not a bridge.
MAX_POSITION_SPREAD_M = 0.005
MAX_ORIENTATION_SPREAD_DEG = 0.5
MIN_KINEMATIC_SAMPLES = 12


def sample_joint_states(seed: int, count: int) -> list[list[float]]:
    """Joint states spread over a safe interior band of each joint's range.

    Nothing moves, so these need only be kinematically meaningful, not
    reachable-with-the-shelf-there.  Spread matters: poses clustered together
    would make a wrong bridge look consistent.
    """
    if isinstance(count, bool) or int(count) < MIN_KINEMATIC_SAMPLES:
        raise SafetyAbort(
            f"左臂运动学标定至少需要 {MIN_KINEMATIC_SAMPLES} 个分散样本"
        )
    rng = np.random.default_rng(seed)
    # Conservative interior of an RM75's travel, in degrees.
    spans = [(-150, 150), (-120, 120), (-150, 150), (-120, 120), (-150, 150),
             (-120, 120), (-150, 150)]
    states = [[0.0] * 7]
    for _ in range(count - 1):
        states.append(
            [float(rng.uniform(low, high)) for low, high in spans]
        )
    return states


def moveit_link7_poses(
    moveit: MoveItPlanner, states: list[list[float]], planning_frame: str
) -> list[np.ndarray]:
    request = moveit.run_dir / "left_bridge_fk_request.json"
    output = moveit.run_dir / "left_bridge_fk.json"
    request.write_text(
        json.dumps(
            {
                "planning_group": "left_arm",
                "joint_states_deg": states,
                "planning_frame": planning_frame,
            }
        ),
        encoding="utf-8",
    )
    helper = ROOT / "shelf_dispenser" / "ros" / "link7_fk.py"
    result = subprocess.run(
        [
            "bash",
            "-lc",
            moveit.ros_prefix + f"exec python3 '{helper}' '{request}' '{output}'",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if not output.exists():
        raise SafetyAbort(
            f"MoveIt 左臂 FK 失败: rc={result.returncode}; {result.stderr[-600:]}"
        )
    poses = json.loads(output.read_text(encoding="utf-8"))["poses"]
    out = []
    for pose in poses:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(pose["quaternion_xyzw"]).as_matrix()
        matrix[:3, 3] = pose["position"]
        out.append(matrix)
    return out


def average_transform(transforms: list[np.ndarray]) -> np.ndarray:
    """Mean of rigid transforms: translations averaged, rotations via SVD."""
    stacked = np.stack([t[:3, :3] for t in transforms])
    u, _, vt = np.linalg.svd(stacked.mean(axis=0))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:  # keep it proper
        u[:, -1] *= -1
        rotation = u @ vt
    mean = np.eye(4)
    mean[:3, :3] = rotation
    mean[:3, 3] = np.mean([t[:3, 3] for t in transforms], axis=0)
    return mean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "left_arm_bridge")
    )
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    states = sample_joint_states(cli.seed, cli.samples)
    run_dir = Path(cli.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    left_tool = profile.left_tool_mount_calibration
    if left_tool is None:
        raise SafetyAbort("profile 缺少 left_tool_mount_calibration")
    link7_to_flange, _ = left_tool.require_transforms()

    left = open_left_arm(cfg, params, profile, take_control=False)
    moveit = MoveItPlanner(project_root=ROOT, run_dir=run_dir)
    try:
        moveit.start()
        moveit_poses = moveit_link7_poses(moveit, states, profile.moveit_frame)
        estimates = []
        for values, moveit_link7 in zip(states, moveit_poses):
            flange = np.asarray(
                left.controller_flange_from_joints(values), dtype=float
            )
            sdk_link7 = flange @ np.linalg.inv(link7_to_flange)
            estimates.append(moveit_link7 @ np.linalg.inv(sdk_link7))
    finally:
        moveit.close()
        left.close()

    bridge = average_transform(estimates)
    positions = np.array([e[:3, 3] for e in estimates])
    position_spread = float(np.max(np.linalg.norm(positions - bridge[:3, 3], axis=1)))
    angles = [
        float(
            np.degrees(
                Rotation.from_matrix(bridge[:3, :3].T @ e[:3, :3]).magnitude()
            )
        )
        for e in estimates
    ]
    orientation_spread = float(np.max(angles))

    print(f"样本 {len(estimates)} 个关节状态（无运动，纯正运动学）\n")
    print("解出的左臂坐标桥 T_moveit_from_left_profile：")
    for row in bridge:
        print("   " + "  ".join(f"{v: .9f}" for v in row))
    print(f"\n一致性：位置离散 {position_spread * 1000:7.3f} mm "
          f"（上限 {MAX_POSITION_SPREAD_M * 1000:.1f}）")
    print(f"          姿态离散 {orientation_spread:7.3f}°   "
          f"（上限 {MAX_ORIENTATION_SPREAD_DEG:.1f}）")

    passed = (
        position_spread <= MAX_POSITION_SPREAD_M
        and orientation_spread <= MAX_ORIENTATION_SPREAD_DEG
    )
    if not passed:
        print(
            "\n✗ 不是一个固定变换。两套运动学的分歧不止一个常量坐标系——"
            "多半是关节符号或顺序不一致，再多样本也解不出来。"
        )
        return 1
    print("\n✓ 各样本一致，是一个固定刚体变换。")
    print("该工具只报告桥接诊断；完整模型写入请使用 solve_left_arm_model.py --write。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
