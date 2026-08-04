#!/usr/bin/env python3
"""Solve the left arm's SDK-to-MoveIt correspondence, including its tool offset.

Three findings, in the order they arrived:

  * A single constant transform does not relate the two kinematics: solving for
    one across twelve joint states left an 889 mm spread.
  * Flipping the sign of joints 2, 4 and 6 collapses the orientation spread to
    0.11 deg.  Alternating signs are the usual convention for a mirrored arm,
    and the URDF agrees: r_joint1 carries a 3.1415 yaw that l_joint1 does not.
  * Position still spread by 36 mm, because l_joint7 carries a 3.14 the right
    arm has at joint1 instead.  A constant on the tool side multiplies from the
    right, and no left-multiplied bridge can absorb it.

So the model is ``moveit(q) = Bridge @ sdk(q_mapped) @ Tool``.  With the
orientation already constant, Tool's rotation is identity to within the noise
and only its translation is unknown, which makes the whole thing linear:

    moveit_p(q) = Bridge_R @ (sdk_R(q) @ t + sdk_p(q)) + Bridge_t

Least squares over the samples gives ``t`` and ``Bridge_t`` together.  If the
residual does not close, the extra structure is somewhere else and this says so
rather than reporting a fit nobody should use.

Commands nothing; both sides are forward kinematics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from measure_left_arm_bridge import (  # noqa: E402  (same directory)
    average_transform,
    moveit_link7_poses,
    sample_joint_states,
)
from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.left_arm import open_left_arm
from shelf_dispenser.planner import MoveItPlanner
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config

LOG = logging.getLogger("solve_left_arm_model")

# Signs that took the orientation spread from 178 deg to 0.11 deg.
JOINT_SIGNS = (1, -1, 1, -1, 1, -1, 1)
MAX_RESIDUAL_M = 0.003
MAX_ORIENTATION_RESIDUAL_DEG = 0.5


def solve(sdk: list[np.ndarray], moveit: list[np.ndarray]):
    """Return (bridge, tool_translation, position_residual, angle_residual)."""
    # Rotation first: it does not involve the tool translation at all.
    bridge_rotation = average_transform(
        [m @ np.linalg.inv(s) for m, s in zip(moveit, sdk)]
    )[:3, :3]

    # moveit_p = Bridge_R @ sdk_R @ t + Bridge_R @ sdk_p + Bridge_t
    rows, targets = [], []
    for s, m in zip(sdk, moveit):
        block = np.zeros((3, 6))
        block[:, :3] = bridge_rotation @ s[:3, :3]
        block[:, 3:] = np.eye(3)
        rows.append(block)
        targets.append(m[:3, 3] - bridge_rotation @ s[:3, 3])
    solution, *_ = np.linalg.lstsq(
        np.vstack(rows), np.concatenate(targets), rcond=None
    )
    tool_translation = solution[:3]
    bridge = np.eye(4)
    bridge[:3, :3] = bridge_rotation
    bridge[:3, 3] = solution[3:]

    tool = np.eye(4)
    tool[:3, 3] = tool_translation
    position_residual = 0.0
    angle_residual = 0.0
    for s, m in zip(sdk, moveit):
        predicted = bridge @ s @ tool
        position_residual = max(
            position_residual,
            float(np.linalg.norm(predicted[:3, 3] - m[:3, 3])),
        )
        angle_residual = max(
            angle_residual,
            float(
                np.degrees(
                    Rotation.from_matrix(
                        predicted[:3, :3].T @ m[:3, :3]
                    ).magnitude()
                )
            ),
        )
    return bridge, tool, position_residual, angle_residual


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
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--write", action="store_true")
    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run_dir = Path(cli.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    link7_to_flange, _ = profile.left_tool_mount_calibration.require_transforms()

    states = sample_joint_states(0, cli.samples)
    mapped = [[s * v for s, v in zip(JOINT_SIGNS, q)] for q in states]

    left = open_left_arm(cfg, params, profile, take_control=False)
    moveit_planner = MoveItPlanner(project_root=ROOT, run_dir=run_dir)
    try:
        moveit_planner.start()
        moveit_poses = moveit_link7_poses(
            moveit_planner, mapped, profile.moveit_frame
        )
        sdk_poses = [
            np.asarray(left.controller_flange_from_joints(q), dtype=float)
            @ np.linalg.inv(link7_to_flange)
            for q in states
        ]
    finally:
        moveit_planner.close()
        left.close()

    bridge, tool, position_residual, angle_residual = solve(sdk_poses, moveit_poses)

    print(f"样本 {len(states)} 个关节状态，关节符号 {list(JOINT_SIGNS)}\n")
    print("Bridge（左臂控制器基座 → MoveIt 帧）:")
    for row in bridge:
        print("   " + "  ".join(f"{v: .9f}" for v in row))
    print(f"\n工具侧平移 t = {np.round(tool[:3, 3], 6).tolist()} m")
    print(f"\n残差：位置 {position_residual * 1000:7.3f} mm "
          f"（上限 {MAX_RESIDUAL_M * 1000:.1f}）")
    print(f"       姿态 {angle_residual:7.3f}°   "
          f"（上限 {MAX_ORIENTATION_RESIDUAL_DEG:.1f}）")

    if (
        position_residual > MAX_RESIDUAL_M
        or angle_residual > MAX_ORIENTATION_RESIDUAL_DEG
    ):
        print("\n✗ 残差没有收敛，模型里还有这三项之外的结构，不要采用这个拟合。")
        return 1

    print("\n✓ 模型闭合：符号 + 桥 + 工具平移三项就能解释全部样本。")
    if not cli.write:
        print("加 --write 写入 profile。")
        return 0

    path = Path(cli.safety_config)
    data = json.loads(path.read_text(encoding="utf-8"))
    shelf = data["profiles"]["shelf_template"]
    shelf["left_arm_model"] = {
        "joint_signs": list(JOINT_SIGNS),
        "T_moveit_from_left_profile": [[float(v) for v in row] for row in bridge],
        "T_left_link7_tool_offset": [[float(v) for v in row] for row in tool],
        "measured_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "samples": len(states),
        "max_position_residual_m": position_residual,
        "max_orientation_residual_deg": angle_residual,
        "method": (
            "Forward kinematics only, no motion commanded. moveit(q) = Bridge @ "
            "sdk(q with joints 2,4,6 negated) @ Tool, with Bridge's rotation "
            "from the mean of per-sample estimates and the two translations "
            "from one least squares. The signs and the tool-side constant match "
            "the URDF: r_joint1 carries a 3.1415 yaw that l_joint1 does not, and "
            "l_joint7 carries one the right arm does not."
        ),
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已写入 {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
