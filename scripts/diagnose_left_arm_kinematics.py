#!/usr/bin/env python3
"""Find out why the left arm's two kinematics do not differ by a fixed frame.

``measure_left_arm_bridge`` reported an 889 mm / 178 deg spread across joint
states, which rules out a constant transform.  The usual causes are a joint
sign convention or an ordering that differs between the SDK and the URDF, and
both are cheap to test: apply a candidate mapping to the joint values handed to
one side, re-solve for the bridge, and see whether the spread collapses.

A mapping that makes twelve independent poses agree to a millimetre is not a
coincidence.  One that does not is evidence against the whole family, which is
worth just as much -- it says the difference is in the model, not the wiring.

Commands nothing; both sides are forward kinematics.

    python scripts/diagnose_left_arm_kinematics.py
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
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

LOG = logging.getLogger("diagnose_left_arm_kinematics")


def spread(estimates: list[np.ndarray]) -> tuple[float, float]:
    mean = average_transform(estimates)
    positions = np.array([e[:3, 3] for e in estimates])
    position = float(np.max(np.linalg.norm(positions - mean[:3, 3], axis=1)))
    orientation = max(
        float(
            np.degrees(
                Rotation.from_matrix(mean[:3, :3].T @ e[:3, :3]).magnitude()
            )
        )
        for e in estimates
    )
    return position, orientation


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
    left = open_left_arm(cfg, params, profile, take_control=False)
    moveit = MoveItPlanner(project_root=ROOT, run_dir=run_dir)
    try:
        moveit.start()
        # The SDK side is evaluated once per candidate mapping; the MoveIt side
        # is evaluated once per mapping too, since the mapping is applied to
        # what MoveIt is asked about.  Cache what does not change.
        sdk_link7 = []
        for values in states:
            flange = np.asarray(
                left.controller_flange_from_joints(values), dtype=float
            )
            sdk_link7.append(flange @ np.linalg.inv(link7_to_flange))

        results = []
        for signs in itertools.product((1, -1), repeat=7):
            mapped = [[s * v for s, v in zip(signs, state)] for state in states]
            moveit_poses = moveit_link7_poses(moveit, mapped, profile.moveit_frame)
            estimates = [
                m @ np.linalg.inv(s) for m, s in zip(moveit_poses, sdk_link7)
            ]
            position, orientation = spread(estimates)
            results.append((position, orientation, signs))
            if position < 0.005 and orientation < 0.5:
                LOG.info("符号组合 %s 收敛，提前停止", signs)
                break
    finally:
        moveit.close()
        left.close()

    results.sort(key=lambda item: (item[0], item[1]))
    print("\n最一致的 8 个关节符号组合：")
    print("  位置离散(mm)  姿态离散(°)  符号")
    for position, orientation, signs in results[:8]:
        print(
            f"  {position * 1000:11.2f}  {orientation:10.2f}   "
            + " ".join(f"{s:+d}" for s in signs)
        )

    best_position, best_orientation, best_signs = results[0]
    if best_position < 0.005 and best_orientation < 0.5:
        print(
            f"\n✓ 找到了：关节符号 {list(best_signs)} 下两套运动学相差一个固定变换。"
        )
        return 0
    print(
        "\n✗ 128 种符号组合都不收敛。差异不在关节符号上——"
        "可能是关节顺序、URDF 里 l_link7 的定义，或者左臂 SDK 安装角与模型不符。"
    )
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
