#!/usr/bin/env python3
"""Solve the left arm's SDK-to-MoveIt bridge and link7/flange transform.

For every sampled joint state the two forward-kinematics implementations obey

    moveit_link7(q) = Bridge @ sdk_flange(q) @ inv(Link7ToFlange)

The relative motions form the standard ``A X = X B`` hand-eye problem.  Joint
values are not remapped: the previous 2/4/6 sign flip matched the endpoint while
mirroring the elbow into a self-collision.  This tool commands no motion.
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
    MIN_KINEMATIC_SAMPLES,
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

JOINT_SIGNS = (1,) * 7
MAX_RESIDUAL_M = 0.003
MAX_ORIENTATION_RESIDUAL_DEG = 0.5


def rigid_inverse(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result = np.eye(4)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def solve(sdk_flanges: list[np.ndarray], moveit_link7s: list[np.ndarray]):
    """Return bridge, link7-to-flange, and maximum position/angle residuals."""
    if len(sdk_flanges) != len(moveit_link7s):
        raise SafetyAbort("SDK/MoveIt 左臂标定样本数量不一致")
    if len(sdk_flanges) < MIN_KINEMATIC_SAMPLES:
        raise SafetyAbort(
            f"左臂运动学标定至少需要 {MIN_KINEMATIC_SAMPLES} 个分散样本"
        )
    for label, transforms in (
        ("SDK", sdk_flanges),
        ("MoveIt", moveit_link7s),
    ):
        for index, transform in enumerate(transforms):
            transform = np.asarray(transform, dtype=float)
            if (
                transform.shape != (4, 4)
                or not np.all(np.isfinite(transform))
                or not np.allclose(
                    transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9
                )
                or not np.allclose(
                    transform[:3, :3].T @ transform[:3, :3],
                    np.eye(3),
                    atol=1e-6,
                )
                or not np.isclose(
                    np.linalg.det(transform[:3, :3]), 1.0, atol=1e-6
                )
            ):
                raise SafetyAbort(f"{label} 左臂标定样本 {index} 不是有效刚体变换")

    pairs = []
    rotation_rows = []
    for first in range(len(sdk_flanges) - 1):
        for second in range(first + 1, len(sdk_flanges)):
            sdk_relative = (
                rigid_inverse(sdk_flanges[first]) @ sdk_flanges[second]
            )
            moveit_relative = (
                rigid_inverse(moveit_link7s[first]) @ moveit_link7s[second]
            )
            pairs.append((sdk_relative, moveit_relative))
            rotation_rows.append(
                np.kron(np.eye(3), sdk_relative[:3, :3])
                - np.kron(moveit_relative[:3, :3].T, np.eye(3))
            )

    rotation_system = np.vstack(rotation_rows)
    if np.linalg.matrix_rank(rotation_system, tol=1e-8) < 8:
        raise SafetyAbort("左臂标定样本旋转激励不足，拒绝欠定拟合")
    _, _, right_vectors = np.linalg.svd(rotation_system)
    raw_rotation = right_vectors[-1].reshape((3, 3), order="F")
    if np.linalg.det(raw_rotation) < 0:
        raw_rotation = -raw_rotation
    u, _, vt = np.linalg.svd(raw_rotation)
    x_rotation = u @ vt
    if np.linalg.det(x_rotation) < 0:
        u[:, -1] *= -1
        x_rotation = u @ vt

    translation_rows = []
    translation_targets = []
    for sdk_relative, moveit_relative in pairs:
        translation_rows.append(sdk_relative[:3, :3] - np.eye(3))
        translation_targets.append(
            x_rotation @ moveit_relative[:3, 3] - sdk_relative[:3, 3]
        )
    translation_system = np.vstack(translation_rows)
    if np.linalg.matrix_rank(translation_system, tol=1e-8) < 3:
        raise SafetyAbort("左臂标定样本平移激励不足，拒绝欠定拟合")
    x_translation, *_ = np.linalg.lstsq(
        translation_system,
        np.concatenate(translation_targets),
        rcond=None,
    )
    flange_to_link7 = np.eye(4)
    flange_to_link7[:3, :3] = x_rotation
    flange_to_link7[:3, 3] = x_translation
    link7_to_flange = rigid_inverse(flange_to_link7)
    bridge = average_transform(
        [
            moveit @ rigid_inverse(flange @ flange_to_link7)
            for flange, moveit in zip(sdk_flanges, moveit_link7s)
        ]
    )

    position_residual = 0.0
    angle_residual = 0.0
    for flange, moveit in zip(sdk_flanges, moveit_link7s):
        predicted = bridge @ flange @ flange_to_link7
        position_residual = max(
            position_residual,
            float(np.linalg.norm(predicted[:3, 3] - moveit[:3, 3])),
        )
        angle_residual = max(
            angle_residual,
            float(
                np.degrees(
                    Rotation.from_matrix(
                        predicted[:3, :3].T @ moveit[:3, :3]
                    ).magnitude()
                )
            ),
        )
    return bridge, link7_to_flange, position_residual, angle_residual


def update_profile_records(
    shelf: dict,
    *,
    bridge: np.ndarray,
    link7_to_flange: np.ndarray,
    tool_offset: np.ndarray,
    measured_at: str,
    samples: int,
    position_residual: float,
    angle_residual: float,
) -> None:
    """Write the inseparable tool/model records from one accepted solution."""
    evidence = (
        f"Forward-kinematics hand-eye AX=XB over {samples} joint states; "
        f"no joint remapping and no motion commanded. Residual "
        f"{position_residual * 1000:.3f} mm / {angle_residual:.3f} deg. "
        "No left-arm grasp has held, so this remains free-space only."
    )
    left_tool = shelf["left_tool_mount_calibration"]
    left_tool["T_link7_controller_flange"] = [
        [float(value) for value in row] for row in link7_to_flange
    ]
    left_tool["measured_at_utc"] = measured_at
    left_tool["evidence_id"] = evidence
    shelf["left_arm_model"] = {
        "joint_signs": list(JOINT_SIGNS),
        "T_moveit_from_left_profile": [
            [float(value) for value in row] for row in bridge
        ],
        "T_left_link7_tool_offset": [
            [float(value) for value in row] for row in tool_offset
        ],
        "measured_at_utc": measured_at,
        "samples": samples,
        "max_position_residual_m": position_residual,
        "max_orientation_residual_deg": angle_residual,
        "method": (
            "moveit_link7(q) = Bridge @ sdk_flange(q) @ inv(Link7ToFlange). "
            "Link7ToFlange is solved by hand-eye AX=XB over relative motions; "
            "Bridge is averaged from the resulting absolute transforms. "
            "No joint remapping and no motion commanded."
        ),
    }


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
    states = sample_joint_states(0, cli.samples)
    run_dir = Path(cli.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )

    left = open_left_arm(cfg, params, profile, take_control=False)
    moveit_planner = MoveItPlanner(project_root=ROOT, run_dir=run_dir)
    try:
        moveit_planner.start()
        moveit_poses = moveit_link7_poses(
            moveit_planner, states, profile.moveit_frame
        )
        sdk_flanges = [
            np.asarray(left.controller_flange_from_joints(q), dtype=float)
            for q in states
        ]
    finally:
        moveit_planner.close()
        left.close()

    bridge, link7_to_flange, position_residual, angle_residual = solve(
        sdk_flanges, moveit_poses
    )
    nominal_link7_to_flange = np.eye(4)
    nominal_link7_to_flange[2, 3] = params.moveit_link7_to_controller_flange_m
    tool_offset = rigid_inverse(link7_to_flange) @ nominal_link7_to_flange

    print(f"样本 {len(states)} 个关节状态，关节符号 {list(JOINT_SIGNS)}\n")
    print("Bridge（左臂控制器基座 → MoveIt 帧）:")
    for row in bridge:
        print("   " + "  ".join(f"{v: .9f}" for v in row))
    print("\nLink7 → 控制器法兰:")
    for row in link7_to_flange:
        print("   " + "  ".join(f"{v: .9f}" for v in row))
    print(f"\n残差：位置 {position_residual * 1000:7.3f} mm "
          f"（上限 {MAX_RESIDUAL_M * 1000:.1f}）")
    print(f"       姿态 {angle_residual:7.3f}°   "
          f"（上限 {MAX_ORIENTATION_RESIDUAL_DEG:.1f}）")

    if (
        position_residual > MAX_RESIDUAL_M
        or angle_residual > MAX_ORIENTATION_RESIDUAL_DEG
    ):
        print("\n✗ 残差没有收敛，不写入安全档。")
        return 1

    print("\n✓ 模型闭合：无需关节重映射，桥和工具侧刚体变换解释全部样本。")
    if not cli.write:
        print("加 --write 写入 profile。")
        return 0

    path = Path(cli.safety_config)
    data = json.loads(path.read_text(encoding="utf-8"))
    shelf = data["profiles"]["shelf_template"]
    measured_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    update_profile_records(
        shelf,
        bridge=bridge,
        link7_to_flange=link7_to_flange,
        tool_offset=tool_offset,
        measured_at=measured_at,
        samples=len(states),
        position_residual=position_residual,
        angle_residual=angle_residual,
    )
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
