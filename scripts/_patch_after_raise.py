#!/usr/bin/env python3
"""Refresh a place-only scenario after the arm physically raises its payload."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _transform(pose: dict) -> np.ndarray:
    transform = np.eye(4)
    xyz = np.asarray(pose["xyz"], dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("pose.xyz must contain three finite values")
    if "quat_xyzw" in pose:
        rotation = Rotation.from_quat(pose["quat_xyzw"])
    elif "rpy_deg" in pose:
        rotation = Rotation.from_euler("xyz", pose["rpy_deg"], degrees=True)
    else:
        raise ValueError("pose needs quat_xyzw or rpy_deg")
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = xyz
    return transform


def _pose(transform: np.ndarray) -> dict:
    return {
        "xyz": transform[:3, 3].astype(float).tolist(),
        "quat_xyzw": Rotation.from_matrix(
            transform[:3, :3]
        ).as_quat().tolist(),
    }


def rebase_held_scenario(scenario: dict, new_source_pose: dict) -> dict:
    bottle_in_tcp = np.linalg.inv(
        _transform(scenario["source_grasp_pose"])
    ) @ _transform(scenario["bottle"]["pose"])
    scenario["bottle"]["pose"] = _pose(
        _transform(new_source_pose) @ bottle_in_tcp
    )
    scenario["source_grasp_pose"] = new_source_pose
    scenario["source_grasp_candidates"] = [
        {"id": "held_bottle", "pose": new_source_pose}
    ]
    scenario["target_transit_raise_m"] = 0.0
    return scenario


def main(argv: list[str] | None = None) -> int:
    from shelf_dispenser.core import DemoParams
    from shelf_dispenser.live_arm import open_live_arm
    from shelf_dispenser.safety import load_safety_profile
    from utils.config import load_config

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit("usage: _patch_after_raise.py SOURCE.yaml DESTINATION.yaml")
    source, destination = map(Path, args)
    config = load_config(ROOT / "config.yaml")
    profile = load_safety_profile(
        ROOT / "shelf_dispenser/safety_profiles.json",
        "shelf_template",
        require_verified=False,
    )
    robot, view = open_live_arm(
        config, DemoParams(), profile, "right_arm", take_control=False
    )
    try:
        tcp = view.pose_to_moveit(
            np.asarray(robot.tcp_from_joints(robot.joints_deg()), dtype=float)
        )
    finally:
        robot.close()
    pose = {
        "xyz": tcp[:3, 3].astype(float).tolist(),
        "quat_xyzw": Rotation.from_matrix(tcp[:3, :3]).as_quat().tolist(),
    }
    scenario = yaml.safe_load(source.read_text(encoding="utf-8"))
    old_xyz = scenario["source_grasp_pose"]["xyz"]
    rebase_held_scenario(scenario, pose)
    destination.write_text(
        yaml.safe_dump(scenario, allow_unicode=True), encoding="utf-8"
    )
    print(
        "手持位姿 %s -> %s  (抬升 %.0f mm)"
        % (
            [round(value, 3) for value in old_xyz],
            [round(value, 3) for value in pose["xyz"]],
            1000 * (pose["xyz"][2] - old_xyz[2]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
