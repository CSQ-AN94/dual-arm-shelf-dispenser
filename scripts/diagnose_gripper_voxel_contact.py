#!/usr/bin/env python3
"""Which RGB-D voxel does the CTAG2F90D envelope hit, and where.

Plan-only and offline: no ROS, no arms, no camera.  It reads a pick-only
scenario YAML and rebuilds exactly the geometry MoveIt sees -- the two
CTAG2F90D collision boxes from
``grabber_robot_state_bridge.robot_description`` carried along the scenario's
own approach line -- then names every obstacle voxel that overlaps, with its
coordinates and how deep the overlap is on each gripper axis.

That last number is the point of the tool.  "IK rejected" says only that
something touched; the per-axis overlap says *which* axis is tight, so the
three candidate explanations separate on evidence instead of argument:

  * overlap only on the opening axis, and the voxel sits beside the bottle
    -> the 151 mm fully-open envelope is what is failing, not the scene;
  * the voxel sits on the bottle axis -> the target mask leaked;
  * the voxel sits on a shelf upright/side plane -> a real obstacle, and no
    amount of model tuning may delete it.

ponytail: the hand's two boxes only.  Forearm and wrist links are not carried
here, so this tool can exonerate the gripper but never the whole arm; if it
reports zero contacts and MTC still refuses, the contact is upstream of the
hand and needs the ROS-side check_state_validity path instead.

    python3 scripts/diagnose_gripper_voxel_contact.py \\
        outputs/mtc_direct_pick/<run>/mtc_direct_pick.yaml \\
        --out /tmp/gripper_voxel_contact.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(ROOT / "mtc_ws/src/grabber_robot_state_bridge")
)

from grabber_robot_state_bridge.robot_description import (  # noqa: E402
    COLLISION_BOXES,
    HAND_LINK_Z_FROM_IK_LINK_M,
    MOUNT_YAW_RAD,
)

DEFAULT_ARMS_CONFIG = ROOT / "mtc_ws/src/grabber_mtc_planner/config/dual_rm75_arms.yaml"

# The gripper boxes are authored in the hand frame with a +90 deg yaw, so their
# local axes carry a fixed physical meaning.  Naming them is what turns a raw
# overlap number into an answer.
BOX_AXIS_NAMES = ("opening", "palm_normal", "approach")


def _quat_to_matrix(quat_xyzw):
    x, y, z, w = (float(v) for v in quat_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 0.0:
        raise SystemExit("quat_xyzw 必须是有限的非零四元数")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _pose_matrix(pose):
    out = np.eye(4)
    out[:3, :3] = _quat_to_matrix(pose["quat_xyzw"])
    out[:3, 3] = np.asarray(pose["xyz"], dtype=float)
    return out


def gripper_boxes_in_hand_frame():
    """Return (half_extents, center, rotation) per CTAG2F90D collision box."""

    yaw = MOUNT_YAW_RAD
    rot = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    boxes = []
    for name, (size, xyz) in zip(("palm", "fingers"), COLLISION_BOXES):
        half = np.asarray([float(v) for v in size.split()], dtype=float) / 2.0
        center = np.asarray([float(v) for v in xyz.split()], dtype=float)
        boxes.append({"name": name, "half": half, "center": center, "rot": rot})
    return boxes


def obb_aabb_overlap(box_center, box_rot, box_half, centers, halves):
    """Separating-axis test of one OBB against many axis-aligned boxes.

    Returns (gap, per-box-axis penetration).  ``gap`` is the widest separation
    any single axis reports: positive means clear by at least that much,
    non-positive means contact.  Reporting it rather than a bare boolean is
    what distinguishes "the envelope fits" from "the envelope fits by 2 mm".

    The penetration is measured on the gripper box's own three axes, which is
    what makes it readable: it is how much narrower/shorter/thinner that box
    would have to be on that axis for this particular contact to disappear.
    """

    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    halves = np.asarray(halves, dtype=float).reshape(-1, 3)
    if len(centers) == 0:
        return np.zeros(0), np.zeros((0, 3))

    # R[i, j] = component of world axis j along gripper-box axis i.
    rot = box_rot.T
    abs_rot = np.abs(rot) + 1e-9
    delta = (centers - box_center) @ box_rot  # separation in the box's frame

    # Gripper-box axes.
    reach_a = halves @ abs_rot.T  # each world box's support on each box axis
    overlap_a = box_half[None, :] + reach_a - np.abs(delta)
    gap = (-overlap_a).max(axis=1)

    # World axes.
    delta_world = centers - box_center
    reach_b = np.abs(box_half[None, :] @ abs_rot)  # box support on world axes
    gap = np.maximum(gap, (np.abs(delta_world) - halves - reach_b).max(axis=1))

    # The nine cross-product axes.  Without them a corner-to-corner near miss
    # reads as a contact, and this tool exists to be believed about near misses.
    for i in range(3):
        for j in range(3):
            i1, i2 = (i + 1) % 3, (i + 2) % 3
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            radius_a = (
                box_half[i1] * abs_rot[i2, j] + box_half[i2] * abs_rot[i1, j]
            )
            radius_b = halves[:, j1] * abs_rot[i, j2] + halves[:, j2] * abs_rot[i, j1]
            distance = np.abs(
                delta[:, i2] * rot[i1, j] - delta[:, i1] * rot[i2, j]
            )
            gap = np.maximum(gap, distance - (radius_a + radius_b))

    return gap, overlap_a


def _support_top_z(shelf_boxes, support_id):
    for box in shelf_boxes:
        if str(box.get("id")) == support_id:
            return float(box["pose"]["xyz"][2]) + float(box["size"][2]) / 2.0
    return None


def analyze(scenario, arms_config, stations_step_m=0.005):
    frame_id = scenario.get("frame_id", "")
    voxels = np.asarray(scenario.get("obstacle_voxels") or [], dtype=float).reshape(-1, 3)
    voxel_size = float(scenario.get("obstacle_voxel_size_m", 0.0))
    if len(voxels) and voxel_size <= 0.0:
        raise SystemExit("scenario 有 obstacle_voxels 却没有正的 obstacle_voxel_size_m")
    shelf_boxes = scenario.get("shelf_boxes") or []

    arm_id = scenario.get("planning_arm_id", "right_arm")
    arm = next(
        (a for a in arms_config["arms"] if a["arm_id"] == arm_id), None
    )
    if arm is None:
        raise SystemExit(f"arm config 里没有 {arm_id}")
    tcp_from_ik = np.asarray(arm["tcp_transform_from_ik_link"], dtype=float)
    hand_from_ik = np.eye(4)
    hand_from_ik[2, 3] = HAND_LINK_Z_FROM_IK_LINK_M
    # TCP -> ik_link -> hand link, so the hand pose follows from the commanded
    # TCP pose alone.  No IK and no joint state are involved.
    hand_from_tcp = np.linalg.inv(tcp_from_ik) @ hand_from_ik

    approach = np.asarray(scenario["source_approach_direction"], dtype=float)
    approach = approach / np.linalg.norm(approach)
    pregrasp_offset = float(scenario["source_pregrasp_offset_m"])

    bottle = scenario.get("bottle") or {}
    bottle_center = np.asarray(bottle.get("pose", {}).get("xyz", [0, 0, 0]), dtype=float)
    bottle_radius = float(bottle.get("radius_m", 0.0))
    floor_z = _support_top_z(shelf_boxes, str(scenario.get("source_support_surface_id", "")))

    obstacles = []
    if len(voxels):
        obstacles.append(
            (
                scenario.get("dynamic_obstacle_id", "head_rgbd_non_target"),
                voxels,
                np.full((len(voxels), 3), voxel_size / 2.0),
            )
        )
    if shelf_boxes:
        obstacles.append(
            (
                "shelf_boxes",
                np.asarray([b["pose"]["xyz"] for b in shelf_boxes], dtype=float),
                np.asarray([b["size"] for b in shelf_boxes], dtype=float) / 2.0,
                )
        )
    shelf_ids = [str(b.get("id")) for b in shelf_boxes]

    candidates = scenario.get("source_grasp_candidates") or [
        {"id": "primary", "pose": scenario["source_grasp_pose"]}
    ]
    gripper = gripper_boxes_in_hand_frame()

    stations = np.arange(0.0, pregrasp_offset + 1e-9, stations_step_m)
    report = {
        "frame_id": frame_id,
        "planning_arm_id": arm_id,
        "obstacle_voxel_count": int(len(voxels)),
        "obstacle_voxel_size_m": voxel_size,
        "support_surface_top_z": floor_z,
        "pregrasp_offset_m": pregrasp_offset,
        "station_step_m": stations_step_m,
        "candidates": [],
    }

    for candidate in candidates:
        grasp = _pose_matrix(candidate["pose"])
        contacts = {}
        closest = {"gap_m": math.inf, "obstacle": None, "center_xyz": None}
        for inserted in stations:
            tcp = grasp.copy()
            tcp[:3, 3] -= approach * (pregrasp_offset - inserted)
            hand = tcp @ hand_from_tcp
            for box in gripper:
                rot = hand[:3, :3] @ box["rot"]
                center = hand[:3, 3] + hand[:3, :3] @ box["center"]
                for obstacle in obstacles:
                    name, obs_centers, obs_halves = obstacle
                    gap, overlap = obb_aabb_overlap(
                        center, rot, box["half"], obs_centers, obs_halves
                    )
                    nearest = int(np.argmin(gap))
                    if float(gap[nearest]) < closest["gap_m"]:
                        closest = {
                            "gap_m": float(gap[nearest]),
                            "obstacle": name,
                            "gripper_box": box["name"],
                            "insertion_m": float(inserted),
                            "center_xyz": [float(v) for v in obs_centers[nearest]],
                        }
                    for index in np.nonzero(gap <= 0.0)[0]:
                        key = (name, int(index))
                        record = contacts.get(key)
                        if record is None:
                            point = obs_centers[index]
                            radial = float(
                                np.linalg.norm((point - bottle_center)[:2])
                            )
                            record = {
                                "obstacle": name,
                                "obstacle_index": int(index),
                                "id": shelf_ids[index] if name == "shelf_boxes" else None,
                                "center_xyz": [float(v) for v in point],
                                "distance_to_grasp_m": float(
                                    np.linalg.norm(point - grasp[:3, 3])
                                ),
                                "radial_distance_to_bottle_axis_m": radial,
                                "inside_bottle_radius": bool(radial <= bottle_radius),
                                "height_above_support_m": (
                                    None if floor_z is None else float(point[2] - floor_z)
                                ),
                                "first_contact_insertion_m": float(inserted),
                                "gripper_boxes": {},
                            }
                            contacts[key] = record
                        per_axis = record["gripper_boxes"].setdefault(
                            box["name"], {n: 0.0 for n in BOX_AXIS_NAMES}
                        )
                        for axis, axis_name in enumerate(BOX_AXIS_NAMES):
                            per_axis[axis_name] = max(
                                per_axis[axis_name], float(overlap[index, axis])
                            )

        rows = sorted(contacts.values(), key=lambda r: r["first_contact_insertion_m"])
        # The one number that decides "envelope too wide": the deepest opening-axis
        # bite over all voxel contacts.  Shrink the finger box by twice that and
        # every one of these contacts is gone -- if it is the only tight axis.
        opening_bites = [
            per_axis["opening"]
            for row in rows
            if row["obstacle"] != "shelf_boxes"
            for per_axis in row["gripper_boxes"].values()
        ]
        report["candidates"].append(
            {
                "id": candidate.get("id", ""),
                "pregrasp_clear": not any(
                    row["first_contact_insertion_m"] <= 1e-9 for row in rows
                ),
                "contact_count": len(rows),
                "max_opening_axis_overlap_m": max(opening_bites, default=0.0),
                "closest_approach": closest,
                "contacts": rows,
            }
        )
    return report


def _print(report):
    print(f"frame={report['frame_id']}  arm={report['planning_arm_id']}")
    print(
        f"非目标体素 {report['obstacle_voxel_count']} 个，"
        f"边长 {report['obstacle_voxel_size_m'] * 1000:.0f} mm；"
        f"预抓偏置 {report['pregrasp_offset_m'] * 1000:.0f} mm"
    )
    for candidate in report["candidates"]:
        print(f"\n=== 抓取候选 {candidate['id']} ===")
        print(
            f"  预抓位{'干净' if candidate['pregrasp_clear'] else '就已经碰撞'}，"
            f"整条插入线上共 {candidate['contact_count']} 个接触体"
        )
        if candidate["contact_count"]:
            print(
                "  张开轴最深啃入 "
                f"{candidate['max_opening_axis_overlap_m'] * 1000:.1f} mm"
            )
        near = candidate["closest_approach"]
        if near["center_xyz"] is not None:
            xyz = near["center_xyz"]
            print(
                f"  最近点：{near['obstacle']} @ "
                f"({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f})，"
                f"离 {near['gripper_box']} 表面 {near['gap_m'] * 1000:+.1f} mm"
                f"（插入 {near['insertion_m'] * 1000:.0f} mm 处）"
            )
        for row in candidate["contacts"][:20]:
            label = row["id"] or f"{row['obstacle']}[{row['obstacle_index']}]"
            xyz = row["center_xyz"]
            height = row["height_above_support_m"]
            print(
                f"  - {label}: xyz=({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f})"
                f"  距抓取点 {row['distance_to_grasp_m'] * 1000:.0f} mm"
                f"  距瓶轴 {row['radial_distance_to_bottle_axis_m'] * 1000:.0f} mm"
                + ("" if height is None else f"  底板上方 {height * 1000:.0f} mm")
                + f"  插入 {row['first_contact_insertion_m'] * 1000:.0f} mm 时首次接触"
            )
            for box_name, per_axis in row["gripper_boxes"].items():
                print(
                    f"      {box_name}: "
                    + "  ".join(
                        f"{axis}={value * 1000:+.1f} mm"
                        for axis, value in per_axis.items()
                    )
                )
        if candidate["contact_count"] > 20:
            print(f"  ... 其余 {candidate['contact_count'] - 20} 个见 JSON")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_yaml", type=Path)
    parser.add_argument("--arms-config", type=Path, default=DEFAULT_ARMS_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--station-step-m",
        type=float,
        default=0.005,
        help="沿插入方向的采样步长；默认 5 mm",
    )
    args = parser.parse_args()

    scenario = yaml.safe_load(args.scenario_yaml.read_text(encoding="utf-8"))
    arms_config = yaml.safe_load(args.arms_config.read_text(encoding="utf-8"))
    report = analyze(scenario, arms_config, args.station_step_m)
    _print(report)
    if args.out:
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
