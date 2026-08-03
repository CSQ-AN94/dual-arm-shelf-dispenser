#!/usr/bin/env python3
"""Replay an MTC pick/place plan on the complete dual-RM75 MuJoCo model.

This is a software-only digital twin: exact installed robot/gripper meshes,
synthetic ground-truth head segmentation, and a preplanned MTC trajectory.
The segmentation is a replay gate, not the planner input.  This never imports
the RealMan SDK or opens a robot connection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mujoco_grabber_sim import (
    _body_transform,
    _sample_replay,
    _set_bottle_pose,
    _site_transform,
    load_replay,
    load_scenario,
)

ASSETS = ROOT / ".mujoco_assets/dual_rm_75b_description"
URDF = ASSETS / "urdf/dual_rm_75b_moveit_expanded.urdf"
DEFAULT_SCENARIO = (
    ROOT
    / "mtc_ws/src/grabber_mtc_planner/scenarios/mujoco_shelf_workflow.yaml"
)
PLATFORM_LIFT_M = 0.647
PLATFORM_ORIGIN = np.array([0.0, -0.11663, 0.271 + PLATFORM_LIFT_M])
# Calibrated against the installed RMG24 meshes.  Their pad surface sits
# 1.9 mm behind the slide-joint origin; 0.2 mm closure keeps bilateral
# contact without exceeding the 0.5 mm penetration gate.
FINGER_MESH_INNER_PAD_OFFSET_M = 0.0019
FINGER_CONTACT_COMPRESSION_M = 0.0002
FINGER_TRAVEL_M = 0.0369
MAX_CONTACT_PENETRATION_M = 0.0005
ROBOT_CONTACT_BIT = 1
SHELF_CONTACT_BIT = 2
BOTTLE_CONTACT_BIT = 4
ALL_CONTACT_BITS = (
    ROBOT_CONTACT_BIT | SHELF_CONTACT_BIT | BOTTLE_CONTACT_BIT
)
# The SRDF home state is the only repository-backed left-arm posture known to
# stay upright while the platform lowers. Real execution still uses live state.
LEFT_Q_RAD = np.zeros(7)
SCENE_INITIAL_RIGHT_Q_DEG = np.array(
    [22.523, 115.811, -46.75, 39.085, -9.142, -12.215, -22.785]
)


def _sub(parent: ET.Element, tag: str, **attributes: object) -> ET.Element:
    return ET.SubElement(
        parent, tag, {key: str(value) for key, value in attributes.items()}
    )


def _find(root: ET.Element, xpath: str) -> ET.Element:
    found = root.find(xpath)
    if found is None:
        raise RuntimeError(f"robot model is missing {xpath}")
    return found


def _add_gripper(
    root: ET.Element, side: str, *, tcp_offset_m: float = 0.1682
) -> None:
    asset = _find(root, "asset")
    mesh_dir = ASSETS / "meshes"
    for suffix, filename in (
        ("gripper_base", "rmg24_gripper_base_link.STL"),
        ("finger1", "rmg24_finger1_link.STL"),
        ("finger2", "rmg24_finger2_link.STL"),
    ):
        _sub(
            asset,
            "mesh",
            name=f"{side}_{suffix}_exact",
            file=mesh_dir / filename,
        )
    link7 = _find(root, f".//body[@name='{side}_link7']")
    base = _sub(
        link7,
        "body",
        name=f"{side}_rmg24_gripper_base",
        pos="0 0 0.0445",
        euler="0 0 1.57",
    )
    _sub(
        base,
        "geom",
        type="mesh",
        mesh=f"{side}_gripper_base_exact",
        rgba="0.16 0.18 0.22 1",
        contype=ROBOT_CONTACT_BIT,
        conaffinity=ALL_CONTACT_BITS,
    )
    for index, axis in ((1, "-1 0 0"), (2, "1 0 0")):
        finger = _sub(
            base,
            "body",
            name=f"{side}_rmg24_finger{index}",
            pos="0 0 0.0891",
        )
        _sub(
            finger,
            "joint",
            name=f"{side}_rmg24_finger{index}_joint",
            type="slide",
            axis=axis,
            range=f"0 {FINGER_TRAVEL_M}",
            damping="2",
        )
        _sub(
            finger,
            "geom",
            name=f"{side}_rmg24_finger{index}_geom",
            type="mesh",
            mesh=f"{side}_finger{index}_exact",
            mass="0.2",
            rgba="0.08 0.09 0.12 1",
            contype=ROBOT_CONTACT_BIT,
            conaffinity=ALL_CONTACT_BITS,
        )
    _sub(
        link7,
        "site",
        name=f"{side}_tcp",
        pos=f"0 0 {tcp_offset_m}",
        size="0.008",
        rgba="0.1 1 0.2 1",
    )


def _add_shelf(parent: ET.Element) -> None:
    shelf = _sub(
        parent,
        "body",
        name="full_five_level_shelf",
        pos=" ".join(str(value) for value in PLATFORM_ORIGIN),
    )
    silver = "0.42 0.46 0.50 1"
    for index, z in enumerate((-0.993, -0.603, -0.213, 0.177, 0.567)):
        _sub(
            shelf,
            "geom",
            name=f"shelf_board_{index}",
            type="box",
            pos=f"0.03 -0.72 {z}",
            size="0.35 0.17 0.02",
            rgba=silver,
            contype=SHELF_CONTACT_BIT,
            conaffinity=ROBOT_CONTACT_BIT | BOTTLE_CONTACT_BIT,
        )
    for x in (-0.303, 0.363):
        _sub(
            shelf,
            "geom",
            name=f"shelf_rear_upright_{'left' if x < 0 else 'right'}",
            type="box",
            pos=f"{x} -0.875 -0.213",
            size="0.018 0.018 0.80",
            rgba="0.28 0.31 0.34 1",
            contype=SHELF_CONTACT_BIT,
            conaffinity=ROBOT_CONTACT_BIT | BOTTLE_CONTACT_BIT,
        )
    _sub(
        shelf,
        "geom",
        name="shelf_back_visual",
        type="box",
        pos="0.03 -0.89 -0.213",
        size="0.35 0.012 0.80",
        rgba="0.34 0.37 0.40 1",
        contype=SHELF_CONTACT_BIT,
        conaffinity=ROBOT_CONTACT_BIT | BOTTLE_CONTACT_BIT,
    )


def _add_bottle_geoms(
    body: ET.Element, name: str, color: str, *, target: bool
) -> None:
    _sub(
        body,
        "geom",
        name=name,
        type="cylinder",
        pos="0 0 0",
        size="0.033 0.105",
        mass="0.24" if target else "0",
        rgba=color,
        contype=BOTTLE_CONTACT_BIT,
        conaffinity=ALL_CONTACT_BITS,
    )
    _sub(
        body,
        "geom",
        name=f"{name}_label",
        type="cylinder",
        pos="0 0 -0.005",
        size="0.034 0.035",
        rgba="0.93 0.94 0.90 1",
        contype="0",
        conaffinity="0",
    )
    _sub(
        body,
        "geom",
        name=f"{name}_cap",
        type="cylinder",
        pos="0 0 0.111",
        size="0.018 0.010",
        rgba="0.10 0.22 0.70 1",
        contype="0",
        conaffinity="0",
    )


def build_full_model_xml(scenario: dict) -> str:
    if not URDF.is_file():
        raise FileNotFoundError(
            f"missing exact robot assets: {URDF}\n"
            "Fetch the installed description into .mujoco_assets first."
        )
    urdf = URDF.read_text(encoding="utf-8").replace(
        "package://dual_rm_75b_description/", f"{ASSETS}/"
    ).replace(
        "file:///home/rm/ros2_ws/install/dual_rm_75b_description/"
        "share/dual_rm_75b_description/",
        f"{ASSETS}/",
    )
    imported = mujoco.MjModel.from_xml_string(urdf)
    with tempfile.TemporaryDirectory(prefix="grabber_mujoco_") as temp:
        saved = Path(temp) / "robot.xml"
        mujoco.mj_saveLastXML(str(saved), imported)
        root = ET.fromstring(saved.read_text(encoding="utf-8"))

    root.set("model", "dual_rm75_full_shelf_workflow")
    root.insert(
        1,
        ET.Element(
            "option",
            {
                "timestep": "0.005",
                "gravity": "0 0 0",
                "integrator": "implicitfast",
            },
        ),
    )
    visual = ET.Element("visual")
    _sub(
        visual,
        "global",
        azimuth="135",
        elevation="-16",
        offwidth="1280",
        offheight="800",
    )
    root.insert(2, visual)
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    # The imported URDF geoms are the robot collision model.  Keep both arms,
    # the platform and head active so a replay cannot hide self/shelf contact.
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            geom.set("contype", str(ROBOT_CONTACT_BIT))
            geom.set("conaffinity", str(ALL_CONTACT_BITS))
    # The base URDF carries obsolete hand meshes at both wrists.  The installed
    # robot uses the RMG24 below, so rendering both makes a fake dexterous hand.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("mesh") in {
                "r_hand_base_link",
                "l_hand_base_link",
            }:
                parent.remove(child)
    world = _find(root, "worldbody")
    _sub(
        world,
        "geom",
        name="floor",
        type="plane",
        pos="0 0 -0.205",
        size="3 3 0.1",
        rgba="0.12 0.13 0.15 1",
        contype=SHELF_CONTACT_BIT,
        conaffinity=ROBOT_CONTACT_BIT,
    )
    _sub(
        world,
        "light",
        pos="0 -1 3",
        dir="0 0 -1",
        diffuse="0.9 0.9 0.9",
    )
    _sub(
        world,
        "light",
        pos="1 0 1.7",
        dir="-0.4 -0.8 -0.4",
        diffuse="0.55 0.55 0.55",
    )
    _add_shelf(world)
    obstacle_bottles = scenario.get("simulation_obstacle_bottles") or [
        {"xyz": [-0.220, -0.703, -0.088]},
        {"xyz": [0.300, -0.703, -0.088]},
    ]
    for index, item in enumerate(obstacle_bottles, start=1):
        xyz = PLATFORM_ORIGIN + np.asarray(item["xyz"], dtype=float)
        body = _sub(
            world,
            "body",
            name=f"non_target_bottle_{index}",
            pos=" ".join(f"{value:.9g}" for value in xyz),
        )
        _add_bottle_geoms(
            body,
            f"non_target_bottle_geom_{index}",
            "0.90 0.50 0.08 1",
            target=False,
        )

    bottle_xyz = np.asarray(scenario["bottle"]["pose"]["xyz"], dtype=float)
    bottle_world = PLATFORM_ORIGIN + bottle_xyz
    target = _sub(
        world,
        "body",
        name="target_bottle",
        pos=" ".join(f"{value:.9g}" for value in bottle_world),
    )
    _sub(target, "freejoint", name="target_bottle_free")
    _add_bottle_geoms(
        target,
        "target_bottle_geom",
        "0.05 0.55 0.92 1",
        target=True,
    )
    _add_gripper(root, "r")
    _add_gripper(root, "l")
    head = _find(root, ".//body[@name='head_link2']")
    _sub(
        head,
        "camera",
        name="head_rgbd",
        pos="-0.0032391 -0.051866 0.061606",
        euler="-1.5708 0 3.14159265",
        fovy="70",
    )
    return ET.tostring(root, encoding="unicode")


def _qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo model is missing joint {name}")
    return int(model.jnt_qposadr[joint_id])


def _set_initial_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
    positions: np.ndarray,
    platform_lift_m: float = PLATFORM_LIFT_M,
    left_positions: np.ndarray = LEFT_Q_RAD,
) -> None:
    data.qpos[_qpos_address(model, "platform_joint")] = platform_lift_m
    data.qpos[_qpos_address(model, "head_joint1")] = 0.0
    data.qpos[_qpos_address(model, "head_joint2")] = -0.38
    for index, value in enumerate(left_positions, start=1):
        data.qpos[_qpos_address(model, f"l_joint{index}")] = value
    for name, value in zip(joint_names, positions):
        data.qpos[_qpos_address(model, name)] = value
    for side in ("r", "l"):
        for index in (1, 2):
            data.qpos[
                _qpos_address(model, f"{side}_rmg24_finger{index}_joint")
            ] = FINGER_TRAVEL_M
    mujoco.mj_forward(model, data)


def detect_target_from_head(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rgb_path: Path | None,
) -> dict:
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera="head_rgbd")
    rgb = renderer.render()
    if rgb_path:
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(rgb_path)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="head_rgbd")
    segmentation = renderer.render()
    renderer.disable_segmentation_rendering()
    renderer.close()
    geom_id = model.geom("target_bottle_geom").id
    mask = segmentation[..., 0] == geom_id
    pixels = np.argwhere(mask)
    if not len(pixels):
        raise RuntimeError("head RGB-D segmentation cannot see the target bottle")
    center_vu = pixels.mean(axis=0)
    return {
        "source": "mujoco_ground_truth_head_segmentation",
        "target_geom_id": int(geom_id),
        "pixel_count": int(len(pixels)),
        "centroid_uv": [float(center_vu[1]), float(center_vu[0])],
        "rgb_path": str(rgb_path) if rgb_path else None,
    }


def _validate_grasp_alignment(
    tcp: np.ndarray,
    gripper_base: np.ndarray,
    bottle: np.ndarray,
    bottle_radius_m: float,
) -> tuple[np.ndarray, dict]:
    tcp_to_bottle = np.linalg.inv(tcp) @ bottle
    center_delta = bottle[:3, 3] - tcp[:3, 3]
    horizontal_center_error_m = abs(float(center_delta[0]))
    vertical_offset_m = abs(float(center_delta[2]))
    forward_m = float(center_delta @ tcp[:3, 2])
    center_depth_error_m = abs(forward_m)
    approach_alignment = float(np.clip(tcp[:3, 2] @ [0.0, -1.0, 0.0], -1, 1))
    finger_axis_alignment = abs(
        float(np.clip(gripper_base[:3, 0] @ [1.0, 0.0, 0.0], -1, 1))
    )
    approach_error_deg = math.degrees(math.acos(approach_alignment))
    finger_axis_error_deg = math.degrees(math.acos(finger_axis_alignment))
    if (
        horizontal_center_error_m > 0.01
        or vertical_offset_m > 0.04
        or center_depth_error_m > 0.01
        or approach_error_deg > 5.0
        or finger_axis_error_deg > 5.0
    ):
        raise RuntimeError(
            "attach rejected: gripper is not square to the bottle "
            f"(horizontal={horizontal_center_error_m:.3f} m, "
            f"vertical={vertical_offset_m:.3f} m, "
            f"center_depth={center_depth_error_m:.3f} m, "
            f"approach={approach_error_deg:.1f} deg, "
            f"finger_axis={finger_axis_error_deg:.1f} deg)"
        )
    return tcp_to_bottle, {
        "horizontal_center_error_m": horizontal_center_error_m,
        "vertical_offset_m": vertical_offset_m,
        "tcp_to_bottle_forward_m": forward_m,
        "center_depth_error_m": center_depth_error_m,
        "approach_error_deg": approach_error_deg,
        "finger_axis_error_deg": finger_axis_error_deg,
    }


def _finger_contact_position(bottle_radius_m: float) -> float:
    contact_m = (
        bottle_radius_m
        + FINGER_MESH_INNER_PAD_OFFSET_M
        - FINGER_CONTACT_COMPRESSION_M
    )
    if not 0.0 <= contact_m <= FINGER_TRAVEL_M:
        raise RuntimeError(
            "grasp rejected: bottle diameter is outside the RMG24 opening range"
        )
    return contact_m


def _validate_bilateral_finger_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, side: str
) -> int:
    bottle_id = model.geom("target_bottle_geom").id
    finger_ids = {
        model.geom(f"{side}_rmg24_finger{index}_geom").id for index in (1, 2)
    }
    contacted = set()
    for contact in data.contact[: data.ncon]:
        pair = {int(contact.geom1), int(contact.geom2)}
        if bottle_id in pair:
            if float(contact.dist) < -MAX_CONTACT_PENETRATION_M:
                raise RuntimeError(
                    "attach rejected: gripper penetrates the target bottle"
                )
            contacted.update(pair & finger_ids)
    if contacted != finger_ids:
        raise RuntimeError(
            "attach rejected: target bottle lacks bilateral finger contact"
        )
    return len(contacted)


def _reject_non_target_contacts(
    model: mujoco.MjModel, data: mujoco.MjData
) -> None:
    for contact in data.contact[: data.ncon]:
        body1 = model.body(model.geom_bodyid[int(contact.geom1)]).name
        body2 = model.body(model.geom_bodyid[int(contact.geom2)]).name
        pair = {body1, body2}
        geom1 = model.geom(int(contact.geom1)).name
        geom2 = model.geom(int(contact.geom2)).name
        non_target = next(
            (
                body
                for body in pair
                if body.startswith("non_target_bottle_")
            ),
            None,
        )
        if (
            non_target
            and "full_five_level_shelf" in pair
            and any(name.startswith("shelf_board_") for name in (geom1, geom2))
            and float(contact.dist) >= -MAX_CONTACT_PENETRATION_M
        ):
            continue
        if non_target:
            raise RuntimeError(
                "trajectory rejected: right arm, gripper, or target bottle "
                f"contacts a non-target bottle ({body1} vs {body2})"
            )


def _reject_robot_and_shelf_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    attached: bool,
    released: bool,
    allow_final_grasp_contact: bool = False,
    allow_attached_support_contact: bool = False,
    source_support_geom: str | None = None,
    target_support_geom: str | None = None,
) -> None:
    """Reject contacts MTC also treats as collisions.

    Bottle↔board support is the only world contact permitted.  While held, a
    target bottle must clear the board; this catches the old preview that
    simply slid the bottle through a shelf level.
    """
    for contact in data.contact[: data.ncon]:
        if float(contact.dist) > 0.0:
            continue
        body1 = model.body(model.geom_bodyid[int(contact.geom1)]).name
        body2 = model.body(model.geom_bodyid[int(contact.geom2)]).name
        geom1 = model.geom(int(contact.geom1)).name
        geom2 = model.geom(int(contact.geom2)).name
        pair = {body1, body2}
        if "world" in pair:
            other = next(body for body in pair if body != "world")
            if "wheel" in other and "floor" in (geom1, geom2):
                # Vendor wheel meshes intersect the display floor at rest.
                continue
            raise RuntimeError(
                "trajectory rejected: robot contacts floor "
                f"({body1} vs {body2})"
            )
        if any(body.startswith("non_target_bottle_") for body in pair):
            continue  # handled by _reject_non_target_contacts()
        if "full_five_level_shelf" in pair:
            shelf_geom = (
                geom1
                if body1 == "full_five_level_shelf"
                else geom2
            )
            expected_support = (
                target_support_geom if released else source_support_geom
            )
            if (
                "target_bottle" in pair
                and (not attached or released)
                and shelf_geom == expected_support
                and float(contact.dist) >= -MAX_CONTACT_PENETRATION_M
            ):
                continue
            if (
                "target_bottle" in pair
                and attached
                and allow_attached_support_contact
                and shelf_geom == source_support_geom
                and float(contact.dist) >= -MAX_CONTACT_PENETRATION_M
            ):
                continue
            raise RuntimeError(
                "trajectory rejected: robot or held bottle contacts shelf "
                f"({body1}/{geom1} vs {body2}/{geom2}, "
                f"penetration={max(0.0, -float(contact.dist)):.6f} m)"
            )
        if "target_bottle" in pair:
            other = next(body for body in pair if body != "target_bottle")
            if (
                (attached or allow_final_grasp_contact)
                and "_rmg24_finger" in other
            ):
                continue
            raise RuntimeError(
                "trajectory rejected: target bottle has unintended contact "
                f"({body1} vs {body2})"
            )
        raise RuntimeError(
            "trajectory rejected: robot self-collision "
            f"({body1} vs {body2})"
        )


def _nearest_support_board(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bottle: np.ndarray,
    bottle_half_height_m: float,
) -> str:
    bottle_bottom_z = float(bottle[2, 3] - bottle_half_height_m)
    candidates = []
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if name.startswith("shelf_board_"):
            board_top_z = float(
                data.geom_xpos[geom_id, 2] + model.geom_size[geom_id, 2]
            )
            candidates.append((abs(board_top_z - bottle_bottom_z), name))
    if not candidates or min(candidates)[0] > 0.01:
        raise RuntimeError("bottle pose does not match a shelf support board")
    return min(candidates)[1]


def _validate_release_support(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_support_geom: str | None,
) -> float:
    if not target_support_geom:
        raise RuntimeError("release rejected: target support board is unknown")
    fromto = np.zeros(6, dtype=float)
    distance_m = float(
        mujoco.mj_geomDistance(
            model,
            data,
            model.geom("target_bottle_geom").id,
            model.geom(target_support_geom).id,
            MAX_CONTACT_PENETRATION_M + 1e-6,
            fromto,
        )
    )
    if distance_m < -1e-6:
        raise RuntimeError(
            f"release rejected: bottle penetrates support by {-distance_m:.6f} m"
        )
    if distance_m > MAX_CONTACT_PENETRATION_M:
        raise RuntimeError(
            f"release rejected: bottle floats {distance_m:.6f} m above support"
        )
    return distance_m


def _validate_release_alignment(tcp: np.ndarray, scenario: dict) -> float:
    expected = PLATFORM_ORIGIN + np.asarray(
        scenario["target_place_pose"]["xyz"], dtype=float
    )
    error_m = float(np.linalg.norm(tcp[:3, 3] - expected))
    if error_m > 0.01:
        raise RuntimeError(
            f"release rejected: TCP misses target by {error_m:.3f} m"
        )
    return error_m


def _polyline_metrics(points: np.ndarray) -> tuple[float, float]:
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        raise ValueError("path points must be an Nx3 array with N >= 2")
    path_length_m = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    direct_distance_m = float(np.linalg.norm(points[-1] - points[0]))
    return path_length_m, direct_distance_m


def _validate_platform_arm_interlock(payload: dict) -> np.ndarray:
    lifts_mm = np.asarray(
        [
            point.get("platform_height_mm", PLATFORM_LIFT_M * 1000.0)
            for point in payload["points"]
        ],
        dtype=float,
    )
    positions_deg = np.asarray(
        [point["positions_deg"] for point in payload["points"]], dtype=float
    )
    velocities_deg_s = np.asarray(
        [point["velocities_deg_s"] for point in payload["points"]], dtype=float
    )
    accelerations_deg_s2 = np.asarray(
        [point["accelerations_deg_s2"] for point in payload["points"]],
        dtype=float,
    )
    if (
        lifts_mm.ndim != 1
        or positions_deg.shape != velocities_deg_s.shape
        or positions_deg.shape != accelerations_deg_s2.shape
        or positions_deg.shape != (len(lifts_mm), 7)
        or not np.all(np.isfinite(lifts_mm))
        or not np.all(np.isfinite(positions_deg))
        or not np.all(np.isfinite(velocities_deg_s))
        or not np.all(np.isfinite(accelerations_deg_s2))
        or np.any(lifts_mm < 0.0)
        or np.any(lifts_mm > 1000.0)
    ):
        raise RuntimeError("trajectory platform/arm samples are invalid")
    lift_moves = np.abs(np.diff(lifts_mm)) > 1e-6
    arm_moves = np.max(np.abs(np.diff(positions_deg, axis=0)), axis=1) > 1e-6
    if np.any(lift_moves & arm_moves):
        raise RuntimeError(
            "trajectory rejected: platform and right arm move simultaneously"
        )
    moving_points = np.flatnonzero(
        np.r_[lift_moves, False] | np.r_[False, lift_moves]
    )
    if moving_points.size and np.max(
        np.abs(velocities_deg_s[moving_points])
    ) > 1e-6:
        raise RuntimeError(
            "trajectory rejected: right-arm velocity is non-zero during platform motion"
        )
    if moving_points.size and np.max(
        np.abs(accelerations_deg_s2[moving_points])
    ) > 1e-6:
        raise RuntimeError(
            "trajectory rejected: right-arm acceleration is non-zero during platform motion"
        )
    if payload.get("schema_version") == "grabber.mtc_full_transfer.v1":
        phase = payload.get("platform_lift_phase")
        if not isinstance(phase, dict):
            raise RuntimeError("full-transfer trajectory lacks platform_lift_phase")
        start = phase.get("start_index")
        end = phase.get("end_index")
        source_height = phase.get("source_height_mm")
        target_height = phase.get("target_height_mm")
        declared = [
            item
            for item in payload.get("phase_boundaries", [])
            if isinstance(item, dict) and item.get("name") == "platform_lower"
        ]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end < len(lifts_mm)
            or phase.get("right_arm_stationary") is not True
            or phase.get("left_arm_stationary") is not True
            or isinstance(source_height, bool)
            or not isinstance(source_height, (int, float))
            or not math.isfinite(float(source_height))
            or abs(float(source_height) - 647.0) > 1e-6
            or isinstance(target_height, bool)
            or not isinstance(target_height, (int, float))
            or not math.isfinite(float(target_height))
            or abs(float(target_height) - 250.0) > 1e-6
            or len(declared) != 1
            or declared[0].get("start_index") != start
            or declared[0].get("end_index") != end
        ):
            raise RuntimeError("full-transfer platform phase metadata is invalid")
        if (
            not np.allclose(lifts_mm[: start + 1], 647.0, atol=1e-6, rtol=0.0)
            or not np.allclose(lifts_mm[end:], 250.0, atol=1e-6, rtol=0.0)
            or not np.all(np.diff(lifts_mm[start : end + 1]) < 0.0)
            or not np.allclose(
                positions_deg[start : end + 1],
                positions_deg[start],
                atol=1e-6,
                rtol=0.0,
            )
            or not np.allclose(
                velocities_deg_s[start : end + 1], 0.0, atol=1e-6, rtol=0.0
            )
            or not np.allclose(
                accelerations_deg_s2[start : end + 1],
                0.0,
                atol=1e-6,
                rtol=0.0,
            )
        ):
            raise RuntimeError(
                "full-transfer platform phase must be a stationary-arm monotonic 647->250 mm move"
            )
    return lifts_mm / 1000.0


def _validate_transport_path(
    model: mujoco.MjModel,
    payload: dict,
    scenario: dict,
    joint_names: list[str],
    left_positions: np.ndarray = LEFT_Q_RAD,
) -> dict | None:
    bounds = {
        item["name"]: (item["start_index"], item["end_index"])
        for item in payload["phase_boundaries"]
    }
    if "transport" not in bounds:
        if payload.get("schema_version") == "grabber.mtc_pick.v2":
            return None
        raise RuntimeError("transport validation requires a transport phase")
    data = mujoco.MjData(model)
    tcp_name = f"{joint_names[0][0]}_tcp"
    tcp_positions = []
    for point in payload["points"]:
        platform_lift_m = float(
            point.get("platform_height_mm", PLATFORM_LIFT_M * 1000.0)
        ) / 1000.0
        if not 0.0 <= platform_lift_m <= 1.0:
            raise RuntimeError("trajectory platform height must be within 0..1000 mm")
        _set_initial_state(
            model,
            data,
            joint_names,
            np.radians(point["positions_deg"]),
            platform_lift_m,
            left_positions,
        )
        tcp_positions.append(_site_transform(model, data, tcp_name)[:3, 3])
    start, end = bounds["transport"]
    transport = np.asarray(tcp_positions[start : end + 1]) - PLATFORM_ORIGIN
    path_length_m, direct_distance_m = _polyline_metrics(transport)
    if path_length_m > max(0.25, 3.0 * direct_distance_m):
        raise RuntimeError(
            "transport rejected: path detours "
            f"{path_length_m:.3f} m for a {direct_distance_m:.3f} m move"
        )
    boxes = {item["id"]: item for item in scenario["shelf_boxes"]}
    lower = (
        float(boxes["shelf_bottom"]["pose"]["xyz"][2])
        + float(boxes["shelf_bottom"]["size"][2]) / 2.0
    )
    upper = (
        float(boxes["shelf_top"]["pose"]["xyz"][2])
        - float(boxes["shelf_top"]["size"][2]) / 2.0
    )
    z_min = float(transport[:, 2].min())
    z_max = float(transport[:, 2].max())
    if (
        payload.get("cross_layer_transport") is not True
        and (z_min < lower or z_max > upper)
    ):
        raise RuntimeError(
            "transport rejected: TCP leaves the observed shelf compartment"
        )
    return {
        "path_length_m": path_length_m,
        "direct_distance_m": direct_distance_m,
        "detour_ratio": path_length_m / max(direct_distance_m, 1e-9),
        "z_range_m": [z_min, z_max],
        "cross_layer": payload.get("cross_layer_transport") is True,
    }


def view_scene_only(
    scenario: dict,
    *,
    viewer: bool,
    render_path: Path | None,
    observer_view: bool,
) -> dict:
    """Display scene geometry without inventing a motion trajectory."""
    model = mujoco.MjModel.from_xml_string(build_full_model_xml(scenario))
    data = mujoco.MjData(model)
    joint_names = [f"r_joint{index}" for index in range(1, 8)]
    _set_initial_state(
        model,
        data,
        joint_names,
        np.radians(SCENE_INITIAL_RIGHT_Q_DEG),
        PLATFORM_LIFT_M,
    )
    source_support = _nearest_support_board(
        model,
        data,
        _body_transform(model, data, "target_bottle"),
        float(scenario["bottle"]["height_m"]) / 2.0,
    )
    _reject_non_target_contacts(model, data)
    _reject_robot_and_shelf_contacts(
        model,
        data,
        attached=False,
        released=False,
        source_support_geom=source_support,
    )

    def set_observer(camera) -> None:
        camera.lookat[:] = [0.02, -0.43, 0.72]
        camera.distance = 2.2
        camera.azimuth = 0
        camera.elevation = -12

    if render_path:
        renderer = mujoco.Renderer(model, height=800, width=1280)
        if observer_view:
            camera = mujoco.MjvCamera()
            set_observer(camera)
            renderer.update_scene(data, camera=camera)
        else:
            renderer.update_scene(data, camera="head_rgbd")
        render_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(renderer.render()).save(render_path)
        renderer.close()

    if viewer:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as passive:
            if observer_view:
                set_observer(passive.cam)
            else:
                passive.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                passive.cam.fixedcamid = model.camera("head_rgbd").id
            while passive.is_running():
                passive.sync()
                time.sleep(0.02)

    return {
        "workflow": "scene_only_no_trajectory",
        "scenario_id": scenario.get("scenario_id"),
        "interactive_camera": bool(observer_view),
        "trajectory_loaded": False,
        "hardware_connections": 0,
        "execution_supported": False,
    }


def run_workflow(
    scenario: dict,
    trajectory_path: Path,
    *,
    viewer: bool,
    loop: bool,
    speed: float,
    head_rgb_path: Path | None,
    render_path: Path | None,
    observer_view: bool,
) -> dict:
    if scenario.get("simulation_scene_only") is True:
        raise RuntimeError(
            "scene-only random layout has no MTC trajectory; run the split "
            "pick/lift/place planner before MuJoCo replay"
        )
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if payload.get("simulation_preview_only") is True:
        raise RuntimeError(
            "kinematic preview trajectories are not accepted as MTC output"
        )
    if payload.get("scenario_id") != scenario.get("scenario_id"):
        raise RuntimeError("trajectory scenario_id does not match the scene")
    candidate_ids = {
        item["id"] for item in scenario.get("source_grasp_candidates") or []
    }
    if candidate_ids and payload.get("grasp_candidate_id") not in candidate_ids:
        raise RuntimeError("trajectory grasp candidate does not match the scene")
    replay = load_replay(trajectory_path)
    platform_lifts_m = _validate_platform_arm_interlock(payload)
    joint_names = list(payload["joint_names"])
    left_positions = np.radians(
        np.asarray(
            payload.get("left_joints_deg", np.degrees(LEFT_Q_RAD)),
            dtype=float,
        )
    )
    if left_positions.shape != (7,) or not np.all(np.isfinite(left_positions)):
        raise RuntimeError("trajectory left_joints_deg must contain seven finite values")
    model = mujoco.MjModel.from_xml_string(build_full_model_xml(scenario))
    data = mujoco.MjData(model)
    transport_metrics = _validate_transport_path(
        model, payload, scenario, joint_names, left_positions
    )
    arm_addresses = [_qpos_address(model, name) for name in joint_names]
    finger_addresses = [
        _qpos_address(model, f"{joint_names[0][0]}_rmg24_finger{index}_joint")
        for index in (1, 2)
    ]
    _set_initial_state(
        model,
        data,
        joint_names,
        replay.joints_rad[0],
        platform_lifts_m[0],
        left_positions,
    )
    perception = detect_target_from_head(model, data, head_rgb_path)
    print(
        "[find] head RGB-D locked target: "
        f"{perception['pixel_count']} pixels at {perception['centroid_uv']}"
    )
    phases = [
        (item["name"], float(replay.times[item["start_index"]]))
        for item in payload["phase_boundaries"]
    ]
    phase_start_times = dict(phases)
    duration = max(float(replay.times[-1]), model.opt.timestep)
    initial_bottle = _body_transform(model, data, "target_bottle")
    bottle_half_height_m = float(scenario["bottle"]["height_m"]) / 2.0
    source_support_geom = _nearest_support_board(
        model, data, initial_bottle, bottle_half_height_m
    )
    target_tcp = np.eye(4)
    target_tcp[:3, :3] = Rotation.from_quat(
        scenario["target_place_pose"]["quat_xyzw"]
    ).as_matrix()
    target_tcp[:3, 3] = PLATFORM_ORIGIN + np.asarray(
        scenario["target_place_pose"]["xyz"], dtype=float
    )
    tcp_name = f"{joint_names[0][0]}_tcp"
    bottle_radius_m = float(scenario["bottle"]["radius_m"])
    finger_contact_m = _finger_contact_position(bottle_radius_m)
    last_alignment: dict | None = None
    bilateral_contact_count = 0
    release_target_error_m: float | None = None
    release_support_distance_m: float | None = None
    target_support_geom: str | None = None

    def one_pass(view=None) -> tuple[bool, bool]:
        nonlocal last_alignment, bilateral_contact_count
        nonlocal release_target_error_m, release_support_distance_m
        nonlocal target_support_geom
        _set_initial_state(
            model,
            data,
            joint_names,
            replay.joints_rad[0],
            platform_lifts_m[0],
            left_positions,
        )
        _set_bottle_pose(
            model,
            data,
            initial_bottle,
            joint_name="target_bottle_free",
        )
        attached = False
        released = False
        source_support_cleared = False
        tcp_to_bottle = np.eye(4)
        announced: set[str] = set()
        start = time.monotonic()
        sim_time = 0.0
        while sim_time <= duration:
            position, _ = _sample_replay(replay, sim_time)
            data.qpos[arm_addresses] = position
            data.qpos[_qpos_address(model, "platform_joint")] = np.interp(
                sim_time, replay.times, platform_lifts_m
            )
            if (
                replay.attach_time is not None
                and not attached
                and not released
                and sim_time >= replay.attach_time
            ):
                mujoco.mj_forward(model, data)
                tcp_to_bottle, last_alignment = _validate_grasp_alignment(
                    _site_transform(model, data, tcp_name),
                    _body_transform(
                        model,
                        data,
                        f"{joint_names[0][0]}_rmg24_gripper_base",
                    ),
                    _body_transform(model, data, "target_bottle"),
                    bottle_radius_m,
                )
                data.qpos[finger_addresses] = finger_contact_m
                mujoco.mj_forward(model, data)
                bilateral_contact_count = _validate_bilateral_finger_contacts(
                    model, data, joint_names[0][0]
                )
                target_support_geom = _nearest_support_board(
                    model,
                    data,
                    target_tcp @ tcp_to_bottle,
                    bottle_half_height_m,
                )
                attached = True
            if (
                replay.release_time is not None
                and attached
                and sim_time >= replay.release_time
            ):
                mujoco.mj_forward(model, data)
                tcp = _site_transform(model, data, tcp_name)
                release_target_error_m = _validate_release_alignment(tcp, scenario)
                _set_bottle_pose(
                    model,
                    data,
                    tcp @ tcp_to_bottle,
                    joint_name="target_bottle_free",
                )
                mujoco.mj_forward(model, data)
                release_support_distance_m = _validate_release_support(
                    model, data, target_support_geom
                )
                attached = False
                released = True
            data.qpos[finger_addresses] = (
                finger_contact_m if attached else FINGER_TRAVEL_M
            )
            mujoco.mj_forward(model, data)
            if attached:
                _set_bottle_pose(
                    model,
                    data,
                    _site_transform(model, data, tcp_name) @ tcp_to_bottle,
                    joint_name="target_bottle_free",
                )
                mujoco.mj_forward(model, data)
            elif not released:
                _set_bottle_pose(
                    model,
                    data,
                    initial_bottle,
                    joint_name="target_bottle_free",
                )
                mujoco.mj_forward(model, data)
            _reject_non_target_contacts(model, data)
            tcp = _site_transform(model, data, tcp_name)
            bottle = _body_transform(model, data, "target_bottle")
            if attached and not source_support_cleared and source_support_geom:
                support_id = model.geom(source_support_geom).id
                support_top_z = (
                    float(data.geom_xpos[support_id][2])
                    + float(model.geom_size[support_id][2])
                )
                source_support_cleared = bool(
                    float(bottle[2, 3]) - bottle_half_height_m
                    > support_top_z + MAX_CONTACT_PENETRATION_M
                )
            allow_final_grasp_contact = bool(
                not attached
                and not released
                and "approach" in phase_start_times
                and sim_time >= phase_start_times["approach"]
                and np.linalg.norm(tcp[:3, 3] - bottle[:3, 3])
                <= float(scenario["source_contact_distance_m"]) + 0.002
            )
            _reject_robot_and_shelf_contacts(
                model,
                data,
                attached=attached,
                released=released,
                allow_final_grasp_contact=allow_final_grasp_contact,
                allow_attached_support_contact=bool(
                    attached and not source_support_cleared
                ),
                source_support_geom=source_support_geom,
                target_support_geom=target_support_geom,
            )
            for name, phase_time in phases:
                if sim_time >= phase_time and name not in announced:
                    print(f"[plan] {name}")
                    announced.add(name)
            if view is not None:
                view.sync()
                time.sleep(max(0.0, model.opt.timestep / speed))
                sim_time = (time.monotonic() - start) * speed
            else:
                sim_time += model.opt.timestep
        return attached, released

    attached = released = False
    if viewer:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as passive:
            if observer_view:
                passive.cam.lookat[:] = [0.02, -0.43, 0.72]
                passive.cam.distance = 2.2
                passive.cam.azimuth = 0
                passive.cam.elevation = -12
            else:
                passive.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                passive.cam.fixedcamid = model.camera("head_rgbd").id
            while passive.is_running():
                attached, released = one_pass(passive)
                if not loop:
                    break
    else:
        attached, released = one_pass()

    if render_path:
        renderer = mujoco.Renderer(model, height=800, width=1280)
        if observer_view:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.02, -0.43, 0.72]
            camera.distance = 2.2
            camera.azimuth = 0
            camera.elevation = -12
            renderer.update_scene(data, camera=camera)
        else:
            renderer.update_scene(data, camera="head_rgbd")
        image = renderer.render()
        render_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(render_path)
        renderer.close()
    return {
        "workflow": (
            "synthetic_head_target_gate -> preplanned_mtc_replay -> pick"
            if payload["schema_version"] == "grabber.mtc_pick.v2"
            else "synthetic_head_target_gate -> preplanned_mtc_replay -> "
            "pick -> transport -> place"
        ),
        "robot_model": "dual_rm75_full_shelf_workflow",
        "robot_bodies": int(model.nbody),
        "robot_joints": int(model.njnt),
        "robot_meshes": int(model.nmesh),
        "perception": perception,
        "trajectory_schema": payload["schema_version"],
        "trajectory_points": len(payload["points"]),
        "trajectory_duration_s": duration,
        "platform_height_range_mm": [
            float(platform_lifts_m.min() * 1000.0),
            float(platform_lifts_m.max() * 1000.0),
        ],
        "transport_metrics": transport_metrics,
        "attach_seen": last_alignment is not None,
        "attach_alignment": last_alignment,
        "grasp_validation": "kinematic_pad_enclosure_only",
        "finger_contact_position_m": finger_contact_m,
        "nominal_pad_gap_m": 2.0
        * (finger_contact_m - FINGER_MESH_INNER_PAD_OFFSET_M),
        "bilateral_finger_contacts": bilateral_contact_count,
        "non_target_collision_checks": True,
        "robot_self_collision_checks": True,
        "shelf_collision_checks": True,
        "contact_physics_validated": False,
        "release_seen": released,
        "release_target_error_m": release_target_error_m,
        "release_support_distance_m": release_support_distance_m,
        "hardware_connections": 0,
        "execution_supported": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument(
        "--scene-only",
        action="store_true",
        help="show geometry only; never synthesize or replay a trajectory",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--head-rgb", type=Path, default=Path("/tmp/grabber_head_rgb.png"))
    parser.add_argument("--render", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--observer-view",
        action="store_true",
        help="use an external debug camera instead of the physical head camera",
    )
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0.0:
        parser.error("--speed must be a positive finite number")
    scenario = load_scenario(args.scenario)
    try:
        if args.scene_only:
            if args.trajectory is not None:
                parser.error("--scene-only cannot be combined with --trajectory")
            result = view_scene_only(
                scenario,
                viewer=not args.headless,
                render_path=args.render,
                observer_view=args.observer_view,
            )
        else:
            if args.trajectory is None:
                parser.error("--trajectory is required unless --scene-only is used")
            result = run_workflow(
                scenario,
                args.trajectory,
                viewer=not args.headless,
                loop=args.loop,
                speed=args.speed,
                head_rgb_path=args.head_rgb,
                render_path=args.render,
                observer_view=args.observer_view,
            )
        result["success"] = True
    except RuntimeError as exc:
        message = str(exc)
        failure_stage = next(
            (
                stage
                for marker, stage in (
                    ("head RGB-D", "fixed_head_perception"),
                    ("platform", "platform_lower"),
                    ("transport", "transport_validation"),
                    ("attach", "attach"),
                    ("release", "release"),
                )
                if marker in message
            ),
            "replay_validation",
        )
        failure = {
            "success": False,
            "scenario_id": scenario.get("scenario_id"),
            "failure_stage": failure_stage,
            "error": message,
            "random_scene": {
                "target_bottle_xyz": scenario.get("bottle", {})
                .get("pose", {})
                .get("xyz"),
                "non_target_bottles": scenario.get(
                    "simulation_obstacle_bottles", []
                ),
                "target_place_pose": scenario.get("target_place_pose"),
            },
            "hardware_connections": 0,
            "execution_supported": False,
        }
        if args.result_json:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            args.result_json.write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if args.render:
            view_scene_only(
                scenario,
                viewer=False,
                render_path=args.render,
                observer_view=args.observer_view,
            )
        raise
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
