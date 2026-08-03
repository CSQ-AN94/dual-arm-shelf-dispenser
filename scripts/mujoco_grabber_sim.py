#!/usr/bin/env python3
"""Visualize a synthetic RM75 shelf scene or replay an MTC trajectory in MuJoCo.

This simulator never imports the RealMan SDK and never opens a robot socket.
Arm geometry is deliberately primitive-based; joint origins and axes match the
installed RM75 URDF. Grasp/release follows the MTC semantic boundaries rather
than claiming to validate real finger friction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bottle_grasp.mtc_pick_contract import (
    EXPECTED_JOINTS,
    validate_full_transfer_trajectory,
    validate_pick_trajectory,
    validate_place_trajectory,
)

DEFAULT_SCENARIO = (
    ROOT
    / "mtc_ws/src/grabber_mtc_planner/scenarios/shelf_transfer_fixture.yaml"
)
PLATFORM_WORLD_Z = 1.0
DEFAULT_Q_RAD = np.array(
    [
        0.3929608704338789,
        2.0212308767137754,
        -0.815923981602209,
        0.6821619221630171,
        -0.15957544738457102,
        -0.2131745104443716,
        -0.39765582369860686,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class Replay:
    times: np.ndarray
    joints_rad: np.ndarray
    velocities_rad_s: np.ndarray
    attach_time: float | None
    release_time: float | None
    initially_attached: bool


def _finite_vector(value, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite {size}-vector")
    return result


def load_scenario(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scenario must be a YAML object")
    bottle = payload.get("bottle")
    if not isinstance(bottle, dict):
        raise ValueError("scenario is missing bottle")
    _finite_vector(bottle.get("pose", {}).get("xyz"), 3, "bottle.pose.xyz")
    if not isinstance(payload.get("shelf_boxes"), list):
        raise ValueError("scenario is missing shelf_boxes")
    return payload


def _xml_name(value: object) -> str:
    text = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in str(value)
    )
    return text or "unnamed"


def _orientation_xml(pose: dict, label: str) -> str:
    if "quat_xyzw" in pose:
        quaternion = _finite_vector(
            pose["quat_xyzw"], 4, f"{label}.quat_xyzw"
        )
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-9:
            raise ValueError(f"{label}.quat_xyzw has zero length")
        x, y, z, w = quaternion / norm
        return f'quat="{w:.9g} {x:.9g} {y:.9g} {z:.9g}"'
    rpy_rad = np.radians(
        _finite_vector(
            pose.get("rpy_deg", [0.0, 0.0, 0.0]),
            3,
            f"{label}.rpy_deg",
        )
    )
    return 'euler="{}"'.format(" ".join(f"{value:.9g}" for value in rpy_rad))


def build_model_xml(scenario: dict) -> str:
    shelf_geoms = []
    for index, box in enumerate(scenario["shelf_boxes"]):
        if not isinstance(box, dict):
            raise ValueError(f"shelf_boxes[{index}] must be an object")
        size = _finite_vector(box.get("size"), 3, f"shelf_boxes[{index}].size")
        center = _finite_vector(
            box.get("pose", {}).get("xyz"),
            3,
            f"shelf_boxes[{index}].pose.xyz",
        )
        orientation = _orientation_xml(
            box.get("pose", {}), f"shelf_boxes[{index}].pose"
        )
        if np.any(size <= 0.0):
            raise ValueError(f"shelf_boxes[{index}].size must be positive")
        shelf_geoms.append(
            '<geom name="{name}" type="box" pos="{pos}" {orientation} size="{size}" '
            'rgba="0.55 0.58 0.62 0.22"/>'.format(
                name=html.escape(_xml_name(box.get("id", f"shelf_{index}"))),
                pos=" ".join(f"{value:.9g}" for value in center),
                orientation=orientation,
                size=" ".join(f"{value / 2.0:.9g}" for value in size),
            )
        )

    obstacle_geoms = []
    voxel_size = float(scenario.get("obstacle_voxel_size_m", 0.05))
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("obstacle_voxel_size_m must be positive and finite")
    for index, center in enumerate(scenario.get("obstacle_voxels") or []):
        xyz = _finite_vector(center, 3, f"obstacle_voxels[{index}]")
        obstacle_geoms.append(
            '<geom name="obstacle_{index}" type="box" pos="{pos}" '
            'size="{half} {half} {half}" rgba="0.9 0.25 0.2 0.32"/>'.format(
                index=index,
                pos=" ".join(f"{value:.9g}" for value in xyz),
                half=f"{voxel_size / 2.0:.9g}",
            )
        )

    bottle = scenario["bottle"]
    bottle_xyz = _finite_vector(
        bottle["pose"]["xyz"], 3, "bottle.pose.xyz"
    )
    bottle_world = bottle_xyz + np.array([0.0, 0.0, PLATFORM_WORLD_Z])
    bottle_orientation = _orientation_xml(bottle["pose"], "bottle.pose")
    radius = float(bottle.get("radius_m", 0.033))
    height = float(bottle.get("height_m", 0.21))
    if (
        not math.isfinite(radius)
        or not math.isfinite(height)
        or min(radius, height) <= 0
    ):
        raise ValueError("bottle dimensions must be positive finite values")

    return f"""<mujoco model="grabber_mtc_sim">
  <compiler angle="radian" eulerseq="XYZ" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global azimuth="135" elevation="-18" offwidth="960" offheight="720"/>
    <rgba contactpoint="1 0 0 1" contactforce="1 0.3 0 1"/>
  </visual>
  <default>
    <joint damping="1.5" armature="0.02"/>
    <geom friction="0.9 0.01 0.001" condim="3" rgba="0.24 0.48 0.82 1"/>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="1 1 2" dir="-1 -1 -1" diffuse="0.45 0.45 0.45"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.16 0.18 0.2 1"/>
    <body name="platform_frame" pos="0 0 {PLATFORM_WORLD_Z}">
      <geom name="platform" type="box" pos="0 0 -0.035" size="0.24 0.22 0.035"
            rgba="0.16 0.2 0.28 1" contype="0" conaffinity="0"/>
      {''.join(shelf_geoms)}
      {''.join(obstacle_geoms)}
      <body name="r_base_link1" pos="-0.1 -0.1103 0.031645"
            euler="0 -0.7854 0">
        <geom type="cylinder" pos="0 0 0.09" size="0.065 0.09"/>
        <body name="r_link1" pos="0 0 0.187" euler="-1.5708 0 3.1415">
          <joint name="r_joint1" axis="0 -1 0" range="-3.11 3.11"/>
          <geom type="sphere" size="0.06"/>
          <body name="r_link2" euler="1.5708 0 0">
            <joint name="r_joint2" axis="0 1 0" range="-2.27 2.27"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.256" size="0.047"/>
            <body name="r_link3" pos="0 0 0.256" euler="-1.5708 0 0">
              <joint name="r_joint3" axis="0 -1 0" range="-3.11 3.11"/>
              <geom type="sphere" size="0.053"/>
              <body name="r_link4" euler="1.5708 0 0">
                <joint name="r_joint4" axis="0 1 0" range="-2.36 2.36"/>
                <geom type="capsule" fromto="0 0 0 0 -0.0003 0.21" size="0.043"/>
                <body name="r_link5" pos="0 -0.0003 0.21" euler="-1.5708 0 0">
                  <joint name="r_joint5" axis="0 -1 0" range="-3.11 3.11"/>
                  <geom type="sphere" size="0.047"/>
                  <body name="r_link6" euler="1.5709 0 0">
                    <joint name="r_joint6" axis="0 1 0.00015298" range="-2.23 2.23"/>
                    <geom type="capsule" fromto="0 0 0 0 -0.00028182 0.117"
                          size="0.039"/>
                    <body name="r_link7" pos="0 -0.00028182 0.117"
                          euler="-0.00015298 0 0">
                      <joint name="r_joint7" axis="0 0 1" range="-6.28 6.28"/>
                      <geom type="cylinder" pos="0 0 0.025" size="0.043 0.025"/>
                      <body name="r_hand" pos="0 0 0.0445">
                        <geom name="palm" type="box" pos="0 0 0.045"
                              size="0.055 0.04 0.045" rgba="0.18 0.22 0.3 1"/>
                        <body name="finger_positive" pos="0.06 0 0.09">
                          <joint name="finger_positive_joint" type="slide"
                                 axis="-1 0 0" range="0 0.032"/>
                          <geom type="box" pos="0 0 0.04" size="0.012 0.018 0.065"
                                rgba="0.1 0.13 0.18 1"/>
                        </body>
                        <body name="finger_negative" pos="-0.06 0 0.09">
                          <joint name="finger_negative_joint" type="slide"
                                 axis="1 0 0" range="0 0.032"/>
                          <geom type="box" pos="0 0 0.04" size="0.012 0.018 0.065"
                                rgba="0.1 0.13 0.18 1"/>
                        </body>
                        <site name="tcp" pos="0 0 0.1237" size="0.012"
                              rgba="0.1 1 0.2 1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="bottle" pos="{' '.join(f'{value:.9g}' for value in bottle_world)}"
          {bottle_orientation}>
      <freejoint name="bottle_free"/>
      <geom name="target_bottle" type="cylinder" size="{radius:.9g} {height / 2.0:.9g}"
            mass="0.24" rgba="0.15 0.72 0.95 0.72"/>
    </body>
  </worldbody>
</mujoco>"""


def load_replay(path: Path) -> Replay:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema == "grabber.mtc_pick.v2":
        validate_pick_trajectory(payload)
        attach = next(
            item for item in payload["phase_boundaries"] if item["name"] == "attach"
        )["start_index"]
        release = None
        initially_attached = False
    elif schema == "grabber.mtc_place.v1":
        validate_place_trajectory(payload)
        attach = None
        release = next(
            item
            for item in payload["phase_boundaries"]
            if item["name"] == "release"
        )["start_index"]
        initially_attached = True
    elif schema == "grabber.mtc_full_transfer.v1":
        validate_full_transfer_trajectory(payload)
        attach = next(
            item for item in payload["phase_boundaries"] if item["name"] == "attach"
        )["start_index"]
        release = next(
            item for item in payload["phase_boundaries"] if item["name"] == "release"
        )["start_index"]
        initially_attached = False
    else:
        raise ValueError(f"unsupported trajectory schema: {schema!r}")
    times = np.asarray(
        [point["time_from_start_s"] for point in payload["points"]], dtype=float
    )
    joints = np.radians(
        np.asarray([point["positions_deg"] for point in payload["points"]])
    )
    velocities = np.radians(
        np.asarray([point["velocities_deg_s"] for point in payload["points"]])
    )
    return Replay(
        times=times,
        joints_rad=joints,
        velocities_rad_s=velocities,
        attach_time=None if attach is None else float(times[attach]),
        release_time=None if release is None else float(times[release]),
        initially_attached=initially_attached,
    )


def static_replay(duration_s: float = 5.0) -> Replay:
    return Replay(
        times=np.array([0.0, duration_s], dtype=float),
        joints_rad=np.vstack((DEFAULT_Q_RAD, DEFAULT_Q_RAD)),
        velocities_rad_s=np.zeros((2, 7), dtype=float),
        attach_time=None,
        release_time=None,
        initially_attached=False,
    )


def _joint_qpos_addresses(model: mujoco.MjModel) -> list[int]:
    addresses = []
    for name in EXPECTED_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def _set_bottle_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    transform: np.ndarray,
    *,
    joint_name: str = "bottle_free",
) -> None:
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
    )
    if joint_id < 0:
        raise ValueError(f"MuJoCo model is missing joint {joint_name}")
    address = int(model.jnt_qposadr[joint_id])
    quaternion_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    data.qpos[address : address + 3] = transform[:3, 3]
    data.qpos[address + 3 : address + 7] = quaternion_xyzw[[3, 0, 1, 2]]
    velocity_address = int(model.jnt_dofadr[joint_id])
    data.qvel[velocity_address : velocity_address + 6] = 0.0


def _body_transform(
    model: mujoco.MjModel, data: mujoco.MjData, name: str
) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    transform = np.eye(4)
    transform[:3, :3] = data.xmat[body_id].reshape(3, 3)
    transform[:3, 3] = data.xpos[body_id]
    return transform


def _site_transform(
    model: mujoco.MjModel, data: mujoco.MjData, name: str
) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    transform = np.eye(4)
    transform[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    transform[:3, 3] = data.site_xpos[site_id]
    return transform


def _sample_replay(replay: Replay, time_s: float) -> tuple[np.ndarray, np.ndarray]:
    if time_s <= replay.times[0]:
        return replay.joints_rad[0].copy(), replay.velocities_rad_s[0].copy()
    if time_s >= replay.times[-1]:
        return replay.joints_rad[-1].copy(), replay.velocities_rad_s[-1].copy()
    upper = int(np.searchsorted(replay.times, time_s, side="right"))
    lower = upper - 1
    span = replay.times[upper] - replay.times[lower]
    alpha = (time_s - replay.times[lower]) / span
    start = replay.joints_rad[lower]
    end = replay.joints_rad[upper]
    start_velocity = replay.velocities_rad_s[lower]
    end_velocity = replay.velocities_rad_s[upper]
    alpha2 = alpha * alpha
    alpha3 = alpha2 * alpha
    position = (
        (2.0 * alpha3 - 3.0 * alpha2 + 1.0) * start
        + (alpha3 - 2.0 * alpha2 + alpha) * span * start_velocity
        + (-2.0 * alpha3 + 3.0 * alpha2) * end
        + (alpha3 - alpha2) * span * end_velocity
    )
    velocity = (
        (6.0 * alpha2 - 6.0 * alpha) * start / span
        + (3.0 * alpha2 - 4.0 * alpha + 1.0) * start_velocity
        + (-6.0 * alpha2 + 6.0 * alpha) * end / span
        + (3.0 * alpha2 - 2.0 * alpha) * end_velocity
    )
    return position, velocity


def run_simulation(
    scenario: dict,
    replay: Replay,
    *,
    viewer: bool,
    loop: bool,
    speed: float,
    render_path: Path | None,
) -> dict:
    model = mujoco.MjModel.from_xml_string(build_model_xml(scenario))
    data = mujoco.MjData(model)
    arm_addresses = _joint_qpos_addresses(model)
    arm_velocity_addresses = [
        int(model.jnt_dofadr[model.joint(name).id]) for name in EXPECTED_JOINTS
    ]
    finger_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ("finger_positive_joint", "finger_negative_joint")
    ]
    finger_addresses = [int(model.jnt_qposadr[item]) for item in finger_ids]
    finger_velocity_addresses = [int(model.jnt_dofadr[item]) for item in finger_ids]
    attached = False
    released = False
    tcp_to_bottle = np.eye(4)
    initial_bottle_transform = np.eye(4)
    duration = max(float(replay.times[-1]), model.opt.timestep)
    max_contacts = 0
    max_joint_replay_error_rad = 0.0
    max_attachment_error_m = 0.0
    attach_seen = replay.initially_attached
    release_seen = False

    def reset_pass() -> None:
        nonlocal attached, released, tcp_to_bottle, initial_bottle_transform
        mujoco.mj_resetData(model, data)
        data.qpos[arm_addresses] = replay.joints_rad[0]
        data.qvel[arm_velocity_addresses] = 0.0
        attached = replay.initially_attached
        released = False
        data.qpos[finger_addresses] = 0.03 if attached else 0.0
        data.qvel[finger_velocity_addresses] = 0.0
        mujoco.mj_forward(model, data)
        initial_bottle_transform = _body_transform(model, data, "bottle")
        tcp_to_bottle = np.linalg.inv(
            _site_transform(model, data, "tcp")
        ) @ _body_transform(model, data, "bottle")

    def update(sim_time: float) -> None:
        nonlocal attached, released, tcp_to_bottle
        nonlocal max_contacts, max_joint_replay_error_rad, max_attachment_error_m
        nonlocal attach_seen, release_seen
        arm_position, arm_velocity = _sample_replay(replay, sim_time)
        data.qpos[arm_addresses] = arm_position
        data.qvel[arm_velocity_addresses] = arm_velocity
        if not attached and not released:
            _set_bottle_pose(model, data, initial_bottle_transform)
            mujoco.mj_forward(model, data)
        if (
            replay.attach_time is not None
            and not attached
            and not released
            and sim_time >= replay.attach_time
        ):
            mujoco.mj_forward(model, data)
            tcp_to_bottle = np.linalg.inv(
                _site_transform(model, data, "tcp")
            ) @ _body_transform(model, data, "bottle")
            attached = True
            attach_seen = True
        if (
            replay.release_time is not None
            and attached
            and sim_time >= replay.release_time
        ):
            mujoco.mj_forward(model, data)
            _set_bottle_pose(
                model,
                data,
                _site_transform(model, data, "tcp") @ tcp_to_bottle,
            )
            mujoco.mj_forward(model, data)
            attached = False
            released = True
            release_seen = True
        finger_value = 0.03 if attached else 0.0
        data.qpos[finger_addresses] = finger_value
        data.qvel[finger_velocity_addresses] = 0.0
        if attached:
            mujoco.mj_forward(model, data)
            _set_bottle_pose(
                model,
                data,
                _site_transform(model, data, "tcp") @ tcp_to_bottle,
            )
        mujoco.mj_step(model, data)
        data.qpos[arm_addresses] = arm_position
        data.qvel[arm_velocity_addresses] = arm_velocity
        data.qpos[finger_addresses] = finger_value
        data.qvel[finger_velocity_addresses] = 0.0
        if attached:
            mujoco.mj_forward(model, data)
            expected_bottle = _site_transform(model, data, "tcp") @ tcp_to_bottle
            _set_bottle_pose(model, data, expected_bottle)
        elif not released:
            _set_bottle_pose(model, data, initial_bottle_transform)
        mujoco.mj_forward(model, data)
        max_joint_replay_error_rad = max(
            max_joint_replay_error_rad,
            float(np.max(np.abs(data.qpos[arm_addresses] - arm_position))),
        )
        if attached:
            max_attachment_error_m = max(
                max_attachment_error_m,
                float(
                    np.linalg.norm(
                        _body_transform(model, data, "bottle")[:3, 3]
                        - expected_bottle[:3, 3]
                    )
                ),
            )
        max_contacts = max(max_contacts, int(data.ncon))

    def one_pass(view=None) -> None:
        reset_pass()
        start = time.monotonic()
        sim_time = 0.0
        while sim_time <= duration:
            update(sim_time)
            if view is not None:
                view.sync()
                time.sleep(max(0.0, model.opt.timestep / speed))
            sim_time = (time.monotonic() - start) * speed if view is not None else (
                sim_time + model.opt.timestep
            )

    if viewer:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as passive:
            passive.cam.lookat[:] = [0.0, -0.72, 0.82]
            passive.cam.distance = 1.8
            passive.cam.azimuth = 135
            passive.cam.elevation = -18
            while passive.is_running():
                one_pass(passive)
                if not loop:
                    break
    else:
        one_pass()

    if render_path is not None:
        renderer = mujoco.Renderer(model, height=720, width=960)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, -0.72, 0.82]
        camera.distance = 1.8
        camera.azimuth = 135
        camera.elevation = -18
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
        from PIL import Image

        render_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(render_path)
        renderer.close()

    return {
        "execution_supported": False,
        "hardware_connections": 0,
        "duration_s": duration,
        "max_contacts": max_contacts,
        "max_joint_replay_error_rad": max_joint_replay_error_rad,
        "max_attachment_error_m": max_attachment_error_m,
        "attach_seen": attach_seen,
        "release_seen": release_seen,
        "final_bottle_z_m": float(_body_transform(model, data, "bottle")[2, 3]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--render", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0.0:
        parser.error("--speed must be a positive finite number")
    scenario = load_scenario(args.scenario)
    replay = load_replay(args.trajectory) if args.trajectory else static_replay()
    result = run_simulation(
        scenario,
        replay,
        viewer=not args.headless,
        loop=args.loop or (not args.headless and args.trajectory is None),
        speed=args.speed,
        render_path=args.render,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.trajectory:
        print("未提供 MTC 轨迹：当前仅显示/验证合成场景。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
