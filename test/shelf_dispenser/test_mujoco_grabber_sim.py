from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco
from scipy.spatial.transform import Rotation

from shelf_dispenser.core import SafetyAbort
from shelf_dispenser.mtc_pick_contract import (
    EXPECTED_JOINTS,
    validate_full_transfer_trajectory,
)
from scripts.compose_mujoco_cross_layer_mtc import compose_cross_layer_replay
from scripts.mujoco_full_workflow import (
    ASSETS,
    DEFAULT_SCENARIO as FULL_SCENARIO,
    PLATFORM_ORIGIN,
    _finger_contact_position,
    _polyline_metrics,
    _reject_non_target_contacts,
    _reject_robot_and_shelf_contacts,
    _validate_bilateral_finger_contacts,
    _validate_grasp_alignment,
    _validate_platform_arm_interlock,
    _validate_release_alignment,
    _validate_release_support,
    _validate_transport_path,
    build_full_model_xml,
    run_workflow,
    view_scene_only,
)
from scripts.mujoco_grabber_sim import (
    DEFAULT_Q_RAD,
    DEFAULT_SCENARIO,
    Replay,
    _sample_replay,
    build_model_xml,
    load_replay,
    load_scenario,
    run_simulation,
    static_replay,
)
from scripts.mujoco_pick_to_place_mtc import build_place_scenario


def _trajectory(schema: str) -> dict:
    pick = schema == "grabber.mtc_pick.v2"
    now = datetime.now(timezone.utc).isoformat()
    points = [
        {
            "time_from_start_s": index * 0.01,
            "positions_deg": np.degrees(DEFAULT_Q_RAD).tolist(),
            "velocities_deg_s": [0.0] * 7,
            "accelerations_deg_s2": [0.0] * 7,
        }
        for index in range(6)
    ]
    payload = {
        "schema_version": schema,
        "plan_only": True,
        "execution_supported": False,
        "execution_block_reason": (
            "EXTERNAL_GATED_PICK_EXECUTOR_REQUIRED"
            if pick
            else "EXTERNAL_GATED_PLACE_EXECUTOR_REQUIRED"
        ),
        "mode": "pick_only" if pick else "place_only",
        "scenario_id": "mujoco-smoke",
        "arm_id": "right_arm",
        "scene_captured_at_utc": now,
        "freshness_max_age_s": 45.0,
        "joint_units": "degrees",
        "joint_names": list(EXPECTED_JOINTS),
        "points": points,
    }
    if pick:
        payload.update(
            {
                "grasp_candidate_id": "wrist_roll_180",
                "target_captured_at_utc": now,
                "phase_boundaries": [
                    {"name": "pregrasp", "start_index": 0, "end_index": 2},
                    {"name": "approach", "start_index": 2, "end_index": 4},
                    {"name": "attach", "start_index": 4, "end_index": 4},
                    {"name": "retreat", "start_index": 4, "end_index": 5},
                ],
                "gripper_events": [
                    {
                        "name": "open_before_motion",
                        "point_index": 0,
                        "operation": "RobotSession.open_gripper",
                        "feedback_required": True,
                    },
                    {
                        "name": "close_at_attach",
                        "point_index": 4,
                        "operation": "RobotSession.close_gripper",
                        "feedback_required": True,
                    },
                ],
            }
        )
    else:
        payload.update(
            {
                "phase_boundaries": [
                    {"name": "transport", "start_index": 0, "end_index": 1},
                    {"name": "approach", "start_index": 1, "end_index": 4},
                    {"name": "release", "start_index": 4, "end_index": 4},
                    {"name": "retreat", "start_index": 4, "end_index": 5},
                ],
                "gripper_events": [
                    {
                        "name": "hold_before_motion",
                        "point_index": 0,
                        "operation": "validate_holding_gripper_feedback",
                        "feedback_required": True,
                    },
                    {
                        "name": "open_at_release",
                        "point_index": 4,
                        "operation": "RobotSession.open_gripper",
                        "feedback_required": True,
                    },
                ],
            }
        )
    return payload


def test_mujoco_scene_and_pick_place_semantics(tmp_path):
    scenario = load_scenario(DEFAULT_SCENARIO)
    scenario = deepcopy(scenario)
    scenario["obstacle_voxels"] = [[0.2, -0.7, 0.0]]
    scenario["shelf_boxes"][0]["pose"] = {
        "xyz": scenario["shelf_boxes"][0]["pose"]["xyz"],
        "quat_xyzw": [0.0, 0.0, 2**-0.5, 2**-0.5],
    }
    model = mujoco.MjModel.from_xml_string(build_model_xml(scenario))
    expected_axes = (
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.00015298],
        [0.0, 0.0, 1.0],
    )
    for name, expected in zip(EXPECTED_JOINTS, expected_axes):
        expected = np.asarray(expected)
        assert np.allclose(model.joint(name).axis, expected / np.linalg.norm(expected))
    assert model.geom("obstacle_0").id >= 0
    assert np.allclose(
        np.abs(model.geom("shelf_bottom").quat),
        [2**-0.5, 0.0, 0.0, 2**-0.5],
    )

    summaries = {}
    for schema in ("grabber.mtc_pick.v2", "grabber.mtc_place.v1"):
        path = tmp_path / f"{schema}.json"
        path.write_text(json.dumps(_trajectory(schema)), encoding="utf-8")
        summaries[schema] = run_simulation(
            scenario,
            load_replay(path),
            viewer=False,
            loop=False,
            speed=1.0,
            render_path=None,
        )

    pick = summaries["grabber.mtc_pick.v2"]
    place = summaries["grabber.mtc_place.v1"]
    assert pick["attach_seen"] is True and pick["release_seen"] is False
    assert place["attach_seen"] is True and place["release_seen"] is True
    assert pick["hardware_connections"] == place["hardware_connections"] == 0
    assert pick["execution_supported"] is place["execution_supported"] is False
    assert pick["max_joint_replay_error_rad"] < 1e-12
    assert pick["max_attachment_error_m"] < 1e-12

    static = run_simulation(
        scenario,
        static_replay(0.02),
        viewer=False,
        loop=False,
        speed=1.0,
        render_path=None,
    )
    expected_z = scenario["bottle"]["pose"]["xyz"][2] + 1.0
    assert static["final_bottle_z_m"] == pytest.approx(expected_z)


def test_replay_uses_exported_velocities():
    replay = Replay(
        times=np.array([0.0, 1.0]),
        joints_rad=np.array([[0.0] * 7, [1.0] * 7]),
        velocities_rad_s=np.array([[1.0] * 7, [1.0] * 7]),
        attach_time=None,
        release_time=None,
        initially_attached=False,
    )
    position, velocity = _sample_replay(replay, 0.25)
    assert np.allclose(position, 0.25)
    assert np.allclose(velocity, 1.0)


def test_full_transfer_contract_and_exact_robot_model(tmp_path):
    points = [
        {
            "time_from_start_s": index * 0.01,
            "positions_deg": np.degrees(DEFAULT_Q_RAD).tolist(),
            "velocities_deg_s": [0.0] * 7,
            "accelerations_deg_s2": [0.0] * 7,
        }
        for index in range(12)
    ]
    payload = {
        "schema_version": "grabber.mtc_full_transfer.v1",
        "plan_only": True,
        "execution_supported": False,
        "execution_block_reason": "PLAN_ONLY_FULL_TRANSFER",
        "mode": "full_transfer",
        "scenario_id": "full-workflow-smoke",
        "arm_id": "right_arm",
        "grasp_candidate_id": "historical_grasp",
        "joint_units": "degrees",
        "joint_names": list(EXPECTED_JOINTS),
        "points": points,
        "phase_boundaries": [
            {"name": "pregrasp", "start_index": 0, "end_index": 1},
            {"name": "approach", "start_index": 1, "end_index": 3},
            {"name": "attach", "start_index": 3, "end_index": 3},
            {"name": "source_retreat", "start_index": 3, "end_index": 5},
            {"name": "platform_lower", "start_index": 5, "end_index": 6},
            {"name": "transport", "start_index": 6, "end_index": 7},
            {"name": "place", "start_index": 7, "end_index": 9},
            {"name": "release", "start_index": 9, "end_index": 9},
            {"name": "target_retreat", "start_index": 9, "end_index": 11},
        ],
        "gripper_events": [
            {"name": "open_before_motion", "point_index": 0},
            {"name": "close_at_attach", "point_index": 3},
            {"name": "open_at_release", "point_index": 9},
        ],
    }
    validate_full_transfer_trajectory(payload)
    bad = deepcopy(payload)
    bad["arm_id"] = "left_arm"
    with pytest.raises(SafetyAbort, match="arm_id"):
        validate_full_transfer_trajectory(bad)
    bad = deepcopy(payload)
    bad["scenario_id"] = ""
    with pytest.raises(SafetyAbort, match="scenario_id"):
        validate_full_transfer_trajectory(bad)
    path = tmp_path / "full.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    replay = load_replay(path)
    assert replay.attach_time == pytest.approx(0.03)
    assert replay.release_time == pytest.approx(0.09)

    if not ASSETS.is_dir():
        pytest.skip("exact installed robot meshes are an untracked local cache")
    model = mujoco.MjModel.from_xml_string(
        build_full_model_xml(load_scenario(FULL_SCENARIO))
    )
    assert model.nmesh == 40
    expected_axes = (
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.00015298],
        [0.0, 0.0, 1.0],
    )
    for name, expected in zip(EXPECTED_JOINTS, expected_axes):
        expected = np.asarray(expected)
        assert np.allclose(
            model.joint(name).axis, expected / np.linalg.norm(expected)
        )
    assert model.camera("head_rgbd").id >= 0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera_rotation = data.cam_xmat[
        model.camera("head_rgbd").id
    ].reshape(3, 3)
    assert camera_rotation[:, 1] @ [0.0, 0.0, 1.0] > 0.9
    assert model.body("l_link7").id >= 0
    assert model.body("r_link7").id >= 0
    assert model.geom("target_bottle_geom").id >= 0
    assert model.geom("non_target_bottle_geom_1").id >= 0
    assert model.geom("shelf_board_2").contype != 0
    assert model.geom("target_bottle_geom").contype != 0
    assert model.geom(model.body("r_link3").geomadr[0]).contype != 0
    assert model.geom(model.body("l_link3").geomadr[0]).contype != 0
    assert model.joint("r_rmg24_finger1_joint").id >= 0
    assert model.joint("r_rmg24_finger1_joint").range == pytest.approx(
        [0.0, 0.0369]
    )
    shelf_id = model.body("full_five_level_shelf").id
    assert model.body_parentid[shelf_id] == 0
    platform_address = model.jnt_qposadr[model.joint("platform_joint").id]
    data = mujoco.MjData(model)
    data.qpos[platform_address] = 0.647
    mujoco.mj_forward(model, data)
    shelf_z = data.xpos[shelf_id, 2]
    link_z = data.xpos[model.body("r_link1").id, 2]
    data.qpos[platform_address] = 0.250
    mujoco.mj_forward(model, data)
    assert data.xpos[shelf_id, 2] == pytest.approx(shelf_z)
    assert data.xpos[model.body("r_link1").id, 2] == pytest.approx(
        link_z - 0.397
    )
    scene_summary = view_scene_only(
        load_scenario(FULL_SCENARIO),
        viewer=False,
        render_path=None,
        observer_view=True,
    )
    assert scene_summary["workflow"] == "scene_only_no_trajectory"
    assert scene_summary["interactive_camera"] is True
    assert scene_summary["trajectory_loaded"] is False
    assert scene_summary["hardware_connections"] == 0

    tcp = np.eye(4)
    tcp[:3, :3] = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]
    )
    gripper = tcp.copy()
    gripper[:3, :3] = tcp[:3, :3] @ np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    bottle = np.eye(4)
    _, alignment = _validate_grasp_alignment(tcp, gripper, bottle, 0.033)
    assert alignment["center_depth_error_m"] == pytest.approx(0.0)
    bottle[:3, 3] = tcp[:3, 2] * 0.02
    with pytest.raises(RuntimeError, match="attach rejected"):
        _validate_grasp_alignment(tcp, gripper, bottle, 0.033)
    bottle[:3, 3] = 0.0
    bottle[0, 3] = 0.02
    with pytest.raises(RuntimeError, match="attach rejected"):
        _validate_grasp_alignment(tcp, gripper, bottle, 0.033)
    bottle[0, 3] = 0.0
    rotation = np.array(
        [
            [np.cos(np.radians(10)), -np.sin(np.radians(10)), 0.0],
            [np.sin(np.radians(10)), np.cos(np.radians(10)), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tcp[:3, :3] = rotation @ tcp[:3, :3]
    gripper[:3, :3] = rotation @ gripper[:3, :3]
    with pytest.raises(RuntimeError, match="not square"):
        _validate_grasp_alignment(tcp, gripper, bottle, 0.033)
    vertical_fingers = gripper.copy()
    vertical_fingers[:3, 0] = [0.0, 0.0, 1.0]
    with pytest.raises(RuntimeError, match="finger_axis"):
        _validate_grasp_alignment(tcp, vertical_fingers, bottle, 0.033)
    assert _finger_contact_position(0.033) == pytest.approx(0.0347)
    with pytest.raises(RuntimeError, match="opening range"):
        _finger_contact_position(0.05)

    scenario = load_scenario(FULL_SCENARIO)
    tcp = np.eye(4)
    tcp[:3, 3] = PLATFORM_ORIGIN + np.asarray(
        scenario["target_place_pose"]["xyz"]
    )
    assert _validate_release_alignment(tcp, scenario) == pytest.approx(0.0)
    tcp[0, 3] += 0.02
    with pytest.raises(RuntimeError, match="release rejected"):
        _validate_release_alignment(tcp, scenario)


def test_bilateral_finger_contact_gate():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="target_bottle">
              <joint name="target_shift" type="slide" axis="0 0 1"/>
              <geom name="target_bottle_geom" type="cylinder"
                    size="0.033 0.1" contype="1" conaffinity="1"/>
            </body>
            <body name="r_rmg24_finger1" pos="-0.0349 0 0">
              <joint name="finger1_shift" type="slide" axis="-1 0 0"/>
              <geom name="r_rmg24_finger1_geom" type="box"
                    size="0.002 0.02 0.02" contype="1" conaffinity="1"/>
            </body>
            <body name="r_rmg24_finger2" pos="0.0349 0 0">
              <joint name="finger2_shift" type="slide" axis="1 0 0"/>
              <geom name="r_rmg24_finger2_geom" type="box"
                    size="0.002 0.02 0.02" contype="1" conaffinity="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert _validate_bilateral_finger_contacts(model, data, "r") == 2
    with pytest.raises(RuntimeError, match="unintended contact"):
        _reject_robot_and_shelf_contacts(
            model, data, attached=False, released=False
        )
    _reject_robot_and_shelf_contacts(
        model,
        data,
        attached=False,
        released=False,
        allow_final_grasp_contact=True,
    )
    data.qpos[1] = 0.02
    mujoco.mj_forward(model, data)
    with pytest.raises(RuntimeError, match="bilateral"):
        _validate_bilateral_finger_contacts(model, data, "r")

    data.qpos[:] = 0.0
    model.body_pos[model.body("r_rmg24_finger1").id, 0] = -0.034
    model.body_pos[model.body("r_rmg24_finger2").id, 0] = 0.034
    mujoco.mj_forward(model, data)
    with pytest.raises(RuntimeError, match="penetrates"):
        _validate_bilateral_finger_contacts(model, data, "r")


def test_non_target_bottle_contact_is_always_rejected():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="non_target_bottle_1">
              <geom type="sphere" size="0.03" contype="2" conaffinity="4"/>
            </body>
            <body name="r_link3" pos="0.04 0 0">
              <freejoint/>
              <geom type="sphere" size="0.03" contype="4" conaffinity="2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with pytest.raises(RuntimeError, match="non-target bottle"):
        _reject_non_target_contacts(model, data)


def test_shelf_and_robot_self_contacts_are_rejected():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="full_five_level_shelf">
              <geom type="box" size="0.05 0.05 0.05"
                    contype="2" conaffinity="1"/>
            </body>
            <body name="r_link3" pos="0.04 0 0">
              <freejoint/>
              <geom type="sphere" size="0.03"
                    contype="1" conaffinity="2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with pytest.raises(RuntimeError, match="contacts shelf"):
        _reject_robot_and_shelf_contacts(
            model, data, attached=False, released=False
        )

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1"
                  contype="2" conaffinity="1"/>
            <body name="r_link3" pos="0 0 0.02">
              <freejoint/>
              <geom type="sphere" size="0.03"
                    contype="1" conaffinity="2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with pytest.raises(RuntimeError, match="contacts floor"):
        _reject_robot_and_shelf_contacts(
            model, data, attached=False, released=False
        )

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="full_five_level_shelf">
              <geom name="shelf_back_visual" type="box"
                    size="0.05 0.05 0.05" contype="2" conaffinity="4"/>
            </body>
            <body name="target_bottle" pos="0.04 0 0">
              <freejoint/>
              <geom type="sphere" size="0.03"
                    contype="4" conaffinity="2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with pytest.raises(RuntimeError, match="contacts shelf"):
        _reject_robot_and_shelf_contacts(
            model,
            data,
            attached=False,
            released=False,
            source_support_geom="shelf_board_2",
        )

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="full_five_level_shelf">
              <geom name="shelf_board_2" type="box"
                    size="0.05 0.05 0.01" contype="2" conaffinity="4"/>
            </body>
            <body name="target_bottle" pos="0 0 0.0398">
              <freejoint/>
              <geom type="sphere" size="0.03"
                    contype="4" conaffinity="2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _reject_robot_and_shelf_contacts(
        model,
        data,
        attached=False,
        released=False,
        source_support_geom="shelf_board_2",
    )
    with pytest.raises(RuntimeError, match="contacts shelf"):
        _reject_robot_and_shelf_contacts(
            model,
            data,
            attached=True,
            released=False,
            allow_attached_support_contact=False,
            source_support_geom="shelf_board_2",
        )

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="r_link3">
              <geom type="sphere" size="0.03"
                    contype="1" conaffinity="1"/>
            </body>
            <body name="l_link3" pos="0.04 0 0">
              <freejoint/>
              <geom type="sphere" size="0.03"
                    contype="1" conaffinity="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with pytest.raises(RuntimeError, match="self-collision"):
        _reject_robot_and_shelf_contacts(
            model, data, attached=False, released=False
        )


def test_polyline_metrics_exposes_transport_detours():
    direct = np.array([[0.0, 0.0, 0.0], [0.06, 0.04, 0.0]])
    length, distance = _polyline_metrics(direct)
    assert length == pytest.approx(distance)
    detour = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.06, 0.04, 0.0]]
    )
    detour_length, detour_distance = _polyline_metrics(detour)
    assert detour_length > 0.9
    assert detour_distance == pytest.approx(distance)


def test_pick_only_replay_does_not_require_a_transport_phase():
    payload = {
        "schema_version": "grabber.mtc_pick.v2",
        "phase_boundaries": [
            {"name": "pregrasp", "start_index": 0, "end_index": 1},
            {"name": "approach", "start_index": 1, "end_index": 2},
            {"name": "attach", "start_index": 2, "end_index": 2},
            {"name": "retreat", "start_index": 2, "end_index": 3},
        ],
    }
    assert _validate_transport_path(None, payload, {}, [], np.zeros(7)) is None

    payload["schema_version"] = "grabber.mtc_full_transfer.v1"
    with pytest.raises(RuntimeError, match="requires a transport phase"):
        _validate_transport_path(None, payload, {}, [], np.zeros(7))


def test_platform_and_arm_motion_must_be_sequential():
    point = {
        "positions_deg": [0.0] * 7,
        "velocities_deg_s": [0.0] * 7,
        "accelerations_deg_s2": [0.0] * 7,
        "platform_height_mm": 647.0,
    }
    payload = {"points": [deepcopy(point), deepcopy(point)]}
    payload["points"][1]["platform_height_mm"] = 250.0
    assert _validate_platform_arm_interlock(payload).tolist() == [
        pytest.approx(0.647),
        pytest.approx(0.250),
    ]

    payload["points"][1]["positions_deg"][0] = 1.0
    with pytest.raises(RuntimeError, match="simultaneously"):
        _validate_platform_arm_interlock(payload)

    payload["points"][1]["positions_deg"][0] = 0.0
    payload["points"][0]["velocities_deg_s"][0] = 1.0
    with pytest.raises(RuntimeError, match="velocity"):
        _validate_platform_arm_interlock(payload)

    full = {
        "schema_version": "grabber.mtc_full_transfer.v1",
        "points": [deepcopy(point) for _ in range(5)],
        "phase_boundaries": [
            {"name": "platform_lower", "start_index": 1, "end_index": 3}
        ],
        "platform_lift_phase": {
            "start_index": 1,
            "end_index": 3,
            "source_height_mm": 647.0,
            "target_height_mm": 250.0,
            "right_arm_stationary": True,
            "left_arm_stationary": True,
        },
    }
    for item, height in zip(full["points"], [647, 647, 448.5, 250, 250]):
        item["platform_height_mm"] = height
    assert _validate_platform_arm_interlock(full).tolist() == pytest.approx(
        [0.647, 0.647, 0.4485, 0.250, 0.250]
    )

    non_monotonic = deepcopy(full)
    non_monotonic["points"][2]["platform_height_mm"] = 648
    with pytest.raises(RuntimeError, match="647->250"):
        _validate_platform_arm_interlock(non_monotonic)

    moving_acceleration = deepcopy(full)
    moving_acceleration["points"][2]["accelerations_deg_s2"][0] = 1.0
    with pytest.raises(RuntimeError, match="acceleration"):
        _validate_platform_arm_interlock(moving_acceleration)


def test_release_requires_exact_bottle_support_contact():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <geom name="board" type="box" size=".2 .2 .01"/>
          <body name="target_bottle" pos="0 0 .04">
            <freejoint/>
            <geom name="target_bottle_geom" type="cylinder" size=".03 .03"/>
          </body>
        </worldbody></mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert _validate_release_support(model, data, "board") == pytest.approx(0.0)

    data.qpos[2] = 0.0406
    mujoco.mj_forward(model, data)
    with pytest.raises(RuntimeError, match="floats"):
        _validate_release_support(model, data, "board")

    data.qpos[2] = 0.039
    mujoco.mj_forward(model, data)
    with pytest.raises(RuntimeError, match="penetrates"):
        _validate_release_support(model, data, "board")


def test_real_mtc_exports_compose_with_stationary_arm_lift():
    scene = {
        "scenario_id": "random-scene-23",
        "simulation_scene_only": True,
        "target_place_pose": {
            "xyz": [0.0, -0.62, -0.467],
            "quat_xyzw": [-0.5, 0.5, -0.5, -0.5],
        },
    }
    manifest = {
        "scenario_id": "random-scene-23",
        "coordinate_contract": {
            "visualization_reference_lift_mm": 647,
            "place_planning_lift_mm": 250,
            "place_frame_z_shift_m": 0.397,
            "target_place_xyz_at_250mm": [0.0, -0.62, -0.070],
        },
    }
    pick = _trajectory("grabber.mtc_pick.v2")
    place = _trajectory("grabber.mtc_place.v1")
    pick_scenario = {
        "scenario_id": pick["scenario_id"],
        "simulation_scene_id": scene["scenario_id"],
        "simulation_source": True,
    }
    place_scenario = {
        "scenario_id": place["scenario_id"],
        "simulation_scene_id": scene["scenario_id"],
        "simulation_source": True,
        "target_place_pose": {
            "xyz": [0.0, -0.62, -0.070],
            "quat_xyzw": [0.5, 0.5, -0.5, 0.5],
        },
    }

    replay_scene, trajectory = compose_cross_layer_replay(
        scene=scene,
        manifest=manifest,
        pick_scenario=pick_scenario,
        pick=pick,
        place_scenario=place_scenario,
        place=place,
        left_joints_deg=[0.0] * 7,
        lift_duration_s=0.2,
        lift_sample_s=0.1,
    )

    validate_full_transfer_trajectory(trajectory)
    assert replay_scene["simulation_scene_only"] is False
    assert replay_scene["target_place_pose"] == {
        "xyz": [0.0, -0.62, -0.467],
        "quat_xyzw": [0.5, 0.5, -0.5, 0.5],
    }
    assert trajectory["simulation_replay_only"] is True
    assert trajectory["cross_layer_transport"] is True
    assert [item["name"] for item in trajectory["phase_boundaries"]] == [
        "pregrasp",
        "approach",
        "attach",
        "source_retreat",
        "platform_lower",
        "transport",
        "place",
        "release",
        "target_retreat",
    ]
    lift = trajectory["platform_lift_phase"]
    lift_points = trajectory["points"][
        lift["start_index"] : lift["end_index"] + 1
    ]
    assert lift_points[0]["platform_height_mm"] == pytest.approx(647)
    assert lift_points[-1]["platform_height_mm"] == pytest.approx(250)
    assert all(
        point["positions_deg"] == pick["points"][-1]["positions_deg"]
        for point in lift_points
    )
    assert np.all(np.diff([
        point["time_from_start_s"] for point in trajectory["points"]
    ]) > 0.0)

    place["points"][0]["positions_deg"][0] += 1.0
    with pytest.raises(SafetyAbort, match="起点"):
        compose_cross_layer_replay(
            scene=scene,
            manifest=manifest,
            pick_scenario=pick_scenario,
            pick=pick,
            place_scenario=place_scenario,
            place=place,
            left_joints_deg=[0.0] * 7,
            lift_duration_s=0.2,
            lift_sample_s=0.1,
        )


def test_full_scenario_candidates_match_the_real_gripper_mount():
    scenario = load_scenario(FULL_SCENARIO)
    mount = Rotation.from_euler("z", 1.57).as_matrix()
    assert {
        item["id"] for item in scenario["source_grasp_candidates"]
    } == {"horizontal_fingers_roll_0", "horizontal_fingers_roll_180"}
    for candidate in scenario["source_grasp_candidates"]:
        tcp = Rotation.from_quat(
            candidate["pose"]["quat_xyzw"]
        ).as_matrix()
        assert tcp[:, 2] @ [0.0, -1.0, 0.0] > np.cos(np.radians(1.0))
        assert abs((tcp @ mount)[:, 0] @ [1.0, 0.0, 0.0]) > np.cos(
            np.radians(1.0)
        )


def test_full_replay_rejects_a_different_grasp_candidate(tmp_path):
    scenario = load_scenario(FULL_SCENARIO)
    path = tmp_path / "wrong_candidate.json"
    path.write_text(
        json.dumps(
            {
                "scenario_id": scenario["scenario_id"],
                "grasp_candidate_id": "primary",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="grasp candidate"):
        run_workflow(
            scenario,
            path,
            viewer=False,
            loop=False,
            speed=1.0,
            head_rgb_path=None,
            render_path=None,
            observer_view=False,
        )


def test_random_two_layer_scene_selects_blue_and_an_empty_patch():
    root = Path(__file__).resolve().parents[2]
    seed = 7
    subprocess.run(
        [sys.executable, "scripts/mujoco_random_shelf_workflow.py"],
        cwd=root,
        env={**os.environ, "GRABBER_SIM_SEED": str(seed)},
        check=True,
        capture_output=True,
        text=True,
    )
    scenario = load_scenario(
        Path(f"/tmp/grabber_lower_place_seed_{seed}.yaml")
    )
    pick_scenario = load_scenario(
        Path(f"/tmp/grabber_pick_seed_{seed}.yaml")
    )
    pick_fixture = json.loads(
        Path(f"/tmp/grabber_pick_fixture_seed_{seed}.json").read_text(
            encoding="utf-8"
        )
    )
    payload = json.loads(
        Path(f"/tmp/grabber_lower_place_seed_{seed}.json").read_text(
            encoding="utf-8"
        )
    )
    selection = payload["placement_selection"]
    assert len(selection["source_bottles_xy"]) == 3
    assert len(selection["obstacle_bottles_xy"]) == 3
    assert selection["blue_target_xy"] == selection["source_bottles_xy"][
        selection["blue_target_index"]
    ]
    assert selection["nearest_obstacle_m"] >= 0.13
    assert selection["offline_workspace_region"] == {
        "frame_id": "platform_base_link",
        "shelf_center_xy_m": [0.03, -0.72],
        "platform_heights_mm": [647, 250],
        "source_and_place_candidate_x_band_m": [-0.28, 0.01],
        "lower_obstacle_x_band_m": [-0.28, 0.28],
        "shared_y_band_m": [-0.64, -0.6],
        "shared_lateral_offsets_from_shelf_center_m": [-0.31, -0.02],
        "shared_depth_offsets_from_shelf_center_m": [0.08, 0.12],
        "robot_facing_layout_cm": {
            "bottle_center_from_left_edge": [4.0, 33.0],
            "bottle_center_from_right_edge": [37.0, 66.0],
            "bottle_center_inward_from_front_edge": [5.0, 9.0],
        },
        "continuous_region": True,
        "lateral_grasp_corridor": {
            "bottle_diameter_m": 0.066,
            "open_gripper_inner_width_m": 0.07,
            "extra_margin_m": 0.03,
            "minimum_center_separation_m": 0.1,
            "camera_projection_validated": False,
        },
        "minimum_bottle_center_clearance_m": 0.13,
    }
    for x, y in selection["source_bottles_xy"] + [selection["selected_xy"]]:
        assert -0.28 <= x <= 0.01
        assert -0.64 <= y <= -0.60
    for x, y in selection["obstacle_bottles_xy"]:
        assert -0.28 <= x <= 0.28
        assert -0.64 <= y <= -0.60
    source_x = sorted(x for x, _ in selection["source_bottles_xy"])
    assert np.min(np.diff(source_x)) >= 0.1
    lower_x = sorted(x for x, _ in selection["obstacle_bottles_xy"])
    if len(lower_x) > 1:
        assert np.min(np.diff(lower_x)) >= 0.1
    selected = np.asarray(selection["selected_xy"])
    assert all(
        np.linalg.norm(selected - np.asarray(bottle)) >= 0.13
        for bottle in selection["obstacle_bottles_xy"]
    )
    assert len(scenario["simulation_obstacle_bottles"]) == (
        2 + len(selection["obstacle_bottles_xy"])
    )
    assert scenario["local_motion_planner"] == "pilz_lin"
    assert scenario["cartesian_transport"] is False
    assert scenario["mode"] == "simulation_scene_only"
    assert scenario["source_grasp_pose"]["xyz"] == pytest.approx(
        scenario["bottle"]["pose"]["xyz"]
    )
    assert scenario["source_support_surface_id"] == "shelf_bottom"
    assert scenario["target_support_surface_id"] == "second_shelf_board"
    assert {
        item["id"] for item in scenario["shelf_boxes"]
    } >= {
        "shelf_bottom",
        "shelf_top",
        "shelf_back",
        "second_shelf_board",
        "second_shelf_back",
    }
    assert payload["schema_version"] == "grabber.mujoco_scene.v2"
    assert payload["simulation_scene_only"] is True
    assert payload["planning_required"] is True
    assert payload["trajectory"] is None
    coordinates = payload["coordinate_contract"]
    assert coordinates["visualization_reference_lift_mm"] == 647
    assert coordinates["place_planning_lift_mm"] == 250
    assert coordinates["place_frame_z_shift_m"] == pytest.approx(0.397)
    assert coordinates["target_place_xyz_at_250mm"] == pytest.approx(
        [
            selection["selected_xy"][0],
            selection["selected_xy"][1],
            -0.081,
        ]
    )
    assert coordinates["second_shelf_board_z_at_250mm"] == pytest.approx(
        -0.206
    )
    assert pick_scenario["mode"] == "pick_only"
    assert pick_scenario["scenario_id"] == f"mujoco_pick_seed_{seed}"
    assert pick_scenario["simulation_scene_id"] == scenario["scenario_id"]
    assert pick_scenario["simulation_source"] is True
    assert pick_scenario["fixture_source"] is True
    assert pick_scenario["source_lift_direction"] == [0.0, 0.0, 1.0]
    assert pick_scenario["source_lift_distance_m"] == pytest.approx(0.05)
    assert pick_scenario["tcp_path_workspace"]["id"] == "tcp_path_workspace"
    assert pick_fixture["schema_version"] == "grabber.mtc_fixture_joint_state.v1"
    assert pick_fixture["simulation_only"] is True
    assert pick_fixture["hardware_connections"] == 0
    assert pick_fixture["platform_height_mm"] == 647
    assert pick_fixture["left_joints_deg"] == [0.0] * 7
    assert pick_fixture["right_joints_deg"] == pytest.approx(
        [22.523, 115.811, -46.75, 39.085, -9.142, -12.215, -22.785]
    )

    with pytest.raises(RuntimeError, match="has no MTC trajectory"):
        run_workflow(
            scenario,
            Path(f"/tmp/grabber_lower_place_seed_{seed}.json"),
            viewer=False,
            loop=False,
            speed=1.0,
            head_rgb_path=None,
            render_path=None,
            observer_view=False,
        )


def test_pick_endpoint_builds_a_geometry_bound_place_scenario():
    root = Path(__file__).resolve().parents[2]
    seed = 271
    subprocess.run(
        [sys.executable, "scripts/mujoco_random_shelf_workflow.py"],
        cwd=root,
        env={**os.environ, "GRABBER_SIM_SEED": str(seed)},
        check=True,
        capture_output=True,
        text=True,
    )
    scene = load_scenario(Path(f"/tmp/grabber_lower_place_seed_{seed}.yaml"))
    manifest = json.loads(
        Path(f"/tmp/grabber_lower_place_seed_{seed}.json").read_text(
            encoding="utf-8"
        )
    )
    pick_scenario = load_scenario(Path(f"/tmp/grabber_pick_seed_{seed}.yaml"))
    pick = _trajectory("grabber.mtc_pick.v2")
    pick["scenario_id"] = pick_scenario["scenario_id"]
    pick["grasp_candidate_id"] = "horizontal_fingers_roll_0"

    place = build_place_scenario(
        scene=scene,
        manifest=manifest,
        pick_scenario=pick_scenario,
        pick=pick,
    )

    assert place["mode"] == "place_only"
    assert place["simulation_scene_id"] == scene["scenario_id"]
    assert place["simulation_source"] is True
    assert place["fixture_source"] is True
    assert place["target_insert_direction"] == [0.0, 0.0, -1.0]
    assert place["target_retreat_direction"] == [0.0, 0.0, 1.0]
    assert place["target_retreat_distance_m"] == pytest.approx(0.2)
    assert place["target_place_pose"]["xyz"] == pytest.approx(
        manifest["coordinate_contract"]["target_place_xyz_at_250mm"]
    )
    assert place["target_place_pose"]["quat_xyzw"] == pytest.approx(
        pick_scenario["source_grasp_candidates"][0]["pose"]["quat_xyzw"]
    )
    assert place["post_place_home_joints_deg"] == pytest.approx(
        pick["points"][-1]["positions_deg"]
    )
    shift = manifest["coordinate_contract"]["place_frame_z_shift_m"]
    assert [item["pose"]["xyz"][2] for item in place["shelf_boxes"]] == pytest.approx(
        [item["pose"]["xyz"][2] + shift for item in scene["shelf_boxes"]]
    )
    assert np.allclose(
        np.asarray(place["obstacle_voxels"])[:, 2],
        np.asarray(scene["obstacle_voxels"])[:, 2] + shift,
    )

    mismatched = deepcopy(pick_scenario)
    mismatched["bottle"]["pose"]["xyz"][0] += 0.01
    with pytest.raises(SafetyAbort, match="瓶体几何"):
        build_place_scenario(
            scene=scene,
            manifest=manifest,
            pick_scenario=mismatched,
            pick=pick,
        )
