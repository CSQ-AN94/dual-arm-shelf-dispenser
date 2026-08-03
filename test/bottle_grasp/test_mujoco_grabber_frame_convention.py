"""Regression for the 2026-08-03 reversed MuJoCo pick preview."""

import mujoco
import numpy as np

from scripts.mujoco_grabber_sim import build_model_xml


def test_real_pick_attach_state_reaches_the_scenario_bottle():
    bottle = np.array([-0.0746825486, -0.7203586841, -0.1071678319])
    scenario = {
        "bottle": {
            "pose": {"xyz": bottle.tolist()},
            "radius_m": 0.033,
            "height_m": 0.21,
        },
        "shelf_boxes": [],
    }
    joints_deg = [
        75.4445663,
        125.205665,
        174.309907,
        -17.1672889,
        164.217547,
        -56.8002005,
        149.044396,
    ]
    model = mujoco.MjModel.from_xml_string(build_model_xml(scenario))
    data = mujoco.MjData(model)
    for name, value in zip(
        [f"r_joint{index}" for index in range(1, 8)], joints_deg
    ):
        data.qpos[model.joint(name).qposadr[0]] = np.radians(value)
    mujoco.mj_forward(model, data)

    tcp_in_platform = data.site("tcp").xpos - [0.0, 0.0, 1.0]
    assert np.linalg.norm(tcp_in_platform - bottle) < 0.001
