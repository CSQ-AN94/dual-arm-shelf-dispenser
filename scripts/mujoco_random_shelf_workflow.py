#!/usr/bin/env python3
"""Generate a random two-layer scene for the MTC→MuJoCo workflow."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.delivery_table import observe_output_table
DEFAULT_SCENARIO = (
    ROOT
    / "mtc_ws/src/grabber_mtc_planner/scenarios/mujoco_shelf_workflow.yaml"
)


seed = int(os.environ.get("GRABBER_SIM_SEED", "7"))
rng = np.random.default_rng(seed)
WORKSPACE_X_BAND_M = (-0.28, 0.01)
THREE_OBSTACLE_X_BAND_M = (-0.28, 0.28)
# Bottle centres may be anywhere in this rectangle.  The front edge is
# y=-0.55 m, so this is 5--9 cm inward: radius + edge margin at the near side,
# without the needless reach/occlusion of the old 14.8--15.8 cm strip.
WORKSPACE_Y_BAND_M = (-0.64, -0.60)
BOTTLE_DIAMETER_M = 0.066
OPEN_GRIPPER_INNER_WIDTH_M = 0.070
GRASP_LANE_MARGIN_M = 0.030
MIN_LATERAL_CENTER_SEPARATION_M = (
    max(BOTTLE_DIAMETER_M, OPEN_GRIPPER_INNER_WIDTH_M)
    + GRASP_LANE_MARGIN_M
)
MIN_PLACE_CLEARANCE_M = 0.13
SOURCE_LIFT_MM = 647
TARGET_LIFT_MM = 250
INITIAL_RIGHT_JOINTS_DEG = [
    22.523,
    115.811,
    -46.75,
    39.085,
    -9.142,
    -12.215,
    -22.785,
]
PLACE_FRAME_Z_SHIFT_M = (SOURCE_LIFT_MM - TARGET_LIFT_MM) / 1000.0
SHELF_CENTER_XY_M = (0.03, -0.72)
SHELF_LEFT_X_M = SHELF_CENTER_XY_M[0] - 0.35
SHELF_RIGHT_X_M = SHELF_CENTER_XY_M[0] + 0.35
SHELF_FRONT_Y_M = SHELF_CENTER_XY_M[1] + 0.17


def sample_laterally_separated_bottles(
    count: int,
    *,
    x_band: tuple[float, float] = WORKSPACE_X_BAND_M,
) -> np.ndarray:
    """Sample a continuous front row with disjoint lateral grasp corridors."""
    for _ in range(10_000):
        bottles = rng.uniform(
            [x_band[0], WORKSPACE_Y_BAND_M[0]],
            [x_band[1], WORKSPACE_Y_BAND_M[1]],
            size=(count, 2),
        )
        lateral = np.sort(bottles[:, 0])
        if (
            count < 2
            or np.min(np.diff(lateral))
            >= MIN_LATERAL_CENTER_SEPARATION_M
        ):
            return bottles
    raise RuntimeError(
        "failed to sample continuous bottle positions without depth overlap"
    )


scenario_path = Path(f"/tmp/grabber_lower_place_seed_{seed}.yaml")
pick_scenario_path = Path(f"/tmp/grabber_pick_seed_{seed}.yaml")
pick_fixture_path = Path(f"/tmp/grabber_pick_fixture_seed_{seed}.json")
output = Path(f"/tmp/grabber_lower_place_seed_{seed}.json")

# Reuse the real empty-patch selector with a synthetic lower-shelf point cloud.
surface_z = -0.583
support_x, support_y = np.meshgrid(
    np.linspace(-0.30, 0.36, 67), np.linspace(-0.88, -0.56, 33)
)
support = np.column_stack(
    (
        support_x.ravel(),
        support_y.ravel(),
        np.full(support_x.size, surface_z),
    )
)
lower_bottle_count = int(rng.integers(1, 4))
lower_x_band = (
    THREE_OBSTACLE_X_BAND_M
    if lower_bottle_count == 3
    else WORKSPACE_X_BAND_M
)
for _ in range(10_000):
    bottles = sample_laterally_separated_bottles(
        lower_bottle_count, x_band=lower_x_band
    )
    candidate_x = np.arange(
        WORKSPACE_X_BAND_M[0], WORKSPACE_X_BAND_M[1] + 1e-9, 0.01
    )
    candidate_y = np.arange(
        WORKSPACE_Y_BAND_M[0], WORKSPACE_Y_BAND_M[1] + 1e-9, 0.01
    )
    if any(
        all(
            np.linalg.norm(np.asarray([x, y]) - bottle)
            >= MIN_PLACE_CLEARANCE_M
            for bottle in bottles
        )
        for x in candidate_x
        for y in candidate_y
    ):
        break
else:
    raise RuntimeError("failed to sample a lower shelf with a free patch")
columns = np.vstack(
    [
        np.column_stack(
            (
                np.full(20, xy[0]),
                np.full(20, xy[1]),
                np.linspace(surface_z + 0.03, surface_z + 0.21, 20),
            )
        )
        for xy in bottles
    ]
)
place_edge_m = 0.04
place_config = SimpleNamespace(
    # Apply the same conservative horizontal workspace to every aligned layer.
    table_roi_min=(
        WORKSPACE_X_BAND_M[0] - place_edge_m,
        WORKSPACE_Y_BAND_M[0] - place_edge_m,
        -0.60,
    ),
    table_roi_max=(
        WORKSPACE_X_BAND_M[1] + place_edge_m,
        WORKSPACE_Y_BAND_M[1] + place_edge_m,
        -0.34,
    ),
    table_height_bin_m=0.01,
    table_inlier_band_m=0.012,
    table_min_inliers=150,
    table_frame_agreement_m=0.012,
    table_edge_margin_m=place_edge_m,
    table_support_radius_m=0.06,
    table_min_patch_points=8,
    place_clearance_radius_m=MIN_PLACE_CLEARANCE_M,
    place_grid_m=0.01,
    obstacle_min_height_m=0.025,
    obstacle_max_height_m=0.30,
    max_place_candidates=8,
)
observation = observe_output_table(
    [np.vstack((support, columns))] * 3, place_config
)
target_x, target_y = observation.best.xy_base
target_z = surface_z + 0.105

# All third-layer bottles are sampled from the same continuous rectangle. One
# is selected at random as the blue semantic target.
source_bottles = sample_laterally_separated_bottles(3)
blue_index = int(rng.integers(len(source_bottles)))
blue_xy = source_bottles[blue_index]
source_center_z = -0.088398
source_tcp_xyz = np.asarray(
    [blue_xy[0], blue_xy[1], source_center_z]
)

scenario = yaml.safe_load(DEFAULT_SCENARIO.read_text(encoding="utf-8"))
scenario["scenario_id"] = f"mujoco_lower_place_seed_{seed}"
# This combined scene is an input bundle for the future pick/lift/place
# orchestrator, not a physically valid single MTC task: the platform height
# changes between source and target layers.
scenario["mode"] = "simulation_scene_only"
scenario["simulation_scene_only"] = True
scenario["source_grasp_pose"]["xyz"] = source_tcp_xyz.tolist()
for candidate in scenario["source_grasp_candidates"]:
    candidate["pose"]["xyz"] = source_tcp_xyz.tolist()
scenario["bottle"]["pose"]["xyz"] = [
    float(blue_xy[0]),
    float(blue_xy[1]),
    source_center_z,
]
scenario["target_layer_id"] = "simulated_second_shelf_empty_patch"
scenario["target_support_surface_id"] = "second_shelf_board"
scenario["target_place_pose"]["xyz"] = [target_x, target_y, target_z]
scenario["target_preplace_offset_m"] = 0.20
scenario["cartesian_transport"] = False
scenario["simulation_obstacle_bottles"] = [
    {"xyz": [float(x), float(y), source_center_z]}
    for index, (x, y) in enumerate(source_bottles)
    if index != blue_index
] + [
    {"xyz": [float(x), float(y), surface_z + 0.105]}
    for x, y in bottles
]
scenario["shelf_boxes"] = [
    item
    for item in scenario["shelf_boxes"]
    if not item["id"].startswith("unknown_")
] + [
    {
        "id": "second_shelf_board",
        "size": [0.700, 0.340, 0.040],
        "pose": {"xyz": [0.030, -0.720, -0.603], "rpy_deg": [0, 0, 0]},
    },
    {
        "id": "second_shelf_back",
        "size": [0.700, 0.030, 0.390],
        "pose": {"xyz": [0.030, -0.875, -0.408], "rpy_deg": [0, 0, 0]},
    },
]
scenario["obstacle_voxels"] = [
    [float(x), float(y), float(z)]
    for index, (x, y) in enumerate(source_bottles)
    if index != blue_index
    for z in np.arange(-0.168, -0.017, 0.05)
] + [
    [float(x), float(y), float(z)]
    for x, y in bottles
    for z in np.arange(surface_z + 0.025, surface_z + 0.211, 0.05)
]
scenario_path.write_text(
    yaml.safe_dump(scenario, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)

stamp = datetime.now(timezone.utc).isoformat()
pick_scenario = deepcopy(scenario)
pick_scenario.update(
    {
        "scenario_id": f"mujoco_pick_seed_{seed}",
        "simulation_scene_id": scenario["scenario_id"],
        "simulation_source": True,
        "simulation_scene_only": False,
        "mode": "pick_only",
        "target_captured_at_utc": stamp,
        "scene_captured_at_utc": stamp,
        "freshness_max_age_s": 3600.0,
        "source_lift_direction": [0.0, 0.0, 1.0],
        "source_lift_distance_m": 0.05,
        "source_pregrasp_offset_m": 0.148,
        "source_contact_distance_m": 0.135,
        "source_retreat_distance_m": 0.15,
        "tcp_path_workspace": {
            "id": "tcp_path_workspace",
            "size": [1.19, 1.24, 1.34],
            "pose": {
                "xyz": [-0.037292, -0.447784, -0.041684],
                "rpy_deg": [0.0, 0.0, 0.0],
            },
        },
    }
)
pick_scenario_path.write_text(
    yaml.safe_dump(pick_scenario, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
pick_fixture_path.write_text(
    json.dumps(
        {
            "schema_version": "grabber.mtc_fixture_joint_state.v1",
            "simulation_only": True,
            "hardware_connections": 0,
            "pick_scenario_id": pick_scenario["scenario_id"],
            "platform_height_mm": float(SOURCE_LIFT_MM),
            "head_joints_rad": [0.0, 0.0],
            "left_joints_deg": [0.0] * 7,
            "right_joints_deg": INITIAL_RIGHT_JOINTS_DEG,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

placement_selection = {
    "seed": seed,
    "source_bottles_xy": source_bottles.tolist(),
    "blue_target_index": blue_index,
    "blue_target_xy": blue_xy.tolist(),
    "obstacle_bottles_xy": bottles.tolist(),
    "selected_xy": [target_x, target_y],
    "nearest_obstacle_m": observation.best.nearest_obstacle_m,
    "offline_workspace_region": {
        "frame_id": "platform_base_link",
        "shelf_center_xy_m": list(SHELF_CENTER_XY_M),
        "platform_heights_mm": [SOURCE_LIFT_MM, TARGET_LIFT_MM],
        "source_and_place_candidate_x_band_m": list(WORKSPACE_X_BAND_M),
        "lower_obstacle_x_band_m": list(lower_x_band),
        "shared_y_band_m": list(WORKSPACE_Y_BAND_M),
        "shared_lateral_offsets_from_shelf_center_m": [
            round(WORKSPACE_X_BAND_M[0] - SHELF_CENTER_XY_M[0], 3),
            round(WORKSPACE_X_BAND_M[1] - SHELF_CENTER_XY_M[0], 3),
        ],
        "shared_depth_offsets_from_shelf_center_m": [
            round(WORKSPACE_Y_BAND_M[0] - SHELF_CENTER_XY_M[1], 3),
            round(WORKSPACE_Y_BAND_M[1] - SHELF_CENTER_XY_M[1], 3),
        ],
        "robot_facing_layout_cm": {
            "bottle_center_from_left_edge": [
                round(100 * (WORKSPACE_X_BAND_M[0] - SHELF_LEFT_X_M), 1),
                round(100 * (WORKSPACE_X_BAND_M[1] - SHELF_LEFT_X_M), 1),
            ],
            "bottle_center_from_right_edge": [
                round(100 * (SHELF_RIGHT_X_M - WORKSPACE_X_BAND_M[1]), 1),
                round(100 * (SHELF_RIGHT_X_M - WORKSPACE_X_BAND_M[0]), 1),
            ],
            "bottle_center_inward_from_front_edge": [
                round(100 * (SHELF_FRONT_Y_M - WORKSPACE_Y_BAND_M[1]), 1),
                round(100 * (SHELF_FRONT_Y_M - WORKSPACE_Y_BAND_M[0]), 1),
            ],
        },
        "continuous_region": True,
        "lateral_grasp_corridor": {
            "bottle_diameter_m": BOTTLE_DIAMETER_M,
            "open_gripper_inner_width_m": OPEN_GRIPPER_INNER_WIDTH_M,
            "extra_margin_m": GRASP_LANE_MARGIN_M,
            "minimum_center_separation_m": (
                MIN_LATERAL_CENTER_SEPARATION_M
            ),
            "camera_projection_validated": False,
        },
        "minimum_bottle_center_clearance_m": MIN_PLACE_CLEARANCE_M,
    },
}
manifest = {
    "schema_version": "grabber.mujoco_scene.v2",
    "scenario_id": scenario["scenario_id"],
    "simulation_scene_only": True,
    "planning_required": True,
    "trajectory": None,
    "blocked_reason": (
        "MTC_PICK_AND_PLACE_TRAJECTORIES_REQUIRED; "
        "cross-layer IK trajectory synthesis is forbidden"
    ),
    "coordinate_contract": {
        "visualization_frame_id": "platform_base_link",
        "visualization_reference_lift_mm": SOURCE_LIFT_MM,
        "place_planning_frame_id": "platform_base_link",
        "place_planning_lift_mm": TARGET_LIFT_MM,
        "place_frame_z_shift_m": PLACE_FRAME_Z_SHIFT_M,
        "target_place_xyz_at_250mm": [
            target_x,
            target_y,
            target_z + PLACE_FRAME_Z_SHIFT_M,
        ],
        "second_shelf_board_z_at_250mm": -0.603
        + PLACE_FRAME_Z_SHIFT_M,
        "second_shelf_back_z_at_250mm": -0.408
        + PLACE_FRAME_Z_SHIFT_M,
        "lower_obstacle_bottles_xyz_at_250mm": [
            [float(x), float(y), surface_z + 0.105 + PLACE_FRAME_Z_SHIFT_M]
            for x, y in bottles
        ],
    },
    "placement_selection": placement_selection,
}
output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "scenario": str(scenario_path),
            "pick_scenario": str(pick_scenario_path),
            "pick_fixture_state": str(pick_fixture_path),
            "manifest": str(output),
            "source_bottles_xy": source_bottles.tolist(),
            "blue_target_xy": blue_xy.tolist(),
            "lower_bottles_xy": bottles.tolist(),
            "selected_empty_xy": [target_x, target_y],
            "clearance_m": observation.best.nearest_obstacle_m,
            "offline_workspace_region": placement_selection[
                "offline_workspace_region"
            ],
            "planning_required": True,
        },
        indent=2,
    )
)
