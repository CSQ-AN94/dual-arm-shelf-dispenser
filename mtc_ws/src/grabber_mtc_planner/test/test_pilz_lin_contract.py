from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
SCENARIO_HPP = (PKG / "src" / "scenario.hpp").read_text(encoding="utf-8")
SCENARIO_CPP = (PKG / "src" / "scenario.cpp").read_text(encoding="utf-8")
PLANNER_CPP = (PKG / "src" / "plan_shelf_transfer.cpp").read_text(encoding="utf-8")
LAUNCH = (
    PKG / "launch" / "plan_shelf_transfer_experimental.launch.py"
).read_text(encoding="utf-8")
OLD_SCENARIO = (
    PKG / "scenarios" / "shelf_transfer_fixture.yaml"
).read_text(encoding="utf-8")
PACKAGE_XML = (PKG / "package.xml").read_text(encoding="utf-8")


def test_cartesian_remains_the_compatible_default_and_invalid_values_fail():
    assert "local_motion_planner:" not in OLD_SCENARIO
    assert 'local_motion_planner{ "cartesian" }' in SCENARIO_HPP
    assert 'root["local_motion_planner"].as<std::string>' in SCENARIO_CPP
    assert 's.local_motion_planner != "cartesian"' in SCENARIO_CPP
    assert 's.local_motion_planner != "pilz_lin"' in SCENARIO_CPP
    assert "local_motion_planner must be" in SCENARIO_CPP


def test_pilz_lin_is_local_while_connect_keeps_ompl_sampling():
    assert "PipelinePlanner>(node);" in PLANNER_CPP
    assert "sampling->setPlannerId(scenario.planner_id);" in PLANNER_CPP
    assert 'node, "pilz_industrial_motion_planner");' in PLANNER_CPP
    assert 'pilz_lin->setPlannerId("LIN");' in PLANNER_CPP
    assert "GroupPlannerVector{ { arm.planning_group, sampling } }" in PLANNER_CPP
    assert PLANNER_CPP.count("sampling, local_motion") == 3


def test_launch_supplies_pilz_pipeline_and_cartesian_limits():
    assert '"planning_plugin": "pilz_industrial_motion_planner/CommandPlanner"' in LAUNCH
    assert '"max_trans_vel":' in LAUNCH
    assert '"max_trans_acc":' in LAUNCH
    assert '"max_trans_dec":' in LAUNCH
    assert '"max_rot_vel":' in LAUNCH
    assert "pilz_pipeline," in LAUNCH
    assert "<exec_depend>pilz_industrial_motion_planner</exec_depend>" in PACKAGE_XML


def test_plan_only_robot_model_covers_controller_limit_discrepancy():
    assert "JOINT_POSITION_MARGIN_RAD = math.radians(3.5)" in LAUNCH
    assert "controller reports J3's hard upper bound as 178.00 deg" in LAUNCH
    assert 'name.startswith(("l_joint", "r_joint"))' in LAUNCH
    assert '"has_position_limits": True' in LAUNCH
    assert '"min_position": lower' in LAUNCH
    assert '"max_position": upper' in LAUNCH


def test_trajectory_export_keeps_timing_and_velocity_contract():
    assert "time_from_start_s" in SCENARIO_CPP
    assert "velocities_deg_s" in SCENARIO_CPP
    assert "accelerations_deg_s2" in SCENARIO_CPP
    assert "out.points.back().time_from_start_s" in PLANNER_CPP
    assert "offset + durationSeconds(point.time_from_start)" in PLANNER_CPP
    assert "point.accelerations" in PLANNER_CPP
    assert "validateTrajectoryTiming(out.points" in PLANNER_CPP
