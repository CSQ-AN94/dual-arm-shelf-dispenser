#!/usr/bin/env python3
"""Offline safety-contract checks for grabber_mtc_planner.

These cannot prove planning success — that needs a ROS 2 Humble build and a
model-backed plan-only run.  They fail cheaply if the plan-only boundary,
branch-local ACM/object lifecycle, whole-branch result semantics, full rigid
tool transform, or experimental-only KDL override regresses.

Run: python3 test/test_plan_only_contract.py   (or pytest)
"""

from __future__ import annotations

import math
import pathlib

import yaml

PKG = pathlib.Path(__file__).resolve().parent.parent
PLANNER_CPP = (PKG / "src" / "plan_shelf_transfer.cpp").read_text(encoding="utf-8")
# Comments talk *about* the things the code must not call.
PLANNER_CODE = "\n".join(
    line.split("//", 1)[0] for line in PLANNER_CPP.splitlines()
)
SCENARIO = yaml.safe_load((PKG / "scenarios" / "shelf_transfer_fixture.yaml").read_text(encoding="utf-8"))
TRACE_SCENARIO = yaml.safe_load(
    (PKG / "scenarios" / "right_arm_placeback_trace.yaml").read_text(encoding="utf-8")
)
ARMS = yaml.safe_load((PKG / "config" / "dual_rm75_arms.yaml").read_text(encoding="utf-8"))["arms"]
LAUNCHES = list((PKG / "launch").glob("*.launch.py"))
DIRECT_ENTRY = (
    PKG.parents[2] / "scripts" / "capture_mtc_direct_pick_scene.py"
).read_text(encoding="utf-8")
PLACE_CAPTURE = (
    PKG.parents[2] / "scripts" / "capture_empty_shelf_places.py"
).read_text(encoding="utf-8")
PLACE_CONVERTER = (
    PKG.parents[2] / "scripts" / "empty_shelf_places_to_mtc_scenario.py"
).read_text(encoding="utf-8")
CROSS_LAYER_RUNNER = (
    PKG.parents[2] / "scripts" / "run_mtc_cross_layer_workflow.py"
).read_text(encoding="utf-8")

FORBIDDEN = [
    "task.execute",
    "->execute(",
    "ExecuteTaskSolution",
    "FollowJointTrajectory",
    "rm_api",  # RealMan SDK
    "RM_API",
    "Movej",
    "Movel",
    "MoveJ_Cmd",
    "MoveL_Cmd",
    "set_gripper",
]


def test_no_execution_path():
    for symbol in FORBIDDEN:
        assert symbol not in PLANNER_CODE, f"plan-only node must not reference {symbol!r}"
    assert "--plan-only is required" in PLANNER_CPP


def test_complete_branch_defines_solved():
    # solved comes from the branch container's solutions, after target retreat
    # and collision-policy restoration -- a source-only grasp cannot satisfy it.
    assert "result.complete_solution_count = branch.solutions().size();" in PLANNER_CPP
    assert "result.solved = result.complete_solution_count > 0;" in PLANNER_CPP
    order = [
        "source_pregrasp_ik",
        "source_approach",
        "allow_final_grasp_contact",
        "source_contact",
        "attach_bottle",
        "source_lift",
        "forbid_support_contact_after_lift",
        "source_retreat",
        "transport",
        "target_insert",
        "target_place_ik",
        "detach_bottle",
        "target_retreat",
        "restore_bottle_collision_check",
    ]
    duplicated_by_place = {
        "target_place_ik",
        "detach_bottle",
        "target_retreat",
        "restore_bottle_collision_check",
    }
    positions = [
        (
            PLANNER_CPP.rindex(f'p + "{name}"')
            if name in duplicated_by_place
            else PLANNER_CPP.index(f'p + "{name}"')
        )
        for name in order
    ]
    assert positions == sorted(positions), "branch stages are out of task order"


def test_acm_and_object_lifecycle_are_branch_local():
    assert "allow_gripper_bottle_contact" not in PLANNER_CODE
    assert 'p + "allow_support_contact"' in PLANNER_CODE
    assert 'p + "allow_final_grasp_contact"' in PLANNER_CODE
    assert "stage->allowCollisions(s.bottle_id, arm.touch_links, true);" in PLANNER_CODE
    assert "stage->allowCollisions(s.bottle_id, arm.touch_links, false);" in PLANNER_CODE
    assert (
        PLANNER_CODE.count(
            "stage->allowCollisions(s.bottle_id, arm.touch_links, false);"
        )
        >= 3
    )
    for surface in ("source_support_surface_id", "target_support_surface_id"):
        assert f"stage->allowCollisions(s.bottle_id, s.{surface}, true);" in PLANNER_CODE
        assert f"stage->allowCollisions(s.bottle_id, s.{surface}, false);" in PLANNER_CODE
    detach_block = PLANNER_CODE[
        PLANNER_CODE.rindex('p + "detach_bottle"'):
        PLANNER_CODE.rindex('p + "target_retreat"')
    ]
    assert "detachObject(" in detach_block
    assert "allowCollisions(" not in detach_block

    order = [
        "allow_support_contact",
        "connect_to_source_pregrasp",
        "source_pregrasp_ik",
        "source_approach",
        "allow_final_grasp_contact",
        "source_contact",
        "attach_bottle",
        "source_lift",
        "forbid_support_contact_after_lift",
        "source_retreat",
        "transport",
        "target_insert",
        "target_place_ik",
        "open_gripper_semantic",
        "detach_bottle",
        "target_retreat",
        "restore_bottle_collision_check",
    ]
    duplicated_by_place = {
        "open_gripper_semantic",
        "detach_bottle",
        "target_retreat",
        "restore_bottle_collision_check",
    }
    positions = [
        (
            PLANNER_CODE.rindex(f'p + "{name}"')
            if name in duplicated_by_place
            else PLANNER_CODE.index(f'p + "{name}"')
        )
        for name in order
    ]
    assert positions == sorted(positions), "ACM/object lifecycle is out of order"
    assert "stage->attachObject(s.bottle_id, arm.ik_link);" in PLANNER_CODE
    assert "stage->detachObject(s.bottle_id, arm.ik_link);" in PLANNER_CODE
    assert "stage->setCallback(" in PLANNER_CODE
    assert "scene->getAttachedCollisionObjectMsg(attached, object_id)" in PLANNER_CODE
    assert "attached.touch_links = touch_links;" in PLANNER_CODE
    assert "scene->processAttachedCollisionObjectMsg(attached)" in PLANNER_CODE


def test_pregrasp_ik_is_collision_checked_before_touch_is_allowed():
    pregrasp_ik = PLANNER_CODE.index('p + "source_pregrasp_ik"')
    allow_touch = PLANNER_CODE.index('p + "allow_final_grasp_contact"')
    assert pregrasp_ik < allow_touch
    assert 'p + "allow_final_grasp_contact"' in PLANNER_CODE
    assert "generator->setMonitoredStage(support_scene);" in PLANNER_CODE
    assert "setIgnoreCollisions(true)" not in PLANNER_CODE
    assert "PredicateFilter" not in PLANNER_CODE


def test_pick_only_stops_after_retreat_and_exports_a_blocked_contract():
    pregrasp_ik = PLANNER_CODE.index('p + "source_pregrasp_ik"')
    source_approach = PLANNER_CODE.index('p + "source_approach"')
    allow_touch = PLANNER_CODE.index('p + "allow_final_grasp_contact"')
    source_contact = PLANNER_CODE.index('p + "source_contact"')
    attach = PLANNER_CODE.index('p + "attach_bottle"')
    lift = PLANNER_CODE.index('p + "source_lift"')
    forbid_support = PLANNER_CODE.index(
        'p + "forbid_support_contact_after_lift"'
    )
    retreat = PLANNER_CODE.index('p + "source_retreat"')
    assert (
        pregrasp_ik
        < source_approach
        < allow_touch
        < source_contact
        < attach
        < lift
        < forbid_support
        < retreat
    )
    pick_restore = PLANNER_CODE.index('p + "restore_bottle_collision_check"')
    detach = PLANNER_CODE.rindex('p + "detach_bottle"')
    place_restore = PLANNER_CODE.rindex('p + "restore_bottle_collision_check"')
    assert retreat < pick_restore < detach < place_restore
    assert PLANNER_CODE.count('p + "restore_bottle_collision_check"') == 2
    assert PLANNER_CODE.index('p + "connect_to_source_pregrasp"') < allow_touch
    assert "pick-only export expected exactly five motion segments" in PLANNER_CODE
    assert 'p + "move_to_post_pick_carry"' not in PLANNER_CODE
    connect = PLANNER_CODE.index('p + "connect_to_source_pregrasp"')
    assert connect < pregrasp_ik
    assert 'p + "move_to_source_pregrasp_staging"' not in PLANNER_CODE
    assert "generator->setMonitoredStage(support_scene);" in PLANNER_CODE
    assert "source_pregrasp_staging" not in (
        PKG / "src" / "scenario.cpp"
    ).read_text(encoding="utf-8")
    # What cost 80 s without a solution was a TCP *pose* box as a path
    # constraint: it forces OMPL into projection-based constrained sampling.
    # That stays banned. Joint constraints are a different mechanism —
    # JointConstraintSampler just narrows the sampling bounds — so the elbow
    # branch is allowed to be declared up front rather than filtered after
    # export (see test_elbow_branch_constrains_sampling_instead_of_only_
    # filtering_output).
    assert "position_constraints" not in PLANNER_CODE
    assert "orientation_constraints" not in PLANNER_CODE
    assert "tcpWorkspaceContainsTrajectory" in PLANNER_CODE
    assert "EXECUTION_DENSE_FK_JOINT_STEP_DEG = 1.5" in PLANNER_CODE
    assert "EXECUTION_MAX_JOINT_RANGE_DEG" in PLANNER_CODE
    assert "joint_travel < selected_pick_joint_travel" in PLANNER_CODE
    assert "j4_margin > selected_pick_j4_margin" in PLANNER_CODE
    lift_block = PLANNER_CODE[lift:forbid_support]
    assert (
        "setMinMaxDistance(s.source_lift_distance_m, "
        "s.source_lift_distance_m)" in lift_block
    )
    support_restore_block = PLANNER_CODE[forbid_support:retreat]
    for surface in ("source_support_surface_id", "target_support_surface_id"):
        assert (
            f"stage->allowCollisions(s.bottle_id, s.{surface}, false);"
            in support_restore_block
        )
    assert "arm.touch_links" not in support_restore_block
    assert "if (s.pick_only)" in PLANNER_CODE
    assert "return branch;" in PLANNER_CODE[
        PLANNER_CODE.index("if (s.pick_only)"):
        PLANNER_CODE.index('p + "transport"')
    ]
    assert "source_grasp_candidates" in PLANNER_CPP
    assert "writePickTrajectoryJson" in PLANNER_CODE
    assert "placeTransportTcpMetrics" in PLANNER_CODE
    assert "transport_length <= std::max(0.25, 3.0 * direct_distance)" in PLANNER_CODE
    assert "transport_length < selected_place_transport_length" in PLANNER_CODE


def test_pick_only_export_uses_the_selected_arm_joint_group():
    export_start = PLANNER_CODE.index("exportPickTrajectory(")
    export_end = PLANNER_CODE.index("exportPlaceTrajectory(")
    export_block = PLANNER_CODE[export_start:export_end]
    assert "arm_joint_names" in export_block
    assert '"r_joint' not in export_block

    branch_start = PLANNER_CODE.index("buildArmBranch(")
    branch_end = PLANNER_CODE.index("double bestCost(")
    branch_block = PLANNER_CODE[branch_start:branch_end]
    assert "arm.planning_group" in branch_block
    assert '"r_joint' not in branch_block

    assert '"--planning-arm"' in DIRECT_ENTRY
    assert "planning_arm_id=cli.planning_arm" in DIRECT_ENTRY
    assert "arm.execution_block_reason" in export_block
    left = next(arm for arm in ARMS if arm["arm_id"] == "left_arm")
    assert left["execution_block_reason"] == "LEFT_TOOL_CALIBRATION_REQUIRED"


def test_full_transfer_exports_the_complete_plan_only_sequence():
    assert "writeFullTransferTrajectoryJson" in PLANNER_CODE
    assert "full-transfer export expected seven motion segments" in PLANNER_CPP
    for phase in (
        "pregrasp",
        "approach",
        "attach",
        "source_retreat",
        "transport",
        "place",
        "release",
        "target_retreat",
    ):
        assert f'{{ "{phase}",' in PLANNER_CODE
    assert 'run.execution_block_reason = "PLAN_ONLY_FULL_TRANSFER";' in PLANNER_CODE
    assert (
        "!scenario.planning_arm_id.empty()" in PLANNER_CODE
    ), "full transfer must optionally constrain planning to one arm"
    assert "EXTERNAL_GATED_PICK_EXECUTOR_REQUIRED" in PLANNER_CPP
    assert '"pregrasp"' in PLANNER_CPP
    assert '"approach"' in PLANNER_CPP
    assert '"attach"' in PLANNER_CPP
    assert '"retreat"' in PLANNER_CPP


def test_pick_only_export_rejects_j4_singularity_before_selection():
    assert "EXECUTION_J4_SINGULARITY_DEG = 8.0" in PLANNER_CODE
    assert "std::abs(j4) < EXECUTION_J4_SINGULARITY_DEG" in PLANNER_CODE
    assert "previous_j4 * j4 < 0.0" in PLANNER_CODE
    assert "for (std::size_t i = 0; i < planned_branches.size(); ++i)" in PLANNER_CODE
    assert "selected_pick_solution = candidate.get();" in PLANNER_CODE
    assert "#execution_safe" in PLANNER_CODE


def test_dynamic_scene_object_is_never_in_the_target_acm():
    assert "scenario.obstacle_voxels" in (
        PKG / "src" / "scenario.cpp"
    ).read_text(encoding="utf-8")
    assert "dynamic_obstacle_id" not in "\n".join(
        line for line in PLANNER_CODE.splitlines() if "allowCollisions(" in line
    )


def test_place_only_has_late_support_contact_and_restores_the_acm():
    order = [
        "attach_held_bottle",
        "transport_to_target_preplace",
        "target_preplace_ik",
        "target_approach",
        "allow_final_support_contact",
        "target_contact",
        "open_gripper_semantic",
        "detach_bottle",
        "target_retreat",
        "restore_support_collision_check",
        "move_to_post_place_home",
    ]
    positions = [PLANNER_CODE.index(f'p + "{name}"') for name in order]
    assert positions == sorted(positions)
    before_contact = PLANNER_CODE[
        positions[0]:positions[4]
    ]
    assert "target_support_surface_id, true" not in before_contact
    restore = PLANNER_CODE[positions[-2]:positions[-1]]
    assert "target_support_surface_id, false" in restore
    assert "arm.touch_links, false" in restore
    assert "EXTERNAL_GATED_PLACE_EXECUTOR_REQUIRED" in PLANNER_CPP
    assert "place-only segment contains duplicate joint names" in PLANNER_CPP
    assert "place-only export expected five motion segments" in PLANNER_CPP
    assert "stage->setGoal(home_goal);" in PLANNER_CODE
    for phase in ('"transport"', '"approach"', '"release"', '"retreat"'):
        assert phase in PLANNER_CPP


def test_empty_shelf_entry_is_head_only_fresh_and_keeps_non_target_voxels():
    assert "head_scene_points(" in PLACE_CAPTURE
    assert "observe_output_table(" in PLACE_CAPTURE
    assert "head_lock.is_at_reference(angle)" in PLACE_CAPTURE
    assert "abs(lift.height_mm - cli.expected_lift_mm) > 5" in PLACE_CAPTURE
    assert "--operator-confirms-shelf-obstacles-complete" in PLACE_CAPTURE
    assert "verified_shelf_geometry_operator_obstacle_confirmation" in PLACE_CAPTURE
    assert "--held-tcp-base-rad" in PLACE_CAPTURE
    assert "held_pose.tolist()" in PLACE_CAPTURE
    assert 'payload["held_tcp_base_xyz_rpy_rad"]' in PLACE_CONVERTER
    assert "--held-tcp-base-rad" not in PLACE_CONVERTER
    assert "RobotSession" not in PLACE_CAPTURE
    assert "age_s > 45.0" in PLACE_CONVERTER
    assert '"fixture_source": False' in PLACE_CONVERTER
    # A map is taken either with the arm holding a bottle -- which leaves an
    # occlusion shadow the arm subtraction cannot recover -- or before the
    # pick with the arm clear of the head camera.  Both scripts must state
    # which, so "empty" is never claimed for a region that was never seen.
    assert '"occlusion_regime"' in PLACE_CAPTURE
    assert '"arm_clear_of_view"' in PLACE_CAPTURE
    assert 'payload.get("occlusion_regime", "held_arm_subtracted")' in PLACE_CONVERTER
    assert "held | robot_tool" in PLACE_CONVERTER
    assert '"target_insert_direction": [0.0, 0.0, -1.0]' in PLACE_CONVERTER
    assert '"target_retreat_direction": [0.0, 0.0, 1.0]' in PLACE_CONVERTER
    assert (
        '"post_place_home_joints_deg": list(profile.grasp_start_right_joints_deg)'
        in PLACE_CONVERTER
    )
    assert "--lift-execution-record" in PLACE_CAPTURE
    assert "held_right_joints.tolist()" in PLACE_CAPTURE
    assert '"held_right_joints_deg": held_right_joints.tolist()' in PLACE_CONVERTER


def test_cross_layer_runner_uses_live_state_then_pick_lift_empty_place():
    order = [
        "live_state_plan_only.launch.py",
        "calibrate_mtc_gripper.py",
        "capture_mtc_direct_pick_scene.py",
        '"pick",',
        "execute_mtc_lift_transfer.py",
        "capture_empty_shelf_places.py",
        "empty_shelf_places_to_mtc_scenario.py",
        '"place",',
    ]
    positions = [CROSS_LAYER_RUNNER.index(item) for item in order]
    assert positions == sorted(positions)
    assert '"--execute", action="store_true", required=True' in CROSS_LAYER_RUNNER
    assert "operator-confirms-lower-shelf-obstacles-complete" in CROSS_LAYER_RUNNER
    assert '"--allow-sdk-retiming"' in CROSS_LAYER_RUNNER
    assert CROSS_LAYER_RUNNER.index(
        "load_lift_transfer_contract(cli.lift_contract)"
    ) < CROSS_LAYER_RUNNER.index("subprocess.Popen(")


def test_direct_entry_uses_fixed_head_without_observation_or_wrist_lock():
    assert "demo._fresh_head_target()" in DIRECT_ENTRY
    assert "demo._build_head_scene(target, full_frame=True)" in DIRECT_ENTRY
    assert "demo_args.plan_only = False" in DIRECT_ENTRY
    assert "head_lock.is_at_reference(current_head)" in DIRECT_ENTRY
    assert "head_lock.read_current_angle_direct()" in DIRECT_ENTRY
    for dependency in (
        "_plan_observation",
        "_fresh_wrist_target",
        "_verify_wrist_observation_start",
    ):
        assert dependency not in DIRECT_ENTRY


def test_failure_stage_requires_an_actual_failed_attempt():
    assert "stage.numFailures() == 0" in PLANNER_CODE
    assert "result.earliest_failure_stage = first_failed_leaf;" in PLANNER_CODE
    assert "dynamic_cast<const mtc::stages::Connect*>" not in PLANNER_CODE


def test_arm_selection_is_not_hardcoded():
    assert "arm_result.best_total_cost < selected->best_total_cost" in PLANNER_CPP
    assert "arm_result.solved &&" in PLANNER_CPP
    assert "arm_result.execution_eligible && !selected->execution_eligible" in PLANNER_CPP
    assert "arm_result.execution_eligible == selected->execution_eligible" in PLANNER_CPP
    # No arm id may appear in the planner source: identity lives in YAML only.
    for arm in ARMS:
        assert arm["arm_id"] not in PLANNER_CPP
        assert arm["planning_group"] not in PLANNER_CPP
        assert arm["ik_link"] not in PLANNER_CPP


def test_arm_config_is_complete_and_gated():
    assert len(ARMS) == 2, "both arms must be planned"
    for arm in ARMS:
        for key in ("arm_id", "planning_group", "ik_link", "touch_links", "tcp_transform_from_ik_link"):
            assert arm.get(key), f"{arm.get('arm_id')} missing {key}"
        if not arm["execution_eligible"]:
            assert arm["execution_block_reason"], f"{arm['arm_id']} blocked without a reason"
    left = next(a for a in ARMS if a["arm_id"] == "left_arm")
    assert left["execution_eligible"] is False
    assert left["execution_block_reason"] == "LEFT_TOOL_CALIBRATION_REQUIRED"


def test_tool_transform_is_full_and_rigid():
    for arm in ARMS:
        transform = arm["tcp_transform_from_ik_link"]
        assert len(transform) == 4 and all(len(row) == 4 for row in transform)
        assert transform[3] == [0.0, 0.0, 0.0, 1.0]
        rotation = [row[:3] for row in transform[:3]]
        for i in range(3):
            for j in range(3):
                dot = sum(rotation[k][i] * rotation[k][j] for k in range(3))
                assert math.isclose(dot, float(i == j), abs_tol=1e-6)
        determinant = (
            rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        assert math.isclose(determinant, 1.0, abs_tol=1e-6)
    assert "return arm.tcp_transform_from_ik_link;" in PLANNER_CODE


def test_kdl_50ms_override_is_experimental_only():
    overrides = [
        path for path in LAUNCHES
        if 'group["kinematics_solver_timeout"] = 0.05' in path.read_text(encoding="utf-8")
    ]
    assert len(overrides) == 1
    assert overrides[0].name == "plan_shelf_transfer_experimental.launch.py"


def test_scenario_is_a_declared_fixture():
    assert SCENARIO["fixture_source"] is True, "hand-measured poses must be flagged as a fixture"
    assert SCENARIO["start_state_source"] == "current_state"
    assert SCENARIO["frame_id"] == "platform_base_link"

    for key in (
        "source_approach_direction",
        "source_lift_direction",
        "source_retreat_direction",
        "target_insert_direction",
        "target_retreat_direction",
    ):
        v = SCENARIO[key]
        assert len(v) == 3 and math.isclose(math.sqrt(sum(c * c for c in v)), 1.0, abs_tol=1e-6), key
    assert SCENARIO["source_lift_direction"] == [0.0, 0.0, 1.0]
    assert SCENARIO["source_lift_distance_m"] > 0.0
    # Retreats must undo the insertions, or the gripper leaves through the shelf.
    for insert, retreat in (("source_approach_direction", "source_retreat_direction"),
                            ("target_insert_direction", "target_retreat_direction")):
        dot = sum(a * b for a, b in zip(SCENARIO[insert], SCENARIO[retreat]))
        assert dot < 0.0, f"{retreat} does not back out of {insert}"

    for key in ("source_grasp_pose", "target_place_pose"):
        q = SCENARIO[key]["quat_xyzw"]
        assert math.isclose(math.sqrt(sum(c * c for c in q)), 1.0, abs_tol=1e-6), key


def test_synthetic_pick_fixture_has_an_explicit_non_execution_marker():
    scenario_code = (PKG / "src" / "scenario.cpp").read_text(encoding="utf-8")
    assert 'root["simulation_source"].as<bool>(false)' in scenario_code
    assert "simulation_source && !s.fixture_source" in scenario_code
    assert "s.fixture_source && !simulation_source" in scenario_code


def test_synthetic_place_fixture_has_the_same_plan_only_exception():
    scenario_code = (PKG / "src" / "scenario.cpp").read_text(encoding="utf-8")
    place_checks = scenario_code.split("\tif (s.place_only)", 1)[1].split(
        "\n\ts.planner_id", 1
    )[0]
    assert "s.fixture_source && !simulation_source" in place_checks
    assert "if (s.fixture_source)" not in place_checks


def test_real_grasp_trace_scenario_stays_explicitly_historical():
    assert TRACE_SCENARIO["scenario_id"] == "right_arm_placeback_trace_v1"
    assert TRACE_SCENARIO["fixture_source"] is True
    assert TRACE_SCENARIO["start_state_source"] == "current_state"
    assert TRACE_SCENARIO["frame_id"] == "platform_base_link"
    assert TRACE_SCENARIO["source_grasp_pose"] == TRACE_SCENARIO["target_place_pose"]
    assert TRACE_SCENARIO["shelf_boxes"][0]["id"] == "table_top"
    assert TRACE_SCENARIO["source_support_surface_id"] == "table_top"
    assert TRACE_SCENARIO["target_support_surface_id"] == "table_top"


def test_j4_singularity_gate_is_scoped_to_the_cartesian_legs():
    # Elbow-extended is a Jacobian singularity: it breaks CARTESIAN control.
    # Every MTC segment is executed by rm_movej (rm_movel appears nowhere in
    # bottle_grasp/mtc_execution.py), so the free-space leg inverts no Jacobian
    # and may pass through it. Auditing the whole trajectory rejected paths that
    # execute fine, and tied success to whether the arm happened to be parked on
    # the same elbow branch as the reachable pregrasp.
    assert "const std::size_t j4_audit_begin = trajectory.phases.front().end_index;" in (
        PLANNER_CODE
    )
    assert "for (std::size_t point = j4_audit_begin; point < trajectory.points.size();" in (
        PLANNER_CODE
    )
    # The pregrasp IK endpoint is the first audited point: the Cartesian legs
    # start there, so it must still satisfy the band.
    assert "point > j4_audit_begin && previous_j4 * j4 < 0.0" in PLANNER_CODE
    assert "EXECUTION_J4_SINGULARITY_DEG" in PLANNER_CODE

    # The premise the scoping rests on. If a Cartesian primitive ever enters the
    # MTC execution path, the free-space exemption stops being sound.
    execution = (
        PKG.parents[2] / "bottle_grasp" / "mtc_execution.py"
    ).read_text(encoding="utf-8")
    assert "movel" not in execution
    assert "execute_planned_joints" in execution

    # Wrap and workspace stay whole-trajectory: neither is about Cartesian control.
    assert "for (const auto& point : trajectory.points)" in PLANNER_CODE
    assert "EXECUTION_MAX_JOINT_RANGE_DEG" in PLANNER_CODE


def test_workspace_bound_is_collision_geometry_scoped_to_the_tcp_links():
    # Expressed as collision geometry, not a TCP pose constraint: the pose-box
    # attempt measured 80 s with no solution because it forces projection
    # sampling. Collision checks run natively.
    assert "buildTcpWorkspaceShell" in PLANNER_CODE
    assert "SolidPrimitive::BOX" in PLANNER_CODE

    # Thin walls are tunnellable between discrete validity samples.
    thickness = "TCP_WORKSPACE_SHELL_THICKNESS_M = "
    assert float(PLANNER_CODE.split(thickness)[1].split(";")[0]) >= 0.25

    # The shell must stay looser than the audit it guides: collision checks act
    # on links, and r_hand reaches past the TCP the audit actually bounds.
    assert "TCP_WORKSPACE_SHELL_LINK_REACH_M" in PLANNER_CODE
    assert "tcp_transform_from_ik_link.translation().norm()" in PLANNER_CODE
    assert PLANNER_CODE.count("/ 2.0 + margin") == 3

    # The audit it mirrors bounds the TCP point only. The torso, base and left
    # arm sit outside the certified volume legitimately, so the shell must be
    # allowed against everything and then re-enabled for the TCP links alone.
    shell = PLANNER_CODE.index("stage->addObject(buildTcpWorkspaceShell(scenario, *arm));")
    block = PLANNER_CODE[shell:shell + 700]
    assert "getLinkModelNames(), true" in block
    assert "tcp_links, false" in block
    assert "scenario.bottle_id, true" in block

    # Pick-only only; other modes keep their historic scene.
    assert "scenario.pick_only && scenario.has_tcp_path_workspace" in PLANNER_CODE

    # The post-export dense-FK audit stays as the independent second check.
    assert "tcpWorkspaceContainsTrajectory" in PLANNER_CODE


def test_path_cost_weights_proximal_joints_and_ranking_agrees_with_it():
    # "Looks coordinated" comes from not swinging the shoulder when a wrist turn
    # would do. Equal weights price those the same, which is why the shortest
    # solution kept looking uncoordinated.
    weights = SCENARIO["planning_joint_weights"]
    assert len(weights) == 7
    assert all(value > 0 for value in weights)
    assert weights[0] > weights[-1], "base joint must cost more than the wrist"
    assert weights == sorted(weights, reverse=True), "weights must fall base->wrist"

    # Same table drives the planner's cost term and the post-export ranking.
    # If only one side is weighted the two layers pull against each other.
    assert "mtc::cost::PathLength>(joint_weights)" in PLANNER_CODE
    assert "scenario.planning_joint_weights[joint] *" in PLANNER_CODE
    assert "jointCostWeights(scenario" in PLANNER_CODE

    # Cheap on purpose: MTC expands best-first by cost, so this biases the
    # search without giving up RRTConnect's speed for an anytime optimizer.
    assert SCENARIO["planner_id"] == "RRTConnectkConfigDefault"

    # Direct pick must not turn a historical observation pose into an
    # executable waypoint or an IK ranking reference.
    ik = PLANNER_CODE.index('p + "source_pregrasp_ik"')
    ik_block = PLANNER_CODE[ik:ik + 1200]
    assert "cost::DistanceToReference" not in ik_block
    assert "staging_joints" not in ik_block
    assert "generator->setMonitoredStage(support_scene);" in PLANNER_CODE


def test_start_state_failures_name_the_cause_instead_of_no_solution():
    assert "start state TCP is outside tcp_path_workspace" in PLANNER_CODE
    assert "move the arm to the taught home before planning" in PLANNER_CODE


def test_selected_arm_start_state_has_fresh_joint_state_provenance():
    cmake = (PKG / "CMakeLists.txt").read_text(encoding="utf-8")
    package = (PKG / "package.xml").read_text(encoding="utf-8")
    result_writer = (PKG / "src" / "scenario.cpp").read_text(encoding="utf-8")
    assert "sensor_msgs::msg::JointState" in PLANNER_CPP
    assert '"/joint_states"' in PLANNER_CPP
    assert "selected_arm_complete" in PLANNER_CPP
    assert "joint_state_age_s_at_planning" in PLANNER_CPP
    assert "CurrentState does not match fresh /joint_states" in PLANNER_CPP
    assert "selected_arm_complete" in result_writer
    assert "find_package(sensor_msgs REQUIRED)" in cmake
    assert "<depend>sensor_msgs</depend>" in package


def test_direct_entry_can_pass_a_target_product_class():
    assert '"--target-product"' in DIRECT_ENTRY
    assert "demo_args.target_product = cli.target_product" in DIRECT_ENTRY
    # Set before the ctor, which is where BottleDemo reads it.
    assert DIRECT_ENTRY.index("demo_args.target_product") < DIRECT_ENTRY.index(
        "BottleDemo(demo_args"
    )
    assert "demo._build_head_scene(target, full_frame=True)" in DIRECT_ENTRY


def test_joint_state_wait_is_discovery_patience_not_a_freshness_relaxation():
    # The wait bounds DDS discovery for a freshly spawned process. Freshness is
    # judged separately on the message stamp and must stay tight.
    assert "age_s <= 0.5" in PLANNER_CODE
    assert "std::chrono::seconds(10)" in PLANNER_CODE
    assert "std::chrono::seconds(2)" not in PLANNER_CODE

    # It must still break out as soon as a good sample lands, or the generous
    # deadline would become a fixed startup cost on every scenario.
    wait = PLANNER_CODE.index("const auto deadline = std::chrono::steady_clock::now()")
    block = PLANNER_CODE[wait:wait + 1400]
    assert "break;" in block
    assert "complete" in block


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all offline contract checks passed")
