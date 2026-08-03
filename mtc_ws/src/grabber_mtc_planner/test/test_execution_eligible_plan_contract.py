#!/usr/bin/env python3
"""The MTC success bit must exclude trajectories the real executor rejects."""

from __future__ import annotations

import pathlib


PKG = pathlib.Path(__file__).resolve().parents[1]
REPO = pathlib.Path(__file__).resolve().parents[4]
PLANNER = (PKG / "src/plan_shelf_transfer.cpp").read_text(
    encoding="utf-8"
)
EXECUTOR = (REPO / "shelf_dispenser/arm.py").read_text(encoding="utf-8")


def test_mtc_applies_the_real_controller_contract_before_selecting_a_solution():
    for planner_value, executor_value in (
        (
            "EXECUTION_CONTROLLER_MAX_COMMANDS = 30",
            "CONNECTED_TRAJECTORY_MAX_COMMANDS = 30",
        ),
        (
            "EXECUTION_CONTROLLER_MAX_STEP_DEG = 15.0",
            "CONNECTED_TRAJECTORY_MAX_STEP_DEG = 15.0",
        ),
        (
            "EXECUTION_CONTROLLER_MAX_ERROR_DEG = 0.02",
            "CONNECTED_TRAJECTORY_MAX_ERROR_DEG = 0.02",
        ),
    ):
        assert planner_value in PLANNER
        assert executor_value in EXECUTOR
    assert "controllerTrajectorySafe" in PLANNER
    assert "EXECUTION_PLANNER_MAX_COMMANDS" in PLANNER
    assert "EXECUTION_CONTROLLER_MAX_COMMANDS - 1" in PLANNER
    assert "_compress_connected_joint_path" in EXECUTOR


def test_mtc_rejects_general_jacobian_singularities_before_selection():
    assert "getJacobian" in PLANNER
    assert "JacobiSVD" in PLANNER
    assert "EXECUTION_MIN_JACOBIAN_SINGULAR_VALUE" in PLANNER
    assert "jacobian_safe" in PLANNER


def test_mtc_enforces_the_no_roll_gripper_contract_before_selection():
    assert "EXECUTION_MAX_FINGER_ROLL_DEG = 0.25" in PLANNER
    assert "authoredFingerRollDeg" in PLANNER
    assert "plannedFingerRollDeg" in PLANNER
    assert "roll_safe" in PLANNER


def test_pick_place_and_full_transfer_all_require_execution_safe_candidates():
    assert PLANNER.count("auditExecutionTrajectory(") >= 4
    assert "#execution_safe" in PLANNER
    assert "no execution-safe exportable solution" in PLANNER
