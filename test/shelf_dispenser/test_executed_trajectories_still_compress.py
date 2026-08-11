"""Replay trajectories that really executed through the current compressor.

"What worked today still works tomorrow" is not something unit fixtures can
answer -- they are shaped by whoever wrote them.  These are the exports from
picks that actually ran on the arm, kept under ``outputs/runs/``, and the point
is narrow: a change to the controller-command compressor must not stop any of
them from being executable.

Skipped where the evidence is not present.  ``outputs/`` is gitignored, so the
robot's checkout does not have it; that is a missing input, not a failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shelf_dispenser.arm import (
    CONNECTED_TRAJECTORY_MAX_COMMANDS,
    CONNECTED_TRAJECTORY_MAX_STEP_DEG,
    RobotSession,
)

ROOT = Path(__file__).resolve().parents[2]
EXECUTED = sorted((ROOT / "outputs" / "runs").glob("*/*/*.trajectory.json"))


def _segments(trajectory: dict) -> list[list[list[float]]]:
    """Split at the attach boundary: the gripper closes there, so the arm
    stops there, and each side is queued as its own connected motion."""
    points = [list(map(float, p["positions_deg"])) for p in trajectory["points"]]
    attach = next(
        item["start_index"]
        for item in trajectory["phase_boundaries"]
        if item["name"] == "attach"
    )
    return [points[: attach + 1], points[attach:]]


@pytest.mark.skipif(not EXECUTED, reason="本机没有已执行轨迹证据（outputs/ 未同步）")
@pytest.mark.parametrize("path", EXECUTED, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_executed_trajectory_still_fits_the_controller_queue(path):
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    for index, segment in enumerate(_segments(trajectory)):
        commands = RobotSession._compress_connected_joint_path(
            segment[0], segment[1:]
        )
        assert len(commands) <= CONNECTED_TRAJECTORY_MAX_COMMANDS, (
            f"{path.name} 第 {index} 段压缩后 {len(commands)} 条指令，超过队列上限"
        )
        previous = segment[0]
        for command in commands:
            step = max(abs(a - b) for a, b in zip(command, previous))
            assert step <= CONNECTED_TRAJECTORY_MAX_STEP_DEG + 1e-9, (
                f"{path.name} 第 {index} 段有一条 {step:.2f}° 的指令超过单段上限"
            )
            previous = command
        assert commands[-1] == pytest.approx(segment[-1]), (
            f"{path.name} 第 {index} 段压缩后终点与规划终点不一致"
        )
