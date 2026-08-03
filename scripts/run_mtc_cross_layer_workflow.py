#!/usr/bin/env python3
"""Run the live head-camera MTC pick→647/250 lift→empty-place workflow."""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bottle_grasp.core import SafetyAbort
from bottle_grasp.mtc_execution import load_lift_transfer_contract


def run(command: list[str], *, timeout: float) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, timeout=timeout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument(
        "--allow-sdk-retiming",
        action="store_true",
        required=True,
        help="明确接受 RealMan SDK movej 不保留 MTC/Pilz timing",
    )
    parser.add_argument(
        "--operator-confirms-lower-shelf-obstacles-complete",
        action="store_true",
        required=True,
    )
    parser.add_argument(
        "--lift-contract",
        type=Path,
        default=ROOT / "bottle_grasp/lift_transfer_647_to_250.json",
    )
    parser.add_argument("--arm-speed", type=int, default=100)
    parser.add_argument("--lift-speed", type=int, default=30)
    parser.add_argument(
        "--lower-roi-min", type=float, nargs=3, default=(-0.37, 0.58, -0.25)
    )
    parser.add_argument(
        "--lower-roi-max", type=float, nargs=3, default=(0.14, 0.78, 0.18)
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/mtc_cross_layer",
    )
    cli = parser.parse_args(argv)
    if not all(
        math.isfinite(low) and math.isfinite(high) and low < high
        for low, high in zip(cli.lower_roi_min, cli.lower_roi_max)
    ):
        parser.error("--lower-roi-min 每一维都必须小于有限的 --lower-roi-max")
    load_lift_transfer_contract(cli.lift_contract)
    run_dir = cli.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    python = sys.executable
    pick_scenario = run_dir / "pick.yaml"
    pick_result = run_dir / "pick_result.json"
    pick_trajectory = Path(str(pick_result) + ".trajectory.json")
    pick_execution = run_dir / "pick_execution.json"
    lift_execution = run_dir / "lift_execution.json"
    empty_observation = run_dir / "lower_empty_places.json"
    place_scenario = run_dir / "place.yaml"
    place_result = run_dir / "place_result.json"
    place_trajectory = Path(str(place_result) + ".trajectory.json")
    bridge_status = run_dir / "bridge_status.json"
    gripper_calibration = run_dir / "gripper_calibration.json"
    stack = subprocess.Popen(
        [
            "ros2",
            "launch",
            "grabber_robot_state_bridge",
            "live_state_plan_only.launch.py",
            f"bridge_status_file:={bridge_status}",
        ],
        cwd=ROOT,
    )

    def stop_stack() -> None:
        if stack.poll() is not None:
            return
        stack.send_signal(signal.SIGINT)
        try:
            stack.wait(timeout=12)
        except subprocess.TimeoutExpired:
            stack.terminate()
            stack.wait(timeout=5)

    atexit.register(stop_stack)
    deadline = time.monotonic() + 35.0
    while time.monotonic() < deadline:
        if stack.poll() is not None:
            raise RuntimeError(
                f"实时状态/MoveIt plan-only 栈提前退出: rc={stack.returncode}"
            )
        try:
            status = json.loads(bridge_status.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(0.5)
            continue
        if (
            status.get("read_only") is True
            and status.get("publishing") is True
            and status.get("lift_motion_ready") is True
            and int(status.get("published", 0)) >= 3
        ):
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("实时双臂/升降 joint-state 栈 35 秒内未就绪")

    run(
        [
            python,
            "scripts/calibrate_mtc_gripper.py",
            "--record",
            str(gripper_calibration),
            "--execute",
        ],
        timeout=30,
    )
    run(
        [
            python,
            "scripts/capture_mtc_direct_pick_scene.py",
            "--scenario-out",
            str(pick_scenario),
            "--output-dir",
            str(run_dir / "pick_capture"),
        ],
        timeout=90,
    )
    run(
        [
            "ros2",
            "launch",
            "grabber_mtc_planner",
            "plan_shelf_transfer_experimental.launch.py",
            f"scenario:={pick_scenario}",
            f"out:={pick_result}",
            "hold_seconds:=0",
        ],
        timeout=90,
    )
    run(
        [
            python,
            "scripts/execute_mtc_trajectory.py",
            "pick",
            "--result",
            str(pick_result),
            "--trajectory",
            str(pick_trajectory),
            "--scenario",
            str(pick_scenario),
            "--record",
            str(pick_execution),
            "--speed",
            str(cli.arm_speed),
            "--gripper-calibration-record",
            str(gripper_calibration),
            "--execute",
            "--allow-sdk-retiming",
        ],
        timeout=120,
    )
    run(
        [
            python,
            "scripts/execute_mtc_lift_transfer.py",
            str(pick_execution),
            "--contract",
            str(cli.lift_contract),
            "--record",
            str(lift_execution),
            "--speed",
            str(cli.lift_speed),
            "--execute",
        ],
        timeout=150,
    )
    run(
        [
            python,
            "scripts/capture_empty_shelf_places.py",
            "--expected-lift-mm",
            "250",
            "--roi-min",
            *map(str, cli.lower_roi_min),
            "--roi-max",
            *map(str, cli.lower_roi_max),
            "--lift-execution-record",
            str(lift_execution),
            "--operator-confirms-shelf-obstacles-complete",
            "--output",
            str(empty_observation),
        ],
        timeout=90,
    )
    run(
        [
            python,
            "scripts/empty_shelf_places_to_mtc_scenario.py",
            str(empty_observation),
            str(place_scenario),
        ],
        timeout=20,
    )
    run(
        [
            "ros2",
            "launch",
            "grabber_mtc_planner",
            "plan_shelf_transfer_experimental.launch.py",
            f"scenario:={place_scenario}",
            f"out:={place_result}",
            "hold_seconds:=0",
        ],
        timeout=90,
    )
    run(
        [
            python,
            "scripts/execute_mtc_trajectory.py",
            "place",
            "--result",
            str(place_result),
            "--trajectory",
            str(place_trajectory),
            "--scenario",
            str(place_scenario),
            "--record",
            str(run_dir / "place_execution.json"),
            "--speed",
            str(cli.arm_speed),
            "--execute",
            "--allow-sdk-retiming",
        ],
        timeout=120,
    )
    print(f"跨层抓放完成，证据目录: {run_dir}")
    stop_stack()
    atexit.unregister(stop_stack)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        RuntimeError,
        SafetyAbort,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"跨层抓放中止: {exc}", file=sys.stderr)
        raise SystemExit(2)
