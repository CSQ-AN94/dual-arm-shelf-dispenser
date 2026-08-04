#!/usr/bin/env python3
"""Put the right arm and lift on the taught shelf-pick admission state.

Every shelf pick is planned from wherever the arm happens to be.  Without this
step each run's transit leg is a different, unvalidated path, and the executor
refuses the run anyway because the profile freezes this posture as an
admission gate -- so reaching it belongs to the task, not to an operator's
hands.  Run this before each capture/plan cycle.

Moves the right arm through the same MoveIt plan, dense collision recheck and
electronic-fence recheck as every other taught-pose move, and drives the lift
to the taught height.  The left arm is only reported: it is captured live into
each plan's collision scene and separately required not to drift during
execution, which is what makes the plan sound.

Defaults to a dry run.  Nothing moves without --execute.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.orchestrator import RunOrchestrator
from shelf_dispenser.mobile_body import LiftSocketAdapter
from shelf_dispenser.arm import ArmJointReader
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config

LIFT_TOLERANCE_MM = 5


def _report(label: str, current, target, tolerance_deg: float) -> float:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    error = float(np.max(np.abs(current - target)))
    mark = "OK" if error <= tolerance_deg else "需归位"
    print(f"{label:6s} 最大关节偏差 {error:7.2f}°  {mark}")
    print(f"       当前 {' '.join(f'{v:7.1f}' for v in current)}")
    print(f"       目标 {' '.join(f'{v:7.1f}' for v in target)}")
    return error


def _stamp_tuck(path: Path, joints: list[float], error_deg: float) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    completion = record.get("completion")
    if record.get("mode") != "pick" or not isinstance(completion, dict):
        raise SafetyAbort(f"{path} 不是 pick 执行证据，不能盖收拢戳")
    completion["post_pick_tuck"] = {
        "right_joints_deg": joints,
        "max_error_deg": error_deg,
        "recorded_at": time.time(),
    }
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "grasp_start")
    )
    parser.add_argument("--port", type=int, default=8886)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move the right arm and lift; otherwise report only",
    )
    parser.add_argument(
        "--right-and-lift-only",
        action="store_true",
        help=(
            "Prepare the right arm and calibrated lift height before the "
            "separate left-arm normalizer runs; skip only the final atomic "
            "dual-arm gate"
        ),
    )
    parser.add_argument(
        "--pick-record",
        type=Path,
        help=(
            "carry_home only: the pick execution record to stamp with this "
            "tuck.  The lift-transfer contract requires the arm tucked, but a "
            "pick's trajectory ends at its retreat, so without this the "
            "evidence chain still shows the retreat pose and the lift refuses"
        ),
    )
    parser.add_argument(
        "--target",
        choices=("grasp_start", "carry_home"),
        default="grasp_start",
        help=(
            "Both targets are the same taught posture -- the operator's "
            "locked shelf-pick start.  They differ only in when they run and "
            "what they record: grasp_start before a pick with an empty "
            "gripper, carry_home after one with the bottle held, stamping the "
            "pick record so the lift transfer can see the arm got back"
        ),
    )
    cli = parser.parse_args()
    if cli.right_and_lift_only and cli.target != "grasp_start":
        parser.error("--right-and-lift-only 只适用于 grasp_start")
    cli.safety_profile = "shelf_template"
    Path(cli.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    cfg = load_config(cli.config)
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=True
    )
    if (
        not profile.grasp_start_right_joints_deg
        or not profile.grasp_start_left_joints_deg
        or profile.grasp_start_lift_height_mm is None
    ):
        raise SafetyAbort("shelf_template 缺少完整的抓取起点（双臂 + 升降）")

    tolerance = float(DemoParams().planned_start_tolerance_deg)
    left_reader = ArmJointReader(
        cfg.connections.left_arm_ip, cfg.connections.arm_port
    )
    lift = LiftSocketAdapter(
        cfg.connections.left_arm_ip, cfg.connections.arm_port
    )
    lift_state = lift.state()
    left_error = _report(
        "左臂",
        left_reader.joints_deg(),
        profile.grasp_start_left_joints_deg,
        tolerance,
    )
    lift_error = abs(
        int(lift_state.height_mm) - int(profile.grasp_start_lift_height_mm)
    )
    print(
        f"升降   {lift_state.height_mm} mm -> "
        f"{profile.grasp_start_lift_height_mm} mm  差 {lift_error} mm  "
        f"{'OK' if lift_error <= LIFT_TOLERANCE_MM else '需归位'}"
    )

    demo = RunOrchestrator(
        SimpleNamespace(
            task_mode=None,
            execute=bool(cli.execute),
            plan_only=not cli.execute,
            config=cli.config,
            safety_config=cli.safety_config,
            safety_profile=cli.safety_profile,
            stop_after_observation=False,
            confirm_before_grasp=False,
            place_back=False,
            return_home=False,
            restore_teleop=False,
            resume_at_wrist=False,
            finish_from_current=False,
            host="127.0.0.1",
            port=cli.port,
            output_dir=cli.output_dir,
            observe_seconds=0.0,
        ),
        cfg,
    )
    try:
        demo.initialize()
        if cli.target == "carry_home":
            # The post-pick carry pose is the operator's locked start pose, not
            # profile.home_joints_deg -- that home belongs to the side-table
            # delivery flow, which requires the shelf and table profiles to
            # share it, and it was taught for a different task entirely.
            carry = profile.grasp_start_right_joints_deg
            right_error = _report(
                "右臂", demo.robot.joints_deg(), carry, tolerance
            )
            if not cli.execute:
                print(f"\n干跑：未下发任何运动。右臂距收拢位 {right_error:.2f}°。")
                return 0
            demo.normalize_to_grasp_start()
            reached = [float(v) for v in demo.robot.joints_deg()]
            error = float(
                np.max(
                    np.abs(np.asarray(reached) - np.asarray(carry, dtype=float))
                )
            )
            if error > tolerance:
                raise SafetyAbort(
                    f"收拢位到位偏差 {error:.2f}° 超过 {tolerance:.2f}°"
                )
            if cli.pick_record is not None:
                _stamp_tuck(cli.pick_record, reached, error)
                print(f"已把收拢证据写回 {cli.pick_record}")
            print(f"\n已到达升降收拢位，偏差 {error:.2f}°。")
            return 0
        right_error = _report(
            "右臂",
            demo.robot.joints_deg(),
            profile.grasp_start_right_joints_deg,
            tolerance,
        )
        if not cli.execute:
            print(
                "\n干跑：未下发任何运动。"
                f"右臂 {right_error:.2f}°、左臂 {left_error:.2f}°、"
                f"升降 {lift_error} mm。加 --execute 才会移动右臂和升降。"
            )
            return 0
        if lift_error > LIFT_TOLERANCE_MM:
            lift.move_to(
                int(profile.grasp_start_lift_height_mm),
                speed=demo.params.final_speed,
            )
        right_error = demo.normalize_to_grasp_start()
        if cli.right_and_lift_only:
            settled_lift = lift.state()
            settled_lift_error = abs(
                int(settled_lift.height_mm)
                - int(profile.grasp_start_lift_height_mm)
            )
            if right_error > tolerance or settled_lift_error > LIFT_TOLERANCE_MM:
                raise SafetyAbort(
                    "右臂/升降预归位未到位: "
                    f"右臂={right_error:.2f}°，升降={settled_lift_error} mm"
                )
            print("\n右臂和升降已到位；左臂仍须单独规划，尚未通过双臂原子门禁。")
            return 0
        # The gate the executor will apply, run here so a partial normalization
        # cannot be mistaken for a ready robot.
        profile.assert_grasp_start(
            right_joints_deg=demo.robot.joints_deg(),
            left_joints_deg=left_reader.joints_deg(),
            lift_height_mm=lift.state().height_mm,
            lift_mode=lift.state().mode,
            joint_tolerance_deg=tolerance,
        )
        print("\n已到达抓取起点，且通过执行器的同一道 assert_grasp_start 门禁。")
    finally:
        demo.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
