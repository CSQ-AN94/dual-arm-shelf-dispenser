#!/usr/bin/env python3
"""One grasp, judged, on either arm.  Never moves a joint.

``close_gripper`` is the grasp command, and it refuses to answer "did I get
it?" without this round's measured empty-close baseline -- a stale constant
standing in for today's calibration is a silent wrong answer, not a fallback
(see ``validate_holding_gripper_feedback``).  So the sequence is fixed:

    open -> calibrate_empty_close (nothing between the fingers)
         -> operator puts the bottle in -> close_gripper -> judge -> release

The baseline lives on the session, so calibration and grasp must happen in one
run.  Splitting them across two invocations throws it away.

    python scripts/gripper_grasp_test.py --arm left_arm
    python scripts/gripper_grasp_test.py --arm right_arm --keep-holding

The left arm reaches its gripper through ``ArmProxy``: the whitelist in
``arm_worker`` admits this whole cycle.  What it still cannot do is derive a
grasp *point* from the left tool transform -- provenance is
``nominal_unvalidated`` and ``require_grasping_transforms`` refuses.  This is a
demonstrated grasp: you place the object, so nothing is derived.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.live_arm import ARMS, open_live_arm
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config

LOG = logging.getLogger("gripper_grasp_test")


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] > ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="left_arm")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    parser.add_argument(
        "--keep-holding",
        action="store_true",
        help="判定通过后保持夹持，不自动松开（默认结束时松开并收拢）",
    )
    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=False
    )
    robot, view = open_live_arm(
        cfg, params, profile, cli.arm, take_control=True
    )
    # ArmProxy crosses a JSON boundary; its worker builds the same default
    # DemoParams on the far side, so never serialize this one.
    args = () if cli.arm == "left_arm" else (params,)
    try:
        if cli.arm == "right_arm":
            robot.recover_transient_joint_frame_loss()
        robot.assert_arm_healthy()
        joints = robot.joints_deg()
        tcp = np.asarray(robot.tcp_from_joints(joints), dtype=float)
        view.assert_tcp_point(tcp[:3, 3], label=f"{cli.arm} 抓取测试 TCP")
        print(f"{cli.arm} 已连接，关节角 {[round(v, 1) for v in joints]}")

        if not _confirm("手指之间现在是空的吗？（标定空夹基线需要空手）"):
            print("已取消，未发出任何夹爪指令。")
            return 1
        baseline = robot.calibrate_empty_close(*args)
        threshold = baseline + params.gripper_object_margin
        print(f"✓ 本轮空夹基线 pos={baseline}，抓取判定阈值 pos>{threshold}")

        if not _confirm("把物体放进手指之间，放好了吗？"):
            print("已取消；夹爪停在张开状态。")
            return 1
        try:
            state = robot.close_gripper(*args)
        except SafetyAbort as exc:
            print(f"\n✗ 抓取判定未通过：{exc}")
            print("夹爪保持当前状态；确认现场后再决定是否松开。")
            return 2
        print(
            f"\n✓ 抓取判定通过："
            f"pos={int(state['pos'][0])} > {threshold}，"
            f"state={int(state['dof_state'][0])}（内部夹持力已建立），"
            f"current={int(state['current'][0])}"
        )
        if cli.keep_holding:
            print("按 --keep-holding 保持夹持，未松开。")
            return 0
        if not _confirm("松开并收拢夹爪？（物体会掉）"):
            print("保持夹持退出。")
            return 0
        robot.open_gripper(*args)
        robot.close_empty_gripper(*args)
        print("✓ 已松开并收拢。")
        return 0
    finally:
        robot.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SafetyAbort as exc:
        print(f"拒绝: {exc}", file=sys.stderr)
        raise SystemExit(2)
