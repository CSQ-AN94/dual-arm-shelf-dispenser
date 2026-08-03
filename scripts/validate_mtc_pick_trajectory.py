#!/usr/bin/env python3
"""Validate an MTC pick-only export without executing it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bottle_grasp.core import SafetyAbort
from bottle_grasp.mtc_pick_contract import (
    load_pick_trajectory,
    validate_attach_gate,
    validate_pre_motion_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_json", type=Path)
    parser.add_argument(
        "--pre-motion-state",
        type=Path,
        help=(
            "可选运行时门禁 JSON：current_state、gripper_open_feedback；"
            "缺少时只验证离线契约"
        ),
    )
    parser.add_argument(
        "--attach-state",
        type=Path,
        help=(
            "可选 attach 门禁 JSON：point_index、gripper_close_feedback、"
            "empty_close_pos"
        ),
    )
    args = parser.parse_args()
    try:
        trajectory = load_pick_trajectory(args.trajectory_json)
        if args.pre_motion_state:
            state = json.loads(args.pre_motion_state.read_text(encoding="utf-8"))
            validate_pre_motion_gate(
                trajectory,
                current_state=state["current_state"],
                gripper_open_feedback=state["gripper_open_feedback"],
            )
        if args.attach_state:
            state = json.loads(args.attach_state.read_text(encoding="utf-8"))
            validate_attach_gate(
                trajectory,
                point_index=int(state["point_index"]),
                gripper_close_feedback=state["gripper_close_feedback"],
                empty_close_pos=int(state["empty_close_pos"]),
            )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        SafetyAbort,
    ) as exc:
        parser.exit(2, f"拒绝 MTC pick 轨迹: {exc}\n")

    if args.pre_motion_state or args.attach_state:
        print("指定的 MTC pick 门禁证据通过；本验证器不会发送运动。")
    else:
        print("MTC pick 离线导出契约有效；execution_supported=false。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
