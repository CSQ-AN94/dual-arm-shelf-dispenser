#!/usr/bin/env python3
"""Record one hand-guided arm trajectory without commanding any motion.

The process samples the selected arm until Enter is pressed.  It stores the
raw joint path and TCP path; converting two demonstrations into planner
constraints is deliberately a separate, offline step.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_dispenser.core import DemoParams, SafetyAbort
from shelf_dispenser.live_arm import ARMS, open_live_arm
from shelf_dispenser.mobile_body import LiftSocketAdapter
from shelf_dispenser.safety import load_safety_profile
from utils.config import load_config


SCHEMA = "grabber.demonstrated_joint_trajectory.v1"


def summarize_samples(samples: list[dict], *, min_motion_deg: float) -> dict:
    if len(samples) < 2:
        raise SafetyAbort("示教轨迹至少需要两个采样点")
    times = np.asarray([item.get("t_s") for item in samples], dtype=float)
    joints = np.asarray([item.get("joints_deg") for item in samples], dtype=float)
    if (
        times.shape != (len(samples),)
        or joints.shape != (len(samples), 7)
        or not np.all(np.isfinite(times))
        or not np.all(np.isfinite(joints))
        or times[0] < 0.0
        or np.any(np.diff(times) <= 0.0)
    ):
        raise SafetyAbort("示教轨迹时间或七关节采样无效")
    spans = np.ptp(joints, axis=0)
    maximum_span = float(np.max(spans))
    if maximum_span < float(min_motion_deg):
        raise SafetyAbort(
            f"示教轨迹运动不足: 最大关节跨度 {maximum_span:.2f}° "
            f"< {float(min_motion_deg):.2f}°"
        )
    steps = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
    return {
        "duration_s": float(times[-1] - times[0]),
        "sample_count": len(samples),
        "joint_span_deg": spans.tolist(),
        "maximum_joint_span_deg": maximum_span,
        "maximum_sample_step_deg": float(np.max(steps)),
        "start_joints_deg": joints[0].tolist(),
        "end_joints_deg": joints[-1].tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arm", choices=ARMS, default="right_arm")
    parser.add_argument("--label", required=True)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-duration-s", type=float, default=180.0)
    parser.add_argument("--min-motion-deg", type=float, default=3.0)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument(
        "--safety-config",
        default=str(ROOT / "shelf_dispenser" / "safety_profiles.json"),
    )
    cli = parser.parse_args(argv)
    if cli.output.exists():
        raise SafetyAbort(f"输出已存在，拒绝覆盖: {cli.output}")
    if not 1.0 <= cli.rate_hz <= 50.0:
        raise SafetyAbort("采样率必须在 1..50 Hz")
    if not 5.0 <= cli.max_duration_s <= 600.0:
        raise SafetyAbort("最长录制时间必须在 5..600 秒")
    if not 0.1 <= cli.min_motion_deg <= 30.0:
        raise SafetyAbort("最小运动跨度必须在 0.1..30 度")

    cfg = load_config(cli.config)
    params = DemoParams()
    profile = load_safety_profile(
        cli.safety_config, "shelf_template", require_verified=False
    )
    lift = LiftSocketAdapter(
        cfg.connections.left_arm_ip, cfg.connections.arm_port
    ).state()
    if lift.mode != 0:
        raise SafetyAbort(f"升降仍在运动，mode={lift.mode}")

    robot, view = open_live_arm(
        cfg, params, profile, cli.arm, take_control=False
    )
    stop = threading.Event()
    input_thread = threading.Thread(
        target=lambda: (sys.stdin.readline(), stop.set()), daemon=True
    )
    samples: list[dict] = []
    violations: list[dict] = []
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    period_s = 1.0 / cli.rate_hz
    try:
        print(
            f"开始录制 {cli.arm}/{cli.label}，只读 {cli.rate_hz:g} Hz；"
            "拖动完成后按 Enter 结束。",
            flush=True,
        )
        input_thread.start()
        while not stop.is_set():
            loop_started = time.monotonic()
            elapsed = loop_started - started
            if elapsed >= cli.max_duration_s:
                break
            joints = list(map(float, robot.joints_deg()))
            tcp_arm = np.asarray(robot.tcp_from_joints(joints), dtype=float)
            tcp_moveit = view.pose_to_moveit(tcp_arm)
            safe = True
            try:
                view.assert_tcp_point(
                    tcp_arm[:3, 3], label=f"{cli.arm} 示教轨迹 TCP"
                )
            except SafetyAbort as exc:
                safe = False
                violations.append(
                    {"sample_index": len(samples), "reason": str(exc)}
                )
            samples.append(
                {
                    "t_s": elapsed,
                    "joints_deg": joints,
                    "tcp_pose_moveit": {
                        "xyz": tcp_moveit[:3, 3].tolist(),
                        "quat_xyzw": Rotation.from_matrix(
                            tcp_moveit[:3, :3]
                        ).as_quat().tolist(),
                    },
                    "inside_electronic_fence": safe,
                }
            )
            remaining = period_s - (time.monotonic() - loop_started)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        robot.close()

    ended_at = datetime.now(timezone.utc)
    summary = summarize_samples(samples, min_motion_deg=cli.min_motion_deg)
    payload = {
        "schema_version": SCHEMA,
        "label": cli.label,
        "arm_id": cli.arm,
        "pose_frame": profile.moveit_frame,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "requested_rate_hz": cli.rate_hz,
        "lift_height_mm": int(lift.height_mm),
        "summary": summary,
        "fence_violations": violations,
        "usable_for_planning_constraints": not violations,
        "samples": samples,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"已写入 {cli.output}：{summary['sample_count']} 点，"
        f"{summary['duration_s']:.1f}s，围栏违规 {len(violations)} 次。",
        flush=True,
    )
    if violations:
        print("轨迹已保留，但不得生成规划约束。", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, SafetyAbort) as exc:
        print(f"拒绝记录示教轨迹: {exc}", file=sys.stderr)
        raise SystemExit(2)
