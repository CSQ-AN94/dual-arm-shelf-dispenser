from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from shelf_dispenser.core import SafetyAbort

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "record_demonstrated_trajectory",
    ROOT / "scripts" / "record_demonstrated_trajectory.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sample(t_s, joint):
    return {"t_s": t_s, "joints_deg": [joint, 0, 0, 0, 0, 0, 0]}


def test_summary_preserves_real_motion_extent():
    summary = MODULE.summarize_samples(
        [_sample(0.0, 0.0), _sample(0.1, 2.0), _sample(0.2, 5.0)],
        min_motion_deg=3.0,
    )

    assert summary["duration_s"] == pytest.approx(0.2)
    assert summary["maximum_joint_span_deg"] == pytest.approx(5.0)
    assert summary["maximum_sample_step_deg"] == pytest.approx(3.0)


def test_stationary_or_non_monotonic_capture_is_rejected():
    with pytest.raises(SafetyAbort, match="运动不足"):
        MODULE.summarize_samples(
            [_sample(0.0, 0.0), _sample(0.1, 0.2)], min_motion_deg=3.0
        )
    with pytest.raises(SafetyAbort, match="时间或七关节"):
        MODULE.summarize_samples(
            [_sample(0.1, 0.0), _sample(0.1, 5.0)], min_motion_deg=3.0
        )
