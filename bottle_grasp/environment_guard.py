"""Task-lifetime guards for robot state that participates in collision checks."""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

from .core import SafetyAbort


class LeftArmStabilityGuard:
    """Keep the non-commanded left arm fixed for the entire right-arm task.

    MoveIt treats the left arm as a collision object at a measured joint
    snapshot.  A check around only the long global move is insufficient: the
    same left arm can collide during pregrasp, lift, placement, or retreat.
    This guard owns one reference and requests the controller slow-stop via the
    shared event as soon as that contract is lost.
    """

    def __init__(
        self,
        *,
        left_reader,
        stop_event: threading.Event,
        tolerance_deg: float,
        poll_s: float = 0.5,
    ):
        self.left_reader = left_reader
        self.stop_event = stop_event
        self.tolerance_deg = float(tolerance_deg)
        self.poll_s = float(poll_s)
        self.reference: np.ndarray | None = None
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: SafetyAbort | None = None

    @staticmethod
    def _joints(values: Sequence[float], *, label: str) -> np.ndarray:
        joints = np.asarray(values, dtype=float)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise SafetyAbort(f"{label}左臂关节含非有限数或维度无效")
        return joints

    def _sample(self) -> np.ndarray:
        return self._joints(self.left_reader.joints_deg(), label="实时")

    def _check_once(self) -> None:
        if self.reference is None:
            raise SafetyAbort("左臂任务守卫尚未建立参考快照")
        actual = self._sample()
        error = float(np.max(np.abs(actual - self.reference)))
        if not np.isfinite(error) or error > self.tolerance_deg:
            raise SafetyAbort(
                "左臂已偏离任务碰撞快照，停止右臂任务: "
                f"最大关节差={error:.2f}°，上限={self.tolerance_deg:.2f}°"
            )

    def start(self) -> list[float]:
        if self._thread is not None:
            raise SafetyAbort("左臂任务守卫重复启动")
        if (
            not np.isfinite(self.tolerance_deg)
            or self.tolerance_deg <= 0
            or not np.isfinite(self.poll_s)
            or self.poll_s <= 0
        ):
            raise SafetyAbort("左臂任务守卫容差/周期无效")
        self.reference = self._sample()

        def monitor() -> None:
            while not self._done.wait(self.poll_s):
                try:
                    self._check_once()
                except SafetyAbort as exc:
                    self._error = exc
                    setattr(self.stop_event, "source", "left_arm_drift")
                    self.stop_event.set()
                    return

        self._thread = threading.Thread(
            target=monitor,
            name="bottle-task-left-arm-guard",
            daemon=True,
        )
        self._thread.start()
        return self.reference.tolist()

    def check(self) -> None:
        if self._error is not None:
            raise self._error
        self._check_once()

    def close(self) -> None:
        if self._thread is None:
            return
        self._done.set()
        self._thread.join(timeout=9.0)
        if self._thread.is_alive():
            setattr(self.stop_event, "source", "left_arm_guard_timeout")
            self.stop_event.set()
            raise SafetyAbort("左臂任务守卫读取超时，已请求右臂缓停")
        self.check()
        self._thread = None
