"""Read-only RealMan access: joint angles and lift height.

Every call in this file is a query.  The only SDK entry points used are
``rm_create_robot_arm``, ``rm_get_joint_degree`` and ``rm_delete_robot_arm``,
plus one JSON socket command ``get_lift_state``.  No motion API, no tool
voltage, no TCP frame write, no teleop shutdown -- reading joint degrees was
measured to work with ``atom`` running (2026-07-27, 20/20 reads rc=0,
4-15 ms each), so this bridge never needs to take control.

test_motion_api_audit.py enforces the above statically over this whole package.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass

from .joint_map import ARM_JOINT_COUNT, deg_to_rad, mm_to_m


class StateReadError(RuntimeError):
    """A read failed.  Callers must stop publishing rather than reuse old data."""


@dataclass(frozen=True)
class ArmSample:
    arm: str
    positions_rad: list
    raw_degrees: list
    monotonic: float


@dataclass(frozen=True)
class LiftSample:
    height_mm: int
    position_m: float
    enabled: bool
    error_flag: int
    monotonic: float

    @property
    def motion_ready(self) -> bool:
        return self.enabled and self.error_flag == 0


def validate_lift_sample(
    sample: LiftSample, *, allow_faulted_position: bool = False
) -> None:
    """Accept a measured position only when it is safe for this use.

    A disabled/faulted drive can still report a stable measured height.  The
    explicit override is diagnostic plan-only: it never makes the drive
    motion-ready.
    """
    if not 0.0 <= sample.position_m <= 1.0:
        raise StateReadError(
            f"lift position {sample.position_m:.3f}m outside RobotModel [0, 1]m"
        )
    if not sample.motion_ready and not allow_faulted_position:
        raise StateReadError(
            f"lift not healthy: enabled={sample.enabled} "
            f"err_flag={sample.error_flag}"
        )


class ArmReader:
    """One persistent read-only SDK handle per arm controller."""

    def __init__(self, arm: str, ip: str, port: int = 8080):
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

        self.arm = arm
        self.ip = ip
        self.port = int(port)
        self._handle_api = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self._handle_api.rm_create_robot_arm(ip, self.port)
        if handle.id == -1:
            raise StateReadError(f"{arm} arm SDK connect failed at {ip}:{port}")
        self.handle_id = handle.id

    def sample(self) -> ArmSample:
        stamp = time.monotonic()
        rc, degrees = self._handle_api.rm_get_joint_degree()
        if rc != 0:
            raise StateReadError(f"{self.arm} rm_get_joint_degree rc={rc}")
        if degrees is None or len(degrees) != ARM_JOINT_COUNT:
            raise StateReadError(
                f"{self.arm} rm_get_joint_degree returned {degrees!r}"
            )
        raw = [float(value) for value in degrees]
        if not all(value == value and abs(value) < 1e6 for value in raw):
            raise StateReadError(f"{self.arm} joint degrees not finite: {raw}")
        return ArmSample(self.arm, [deg_to_rad(v) for v in raw], raw, stamp)

    def close(self) -> None:
        try:
            self._handle_api.rm_delete_robot_arm()
        except Exception:  # noqa: BLE001 - teardown must never mask a result
            pass


class LiftReader:
    """Query-only lift state over the controller's JSON socket protocol."""

    def __init__(self, host: str, port: int = 8080, timeout_s: float = 5.0):
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def sample(self) -> LiftSample:
        payload = {"command": "get_lift_state"}
        wire = (json.dumps(payload) + "\r\n").encode("utf-8")
        stamp = time.monotonic()
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.timeout_s
            ) as connection:
                connection.settimeout(self.timeout_s)
                connection.sendall(wire)
                received = b""
                while True:
                    chunk = connection.recv(4096)
                    if not chunk:
                        raise StateReadError("lift socket closed before a full reply")
                    received += chunk
                    try:
                        reply = json.loads(received.decode("utf-8").strip())
                    except json.JSONDecodeError:
                        continue
                    break
        except (OSError, TimeoutError) as exc:
            raise StateReadError(f"lift query failed: {exc}") from exc
        if not isinstance(reply, dict) or reply.get("state") != "lift_state":
            raise StateReadError(f"unexpected lift reply: {reply!r}")
        for key in ("height", "en_flag", "err_flag"):
            if key not in reply:
                raise StateReadError(f"lift reply missing {key}: {reply!r}")
        height_mm = int(reply["height"])
        return LiftSample(
            height_mm=height_mm,
            position_m=mm_to_m(height_mm),
            enabled=bool(int(reply["en_flag"])),
            error_flag=int(reply["err_flag"]),
            monotonic=stamp,
        )
