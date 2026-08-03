"""One arm, one process -- the only way both arms can be driven at once.

The RealMan Python SDK's ``Algo`` is process-global, not per-handle: install
angle, tool frame and joint limits are set on the library, so a second
``RobotSession`` in the same process silently overwrites the first one's
kinematics.  That is why the left arm has only ever been readable, through a
short-lived subprocess per read (``ArmJointReader``).

This keeps that isolation but makes it persistent and two-way: a worker process
owns exactly one ``RobotSession`` and answers newline-delimited JSON commands on
stdin.  The parent talks to it through ``ArmProxy``, which exposes the subset of
``RobotSession`` the second arm is allowed to use.

The whitelist is the safety boundary, not a convenience: a typo cannot reach a
method nobody vetted for the non-primary arm.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .core import SafetyAbort

# Methods the parent may invoke on a worker-owned arm.  Read-only queries and
# the taught-pose motion path; anything that would need a live camera, a scene
# or the gripper's force loop stays with the primary arm for now.
ALLOWED_METHODS = frozenset(
    {
        "joints_deg",
        "current_tcp",
        "current_flange",
        "tcp_from_joints",
        "controller_flange_from_joints",
        "solve_flange_ik",
        "validate_planned_joints",
        "execute_planned_joints",
        "assert_arm_healthy",
        "controller_fence_status",
        "gripper_state",
    }
)

_REQUEST_TIMEOUT_S = 60.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _revive(value: Any) -> Any:
    if isinstance(value, dict):
        if "__ndarray__" in value:
            return np.asarray(value["__ndarray__"], dtype=float)
        return {k: _revive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_revive(v) for v in value]
    return value


def serve(config: dict) -> int:
    """Worker entry point.  Owns one RobotSession and answers stdin commands."""
    from .robot import RobotSession

    session = RobotSession(
        config["ip"],
        int(config["port"]),
        threading.Event(),
        float(config["tcp_z_m"]),
        float(config["model_flange_offset_m"]),
        take_control=bool(config.get("take_control", False)),
        tcp_transform=config.get("tcp_transform"),
        link7_to_controller_flange=config.get("link7_to_controller_flange"),
    )
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                method = request["method"]
                if method not in ALLOWED_METHODS:
                    raise SafetyAbort(f"{method} 不在从臂允许的方法白名单里")
                args = [_revive(a) for a in request.get("args", [])]
                kwargs = {
                    k: _revive(v) for k, v in request.get("kwargs", {}).items()
                }
                reply = {"ok": _jsonable(getattr(session, method)(*args, **kwargs))}
            except Exception as exc:  # reported to the parent, never swallowed
                reply = {"error": f"{type(exc).__name__}: {exc}"}
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    finally:
        session.close()
    return 0


class ArmProxy:
    """Parent-side handle on a worker-owned arm.

    Method calls are forwarded verbatim, so a caller that already knows
    ``RobotSession`` needs to learn nothing new.
    """

    def __init__(
        self,
        ip: str,
        port: int,
        tcp_z_m: float,
        *,
        model_flange_offset_m: float = 0.0172,
        take_control: bool = False,
        tcp_transform: Sequence[Sequence[float]] | None = None,
        link7_to_controller_flange: Sequence[Sequence[float]] | None = None,
        python: str | None = None,
        project_root: Path | None = None,
    ):
        def as_list(matrix):
            return None if matrix is None else np.asarray(matrix, float).tolist()

        config = {
            "ip": ip,
            "port": int(port),
            "tcp_z_m": float(tcp_z_m),
            "model_flange_offset_m": float(model_flange_offset_m),
            "take_control": bool(take_control),
            "tcp_transform": as_list(tcp_transform),
            "link7_to_controller_flange": as_list(link7_to_controller_flange),
        }
        root = project_root or Path(__file__).resolve().parents[1]
        self._process = subprocess.Popen(
            [
                python or sys.executable,
                "-c",
                "import sys,json;sys.path.insert(0,sys.argv[1]);"
                "from bottle_grasp.arm_worker import serve;"
                "sys.exit(serve(json.loads(sys.argv[2])))",
                str(root),
                json.dumps(config),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        handshake = self._read_line(_REQUEST_TIMEOUT_S)
        if not handshake.get("ready"):
            raise SafetyAbort(f"从臂进程启动失败: {handshake}")

    def _read_line(self, timeout_s: float) -> dict:
        line = self._process.stdout.readline()
        if not line:
            stderr = (self._process.stderr.read() or "").strip()
            raise SafetyAbort(f"从臂进程已退出: {stderr}")
        return json.loads(line)

    def _call(self, method: str, *args, **kwargs):
        if method not in ALLOWED_METHODS:
            raise SafetyAbort(f"{method} 不在从臂允许的方法白名单里")
        payload = {
            "method": method,
            "args": [_jsonable(a) for a in args],
            "kwargs": {k: _jsonable(v) for k, v in kwargs.items()},
        }
        with self._lock:
            if self._process.poll() is not None:
                raise SafetyAbort("从臂进程已退出，拒绝下发指令")
            self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
            reply = self._read_line(_REQUEST_TIMEOUT_S)
        if "error" in reply:
            raise SafetyAbort(f"从臂 {method} 失败: {reply['error']}")
        return _revive(reply["ok"])

    def __getattr__(self, name: str):
        if name not in ALLOWED_METHODS:
            raise AttributeError(name)
        return lambda *args, **kwargs: self._call(name, *args, **kwargs)

    def close(self):
        if self._process.poll() is None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
