"""Live lift/chassis state and fail-closed side-table body positioning.

The head camera and both arm bases are children of ``platform_base_link``.
Consequently the calibrated right-arm-base -> head-camera transform is
invariant while ``platform_joint`` moves.  This module deliberately never
accepts or produces a "lift-corrected" camera extrinsic; lift height belongs
only to the platform's pose relative to the chassis/world.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
from typing import Literal, Protocol

import numpy as np

from .core import SafetyAbort, stop_reason


def wrap_angle_rad(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


@dataclass(frozen=True)
class ChassisState:
    x_m: float
    y_m: float
    yaw_rad: float
    linear_mps: float
    angular_radps: float
    control_mode: str
    robot_state: str
    captured_monotonic: float

    def __post_init__(self) -> None:
        values = (
            self.x_m,
            self.y_m,
            self.yaw_rad,
            self.linear_mps,
            self.angular_radps,
            self.captured_monotonic,
        )
        if not all(math.isfinite(value) for value in values):
            raise SafetyAbort("底盘实时状态包含非有限数")


@dataclass(frozen=True)
class LiftState:
    height_mm: int
    enabled: bool
    error_flag: int
    mode: int
    captured_monotonic: float

    def __post_init__(self) -> None:
        if not (0 <= int(self.height_mm) <= 2600):
            raise SafetyAbort(f"升降实时高度越界: {self.height_mm} mm")
        if not math.isfinite(float(self.captured_monotonic)):
            raise SafetyAbort("升降状态时间戳无效")


@dataclass(frozen=True)
class BodySnapshot:
    chassis: ChassisState
    lift: LiftState

    def world_from_platform(self, reference_lift_height_mm: int = 0) -> np.ndarray:
        """Return the live world <- platform rigid transform.

        The odometry origin is the world frame.  Lift changes platform Z once.
        Callers must not apply this Z delta to ``T_base_head_camera`` because
        the right arm base rises with the same platform.
        """
        yaw = self.chassis.yaw_rad
        c, s = math.cos(yaw), math.sin(yaw)
        transform = np.eye(4)
        transform[:3, :3] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
        transform[:3, 3] = (
            self.chassis.x_m,
            self.chassis.y_m,
            (self.lift.height_mm - int(reference_lift_height_mm)) / 1000.0,
        )
        return transform


class ChassisAdapter(Protocol):
    def prepare_for_motion(self) -> bool: ...

    def state(self) -> ChassisState: ...

    def rotate_relative(
        self,
        yaw_rad: float,
        *,
        max_angular_speed_radps: float,
        yaw_tolerance_rad: float,
        max_translation_m: float,
        timeout_s: float,
    ) -> ChassisState: ...

    def stop(self) -> None: ...


class LiftAdapter(Protocol):
    def state(self) -> LiftState: ...

    def move_to(self, height_mm: int, *, speed: int) -> LiftState: ...


class LiftSocketAdapter:
    """RealMan controller JSON adapter for the shared body lift."""

    def __init__(
        self,
        host: str = "169.254.128.18",
        port: int = 8080,
        timeout_s: float = 90.0,
    ):
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def _request(self, payload: dict) -> dict:
        wire = (json.dumps(payload) + "\r\n").encode("utf-8")
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
                        raise SafetyAbort("升降控制器在完整 JSON 响应前断开")
                    received += chunk
                    try:
                        result = json.loads(received.decode("utf-8").strip())
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(result, dict):
                        raise SafetyAbort("升降控制器响应不是 JSON 对象")
                    return result
        except (OSError, TimeoutError) as exc:
            raise SafetyAbort(f"升降控制器通信失败: {exc}") from exc

    @staticmethod
    def _parse_state(payload: dict) -> LiftState:
        if payload.get("state") != "lift_state":
            raise SafetyAbort(f"升降状态响应类型异常: {payload}")
        required = ("height", "en_flag", "err_flag", "mode")
        if any(key not in payload for key in required):
            raise SafetyAbort(f"升降状态字段不完整: {payload}")
        state = LiftState(
            height_mm=int(payload["height"]),
            enabled=bool(int(payload["en_flag"])),
            error_flag=int(payload["err_flag"]),
            mode=int(payload["mode"]),
            captured_monotonic=time.monotonic(),
        )
        if not state.enabled or state.error_flag != 0:
            raise SafetyAbort(
                "升降机构不可执行: "
                f"enabled={state.enabled}, error_flag={state.error_flag}"
            )
        return state

    def state(self) -> LiftState:
        return self._parse_state(self._request({"command": "get_lift_state"}))

    def move_to(self, height_mm: int, *, speed: int) -> LiftState:
        height_mm, speed = int(height_mm), int(speed)
        if not (0 <= height_mm <= 2600):
            raise SafetyAbort(f"升降目标高度越界: {height_mm} mm")
        if not (1 <= speed <= 30):
            raise SafetyAbort(f"升降速度必须在 1..30: {speed}")
        before = self.state()
        if abs(before.height_mm - height_mm) <= 3:
            return before
        response = self._request(
            {
                "command": "set_lift_height",
                "speed": speed,
                "height": height_mm,
                "block": 1,
            }
        )
        # Firmware versions do not use one stable acknowledgement schema.
        # Never infer success from it: the fresh state below is authoritative.
        if response.get("error_code") not in (None, 0):
            raise SafetyAbort(f"升降命令被控制器拒绝: {response}")
        after = self.state()
        if abs(after.height_mm - height_mm) > 5:
            raise SafetyAbort(
                f"升降未到位: target={height_mm} mm, actual={after.height_mm} mm"
            )
        return after


class WooshChassisAdapter:
    """Adapter around audited robot-side Woosh diagnostic/rotation helpers."""

    _POSE_RE = re.compile(
        r"pose x=(?P<x>[-+0-9.eE]+) y=(?P<y>[-+0-9.eE]+) "
        r"theta=(?P<yaw>[-+0-9.eE]+)"
    )
    _TWIST_RE = re.compile(
        r"twist linear=(?P<linear>[-+0-9.eE]+) "
        r"angular=(?P<angular>[-+0-9.eE]+)"
    )
    _MODE_RE = re.compile(
        r"\[Mode\] ok=true[^\n]*\n\s*ctrl: (?P<mode>k[A-Za-z]+)"
    )
    _STATE_RE = re.compile(
        r"\[RobotState\] ok=true[^\n]*\n\s*state: (?P<state>k[A-Za-z]+)"
    )

    def __init__(
        self,
        *,
        diagnostic_path: str = "/home/rm/agv_debug_tools/agv_diag",
        pose_query_path: str = "/home/rm/agv_debug_tools/agv_pose_query",
        init_helper_path: str = "/home/rm/agv_debug_tools/agv_mode_init",
        rotate_helper_path: str = "/home/rm/agv_debug_tools/grabber_rotate_relative",
        stop_event: threading.Event | None = None,
    ):
        self.diagnostic_path = str(diagnostic_path)
        self.pose_query_path = str(pose_query_path)
        self.init_helper_path = str(init_helper_path)
        self.rotate_helper_path = str(rotate_helper_path)
        self.stop_event = stop_event

    def assert_ready(self) -> None:
        for label, path in (
            ("底盘诊断工具", self.diagnostic_path),
            ("底盘位姿工具", self.pose_query_path),
            ("底盘初始化工具", self.init_helper_path),
            ("闭环旋转工具", self.rotate_helper_path),
        ):
            candidate = Path(path)
            if not candidate.is_file() or not candidate.stat().st_mode & 0o111:
                raise SafetyAbort(f"{label}不存在或不可执行: {path}")

    @staticmethod
    def _run(command: list[str], timeout_s: float) -> str:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SafetyAbort(f"底盘工具调用失败 {command[0]}: {exc}") from exc
        if result.returncode != 0:
            raise SafetyAbort(
                f"底盘工具失败 rc={result.returncode}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        return result.stdout

    def _run_motion(self, command: list[str], timeout_s: float) -> str:
        """Run rotation while forwarding the task's shared STOP event."""
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            raise SafetyAbort(f"底盘旋转工具无法启动: {exc}") from exc
        deadline = time.monotonic() + float(timeout_s)
        stop_forwarded = False
        timed_out = False
        while process.poll() is None:
            if self.stop_event is not None and self.stop_event.is_set():
                process.send_signal(signal.SIGINT)
                stop_forwarded = True
                break
            if time.monotonic() >= deadline:
                process.send_signal(signal.SIGINT)
                timed_out = True
                break
            time.sleep(0.05)
        try:
            stdout, stderr = process.communicate(timeout=8.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            stdout, stderr = process.communicate(timeout=3.0)
        if stop_forwarded:
            raise SafetyAbort(stop_reason(self.stop_event))
        if timed_out:
            # A helper that happens to return zero after SIGINT did not finish
            # the requested motion.  Never reinterpret that as a completed
            # closed-loop turn.
            raise SafetyAbort("底盘旋转超时，已请求零速度停车")
        if process.returncode != 0:
            raise SafetyAbort(
                f"底盘旋转工具失败 rc={process.returncode}: "
                f"{stderr.strip() or stdout.strip()}"
            )
        return stdout

    @classmethod
    def _parse_pose(cls, output: str, *, mode: str, state: str) -> ChassisState:
        pose = cls._POSE_RE.search(output)
        twist = cls._TWIST_RE.search(output)
        if pose is None or twist is None:
            raise SafetyAbort(f"底盘位姿输出字段不完整: {output[-1200:]}")
        return ChassisState(
            x_m=float(pose.group("x")),
            y_m=float(pose.group("y")),
            yaw_rad=float(pose.group("yaw")),
            linear_mps=float(twist.group("linear")),
            angular_radps=float(twist.group("angular")),
            control_mode=mode,
            robot_state=state,
            captured_monotonic=time.monotonic(),
        )

    @classmethod
    def _parse_diagnostic(cls, output: str) -> tuple[str, str]:
        mode = cls._MODE_RE.search(output)
        state = cls._STATE_RE.search(output)
        if mode is None or state is None:
            raise SafetyAbort("底盘模式/状态输出字段不完整")
        return mode.group("mode"), state.group("state")

    def state(self) -> ChassisState:
        diagnostic = self._run([self.diagnostic_path], 12.0)
        mode, state = self._parse_diagnostic(diagnostic)
        if mode != "kAuto":
            raise SafetyAbort("底盘物理旋钮不在 kAuto 或模式读取失败")
        if state != "kIdle":
            raise SafetyAbort("底盘不是 kIdle 待命状态")
        pose = self._run([self.pose_query_path], 8.0)
        return self._parse_pose(pose, mode=mode, state=state)

    def prepare_for_motion(self) -> bool:
        """Acquire a clean idle session, then verify fresh state and zero speed.

        ``agv_mode_init`` is the audited robot-side helper: it does not switch
        the physical mode.  It cancels stale tasks, releases stale control,
        emits zero-speed stops, and initializes the current SDK session.  Run
        it even when the reported state is already ``kIdle``: ``kIdle`` alone
        does not prove that a previous client left control ownership cleanly.
        Any state other than kUninit/kIdle remains fail-closed.
        """
        self.assert_ready()
        diagnostic = self._run([self.diagnostic_path], 12.0)
        mode, state = self._parse_diagnostic(diagnostic)
        if mode != "kAuto":
            raise SafetyAbort("底盘物理旋钮不在 kAuto 或模式读取失败")
        if state not in ("kUninit", "kIdle"):
            raise SafetyAbort("底盘既不是 kUninit 也不是 kIdle，拒绝自动初始化")
        self._run([self.init_helper_path], 25.0)

        # Do not trust only the initialization command acknowledgement.  A
        # fresh diagnostic and pose/twist sample are the release condition.
        verified = self.state()
        if abs(verified.linear_mps) > 0.005 or abs(verified.angular_radps) > 0.01:
            raise SafetyAbort("底盘初始化后仍未静止")
        return True

    def rotate_relative(
        self,
        yaw_rad: float,
        *,
        max_angular_speed_radps: float,
        yaw_tolerance_rad: float,
        max_translation_m: float,
        timeout_s: float,
    ) -> ChassisState:
        values = (
            yaw_rad,
            max_angular_speed_radps,
            yaw_tolerance_rad,
            max_translation_m,
            timeout_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise SafetyAbort("底盘旋转参数包含非有限数")
        if not (math.radians(80) <= abs(yaw_rad) <= math.radians(100)):
            raise SafetyAbort("送桌流程只允许约 90° 原地旋转")
        output = self._run_motion(
            [
                self.rotate_helper_path,
                "--rotate-relative-rad",
                f"{yaw_rad:.12g}",
                "--max-angular-radps",
                f"{max_angular_speed_radps:.12g}",
                "--yaw-tolerance-rad",
                f"{yaw_tolerance_rad:.12g}",
                "--max-translation-m",
                f"{max_translation_m:.12g}",
                "--timeout-s",
                f"{timeout_s:.12g}",
            ],
            timeout_s + 10.0,
        )
        state = self._parse_pose(output, mode="kAuto", state="kIdle")
        if abs(state.linear_mps) > 0.005 or abs(state.angular_radps) > 0.01:
            raise SafetyAbort("底盘旋转工具返回时底盘仍未静止")
        return state

    def stop(self) -> None:
        # The helper's signal/exception paths always emit repeated zero twists.
        # Calling it in stop-only mode makes close()/outer exception handling
        # idempotent without using any non-zero velocity.
        self._run([self.rotate_helper_path, "--stop"], 8.0)


@dataclass(frozen=True)
class ReturnAuthorization:
    """Evidence the task must provide before a chassis return is legal.

    Body control does not inspect arm or gripper hardware itself.  Requiring
    this explicit, immutable hand-off keeps the dangerous task-state decision
    in the state machine while making a missing check fail closed here too.
    """

    release_verified: bool
    object_state: Literal["empty", "held", "unknown"]
    right_arm_compact_or_home: bool
    left_arm_stable: bool


@dataclass(frozen=True)
class ShelfReadyLimits:
    """Normalized profile constraints used for every body-loop comparison."""

    x_m: float
    y_m: float
    yaw_rad: float
    lift_height_mm: int
    xy_tolerance_m: float
    yaw_tolerance_rad: float
    lift_tolerance_mm: int


class MobileBodyCoordinator:
    """Closed-loop shelf/side-table body operation with auditable evidence."""

    _MAX_LINEAR_MPS = 0.005
    _MAX_ANGULAR_RADPS = 0.01
    _MAX_LIFT_DRIFT_M = 0.005
    _MAX_LIFT_DRIFT_RAD = math.radians(0.5)
    _STOP_REPETITIONS = 3

    def __init__(
        self,
        *,
        chassis: ChassisAdapter,
        lift: LiftAdapter,
        stop_event: threading.Event,
        evidence_dir: str | Path,
    ):
        self.chassis = chassis
        self.lift = lift
        self.stop_event = stop_event
        self.evidence_dir = Path(evidence_dir)
        # A value-equal snapshot is not an admission token.  Keep the exact
        # object returned by capture_shelf_ready so callers cannot forge the
        # pose and bypass preflight()/prepare_for_motion before a lift/turn.
        self._captured_shelf_ready: BodySnapshot | None = None
        # Return is legal only after this same coordinator completed the
        # outbound lift/turn from that captured token.
        self._delivery_position_start: BodySnapshot | None = None

    @staticmethod
    def _finite(value, *, label: str) -> float:
        if isinstance(value, bool):
            raise SafetyAbort(f"{label} 必须是有限数")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise SafetyAbort(f"{label} 必须是有限数") from exc
        if not math.isfinite(result):
            raise SafetyAbort(f"{label} 必须是有限数")
        return result

    @staticmethod
    def _snapshot_payload(snapshot: BodySnapshot) -> dict:
        return {
            "chassis": asdict(snapshot.chassis),
            "lift": asdict(snapshot.lift),
        }

    def _assert_not_stopped(self) -> None:
        if self.stop_event.is_set():
            raise SafetyAbort(stop_reason(self.stop_event))

    def _assert_body_healthy(self, snapshot: BodySnapshot) -> None:
        chassis, lift = snapshot.chassis, snapshot.lift
        if chassis.control_mode != "kAuto":
            raise SafetyAbort(
                f"底盘控制模式不是 kAuto: {chassis.control_mode!r}"
            )
        if chassis.robot_state != "kIdle":
            raise SafetyAbort(
                f"底盘控制器不是 kIdle: {chassis.robot_state!r}"
            )
        if (
            abs(chassis.linear_mps) > self._MAX_LINEAR_MPS
            or abs(chassis.angular_radps) > self._MAX_ANGULAR_RADPS
        ):
            raise SafetyAbort("底盘尚未静止，拒绝升降/规划")
        if not lift.enabled or lift.error_flag != 0:
            raise SafetyAbort(
                "升降机构不可执行: "
                f"enabled={lift.enabled}, error_flag={lift.error_flag}"
            )

    def snapshot(self) -> BodySnapshot:
        """Read and validate a fresh, stationary chassis/lift state."""
        snapshot = BodySnapshot(
            chassis=self.chassis.state(),
            lift=self.lift.state(),
        )
        self._assert_body_healthy(snapshot)
        return snapshot

    def _stop_repeatedly(self) -> None:
        """Best-effort zero-speed cleanup that never hides the root failure."""
        for _ in range(self._STOP_REPETITIONS):
            try:
                self.chassis.stop()
            except BaseException:
                # The controller may already be unavailable after the event
                # that caused the abort.  Continue trying zero-only stops and
                # preserve the original exception for the task journal.
                pass

    def stop(self) -> None:
        """Idempotently request repeated zero-speed chassis stops."""
        self._stop_repeatedly()

    def close(self) -> None:
        """Safe shutdown hook for the later demo/task composition root."""
        self.stop()

    def preflight(self) -> BodySnapshot:
        """Prepare only the chassis session, then return a live body reading."""
        try:
            self._assert_not_stopped()
            prepare = getattr(self.chassis, "prepare_for_motion", None)
            if callable(prepare):
                prepare()
            else:
                readiness = getattr(self.chassis, "assert_ready", None)
                if callable(readiness):
                    readiness()
            self._assert_not_stopped()
            return self.snapshot()
        except BaseException:
            self.stop()
            raise

    def _validated_shelf_ready(self, config):
        shelf_ready = getattr(config, "shelf_ready", None)
        if shelf_ready is None:
            raise SafetyAbort("side_table_delivery 缺少 SHELF_READY 配置")
        verified_fields = (
            "transport_pose_verified",
            "shelf_ready_verified",
            "lift_transition_verified",
            "table_roi_verified",
            "workspace_verified",
            "keepouts_verified",
            "bottle_tcp_verified",
        )
        missing = [
            field
            for field in verified_fields
            if getattr(config, field, None) is not True
        ]
        if missing:
            raise SafetyAbort(
                "side_table_delivery 尚未完成现场确认: "
                + ", ".join(missing)
            )
        sweep = getattr(config, "rotation_sweep", None)
        if sweep is None:
            raise SafetyAbort("side_table_delivery 缺少正反旋转扫掠量测")
        for direction in ("positive", "negative"):
            if getattr(sweep, f"{direction}_verified", None) is not True:
                raise SafetyAbort(
                    f"{direction} 方向底盘旋转扫掠尚未现场确认，拒绝执行"
                )
            if self._finite(
                getattr(sweep, f"{direction}_clearance_m", None),
                label=f"rotation_sweep.{direction}.clearance_m",
            ) <= 0.0:
                raise SafetyAbort(
                    f"rotation_sweep.{direction}.clearance_m 必须为正"
                )
        # The safety-profile loader is the authoritative complete schema
        # validator.  Rechecking every field used for body movement here keeps
        # a hand-constructed config from bypassing the pre-arm admission gate.
        for field in (
            "target_lift_height_mm",
            "target_lift_tolerance_mm",
            "body_lift_speed",
            "body_rotation_yaw_deg",
            "max_angular_speed_radps",
            "rotation_tolerance_deg",
            "max_base_translation_m",
            "rotation_timeout_s",
        ):
            self._finite(getattr(config, field, None), label=field)
        return shelf_ready

    def _shelf_ready_limits(self, config) -> ShelfReadyLimits:
        shelf = self._validated_shelf_ready(config)
        expected_x = self._finite(shelf.x_m, label="SHELF_READY.x_m")
        expected_y = self._finite(shelf.y_m, label="SHELF_READY.y_m")
        expected_yaw = math.radians(
            self._finite(shelf.yaw_deg, label="SHELF_READY.yaw_deg")
        )
        xy_tolerance = self._finite(
            shelf.xy_tolerance_m, label="SHELF_READY.xy_tolerance_m"
        )
        yaw_tolerance = math.radians(
            self._finite(
                shelf.yaw_tolerance_deg,
                label="SHELF_READY.yaw_tolerance_deg",
            )
        )
        lift_tolerance = int(
            self._finite(
                shelf.lift_tolerance_mm,
                label="SHELF_READY.lift_tolerance_mm",
            )
        )
        expected_lift = int(
            self._finite(shelf.lift_height_mm, label="SHELF_READY.lift_height_mm")
        )
        if xy_tolerance <= 0.0 or yaw_tolerance <= 0.0 or lift_tolerance < 1:
            raise SafetyAbort("SHELF_READY 容差无效")
        source_lift = int(
            self._finite(
                getattr(config, "source_lift_height_mm", None),
                label="source_lift_height_mm",
            )
        )
        if source_lift != expected_lift:
            raise SafetyAbort(
                "source_lift_height_mm 与 SHELF_READY 起始升降高度不一致"
            )
        return ShelfReadyLimits(
            x_m=expected_x,
            y_m=expected_y,
            yaw_rad=expected_yaw,
            lift_height_mm=expected_lift,
            xy_tolerance_m=xy_tolerance,
            yaw_tolerance_rad=yaw_tolerance,
            lift_tolerance_mm=lift_tolerance,
        )

    def _assert_matches_shelf_ready(
        self,
        snapshot: BodySnapshot,
        config,
        *,
        label: str,
    ) -> None:
        limits = self._shelf_ready_limits(config)
        xy_error = math.hypot(
            snapshot.chassis.x_m - limits.x_m,
            snapshot.chassis.y_m - limits.y_m,
        )
        yaw_error = abs(
            wrap_angle_rad(snapshot.chassis.yaw_rad - limits.yaw_rad)
        )
        lift_error = abs(snapshot.lift.height_mm - limits.lift_height_mm)
        if xy_error > limits.xy_tolerance_m:
            raise SafetyAbort(
                f"{label} x/y 超出 SHELF_READY 容差: "
                f"{xy_error * 1000:.1f} mm"
            )
        if yaw_error > limits.yaw_tolerance_rad:
            raise SafetyAbort(
                f"{label} yaw 超出 SHELF_READY 容差: "
                f"{math.degrees(yaw_error):.2f}°"
            )
        if lift_error > limits.lift_tolerance_mm:
            raise SafetyAbort(
                f"{label} 升降高度超出 SHELF_READY 容差: {lift_error} mm"
            )

    def _assert_matches_start(
        self,
        snapshot: BodySnapshot,
        start: BodySnapshot,
        config,
        *,
        label: str,
    ) -> None:
        if not isinstance(start, BodySnapshot):
            raise SafetyAbort(
                "转向/返程必须使用 capture_shelf_ready 返回的 BodySnapshot"
            )
        limits = self._shelf_ready_limits(config)
        xy_error = math.hypot(
            snapshot.chassis.x_m - start.chassis.x_m,
            snapshot.chassis.y_m - start.chassis.y_m,
        )
        yaw_error = abs(
            wrap_angle_rad(snapshot.chassis.yaw_rad - start.chassis.yaw_rad)
        )
        lift_error = abs(snapshot.lift.height_mm - start.lift.height_mm)
        if xy_error > limits.xy_tolerance_m:
            raise SafetyAbort(f"{label} 相对起始快照发生平移: {xy_error * 1000:.1f} mm")
        if yaw_error > limits.yaw_tolerance_rad:
            raise SafetyAbort(
                f"{label} 相对起始快照发生偏航: {math.degrees(yaw_error):.2f}°"
            )
        if lift_error > limits.lift_tolerance_mm:
            raise SafetyAbort(f"{label} 相对起始快照升降偏差: {lift_error} mm")
        self._assert_matches_shelf_ready(snapshot, config, label=label)

    def _require_captured_shelf_ready(self, start: BodySnapshot) -> None:
        """Reject snapshots not issued by this coordinator's latest preflight."""
        if not isinstance(start, BodySnapshot):
            raise SafetyAbort(
                "转向/返程必须使用 capture_shelf_ready 返回的 BodySnapshot"
            )
        if (
            self._captured_shelf_ready is None
            or start is not self._captured_shelf_ready
        ):
            raise SafetyAbort(
                "转向/返程必须使用本 coordinator 本轮 capture_shelf_ready "
                "返回的原始 BodySnapshot，禁止伪造或复用旧快照"
            )

    def _assert_sweep(self, config, requested_yaw_rad: float) -> None:
        if abs(requested_yaw_rad) < math.radians(80):
            raise SafetyAbort("底盘转向必须是经量测的约 90° 正向或反向扫掠")
        if abs(requested_yaw_rad) > math.radians(100):
            raise SafetyAbort("底盘转向偏离经量测的约 90° 扫掠")
        sweep = getattr(config, "rotation_sweep", None)
        direction = "positive" if requested_yaw_rad > 0.0 else "negative"
        if sweep is None:
            raise SafetyAbort("side_table_delivery 缺少正反旋转扫掠量测")
        verified = getattr(sweep, f"{direction}_verified", None)
        clearance = self._finite(
            getattr(sweep, f"{direction}_clearance_m", None),
            label=f"rotation_sweep.{direction}.clearance_m",
        )
        if verified is not True or clearance <= 0.0:
            raise SafetyAbort(
                f"{direction} 方向底盘旋转扫掠尚未现场确认，拒绝执行"
            )

    def _restoration_translation_limit(self, config) -> float:
        """Return the strongest x/y bound the rotation must preserve.

        The turn profile may permit a larger *transient* in-place-rotation
        drift than the shelf admission pose permits.  This coordinator has no
        translational correction primitive, so accepting that larger error
        would knowingly create a state that cannot be restored on return.
        """
        profile_limit = self._finite(
            config.max_base_translation_m,
            label="max_base_translation_m",
        )
        return min(profile_limit, self._shelf_ready_limits(config).xy_tolerance_m)

    def _assert_lift_target(
        self,
        snapshot: BodySnapshot,
        *,
        target_mm: int,
        tolerance_mm: int,
        label: str,
    ) -> None:
        error = abs(snapshot.lift.height_mm - target_mm)
        if error > tolerance_mm:
            raise SafetyAbort(
                f"{label} 未到位: target={target_mm} mm, "
                f"actual={snapshot.lift.height_mm} mm"
            )

    def _assert_lift_kept_chassis_still(
        self,
        before: BodySnapshot,
        after: BodySnapshot,
    ) -> None:
        translation = math.hypot(
            after.chassis.x_m - before.chassis.x_m,
            after.chassis.y_m - before.chassis.y_m,
        )
        yaw = abs(
            wrap_angle_rad(after.chassis.yaw_rad - before.chassis.yaw_rad)
        )
        if (
            translation > self._MAX_LIFT_DRIFT_M
            or yaw > self._MAX_LIFT_DRIFT_RAD
        ):
            raise SafetyAbort("升降期间底盘位姿发生变化，拒绝继续旋转")

    def _rotate(
        self,
        *,
        requested_yaw_rad: float,
        config,
    ) -> None:
        self._assert_sweep(config, requested_yaw_rad)
        self.chassis.rotate_relative(
            requested_yaw_rad,
            max_angular_speed_radps=self._finite(
                config.max_angular_speed_radps,
                label="max_angular_speed_radps",
            ),
            yaw_tolerance_rad=math.radians(
                self._finite(
                    config.rotation_tolerance_deg,
                    label="rotation_tolerance_deg",
                )
            ),
            max_translation_m=self._restoration_translation_limit(config),
            timeout_s=self._finite(
                config.rotation_timeout_s, label="rotation_timeout_s"
            ),
        )

    def _assert_turn_result(
        self,
        final: BodySnapshot,
        *,
        start: BodySnapshot,
        expected_yaw_rad: float,
        config,
        label: str,
    ) -> None:
        translation = math.hypot(
            final.chassis.x_m - start.chassis.x_m,
            final.chassis.y_m - start.chassis.y_m,
        )
        max_translation = self._restoration_translation_limit(config)
        yaw_error = abs(
            wrap_angle_rad(final.chassis.yaw_rad - expected_yaw_rad)
        )
        yaw_tolerance = math.radians(
            self._finite(
                config.rotation_tolerance_deg, label="rotation_tolerance_deg"
            )
        )
        if translation > max_translation:
            raise SafetyAbort(
                f"{label} 产生过大平移: {translation * 1000:.1f} mm"
            )
        if yaw_error > yaw_tolerance:
            raise SafetyAbort(
                f"{label} 未到位: actual={math.degrees(final.chassis.yaw_rad):.2f}°, "
                f"target={math.degrees(expected_yaw_rad):.2f}°"
            )

    def _write_evidence(
        self,
        name: str,
        *,
        config,
        start: BodySnapshot,
        before: BodySnapshot,
        after: BodySnapshot,
        operation: str,
    ) -> None:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "operation": operation,
            "start": self._snapshot_payload(start),
            "before": self._snapshot_payload(before),
            "after": self._snapshot_payload(after),
            "translation_from_start_m": math.hypot(
                after.chassis.x_m - start.chassis.x_m,
                after.chassis.y_m - start.chassis.y_m,
            ),
            "yaw_from_start_rad": wrap_angle_rad(
                after.chassis.yaw_rad - start.chassis.yaw_rad
            ),
            "profile_contract": {
                "shelf_ready": {
                    "x_m": float(config.shelf_ready.x_m),
                    "y_m": float(config.shelf_ready.y_m),
                    "yaw_deg": float(config.shelf_ready.yaw_deg),
                    "lift_height_mm": int(
                        config.shelf_ready.lift_height_mm
                    ),
                    "xy_tolerance_m": float(
                        config.shelf_ready.xy_tolerance_m
                    ),
                    "yaw_tolerance_deg": float(
                        config.shelf_ready.yaw_tolerance_deg
                    ),
                    "lift_tolerance_mm": int(
                        config.shelf_ready.lift_tolerance_mm
                    ),
                },
                "source_lift_height_mm": int(
                    config.source_lift_height_mm
                ),
                "target_lift_height_mm": int(
                    config.target_lift_height_mm
                ),
                "body_rotation_yaw_deg": float(
                    config.body_rotation_yaw_deg
                ),
            },
            "camera_extrinsic_policy": (
                "T_base_right_to_camera_head unchanged; no lift compensation"
            ),
        }
        (self.evidence_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def capture_shelf_ready(self, config) -> BodySnapshot:
        """Body-only pre-arm gate that captures the immutable task origin."""
        # A failed or superseded admission must never leave an old token usable.
        self._captured_shelf_ready = None
        self._delivery_position_start = None
        try:
            self._assert_not_stopped()
            self._validated_shelf_ready(config)
            # Validate the profile locally before touching even the chassis
            # session.  The following preflight is body-only and may send
            # zero-speed stops, never arm or gripper commands.
            self._shelf_ready_limits(config)
            start = self.preflight()
            self._assert_matches_shelf_ready(
                start, config, label="SHELF_READY"
            )
            self._write_evidence(
                "shelf_ready_body_snapshot.json",
                config=config,
                start=start,
                before=start,
                after=start,
                operation="SHELF_READY",
            )
            self._captured_shelf_ready = start
            return start
        except BaseException:
            self._captured_shelf_ready = None
            self._delivery_position_start = None
            self.stop()
            raise

    def position_for_delivery(
        self,
        config,
        *,
        start: BodySnapshot | None = None,
    ) -> BodySnapshot:
        """Lift and turn to the side table using the captured shelf origin."""
        try:
            self._assert_not_stopped()
            self._require_captured_shelf_ready(start)
            before = self.snapshot()
            self._assert_matches_start(
                before, start, config, label="送桌前底盘"
            )
            target_lift = int(
                self._finite(
                    config.target_lift_height_mm,
                    label="target_lift_height_mm",
                )
            )
            target_lift_tolerance = int(
                self._finite(
                    config.target_lift_tolerance_mm,
                    label="target_lift_tolerance_mm",
                )
            )
            self.lift.move_to(
                target_lift,
                speed=int(
                    self._finite(config.body_lift_speed, label="body_lift_speed")
                ),
            )
            after_lift = self.snapshot()
            self._assert_lift_kept_chassis_still(before, after_lift)
            self._assert_lift_target(
                after_lift,
                target_mm=target_lift,
                tolerance_mm=target_lift_tolerance,
                label="送桌升降",
            )
            self._assert_not_stopped()
            expected_yaw = wrap_angle_rad(
                start.chassis.yaw_rad
                + math.radians(
                    self._finite(
                        config.body_rotation_yaw_deg,
                        label="body_rotation_yaw_deg",
                    )
                )
            )
            requested = wrap_angle_rad(
                expected_yaw - after_lift.chassis.yaw_rad
            )
            self._rotate(requested_yaw_rad=requested, config=config)
            final = self.snapshot()
            self._assert_lift_target(
                final,
                target_mm=target_lift,
                tolerance_mm=target_lift_tolerance,
                label="送桌旋转后升降",
            )
            self._assert_turn_result(
                final,
                start=start,
                expected_yaw_rad=expected_yaw,
                config=config,
                label="送桌底盘旋转",
            )
            self._write_evidence(
                "delivery_body_state.json",
                config=config,
                start=start,
                before=before,
                after=final,
                operation="to_side_table",
            )
            self._delivery_position_start = start
            return final
        except BaseException:
            self._captured_shelf_ready = None
            self._delivery_position_start = None
            self.stop()
            raise

    def _assert_return_authorized(
        self, authorization: ReturnAuthorization
    ) -> None:
        if not isinstance(authorization, ReturnAuthorization):
            raise SafetyAbort("返程必须提供 ReturnAuthorization")
        object_state = getattr(authorization.object_state, "value", authorization.object_state)
        if authorization.release_verified is not True:
            raise SafetyAbort("释放未验证，禁止自动返转")
        if object_state != "empty":
            raise SafetyAbort(
                f"物体状态为 {object_state!r}，禁止自动返转或全局 home"
            )
        if authorization.right_arm_compact_or_home is not True:
            raise SafetyAbort("右臂未处于 compact/home，禁止底盘返程")
        if authorization.left_arm_stable is not True:
            raise SafetyAbort("左臂不稳定，禁止底盘返程")

    def return_to_shelf_ready(
        self,
        config,
        *,
        start: BodySnapshot,
        authorization: ReturnAuthorization,
    ) -> BodySnapshot:
        """Reverse the measured sweep and restore the captured shelf state."""
        try:
            self._assert_not_stopped()
            self._require_captured_shelf_ready(start)
            self._assert_return_authorized(authorization)
            if self._delivery_position_start is not start:
                raise SafetyAbort(
                    "返程必须紧接同一 coordinator 成功完成的送桌 body 操作，"
                    "禁止跳过 outbound 状态机"
                )
            current = self.snapshot()
            self._validated_shelf_ready(config)
            target_lift = int(
                self._finite(
                    config.target_lift_height_mm,
                    label="target_lift_height_mm",
                )
            )
            target_lift_tolerance = int(
                self._finite(
                    config.target_lift_tolerance_mm,
                    label="target_lift_tolerance_mm",
                )
            )
            self._assert_lift_target(
                current,
                target_mm=target_lift,
                tolerance_mm=target_lift_tolerance,
                label="返程前升降",
            )
            expected_side_yaw = wrap_angle_rad(
                start.chassis.yaw_rad
                + math.radians(
                    self._finite(
                        config.body_rotation_yaw_deg,
                        label="body_rotation_yaw_deg",
                    )
                )
            )
            side_yaw_error = abs(
                wrap_angle_rad(current.chassis.yaw_rad - expected_side_yaw)
            )
            side_yaw_tolerance = math.radians(
                self._finite(
                    config.rotation_tolerance_deg,
                    label="rotation_tolerance_deg",
                )
            )
            if side_yaw_error > side_yaw_tolerance:
                raise SafetyAbort(
                    "返程起点不在已验证的侧桌朝向，拒绝猜测回转角"
                )
            side_translation = math.hypot(
                current.chassis.x_m - start.chassis.x_m,
                current.chassis.y_m - start.chassis.y_m,
            )
            max_translation = self._restoration_translation_limit(config)
            if side_translation > max_translation:
                # The adapter exposes no translational correction primitive.
                # Turning from a known-bad pose would only hide the error and
                # cannot satisfy the snapshot restoration contract.
                raise SafetyAbort(
                    "返程前已产生过大平移，拒绝在错误 x/y 上命令反向旋转"
                )
            requested = wrap_angle_rad(
                start.chassis.yaw_rad - current.chassis.yaw_rad
            )
            self._rotate(requested_yaw_rad=requested, config=config)
            after_rotate = self.snapshot()
            self._assert_turn_result(
                after_rotate,
                start=start,
                expected_yaw_rad=start.chassis.yaw_rad,
                config=config,
                label="返程底盘旋转",
            )
            source_lift = int(
                self._finite(
                    config.source_lift_height_mm,
                    label="source_lift_height_mm",
                )
            )
            source_lift_tolerance = int(
                self._finite(
                    config.shelf_ready.lift_tolerance_mm,
                    label="SHELF_READY.lift_tolerance_mm",
                )
            )
            self.lift.move_to(
                source_lift,
                speed=int(
                    self._finite(config.body_lift_speed, label="body_lift_speed")
                ),
            )
            final = self.snapshot()
            self._assert_lift_kept_chassis_still(after_rotate, final)
            self._assert_lift_target(
                final,
                target_mm=source_lift,
                tolerance_mm=source_lift_tolerance,
                label="返程升降",
            )
            self._assert_matches_start(
                final, start, config, label="返程完成底盘"
            )
            self._write_evidence(
                "return_body_state.json",
                config=config,
                start=start,
                before=current,
                after=final,
                operation="to_shelf_ready",
            )
            self._captured_shelf_ready = None
            self._delivery_position_start = None
            return final
        except BaseException:
            self._captured_shelf_ready = None
            self._delivery_position_start = None
            self.stop()
            raise
