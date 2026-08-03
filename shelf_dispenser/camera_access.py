"""Exclusive-access preflight for the RealSense cameras used by bottle grasping.

The robot also runs browser preview utilities.  Those utilities open V4L2 nodes
directly and can keep a wrist camera busy even when their command line says
``--camera head`` (the web UI can switch sources after startup).  This module
therefore determines ownership from the file descriptors that are actually
open, never from the requested camera name.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


LOG = logging.getLogger("bottle_demo.camera")

# These are camera-only preview programs.  It is safe for the grasp launcher to
# ask them to exit; arbitrary processes are reported but deliberately not killed.
KNOWN_PREVIEW_PROGRAMS = frozenset(
    {
        "head_camera_control.py",
        "wrist_camera_server.py",
    }
)


class CameraAccessError(RuntimeError):
    """Base class for an actionable camera preflight failure."""


class CameraInUseError(CameraAccessError):
    """Raised when a process still owns one of the target camera's nodes."""


@dataclass(frozen=True)
class CameraOwner:
    pid: int
    command: tuple[str, ...]
    nodes: tuple[Path, ...]
    known_preview: bool

    def describe(self) -> str:
        command = " ".join(self.command) if self.command else "<命令不可读>"
        nodes = ",".join(node.name for node in self.nodes)
        return f"PID {self.pid} ({command}) 持有 {nodes}"


def device_root_from_physical_port(physical_port: str) -> Path:
    """Return the USB-device directory shared by all interfaces of a camera."""
    path = Path(physical_port)
    # .../<usb-device>/<usb-device>:1.0/video4linux/videoN
    if len(path.parents) < 3 or not path.name.startswith("video"):
        raise CameraAccessError(f"无法解析 RealSense physical_port: {physical_port}")
    return path.parents[2]


def _known_preview(command: Sequence[str]) -> bool:
    return any(Path(argument).name in KNOWN_PREVIEW_PROGRAMS for argument in command)


def _read_cmdline(process_dir: Path) -> tuple[str, ...]:
    try:
        raw = (process_dir / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return ()
    return tuple(
        part.decode("utf-8", errors="replace")
        for part in raw.split(b"\0")
        if part
    )


def _fd_target(fd: Path) -> Path | None:
    try:
        raw = os.readlink(fd)
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    target = Path(raw)
    if not target.is_absolute():
        target = fd.parent / target
    return Path(os.path.normpath(str(target)))


def find_camera_owners(
    video_nodes: Iterable[Path],
    *,
    proc_root: Path = Path("/proc"),
    ignore_pid: int | None = None,
) -> list[CameraOwner]:
    """Inspect ``/proc/*/fd`` and return owners of the exact V4L2 nodes."""
    targets = {Path(os.path.normpath(str(node))) for node in video_nodes}
    owners: list[CameraOwner] = []
    if not targets:
        return owners

    try:
        process_dirs = list(proc_root.iterdir())
    except (FileNotFoundError, PermissionError, OSError):
        return owners
    for process_dir in process_dirs:
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        if pid == ignore_pid:
            continue
        held: set[Path] = set()
        try:
            fds = list((process_dir / "fd").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        for fd in fds:
            target = _fd_target(fd)
            if target in targets:
                held.add(target)
        if not held:
            continue
        command = _read_cmdline(process_dir)
        owners.append(
            CameraOwner(
                pid=pid,
                command=command,
                nodes=tuple(sorted(held, key=str)),
                known_preview=_known_preview(command),
            )
        )
    return sorted(owners, key=lambda owner: owner.pid)


def _owners_message(owners: Sequence[CameraOwner]) -> str:
    return "；".join(owner.describe() for owner in owners)


def release_known_preview_owners(
    video_nodes: Iterable[Path],
    *,
    proc_root: Path = Path("/proc"),
    terminate: Callable[[int, signal.Signals], None] = os.kill,
    timeout_s: float = 5.0,
) -> list[int]:
    """Gracefully stop known preview owners; never kill an unknown process."""
    nodes = {Path(node) for node in video_nodes}
    owners = find_camera_owners(nodes, proc_root=proc_root, ignore_pid=os.getpid())
    unknown = [owner for owner in owners if not owner.known_preview]
    if unknown:
        raise CameraInUseError(
            "相机被未知进程占用，已拒绝自动终止：" + _owners_message(unknown)
        )

    released: list[int] = []
    for owner in owners:
        LOG.warning("[相机占用清理] 正在停止已知预览：%s", owner.describe())
        try:
            terminate(owner.pid, signal.SIGTERM)
            released.append(owner.pid)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            raise CameraInUseError(
                f"无权停止相机预览 PID {owner.pid}: {exc}"
            ) from exc

    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        remaining = find_camera_owners(
            nodes, proc_root=proc_root, ignore_pid=os.getpid()
        )
        if not remaining:
            return released
        if time.monotonic() >= deadline:
            raise CameraInUseError(
                "预览进程收到 SIGTERM 后仍未释放相机："
                + _owners_message(remaining)
            )
        time.sleep(0.1)


def camera_video_nodes(
    serial: str,
    *,
    sys_class: Path = Path("/sys/class/video4linux"),
    dev_root: Path = Path("/dev"),
) -> set[Path]:
    """Resolve a RealSense serial number to every V4L2 node it owns."""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraAccessError("pyrealsense2 不可用，无法执行相机占用自检") from exc

    device = None
    available: list[str] = []
    for candidate in rs.context().query_devices():
        candidate_serial = candidate.get_info(rs.camera_info.serial_number)
        available.append(candidate_serial)
        if candidate_serial == serial:
            device = candidate
    if device is None:
        raise CameraAccessError(
            f"未找到 RealSense {serial}；当前设备: {available or '无'}"
        )

    physical_port = device.get_info(rs.camera_info.physical_port)
    device_root = device_root_from_physical_port(physical_port).resolve()
    nodes: set[Path] = set()
    try:
        entries = list(sys_class.glob("video*"))
    except (FileNotFoundError, PermissionError, OSError):
        entries = []
    for entry in entries:
        try:
            resolved = entry.resolve()
            resolved.relative_to(device_root)
        except (FileNotFoundError, RuntimeError, ValueError, OSError):
            continue
        nodes.add(dev_root / entry.name)
    if not nodes:
        raise CameraAccessError(
            f"RealSense {serial} 已枚举，但无法解析对应 /dev/video* 节点"
        )
    return nodes


def prepare_camera_access(serial: str) -> list[int]:
    """Release known preview owners of one camera and return stopped PIDs."""
    nodes = camera_video_nodes(serial)
    released = release_known_preview_owners(nodes)
    if released:
        # V4L2/librealsense may need a short settle period after close().
        time.sleep(0.4)
        LOG.info("[相机占用清理] RealSense %s 已释放", serial)
    else:
        LOG.info("[相机占用检查] RealSense %s 未被其他进程占用", serial)
    return released


def hardware_reset_camera(serial: str, *, settle_s: float = 5.0) -> None:
    """Reset exactly one RealSense after ownership and pipeline retries fail."""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraAccessError("pyrealsense2 不可用，无法重启相机") from exc

    devices = list(rs.context().query_devices())
    target = next(
        (
            device
            for device in devices
            if device.get_info(rs.camera_info.serial_number) == serial
        ),
        None,
    )
    if target is None:
        available = [
            device.get_info(rs.camera_info.serial_number) for device in devices
        ]
        raise CameraAccessError(
            f"无法重启 RealSense {serial}；当前设备: {available or '无'}"
        )
    LOG.warning(
        "[相机硬件恢复] RealSense %s 连续无帧，执行一次 hardware_reset",
        serial,
    )
    try:
        target.hardware_reset()
    except Exception as exc:
        raise CameraAccessError(
            f"RealSense {serial} hardware_reset 失败: {exc}"
        ) from exc

    # The USB device disappears and re-enumerates asynchronously.  A fixed
    # settle avoids accepting the stale pre-reset device object too early.
    time.sleep(max(0.0, settle_s))
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        try:
            serials = [
                device.get_info(rs.camera_info.serial_number)
                for device in rs.context().query_devices()
            ]
        except Exception:
            serials = []
        if serial in serials:
            LOG.info("[相机硬件恢复] RealSense %s 已重新枚举", serial)
            return
        time.sleep(0.5)
    raise CameraAccessError(
        f"RealSense {serial} hardware_reset 后 20 秒仍未重新枚举"
    )


def probe_camera(
    serial: str,
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    attempts: int = 3,
) -> None:
    """Open the same color+depth streams used by the demo, get frames, release."""
    import pyrealsense2 as rs

    error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        pipeline = rs.pipeline()
        started = False
        try:
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            pipeline.start(config)
            started = True
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            if not frames.get_color_frame() or not frames.get_depth_frame():
                raise CameraAccessError("已打开 pipeline，但未同时收到彩色和深度帧")
            LOG.info(
                "[相机打开实测] RealSense %s %dx%d@%dfps 彩色+深度正常",
                serial,
                width,
                height,
                fps,
            )
            return
        except Exception as exc:  # librealsense exposes failures as RuntimeError
            error = exc
            if attempt < max(1, attempts):
                LOG.warning(
                    "[相机打开重试] RealSense %s 第 %d/%d 次失败: %s",
                    serial,
                    attempt,
                    attempts,
                    exc,
                )
                time.sleep(0.5)
        finally:
            if started:
                try:
                    pipeline.stop()
                except Exception:
                    pass
    nodes = camera_video_nodes(serial)
    owners = find_camera_owners(nodes, ignore_pid=os.getpid())
    owner_detail = f"；当前持有者: {_owners_message(owners)}" if owners else ""
    raise CameraAccessError(
        f"RealSense {serial} 连续 {max(1, attempts)} 次打开失败: {error}{owner_detail}"
    ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bottle grasp camera ownership preflight")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--camera",
        action="append",
        required=True,
        choices=("head", "right_wrist", "left_wrist"),
    )
    parser.add_argument("--no-probe", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from utils.config import load_config

    config = load_config(args.config)
    try:
        for camera_name in dict.fromkeys(args.camera):
            serial = config.camera.serial_for(camera_name)
            prepare_camera_access(serial)
            if not args.no_probe:
                probe_camera(
                    serial,
                    width=config.camera.width,
                    height=config.camera.height,
                    fps=config.camera.fps,
                )
    except CameraAccessError as exc:
        LOG.error("[相机自检失败] %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
