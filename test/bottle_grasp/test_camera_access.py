import os
import signal
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from bottle_grasp.camera_access import (
    CameraInUseError,
    device_root_from_physical_port,
    find_camera_owners,
    release_known_preview_owners,
)
from bottle_grasp.demo import BottleDemo
from bottle_grasp.core import SafetyAbort
import bottle_grasp.demo as demo_module


def _fake_owner(proc_root: Path, pid: int, command: list[str], node: str) -> None:
    process = proc_root / str(pid)
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"\0".join(part.encode() for part in command) + b"\0")
    (process / "fd" / "4").symlink_to(node)


def test_device_root_is_shared_by_all_realsense_interfaces():
    physical_port = (
        "/sys/devices/platform/bus@0/3610000.usb/usb2/2-3/2-3.2/"
        "2-3.2.4/2-3.2.4.3/2-3.2.4.3:1.0/video4linux/video16"
    )

    assert device_root_from_physical_port(physical_port) == Path(
        "/sys/devices/platform/bus@0/3610000.usb/usb2/2-3/2-3.2/"
        "2-3.2.4/2-3.2.4.3"
    )


def test_finds_process_holding_any_target_video_node(tmp_path):
    proc_root = tmp_path / "proc"
    _fake_owner(
        proc_root,
        14976,
        ["python3", "head_camera_control.py", "--camera", "head"],
        "/dev/video20",
    )

    owners = find_camera_owners({Path("/dev/video20")}, proc_root=proc_root)

    assert len(owners) == 1
    assert owners[0].pid == 14976
    assert owners[0].command == (
        "python3",
        "head_camera_control.py",
        "--camera",
        "head",
    )
    assert owners[0].known_preview


def test_releases_only_known_preview_process(tmp_path):
    proc_root = tmp_path / "proc"
    _fake_owner(
        proc_root,
        14976,
        ["python3", "/home/rm/test/head_camera_control.py", "--camera", "head"],
        "/dev/video20",
    )
    signals = []

    def terminate(pid, sig):
        signals.append((pid, sig))
        os.unlink(proc_root / str(pid) / "fd" / "4")

    released = release_known_preview_owners(
        {Path("/dev/video20")},
        proc_root=proc_root,
        terminate=terminate,
        timeout_s=0,
    )

    assert released == [14976]
    assert signals == [(14976, signal.SIGTERM)]


def test_unknown_camera_owner_is_reported_and_never_killed(tmp_path):
    proc_root = tmp_path / "proc"
    _fake_owner(
        proc_root,
        8123,
        ["python3", "/home/rm/custom_capture.py"],
        "/dev/video20",
    )

    with pytest.raises(CameraInUseError) as caught:
        release_known_preview_owners(
            {Path("/dev/video20")},
            proc_root=proc_root,
            terminate=lambda *_: pytest.fail("unknown owner must not be killed"),
            timeout_s=0,
        )

    message = str(caught.value)
    assert "PID 8123" in message
    assert "custom_capture.py" in message


@pytest.mark.parametrize(
    ("working_index", "expected_cameras", "expected_resets"),
    [(1, 2, 0), (2, 3, 1)],
)
def test_demo_rebuilds_pipeline_and_resets_hardware_only_after_two_failures(
    monkeypatch, tmp_path, working_index, expected_cameras, expected_resets
):
    cameras = []
    resets = []

    class FakeCamera:
        def __init__(self, **kwargs):
            self.initialization_successful = True
            self.initialization_error = None
            self.index = len(cameras)
            self.stopped = False
            cameras.append(self)

        def start(self):
            return None

        def get_latest_frames(self):
            if self.index < working_index:
                return None, None
            return object(), object()

        def stop(self):
            self.stopped = True

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    class FakePreview:
        def __init__(self, camera, state):
            self.camera = camera

        def start(self):
            return None

    fake_sensors = SimpleNamespace(CameraThread=FakeCamera)
    monkeypatch.setitem(sys.modules, "sensors.camera_thread", fake_sensors)
    monkeypatch.setattr(demo_module, "prepare_camera_access", lambda serial: [])
    monkeypatch.setattr(
        demo_module,
        "hardware_reset_camera",
        lambda serial: resets.append(serial),
    )
    monkeypatch.setattr(demo_module, "PreviewWorker", FakePreview)
    clock = iter(float(value) for value in range(30))
    monkeypatch.setattr(demo_module.time, "time", lambda: next(clock))
    monkeypatch.setattr(demo_module.time, "sleep", lambda _: None)

    demo = BottleDemo.__new__(BottleDemo)
    demo.preview = None
    demo.camera = None
    demo.camera_name = ""
    demo.wrist_detector = object()
    demo.project_root = tmp_path
    demo.params = SimpleNamespace(head_width=640, head_height=480)
    demo.cfg = SimpleNamespace(
        camera=SimpleNamespace(
            serial_for=lambda _: "405622073249",
            width=640,
            height=480,
            fps=30,
        )
    )
    demo.state = SimpleNamespace(update=lambda **kwargs: None)

    demo._start_camera("right_wrist")

    assert len(cameras) == expected_cameras
    assert cameras[0].stopped
    assert demo.camera is cameras[-1]
    assert resets == ["405622073249"] * expected_resets


def test_camera_switch_fails_closed_when_previous_thread_does_not_stop(
    monkeypatch, tmp_path
):
    constructed = []

    class StuckCamera:
        stopped = False

        def stop(self):
            self.stopped = True

        @staticmethod
        def is_alive():
            return True

        @staticmethod
        def join(timeout=None):
            return None

    class NewCamera:
        def __init__(self, **_kwargs):
            constructed.append(self)
            self.initialization_successful = True

        @staticmethod
        def start():
            return None

        @staticmethod
        def get_latest_frames():
            return object(), object()

        @staticmethod
        def is_alive():
            return False

        @staticmethod
        def stop():
            return None

    monkeypatch.setitem(
        sys.modules,
        "sensors.camera_thread",
        SimpleNamespace(CameraThread=NewCamera),
    )
    monkeypatch.setattr(demo_module, "prepare_camera_access", lambda _serial: [])
    monkeypatch.setattr(
        demo_module,
        "PreviewWorker",
        lambda *_args, **_kwargs: SimpleNamespace(start=lambda: None),
    )

    demo = BottleDemo.__new__(BottleDemo)
    demo.preview = None
    demo.camera = StuckCamera()
    demo.camera_name = "head"
    demo.wrist_detector = object()
    demo.project_root = tmp_path
    demo.params = SimpleNamespace(head_width=848, head_height=480)
    demo.cfg = SimpleNamespace(
        camera=SimpleNamespace(
            serial_for=lambda _: "405622073249",
            width=640,
            height=480,
            fps=30,
        )
    )
    demo.state = SimpleNamespace(update=lambda **_kwargs: None)

    with pytest.raises(SafetyAbort, match="相机线程停止超时"):
        demo._start_camera("right_wrist")

    assert demo.camera.stopped
    assert constructed == []
