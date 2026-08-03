"""Force the head-camera servo (pitch/yaw) to its calibrated reference angle.

`config.yaml`'s `calibration.T_base_right_to_camera_head` was solved with the
head pinned at HEAD_REFERENCE (pitch lowest, yaw centered) — see the
eye-to-hand calibration session notes. Any drift off that angle silently
invalidates every 3-D point the head camera computes downstream. The head can
drift for reasons unrelated to this demo (manual `head_camera_control.py`
use, or an SDK side effect on `ArmController`/`RobotSession` init that has
been observed to nudge the servo — see project memory on teleop/SDK
coexistence), so it must be re-forced to the reference angle before anything
else runs, every run, rather than assumed correct.

Protocol lifted from `scripts/head_position_lock.py` (UDP broadcast IO frames
to `head_servo_ctrl.py` + an angle broadcast listener on a separate port),
kept import-safe here so `shelf_dispenser/demo.py` can call it directly instead
of shelling out to a subprocess.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import select
import socket
import subprocess
import termios
import time
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("bottle_demo")

BROADCAST_IP = "169.254.128.255"
CONTROL_PORT = 19999
ANGLE_PORT = 9996

HEAD_CTRL_IO = 5
UP_IO, DOWN_IO, LEFT_IO, RIGHT_IO = 6, 7, 8, 9

# 2026-07-08 标定会话实测基准值：俯仰(angle1)最低、偏航(angle2)居中。
HEAD_REFERENCE = {"angle1": 398, "angle2": 516}
TOLERANCE = 5  # 舵机反馈本身有几个单位的抖动

# head_servo_ctrl.py 里角度查询线程和运动指令共用同一把串口锁：执行运动
# 指令期间角度广播实测会停 2~4 秒（2026-07-15 真机日志）。这是正常间隙，
# 不代表进程死了，必须容忍到远大于间隙的窗口再判定广播中断。
BROADCAST_GAP_PATIENCE = 8.0

# head_servo_ctrl.py 自动重启参数。它 `import serial`，而系统
# /usr/bin/python3 没装 pyserial（用它启动会立刻崩），只能用 conda python
# ——见项目记忆 teleop-sdk-coexist。启动时厂商脚本会先把头回 [500,500]
# 再开始广播，所以重启后要多等一会儿。
HEAD_SERVO_SCRIPT_DIR = "/home/rm/rmc_aida_l_atom/scripts"
HEAD_SERVO_SCRIPT = "head_servo_ctrl.py"
HEAD_SERVO_PYTHON = "/home/rm/miniconda3/bin/python3"
HEAD_SERVO_RESTART_LOG = "/tmp/head_servo_ctrl.autorestart.log"
RESTART_BROADCAST_WAIT = 25.0

# 闭环里连续发了运动指令但角度完全不变，说明控制链路（串口写入侧）卡死，
# 广播还活着也没用——这种也用重启来救。
STAGNANT_STEPS_BEFORE_RESTART = 3

DIRECT_SERIAL_PATHS = ("/dev/rmUSB3", "/dev/ttyUSB0")
DIRECT_ANGLE_QUERY = bytes([0x55, 0x55, 0x05, 0x15, 0x02, 0x01, 0x02])


def _make_io_frame(*pressed_ios: int) -> bytes:
    frame = bytearray(34)
    frame[0] = 0x01
    frame[1] = 0x04
    frame[2] = 0x20
    for io_num in pressed_ios:
        frame[2 + io_num * 2] = 1
    return bytes(frame)


def _open_angle_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", ANGLE_PORT))
    sock.setblocking(False)
    return sock


def _drain_stale_frames(sock: socket.socket) -> None:
    while True:
        ready, _, _ = select.select([sock], [], [], 0)
        if not ready:
            return
        try:
            sock.recvfrom(2048)
        except OSError:
            return


def _wait_fresh_angle(sock: socket.socket, patience: float) -> Optional[dict]:
    """丢掉积压的旧帧，等下一条实时广播；等不到返回 None。

    广播帧是 head_servo_ctrl.py 实时查询串口后立刻发出的，所以"排空积压后
    收到的第一帧"就是当前真实角度，不需要再像旧实现那样固定等满整个窗口
    收集"最新帧"。
    """
    _drain_stale_frames(sock)
    deadline = time.time() + patience
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return None
        ready, _, _ = select.select([sock], [], [], remaining)
        if not ready:
            return None
        data, _ = sock.recvfrom(2048)
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue


def restart_head_servo() -> Optional[str]:
    """杀掉（可能已死或卡住的）head_servo_ctrl.py 并用 conda python 拉起。

    返回 None 表示已发起重启，否则返回不可恢复的原因。厂商脚本启动时会
    先把头回 [500,500] 才开始广播/接受指令，调用方要用
    RESTART_BROADCAST_WAIT 级别的窗口等广播恢复，之后照常闭环回基准。
    """
    script = Path(HEAD_SERVO_SCRIPT_DIR) / HEAD_SERVO_SCRIPT
    if not script.exists():
        return f"{script} 不存在（当前不在机器人上运行？）"
    if not Path(HEAD_SERVO_PYTHON).exists():
        return f"{HEAD_SERVO_PYTHON} 不存在，无法用带 pyserial 的解释器启动"
    LOG.warning(
        "head_servo_ctrl.py 无响应，自动重启：pkill 旧进程 → %s 拉起，日志 %s",
        HEAD_SERVO_PYTHON,
        HEAD_SERVO_RESTART_LOG,
    )
    subprocess.run(["pkill", "-f", HEAD_SERVO_SCRIPT], check=False)
    time.sleep(2.0)  # 等旧进程退出、串口和 19999 端口释放
    with open(HEAD_SERVO_RESTART_LOG, "ab") as log:
        log.write(
            f"\n== {time.strftime('%F %T')} shelf_dispenser 自动重启 ==\n".encode()
        )
        subprocess.Popen(
            [HEAD_SERVO_PYTHON, HEAD_SERVO_SCRIPT],
            cwd=HEAD_SERVO_SCRIPT_DIR,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    return None


def _send_action(action: str, repeat: int = 4, interval: float = 0.05) -> None:
    actions = {"u": UP_IO, "d": DOWN_IO, "l": LEFT_IO, "r": RIGHT_IO}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    press = _make_io_frame(HEAD_CTRL_IO, actions[action])
    release = _make_io_frame()
    for _ in range(repeat):
        sock.sendto(press, (BROADCAST_IP, CONTROL_PORT))
        time.sleep(interval)
    sock.sendto(release, (BROADCAST_IP, CONTROL_PORT))
    sock.close()


def read_current_angle(patience: float = 3.0) -> Optional[dict]:
    sock = _open_angle_socket()
    try:
        return _wait_fresh_angle(sock, patience)
    finally:
        sock.close()


def _decode_direct_angle_response(response: bytes) -> Optional[dict]:
    if (
        len(response) != 11
        or response[:2] != b"\x55\x55"
        or response[2] != 0x09
        or response[3:5] != b"\x15\x02"
    ):
        return None
    angles = {
        response[5]: (response[7] << 8) | response[6],
        response[8]: (response[10] << 8) | response[9],
    }
    if 1 not in angles or 2 not in angles:
        return None
    return {"angle1": angles[1], "angle2": angles[2]}


def _serial_owner_pids(path: str) -> Optional[set[int]]:
    """Return owners, or None when fuser cannot prove the answer."""
    try:
        result = subprocess.run(
            ["fuser", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode == 0:
        owners = {
            int(token) for token in result.stdout.split() if token.isdigit()
        }
        return owners or None
    if (
        result.returncode == 1
        and not result.stdout.strip()
        and not result.stderr.strip()
    ):
        return set()
    return None


def read_current_angle_direct(timeout: float = 2.0) -> Optional[dict]:
    """Read the two servos without starting the vendor service or moving them."""
    serial_paths = tuple(
        path for path in DIRECT_SERIAL_PATHS if Path(path).exists()
    )
    for path in serial_paths:
        owners = _serial_owner_pids(path)
        if owners is None:
            return None
        if owners - {os.getpid()}:
            LOG.warning("头部串口 %s 正被占用，拒绝直接读取", path)
            return None

    for path in serial_paths:
        port = None
        try:
            port = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            fcntl.ioctl(port, termios.TIOCEXCL)
            for checked_path in serial_paths:
                owners = _serial_owner_pids(checked_path)
                if owners is None or owners - {os.getpid()}:
                    raise OSError("head serial ownership changed while opening")
            attributes = termios.tcgetattr(port)
            attributes[0] = 0
            attributes[1] = 0
            attributes[2] = termios.CLOCAL | termios.CREAD | termios.CS8
            attributes[3] = 0
            attributes[4] = termios.B9600
            attributes[5] = termios.B9600
            attributes[6][termios.VMIN] = 0
            attributes[6][termios.VTIME] = 0
            termios.tcsetattr(port, termios.TCSANOW, attributes)
            termios.tcflush(port, termios.TCIFLUSH)
            time.sleep(0.2)
            os.write(port, DIRECT_ANGLE_QUERY)
            deadline = time.monotonic() + timeout
            response = b""
            while len(response) < 11:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ready, _, _ = select.select([port], [], [], remaining)
                if not ready:
                    break
                response += os.read(port, 11 - len(response))
        except (OSError, termios.error):
            continue
        finally:
            if port is not None:
                os.close(port)
        current = _decode_direct_angle_response(response)
        if current is not None:
            return current
    return None


def is_at_reference(current: Optional[dict]) -> bool:
    if not current:
        return False
    d1 = current["angle1"] - HEAD_REFERENCE["angle1"]
    d2 = current["angle2"] - HEAD_REFERENCE["angle2"]
    return abs(d1) <= TOLERANCE and abs(d2) <= TOLERANCE


def restore_reference(max_steps: int = 20, allow_restart: bool = True) -> dict:
    """闭环把头部舵机调回 HEAD_REFERENCE，每个轴独立收敛。

    自愈：角度广播中断超过 BROADCAST_GAP_PATIENCE、或连续发运动指令但角度
    完全不动（控制链路卡死）时，自动重启 head_servo_ctrl.py 一次再继续。
    返回 {"ok": bool, "angle": 最后读到的角度或 None, "reason"/"steps": ...}，
    不抛异常——是否因此中止整个流程由调用方（demo.py）决定。
    """
    sock = _open_angle_socket()
    restarted = False

    def _recover(why: str) -> Optional[dict]:
        nonlocal restarted
        if not allow_restart or restarted:
            return None
        restarted = True
        LOG.warning("头部舵机自愈触发（%s）", why)
        fail = restart_head_servo()
        if fail is not None:
            LOG.error("无法自动重启 head_servo_ctrl.py: %s", fail)
            return None
        fresh = _wait_fresh_angle(sock, RESTART_BROADCAST_WAIT)
        if fresh is not None:
            LOG.info("head_servo_ctrl.py 重启成功，广播已恢复: %s", fresh)
        return fresh

    try:
        current = None
        previous = None
        stagnant = 0
        for step in range(1, max_steps + 1):
            current = _wait_fresh_angle(sock, BROADCAST_GAP_PATIENCE)
            if current is None:
                current = _recover(
                    f"连续 {BROADCAST_GAP_PATIENCE:.0f}s 没收到角度广播"
                )
            if current is None:
                return {
                    "ok": False,
                    "angle": None,
                    "reason": (
                        "角度广播中断，自动重启 head_servo_ctrl.py 也没恢复；"
                        f"看 {HEAD_SERVO_RESTART_LOG}，或手动: "
                        f"cd {HEAD_SERVO_SCRIPT_DIR} && "
                        f"{HEAD_SERVO_PYTHON} {HEAD_SERVO_SCRIPT}"
                        if restarted
                        else f"连续 {BROADCAST_GAP_PATIENCE:.0f}s 没收到角度广播"
                        "（且本次调用不允许自动重启）"
                    ),
                }
            if is_at_reference(current):
                return {"ok": True, "angle": current, "steps": step - 1}

            if previous is not None and (
                abs(current["angle1"] - previous["angle1"]) <= 1
                and abs(current["angle2"] - previous["angle2"]) <= 1
            ):
                stagnant += 1
            else:
                stagnant = 0
            previous = current
            if stagnant >= STAGNANT_STEPS_BEFORE_RESTART:
                recovered = _recover(
                    f"连续 {stagnant} 步发了运动指令但角度不动，控制链路疑似卡死"
                )
                if recovered is not None:
                    current = previous = recovered
                    stagnant = 0
                    if is_at_reference(current):
                        return {"ok": True, "angle": current, "steps": step}

            d1 = current["angle1"] - HEAD_REFERENCE["angle1"]
            d2 = current["angle2"] - HEAD_REFERENCE["angle2"]
            actions = []
            if d1 < -TOLERANCE:
                actions.append("u")
            elif d1 > TOLERANCE:
                actions.append("d")
            if d2 < -TOLERANCE:
                actions.append("l")
            elif d2 > TOLERANCE:
                actions.append("r")
            LOG.info(
                "头部回中 step %d: current=%s delta=(%+d,%+d) actions=%s",
                step,
                current,
                d1,
                d2,
                actions,
            )
            for action in actions:
                _send_action(action)
                time.sleep(0.4)
        return {
            "ok": False,
            "angle": current,
            "reason": f"达到 max_steps={max_steps} 仍未收敛",
        }
    finally:
        sock.close()
