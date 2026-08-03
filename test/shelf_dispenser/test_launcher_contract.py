"""Public bottle-task launcher contract tests.

The launcher normally talks to a real robot.  These tests replace both remote
transport commands with inert recorders, so they can prove that invalid
configuration never reaches transport and that accepted configuration is
forwarded without contacting hardware.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "run_task.sh"


def _fake_transport(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Return an environment whose ssh/rsync commands only append arguments."""

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    call_log = tmp_path / "transport-calls.log"
    recorder = (
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "printf '%s\\n' \"$@\" >> \"${LAUNCHER_CALL_LOG:?}\"\n"
    )
    for command in ("ssh", "rsync"):
        path = fake_bin / command
        path.write_text(recorder, encoding="utf-8")
        path.chmod(0o755)

    env = os.environ.copy()
    for name in (
        "COMMISSIONING_SPEED",
        "BOTTLE_GRASP_TRAJECTORY_MODE",
        "VISUAL_MODE",
        "VISUAL_SERVO",
        "STOP_AFTER_OBSERVATION",
        "CONFIRM_BEFORE_GRASP",
    ):
        env.pop(name, None)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["LAUNCHER_CALL_LOG"] = str(call_log)
    return env, call_log


def _run_launcher(
    env: dict[str, str], confirmation: str = "开始", mode: str = "from-start"
):
    return subprocess.run(
        ["bash", str(LAUNCHER), mode],
        cwd=ROOT,
        env=env,
        input=f"{confirmation}\n",
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def _make_local_provenance_unavailable(env: dict[str, str]) -> None:
    """Put a failing ``python3`` ahead of the local standard-library helper."""

    fake_bin = Path(env["PATH"].split(os.pathsep, 1)[0])
    python = fake_bin / "python3"
    python.write_text("#!/usr/bin/env bash\nexit 70\n", encoding="utf-8")
    python.chmod(0o755)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("COMMISSIONING_SPEED", "0", "COMMISSIONING_SPEED"),
        ("COMMISSIONING_SPEED", "101", "COMMISSIONING_SPEED"),
        ("COMMISSIONING_SPEED", "fast", "COMMISSIONING_SPEED"),
        (
            "BOTTLE_GRASP_TRAJECTORY_MODE",
            "queued",
            "BOTTLE_GRASP_TRAJECTORY_MODE",
        ),
        ("VISUAL_MODE", "enabled", "VISUAL_MODE"),
        ("VISUAL_SERVO", "2", "VISUAL_SERVO"),
        ("STOP_AFTER_OBSERVATION", "2", "STOP_AFTER_OBSERVATION"),
        ("CONFIRM_BEFORE_GRASP", "yes", "CONFIRM_BEFORE_GRASP"),
    ],
)
def test_invalid_public_controls_fail_before_any_transport(
    tmp_path, name, value, message
):
    env, call_log = _fake_transport(tmp_path)
    env[name] = value

    result = _run_launcher(env)

    assert result.returncode == 1
    assert message in result.stderr
    assert "同步当前 bottle task" not in result.stdout
    assert not call_log.exists(), "invalid config must not reach rsync or ssh"


def test_summary_and_operator_cancel_happen_before_any_transport(tmp_path):
    env, call_log = _fake_transport(tmp_path)
    env.update(
        {
            "COMMISSIONING_SPEED": "7",
            "BOTTLE_GRASP_TRAJECTORY_MODE": "blocking",
            "VISUAL_MODE": "shadow",
        }
    )

    result = _run_launcher(env, confirmation="取消")

    assert result.returncode == 2
    assert "commissioning 速度上限: 7%" in result.stdout
    assert "MoveIt 轨迹执行: blocking" in result.stdout
    assert "预抓取视觉闭环: shadow" in result.stdout
    assert not call_log.exists(), "cancelling before start must not transport code"


def test_unavailable_local_provenance_fails_before_any_transport(tmp_path):
    env, call_log = _fake_transport(tmp_path)
    _make_local_provenance_unavailable(env)

    result = _run_launcher(env)

    assert result.returncode == 1
    assert "本地 Git 溯源" in result.stderr
    assert "即将执行真机任务" not in result.stdout
    assert not call_log.exists(), "provenance failure must precede rsync and ssh"


def test_launcher_forwards_low_speed_blocking_shadow_and_stop_gate_without_hardware(
    tmp_path,
):
    env, call_log = _fake_transport(tmp_path)
    env.update(
        {
            "COMMISSIONING_SPEED": "7",
            "BOTTLE_GRASP_TRAJECTORY_MODE": "blocking",
            "VISUAL_MODE": "shadow",
            # An explicit modern mode wins over the compatibility input.
            "VISUAL_SERVO": "1",
            "STOP_AFTER_OBSERVATION": "1",
        }
    )

    result = _run_launcher(env)

    assert result.returncode == 0, result.stderr
    assert "commissioning 速度上限: 7%" in result.stdout
    assert "MoveIt 轨迹执行: blocking" in result.stdout
    assert "预抓取视觉闭环: shadow" in result.stdout
    assert "阶段入口: stop-after-observation" in result.stdout
    assert result.stdout.index("MoveIt 轨迹执行") < result.stdout.index(
        "视频已开始且急停就位后"
    )

    calls = call_log.read_text(encoding="utf-8")
    # The fake commands only record arguments.  No SSH, ROS, or robot command
    # is executed by this test.
    assert "BOTTLE_GRASP_CONTINUOUS_TRAJECTORY='0'" in calls
    assert re.search(
        r"BOTTLE_GRASP_SOURCE_GIT_SHA='[0-9a-f]{40}(?:[0-9a-f]{24})?'",
        calls,
    )
    assert re.search(r"BOTTLE_GRASP_SOURCE_DIRTY='[01]'", calls)
    assert re.search(
        r"BOTTLE_GRASP_SOURCE_DIRTY_DIGEST='[0-9a-f]{64}'", calls
    )
    assert (
        "BOTTLE_GRASP_SOURCE_DIRTY_DIGEST_ALGORITHM="
        "'git-diff-head-plus-untracked-content-v1'"
    ) in calls
    assert "--commissioning-speed '7'" in calls
    assert "--visual-servo-mode 'shadow'" in calls
    assert "--stop-after-observation" in calls


def test_legacy_visual_servo_selects_active_when_visual_mode_is_unset(tmp_path):
    env, call_log = _fake_transport(tmp_path)
    env["VISUAL_SERVO"] = "1"
    env["CONFIRM_BEFORE_GRASP"] = "1"

    result = _run_launcher(env)

    assert result.returncode == 0, result.stderr
    assert "预抓取视觉闭环: active" in result.stdout
    assert "阶段入口: confirm-before-grasp" in result.stdout
    calls = call_log.read_text(encoding="utf-8")
    assert "--visual-servo-mode 'active'" in calls
    assert "BOTTLE_GRASP_CONTINUOUS_TRAJECTORY='1'" in calls
    assert "--commissioning-speed" not in calls
    assert "--confirm-before-grasp" in calls


def test_stop_after_observation_and_confirm_before_grasp_are_mutually_exclusive(
    tmp_path,
):
    env, call_log = _fake_transport(tmp_path)
    env.update(
        {
            "STOP_AFTER_OBSERVATION": "1",
            "CONFIRM_BEFORE_GRASP": "1",
        }
    )

    result = _run_launcher(env)

    assert result.returncode == 1
    assert "互斥" in result.stderr
    assert not call_log.exists()


def test_stop_after_observation_rejects_the_already_past_observation_entry(
    tmp_path,
):
    env, call_log = _fake_transport(tmp_path)
    env["STOP_AFTER_OBSERVATION"] = "1"

    result = _run_launcher(env, mode="from-pregrasp")

    assert result.returncode == 1
    assert "from-pregrasp" in result.stderr
    assert not call_log.exists()
