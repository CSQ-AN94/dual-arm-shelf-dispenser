"""Public CLI contracts for supported task-mode control gates.

These tests only parse and validate arguments.  They never construct a demo,
open a camera, or contact robot hardware.
"""

import importlib
import sys
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def cli(monkeypatch):
    """Load the CLI without requiring robot-side YAML dependencies."""
    monkeypatch.setitem(
        sys.modules, "utils.config", SimpleNamespace(load_config=lambda _path: None)
    )
    monkeypatch.delitem(sys.modules, "scripts.bottle_grasp_demo", raising=False)
    return importlib.import_module("scripts.bottle_grasp_demo")


def _validated(cli, *argv):
    return cli.validate_args(cli.build_parser().parse_args(list(argv)))


def test_task_mode_allows_real_stop_after_observation_not_plan_only(cli):
    args = _validated(
        cli,
        "--execute",
        "--task-mode",
        "from-start",
        "--stop-after-observation",
    )

    assert args.stop_after_observation is True
    assert args.visual_servo_mode == "off"


def test_task_mode_allows_the_same_process_confirmation_gate(cli):
    args = _validated(
        cli,
        "--execute",
        "--task-mode",
        "from-observation",
        "--confirm-before-grasp",
    )

    assert args.confirm_before_grasp is True


def test_task_mode_still_refuses_plan_only_and_mutually_exclusive_gates(cli):
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _validated(cli, "--execute", "--plan-only", "--task-mode", "from-start")
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _validated(
            cli,
            "--execute",
            "--task-mode",
            "from-start",
            "--stop-after-observation",
            "--confirm-before-grasp",
        )
    with pytest.raises(SystemExit, match="已越过观察位"):
        _validated(
            cli,
            "--execute",
            "--task-mode",
            "from-pregrasp",
            "--stop-after-observation",
        )


def test_visual_modes_and_commissioning_cap_have_explicit_validation(cli):
    args = _validated(
        cli,
        "--execute",
        "--task-mode",
        "from-start",
        "--visual-servo-mode",
        "shadow",
        "--commissioning-speed",
        "15",
    )
    assert args.visual_servo_mode == "shadow"
    assert args.visual_servo is False
    assert args.commissioning_speed == 15

    with pytest.raises(SystemExit, match="1-100"):
        _validated(
            cli,
            "--execute",
            "--task-mode",
            "from-start",
            "--commissioning-speed",
            "0",
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ("--visual-servo-step-mm", "nan"),
        ("--visual-servo-total-mm", "inf"),
        ("--visual-servo-convergence-mm", "9", "--visual-servo-step-mm", "8"),
        ("--visual-servo-max-corrections", "0"),
    ],
)
def test_invalid_visual_tunables_are_rejected_by_cli_before_demo_creation(
    cli, arguments
):
    with pytest.raises(SystemExit, match="视觉闭环参数无效"):
        _validated(
            cli,
            "--execute",
            "--task-mode",
            "from-start",
            *arguments,
        )


def test_main_writes_manifest_before_starting_the_task(cli, monkeypatch, tmp_path):
    events = []

    class FakeDemo:
        def __init__(self, args, config):
            self.args = args
            self.cfg = config
            self.project_root = tmp_path
            self.params = SimpleNamespace(
                transit_speed=100,
                travel_speed=15,
                final_speed=15,
            )
            self.run_dir = tmp_path / "evidence"
            self.run_dir.mkdir()
            self.stop_event = threading.Event()
            self.timeline = None

        def run(self):
            events.append("run")

        def close(self):
            events.append("close")

    def write_manifest(run_dir, **_kwargs):
        events.append("manifest")
        path = run_dir / "run_manifest.json"
        path.write_text("{}\n", encoding="utf-8")
        return path

    monkeypatch.setattr(cli, "BottleDemo", FakeDemo)
    monkeypatch.setattr(cli, "write_run_manifest", write_manifest)
    monkeypatch.setattr(cli.console, "install", lambda **_kwargs: None)
    monkeypatch.setattr(cli.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bottle_grasp_demo.py",
            "--execute",
            "--task-mode",
            "from-start",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert cli.main() == 0
    assert events == ["manifest", "run", "close"]
