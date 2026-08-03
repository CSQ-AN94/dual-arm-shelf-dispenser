import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from shelf_dispenser.core import DemoParams, Localization, SafetyAbort, pose_matrix
from shelf_dispenser.orchestrator import RunOrchestrator


def _localization(point):
    return Localization(
        point_camera=list(point),
        point_base=list(point),
        pixel=[320.0, 240.0],
        depth_m=0.085,
        depth_mad_m=0.0,
        position_spread_m=0.001,
        box=[280, 80, 360, 479],
        confidence=0.9,
        frame_count=3,
    )


class _Robot:
    def __init__(self):
        self.tcp = np.eye(4)
        self.moves = []

    def current_tcp(self):
        return self.tcp.copy()

    def move_linear(self, pose, speed):
        self.moves.append((list(pose), speed))
        self.tcp = pose_matrix(pose)


def _demo(*, enabled=True, execute=True, measurements=()):
    demo = RunOrchestrator.__new__(RunOrchestrator)
    demo.params = DemoParams()
    demo.args = SimpleNamespace(
        execute=execute,
        plan_only=not execute,
        visual_servo=enabled,
        visual_servo_max_corrections=None,
        visual_servo_step_mm=None,
        visual_servo_total_mm=None,
        visual_servo_convergence_mm=None,
    )
    demo.robot = _Robot()
    demo.safety = SimpleNamespace(assert_tcp_point=lambda _point, **_kwargs: None)
    demo.stage_calls = []
    demo.stage = lambda name, message="": demo.stage_calls.append((name, message))
    demo._start_camera = lambda name: None
    demo.ensure_bottle_visible = lambda **kwargs: None
    iterator = iter(measurements)
    demo.localize = lambda *args, **kwargs: next(iterator)
    demo.candidate_targets = []
    demo.candidate_path = lambda target: (
        demo.candidate_targets.append(np.asarray(target, dtype=float).copy()),
        ([0.0] * 6, [0.0] * 6, []),
    )[1]
    demo.planned_paths = []
    demo.plan_kwargs = []

    def plan_local(_name, build_path, _params, **_kwargs):
        path = build_path()
        demo.planned_paths.append(path)
        demo.plan_kwargs.append(_kwargs)
        return path

    demo._plan_local_leg = plan_local
    demo.collision_paths = []
    demo.collision_gate = lambda _box, _target, corridor_waypoints_base=None: (
        demo.collision_paths.append(corridor_waypoints_base)
    )
    demo.local_contact_target_base = np.zeros(3)
    return demo


def test_visual_servo_is_a_noop_by_default():
    locked = _localization([0.0, 0.0, 0.0])
    demo = _demo(enabled=False)

    result = demo._run_pregrasp_visual_servo(locked)

    assert result is locked
    assert demo.robot.moves == []
    assert demo.candidate_targets == []


def test_visual_servo_cli_is_opt_in_and_has_tunable_bounds(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "utils.config",
        SimpleNamespace(load_config=lambda _path: None),
    )
    cli = importlib.import_module("scripts.run_pick_place_task")
    defaults = cli.build_parser().parse_args([])
    enabled = cli.build_parser().parse_args(
        [
            "--visual-servo",
            "--visual-servo-max-corrections",
            "1",
            "--visual-servo-step-mm",
            "6",
            "--visual-servo-total-mm",
            "10",
            "--visual-servo-convergence-mm",
            "3",
        ]
    )

    assert defaults.visual_servo is False
    assert enabled.visual_servo is True
    assert enabled.visual_servo_max_corrections == 1
    assert enabled.visual_servo_step_mm == 6.0
    assert enabled.visual_servo_total_mm == 10.0
    assert enabled.visual_servo_convergence_mm == 3.0


def test_visual_servo_never_moves_in_dry_run_even_if_requested():
    locked = _localization([0.0, 0.0, 0.0])
    demo = _demo(enabled=True, execute=False)

    with pytest.raises(SafetyAbort, match="只允许在明确的真机执行模式"):
        demo._run_pregrasp_visual_servo(locked)

    assert demo.robot.moves == []


def test_visual_shadow_observes_once_but_keeps_the_original_motion_path():
    locked = _localization([0.0, 0.0, 0.0])
    demo = _demo(measurements=(_localization([0.012, 0.0, 0.0]),))
    demo.args.visual_servo_mode = "shadow"

    result = demo._run_pregrasp_visual_servo(locked)

    assert result is locked
    assert demo.robot.moves == []
    assert demo.candidate_targets == []
    assert demo.planned_paths == []
    assert demo.collision_paths == []
    assert any(name == "预抓取视觉闭环影子" for name, _ in demo.stage_calls)


def test_visual_servo_clamps_one_correction_and_reobserves_to_convergence():
    locked = _localization([0.0, 0.0, 0.0])
    demo = _demo(
        measurements=(
            _localization([0.012, 0.0, 0.0]),
            _localization([0.012, 0.0, 0.0]),
        )
    )

    result = demo._run_pregrasp_visual_servo(locked)

    assert len(demo.robot.moves) == 1
    assert np.linalg.norm(demo.robot.tcp[:3, 3]) == pytest.approx(0.008)
    np.testing.assert_allclose(result.point_base, [0.008, 0.0, 0.0])
    np.testing.assert_allclose(demo.candidate_targets, [[0.008, 0.0, 0.0]])
    assert len(demo.collision_paths) == 1
    assert demo.plan_kwargs == [{"roll_degrees": (0,)}]
    assert any(name == "预抓取视觉闭环收敛" for name, _ in demo.stage_calls)


def test_visual_servo_rejects_a_target_outside_the_total_correction_envelope():
    locked = _localization([0.0, 0.0, 0.0])
    demo = _demo(measurements=(_localization([0.016, 0.0, 0.0]),))

    with pytest.raises(SafetyAbort, match="超过累计修正上限"):
        demo._run_pregrasp_visual_servo(locked)

    assert demo.robot.moves == []
    assert demo.candidate_targets == []


def test_visual_servo_stops_when_residual_diverges():
    locked = _localization([0.0, 0.0, 0.0])
    demo = _demo(
        measurements=(
            _localization([0.006, 0.0, 0.0]),
            _localization([0.015, 0.0, 0.0]),
        )
    )

    with pytest.raises(SafetyAbort, match="误差发散"):
        demo._run_pregrasp_visual_servo(locked)

    assert len(demo.robot.moves) == 1


def test_visual_servo_propagates_visual_loss_without_motion():
    locked = _localization([0.0, 0.0, 0.0])
    demo = _demo()
    demo.ensure_bottle_visible = lambda **kwargs: (_ for _ in ()).throw(
        SafetyAbort("腕部视觉帧过期")
    )

    with pytest.raises(SafetyAbort, match="视觉帧过期"):
        demo._run_pregrasp_visual_servo(locked)

    assert demo.robot.moves == []


def test_visual_servo_never_moves_a_correction_rejected_by_full_safety_chain():
    locked = _localization([0.0, 0.0, 0.0])
    demo = _demo(measurements=(_localization([0.010, 0.0, 0.0]),))
    demo.candidate_path = lambda _target: (_ for _ in ()).throw(
        SafetyAbort("MoveIt 碰撞复核拒绝")
    )

    with pytest.raises(SafetyAbort, match="碰撞复核拒绝"):
        demo._run_pregrasp_visual_servo(locked)

    assert demo.robot.moves == []
