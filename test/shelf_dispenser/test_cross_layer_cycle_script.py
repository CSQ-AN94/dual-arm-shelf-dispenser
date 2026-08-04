from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_cross_layer_cycle.sh"


def test_cycle_owns_and_cleans_up_its_plan_only_move_group():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "live_state_plan_only.launch.py" in source
    assert "bridge_status_file:=" in source
    assert "'/get_planning_scene'" in source
    assert "trap cleanup_stack EXIT" in source
    assert source.index("live_state_plan_only.launch.py") < source.index(
        "plan_shelf_transfer_experimental.launch.py"
    )
    pick_stack = source.index("start_stack pick")
    right_lift = source.index("--right-and-lift-only")
    left_normalize = source.index("normalize_left_arm.py --execute")
    atomic_gate = source.index(
        "normalize_to_grasp_start.py --execute", left_normalize
    )
    assert right_lift < left_normalize < atomic_gate < pick_stack
    assert source.index("cleanup_stack", pick_stack) < source.index(
        'say "阶段 2.5'
    )
    assert source.index("start_stack place") < source.index('say "阶段 4')
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_robot_drift_tracks_the_left_arm_safety_chain():
    source = (ROOT / "scripts" / "robot_code_drift.py").read_text(
        encoding="utf-8"
    )
    for relative in (
        "shelf_dispenser/ros/scene_helpers.py",
        "shelf_dispenser/ros/validate_path.py",
        "scripts/normalize_left_arm.py",
        "scripts/solve_left_arm_model.py",
        "scripts/run_cross_layer_cycle.sh",
    ):
        assert f'"{relative}"' in source
