from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_cross_layer_cycle.sh"


def test_cycle_owns_and_cleans_up_its_plan_only_move_group():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "source_mtc_workspace()" in source
    assert "set +u" in source
    assert source.index("/opt/ros/humble/setup.bash") < source.index(
        "/home/rm/ros2_ws/install/setup.bash"
    ) < source.index('"$G/mtc_ws/install/setup.bash"')
    assert source.count("source_mtc_workspace") == 4
    assert "live_state_plan_only.launch.py" in source
    assert "setsid ros2 launch" in source
    assert 'kill -INT -- "-$STACK_PID"' in source
    assert 'kill -TERM -- "-$STACK_PID"' in source
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
    capture = source.index("capture_mtc_direct_pick_scene.py")
    row_template = source.index("apply_demonstrated_grasp_to_scenario.py")
    pick_plan = source.index("plan_shelf_transfer_experimental.launch.py")
    assert capture < row_template < pick_plan
    assert '--row-templates "$ROW_TEMPLATES" --layer "$LAYER"' in source
    assert source.count('--row-templates "$ROW_TEMPLATES"') == 2
    assert 'PRODUCT_CODE=${PRODUCT_CODE:-}' in source
    assert '--target-product "$PRODUCT_CODE"' in source
    assert source.count('--product-code "$PRODUCT_CODE"') == 2
    # The tuck reuses the stack that planned the pick instead of booting a
    # second move_group, so teardown moved to after it -- but it still happens
    # before the place stack starts, and the trap still covers every exit.
    tuck = source.index('say "阶段 2.5')
    assert "SHELF_REUSE_MOVEIT=1 $PY scripts/normalize_right_arm.py" in source
    assert tuck < source.index("cleanup_stack", tuck) < source.index(
        "start_stack place"
    )
    assert source.index("start_stack place") < source.index('say "阶段 4')
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_lower_layer_pick_descends_before_planning_and_stops_after_the_tuck():
    """LAYER=lower is the same pick 397 mm down, and it cannot go on to place.

    The lift moves before the plan-only stack starts, so the bridge publishes
    250 mm from its first sample and the plan is made at the height the arm
    will execute at.  There is no layer below the lower one to place into, so
    that combination is refused rather than left to fail three stages later.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "LAYER=${LAYER:-}" in source
    assert '[ "$PICK_ONLY" = 1 ] && LAYER=auto || LAYER=upper' in source
    assert "PICK_ONLY=${PICK_ONLY:-0}" in source
    assert "upper) PICK_LIFT_MM=647 ;;" in source
    assert "lower) PICK_LIFT_MM=250 ;;" in source
    descend = source.index('move_empty_platform_to_layer lower')
    assert source.index("normalize_to_grasp_start.py --execute") < descend
    assert descend < source.index("start_stack pick")
    # Every gate that asks "is the lift where this pick starts?" has to be
    # told which layer, or it asks the profile and refuses the lower one.  The
    # gripper calibration is the one that was missed: six attempts, all
    # refused at 250 against an expected 647, before anything moved.
    assert '--expected-lift-mm "$PICK_LIFT_MM"' in source
    assert source.index("PICK_ONLY_SUCCESS") > source.index('say "阶段 2.5')
    assert source.index("PICK_ONLY_SUCCESS") < source.index("start_stack place")


def test_auto_layer_search_moves_only_on_typed_no_target():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'auto) PICK_LIFT_MM= ;;' in source
    assert 'for CANDIDATE_LAYER in upper lower' in source
    assert 'move_empty_platform_to_layer "$CANDIDATE_LAYER"' in source
    assert 'if [ "$SEARCH_STATUS" = 3 ]' in source
    assert 'SELECTED_LAYER=$CANDIDATE_LAYER' in source
    assert source.index('for CANDIDATE_LAYER in upper lower') < source.index(
        "start_stack pick"
    )
    # Lower-source delivery to the side table is still unmeasured. Search may
    # find it, but must not pick it unless the caller explicitly asked to hold.
    assert '拒绝抓起一个没有安全去处的瓶子' in source


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
        "scripts/apply_demonstrated_grasp_to_scenario.py",
        "mtc_ws/src/grabber_mtc_planner/src/plan_shelf_transfer.cpp",
        "mtc_ws/src/grabber_mtc_planner/src/scenario.cpp",
        "mtc_ws/src/grabber_mtc_planner/src/scenario.hpp",
        "outputs/row_templates.json",
    ):
        assert f'"{relative}"' in source


def test_robot_drift_tracks_every_required_model_asset():
    """A model swap changes a filename, and the sync list is keyed by filename.

    The lock file and config.yaml are both tracked, so a swap that forgets this
    list pushes a lock pointing at a weights file the robot does not have.  The
    robot then fails asset validation on every capture, which looks like a
    perception problem and is not one.  The import-closure test below cannot
    catch this: weights are not importable.
    """
    import json

    source = (ROOT / "scripts" / "robot_code_drift.py").read_text(encoding="utf-8")
    lock = json.loads(
        (ROOT / "shelf_dispenser" / "model_assets.lock.json").read_text(
            encoding="utf-8"
        )
    )
    for asset in lock["assets"]:
        if not asset.get("required"):
            continue
        relative = asset["relative_path"]
        assert f'"{relative}"' in source, f"必需的模型资产不在 TRACKED 里: {relative}"


def test_robot_drift_tracks_every_reachable_runtime_module():
    """An entry point's imports are the entry point.

    The list was hand-kept and fell 18 modules behind, so `--push` reported
    everything in sync while perception.py on the robot still carried the
    alias set from before the trained model existed -- no p01..p06, and no
    case folding on the allowed set.  The six-class model against that copy
    matches nothing, and the cycle fails six captures on localization looking
    nothing like a sync problem.  Recompute the closure instead of trusting
    that whoever adds a module remembers to add it here too.
    """
    import ast
    import re

    source = (ROOT / "scripts" / "robot_code_drift.py").read_text(
        encoding="utf-8"
    )
    tracked = set(
        re.findall(r'"([^"]+)"', source.split("TRACKED = [")[1].split("]")[0])
    )

    seen: set[str] = set()
    stack = [name for name in tracked if name.endswith(".py")]
    while stack:
        relative = stack.pop()
        if relative in seen:
            continue
        seen.add(relative)
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        package = Path(relative).parent
        for node in ast.walk(tree):
            candidates: list[Path] = []
            if isinstance(node, ast.Import):
                candidates = [
                    Path(alias.name.replace(".", "/") + ".py")
                    for alias in node.names
                ]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package
                    for _ in range(node.level - 1):
                        base = base.parent
                    if node.module:
                        candidates = [
                            base / (node.module.replace(".", "/") + ".py")
                        ]
                    else:
                        candidates = [
                            base / (alias.name + ".py") for alias in node.names
                        ]
                elif node.module:
                    module = Path(node.module.replace(".", "/"))
                    candidates = [module.with_suffix(".py")]
                    candidates += [
                        module / (alias.name + ".py") for alias in node.names
                    ]
            for candidate in candidates:
                if (ROOT / candidate).is_file():
                    stack.append(str(candidate))

    missing = sorted(seen - tracked)
    assert not missing, (
        "这些运行时模块能被 TRACKED 的入口 import 到，却不会被同步到机器人："
        + ", ".join(missing)
    )
