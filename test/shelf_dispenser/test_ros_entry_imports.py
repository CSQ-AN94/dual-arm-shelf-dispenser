"""The ros/ entry points must still resolve their imports after the move.

These run under the system Python, invoked by absolute path from ``planner``,
so nothing in this suite imports them and a rename breaks them silently until
the robot is in front of you.  Two things make them work and both are fragile
under a move: Python puts a script's own directory at sys.path[0], which is
what their flat sibling imports rely on, and any ``parents[N]`` in them counts
levels from wherever the file now sits.

Running them for real is the only way to check either.  ROS is stubbed --
the point is not that MoveIt works, it is that the files find each other.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROS_DIR = ROOT / "shelf_dispenser" / "ros"

ENTRIES = [
    "plan_once.py",
    "validate_path.py",
    "detach_object.py",
    "headless.py",
    "collision_selftest.py",
]

STUBBED = [
    "rclpy",
    "moveit_msgs",
    "geometry_msgs",
    "sensor_msgs",
    "shape_msgs",
    "std_msgs",
    "trajectory_msgs",
    "builtin_interfaces",
    "ament_index_python",
    "launch",
    "launch_ros",
    "moveit_configs_utils",
    "xacro",
]

# Any attribute access yields another stub, so `from moveit_msgs.msg import X`
# and `X()` both work without naming a single ROS symbol here.
# Real stub packages on disk, not import hooks.  `from moveit_msgs.msg import
# X` needs `moveit_msgs.msg` to be an actual module; the first version faked it
# with __getattr__, the machinery raised TypeError at the first such import, and
# the test passed having checked nothing.
SUBMODULES = ["msg", "srv", "node", "qos", "actions", "packages"]

STUB_BODY = textwrap.dedent(
    """
    class _Any:
        def __init__(self, *a, **k): pass
        def __getattr__(self, name): return _Any()
        def __call__(self, *a, **k): return _Any()
        def __iter__(self): return iter(())


    def __getattr__(name):     # PEP 562: only consulted when lookup fails
        return _Any
    """
)


@pytest.fixture(scope="module")
def stub_path(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ros_stubs")
    for name in STUBBED:
        package = directory / name
        package.mkdir()
        (package / "__init__.py").write_text(STUB_BODY, encoding="utf-8")
        for submodule in SUBMODULES:
            (package / f"{submodule}.py").write_text(STUB_BODY, encoding="utf-8")
    return directory


@pytest.mark.parametrize("entry", ENTRIES)
def test_ros_entry_resolves_its_imports_when_run_by_path(entry, stub_path, tmp_path):
    """Execute the entry's module level and prove it finished.

    ``run_name`` is not ``__main__``, so this stops after the imports instead of
    trying to reach a robot, and the script's own directory goes on sys.path
    first -- which is what ``python <script>`` does and what the flat sibling
    imports depend on.  Printing after the run is what makes a pass mean
    something: the earlier version asserted only that no import error appeared,
    and a TypeError at line 11 satisfied that while nothing was checked.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            f"""
            import runpy, sys, os
            script = {str(ROS_DIR / entry)!r}
            sys.path.insert(0, os.path.dirname(script))
            sys.argv = [script]
            runpy.run_path(script, run_name="__probe__")
            print("IMPORTS_DONE")
            """
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": f"{stub_path}:{ROOT}",
            "HOME": str(tmp_path),
        },
    )
    combined = result.stdout + result.stderr
    assert "IMPORTS_DONE" in combined, (
        f"{entry} 的模块级 import 没跑完:\n{combined[-1500:]}"
    )


def test_planner_points_at_files_that_exist():
    """A rename that misses planner.py fails only once the robot is running."""
    import re

    source = (ROOT / "shelf_dispenser" / "planner.py").read_text(encoding="utf-8")
    chains = re.findall(r'self\.project_root((?:\s*/\s*"[^"]+")+)', source)
    assert chains, "planner.py 不再按 project_root 拼路径，这个测试需要更新"
    for chain in chains:
        parts = re.findall(r'"([^"]+)"', chain)
        assert ROOT.joinpath(*parts).exists(), f"planner 指向不存在的文件: {parts}"


def test_repo_root_depth_is_right_for_every_file_that_computes_it():
    """``parents[N]`` is counted from where the file sits; a move invalidates it."""
    import re

    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".git", "__pycache__", ".mujoco_assets"} for part in path.parts):
            continue
        for match in re.finditer(
            r"ROOT = Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]", path.read_text(encoding="utf-8")
        ):
            depth = int(match.group(1))
            resolved = path.resolve().parents[depth]
            assert resolved == ROOT, (
                f"{path.relative_to(ROOT)} 的 parents[{depth}] 指向 {resolved}，"
                f"不是仓库根目录"
            )
