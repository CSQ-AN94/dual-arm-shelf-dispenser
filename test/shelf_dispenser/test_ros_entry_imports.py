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
STUB_BODY = textwrap.dedent(
    '''
    import sys, types


    class _Any:
        def __init__(self, *a, **k): pass
        def __getattr__(self, name): return _Any()
        def __call__(self, *a, **k): return _Any()
        def __iter__(self): return iter(())


    class _Module(types.ModuleType):
        def __getattr__(self, name):
            child = f"{self.__name__}.{name}"
            if child not in sys.modules:
                sys.modules[child] = _Module(child)
            return sys.modules[child] if name[:1].islower() else _Any


    sys.modules[__name__] = _Module(__name__)
    '''
)


@pytest.fixture(scope="module")
def stub_path(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ros_stubs")
    for name in STUBBED:
        (directory / f"{name}.py").write_text(STUB_BODY, encoding="utf-8")
    return directory


@pytest.mark.parametrize("entry", ENTRIES)
def test_ros_entry_resolves_its_imports_when_run_by_path(entry, stub_path, tmp_path):
    """Run it the way planner does: absolute path, unrelated working directory."""
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": f"{stub_path}:{ROOT}",
        "HOME": str(tmp_path),
    }
    result = subprocess.run(
        [sys.executable, str(ROS_DIR / entry)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    # The script will fail once it tries to talk to a robot; that is fine and
    # expected.  Failing to find a module is not.
    for failure in ("ModuleNotFoundError", "ImportError", "cannot import name"):
        assert failure not in combined, (
            f"{entry} 按路径调起时 import 失败:\n{combined[-1500:]}"
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
