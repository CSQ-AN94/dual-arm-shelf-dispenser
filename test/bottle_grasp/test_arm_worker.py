"""The worker protocol has to survive without an arm attached."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from bottle_grasp.arm_worker import ALLOWED_METHODS, ArmProxy, _jsonable, _revive
from bottle_grasp.core import SafetyAbort

ROOT = Path(__file__).resolve().parents[2]


def test_ndarray_survives_the_round_trip():
    matrix = np.arange(16, dtype=float).reshape(4, 4)
    revived = _revive(json.loads(json.dumps(_jsonable(matrix))))
    assert isinstance(revived, np.ndarray)
    assert revived == pytest.approx(matrix)

    nested = {"pose": matrix, "joints": [np.float64(1.5), 2.0]}
    out = _revive(json.loads(json.dumps(_jsonable(nested))))
    assert out["pose"] == pytest.approx(matrix)
    assert out["joints"] == [1.5, 2.0]


def test_the_whitelist_is_enforced_on_both_sides():
    """A typo must not reach a method nobody vetted for the second arm."""
    assert "move_linear" not in ALLOWED_METHODS
    assert "open_gripper" not in ALLOWED_METHODS
    assert "joints_deg" in ALLOWED_METHODS

    proxy = ArmProxy.__new__(ArmProxy)
    with pytest.raises(AttributeError):
        proxy.move_linear
    with pytest.raises(SafetyAbort, match="白名单"):
        ArmProxy._call(proxy, "move_linear")


def test_serve_answers_commands_against_a_stubbed_session():
    """Drive the real serve() loop with RobotSession stubbed out."""
    script = textwrap.dedent(
        f"""
        import sys, json, types
        sys.path.insert(0, {str(ROOT)!r})
        import numpy as np

        stub = types.ModuleType("bottle_grasp.robot")

        class RobotSession:
            def __init__(self, ip, port, stop, tcp_z, offset, **kw):
                self.ip = ip
            def joints_deg(self):
                return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
            def tcp_from_joints(self, q):
                out = np.eye(4)
                out[:3, 3] = q[:3]
                return out
            def close(self):
                pass

        stub.RobotSession = RobotSession
        sys.modules["bottle_grasp.robot"] = stub

        from bottle_grasp.arm_worker import serve
        sys.exit(serve({{"ip": "1.2.3.4", "port": 8080, "tcp_z_m": 0.1,
                         "model_flange_offset_m": 0.0172}}))
        """
    )
    commands = "\n".join(
        json.dumps(payload)
        for payload in (
            {"method": "joints_deg"},
            {"method": "tcp_from_joints", "args": [[0.1, 0.2, 0.3, 0, 0, 0, 0]]},
            {"method": "move_linear", "args": []},
            {"method": "joints_deg", "args": [1, 2]},
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=commands + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    replies = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert replies[0] == {"ready": True}
    assert replies[1]["ok"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert _revive(replies[2]["ok"])[:3, 3] == pytest.approx([0.1, 0.2, 0.3])
    # Blocked by the whitelist, and a bad call reports instead of killing the loop.
    assert "白名单" in replies[3]["error"]
    assert "error" in replies[4]


def test_arm_names_derives_both_arms_from_the_group():
    """The planning arm names its own seven joints; the other arm is scene."""
    # The module imports rclpy at load time, so lift the one pure function out
    # by source rather than importing ROS into the unit suite.
    source = (ROOT / "bottle_grasp" / "moveit_plan_once.py").read_text(
        encoding="utf-8"
    )
    start = source.index("def arm_names(")
    end = source.index("def compute_link7_fk(")
    namespace: dict = {}
    exec(compile(source[start:end], "arm_names", "exec"), namespace)
    arm_names = namespace["arm_names"]

    right = arm_names("right_arm")
    assert right["ik_link"] == "r_link7"
    assert right["joints"][0] == "r_joint1"
    assert right["other_joints"][0] == "l_joint1"

    left = arm_names("left_arm")
    assert left["ik_link"] == "l_link7"
    assert left["joints"][0] == "l_joint1"
    assert left["other_joints"][0] == "r_joint1"

    with pytest.raises(RuntimeError, match="unsupported planning group"):
        arm_names("both_arms")
