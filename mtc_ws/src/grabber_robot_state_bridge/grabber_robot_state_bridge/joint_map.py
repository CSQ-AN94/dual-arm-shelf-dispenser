"""SDK -> MoveIt joint mapping for the dual RM75 platform.

This module is pure data plus unit conversion.  It contains no ROS and no SDK
import so the mapping can be dumped and audited without a robot.

Two facts drive every entry below:

* ``rm_get_joint_degree()`` returns seven **degrees**, in joint order 1..7, for
  the controller the handle is connected to.  Left and right arms are separate
  controllers with separate IPs.
* ``{"command": "get_lift_state"}`` returns ``height`` in **millimetres**, and
  the URDF ``platform_joint`` is a prismatic joint limited to 0..1 **metres**.
  647 mm -> 0.647 m sits inside that range, which is the unit cross-check.

The model has 27 movable joints.  Thirteen of them (ten wheel joints and the
two head servo joints) have no read-only query in this stack.  They are
published as an explicitly declared constant so that MoveIt's
CurrentStateMonitor can reach a *complete* RobotState -- their value has no
effect on either arm's kinematics, because both arm chains and
``platform_base_link`` hang off ``body_base_link`` above the wheels and beside
the head.  ``UNMEASURED`` is reported verbatim in current_state_report.json;
nothing here is presented as a measurement.
"""

from __future__ import annotations

import math

ARM_JOINT_COUNT = 7

# SDK index (0-based, joint 1..7) -> MoveIt joint name.
RIGHT_ARM_JOINTS = [f"r_joint{i}" for i in range(1, ARM_JOINT_COUNT + 1)]
LEFT_ARM_JOINTS = [f"l_joint{i}" for i in range(1, ARM_JOINT_COUNT + 1)]
PLANNING_JOINTS = RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS

LIFT_JOINT = "platform_joint"

# No read-only query exists for these in this stack; see the module docstring.
UNMEASURED_JOINTS = [
    "joint_right_wheel",
    "joint_left_wheel",
    "joint_swivel_wheel_1_1",
    "joint_swivel_wheel_1_2",
    "joint_swivel_wheel_2_1",
    "joint_swivel_wheel_2_2",
    "joint_swivel_wheel_3_1",
    "joint_swivel_wheel_3_2",
    "joint_swivel_wheel_4_1",
    "joint_swivel_wheel_4_2",
    "head_joint1",
    "head_joint2",
]
UNMEASURED_VALUE = 0.0

ALL_JOINTS = PLANNING_JOINTS + [LIFT_JOINT] + UNMEASURED_JOINTS


def deg_to_rad(degrees: float) -> float:
    return float(degrees) * math.pi / 180.0


def mm_to_m(millimetres: float) -> float:
    return float(millimetres) / 1000.0


def mapping_document() -> dict:
    """The full, machine-readable mapping, dumped as joint_state_mapping.json."""
    entries = []
    for index, name in enumerate(RIGHT_ARM_JOINTS):
        entries.append(
            {
                "moveit_joint": name,
                "source": "sdk",
                "sdk_call": "rm_get_joint_degree",
                "sdk_endpoint": "right_arm_ip",
                "sdk_index": index,
                "sdk_unit": "degree",
                "ros_unit": "radian",
                "conversion": "value * pi / 180",
            }
        )
    for index, name in enumerate(LEFT_ARM_JOINTS):
        entries.append(
            {
                "moveit_joint": name,
                "source": "sdk",
                "sdk_call": "rm_get_joint_degree",
                "sdk_endpoint": "left_arm_ip",
                "sdk_index": index,
                "sdk_unit": "degree",
                "ros_unit": "radian",
                "conversion": "value * pi / 180",
            }
        )
    entries.append(
        {
            "moveit_joint": LIFT_JOINT,
            "source": "sdk",
            "sdk_call": 'json {"command": "get_lift_state"} -> height',
            "sdk_endpoint": "lift_host",
            "sdk_index": None,
            "sdk_unit": "millimetre",
            "ros_unit": "metre",
            "conversion": "value / 1000",
        }
    )
    for name in UNMEASURED_JOINTS:
        entries.append(
            {
                "moveit_joint": name,
                "source": "UNMEASURED",
                "sdk_call": None,
                "sdk_endpoint": None,
                "sdk_index": None,
                "sdk_unit": None,
                "ros_unit": "radian",
                "conversion": f"constant {UNMEASURED_VALUE}",
                "note": (
                    "No read-only query available.  Published as a declared "
                    "constant so the RobotState is complete; has no effect on "
                    "either arm chain or on platform_base_link."
                ),
            }
        )
    return {
        "robot_model": "dual_rm_75b_description",
        "planning_joints": PLANNING_JOINTS,
        "measured_joints": PLANNING_JOINTS + [LIFT_JOINT],
        "unmeasured_joints": UNMEASURED_JOINTS,
        "published_joints": ALL_JOINTS,
        "entries": entries,
    }


def validate() -> None:
    """Fail closed on a mapping that would publish a malformed JointState."""
    if len(ALL_JOINTS) != len(set(ALL_JOINTS)):
        raise ValueError("joint mapping contains duplicates")
    if len(PLANNING_JOINTS) != 14:
        raise ValueError(f"expected 14 planning joints, got {len(PLANNING_JOINTS)}")


def demo() -> None:
    validate()
    doc = mapping_document()
    assert len(doc["entries"]) == len(ALL_JOINTS) == 27, len(doc["entries"])
    assert abs(deg_to_rad(180.0) - math.pi) < 1e-12
    assert abs(mm_to_m(647) - 0.647) < 1e-12
    # Every planning joint must be sourced from the SDK, never a constant.
    by_name = {entry["moveit_joint"]: entry for entry in doc["entries"]}
    for name in PLANNING_JOINTS:
        assert by_name[name]["source"] == "sdk", name
    assert by_name[LIFT_JOINT]["source"] == "sdk"
    print("joint_map demo OK: 27 joints, 14 planning, no duplicates")


if __name__ == "__main__":
    demo()
