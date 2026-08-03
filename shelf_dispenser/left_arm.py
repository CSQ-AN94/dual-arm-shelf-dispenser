"""Drive the left arm through the same safety chain as the right one.

Two things stood between the left arm and the existing machinery, and neither
was the machinery's fault:

* The RealMan SDK's ``Algo`` is process-global, so a second ``RobotSession``
  overwrites the first one's kinematics.  ``ArmProxy`` gives the left arm its
  own process.
* Every fence box in ``safety_profiles.json`` is expressed in the *right* arm's
  base frame, while the left arm's own FK reports poses in the *left* base
  frame.  Checking one against the other would silently compare two different
  origins 120 mm apart.

``left_view`` solves the second by moving the fence into the left arm's frame
once, rather than converting every dense trajectory point.  ``SafeMotionPlanner``
then needs no knowledge that it is planning the other arm: it only ever asks the
robot object for ``joints_deg``, ``controller_flange_from_joints`` and
``validate_planned_joints``, all of which the proxy forwards.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from .core import SafetyAbort
from .safety import FenceBox, SafetyProfile


def transform_box(box: FenceBox, transform: np.ndarray) -> FenceBox:
    """Re-express an axis-aligned box in another frame.

    The result is the axis-aligned bound of the eight transformed corners.  The
    two arm bases are within about a degree of parallel, so the slack this adds
    is a couple of centimetres: allowed zones end up marginally more permissive
    and keepout boxes marginally larger, which is the safe direction for both.
    """
    lower = np.asarray(box.minimum, dtype=float)
    upper = np.asarray(box.maximum, dtype=float)
    corners = np.array(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ]
    )
    moved = (transform[:3, :3] @ corners.T).T + transform[:3, 3]
    return FenceBox(
        id=box.id,
        minimum=tuple(map(float, moved.min(axis=0))),
        maximum=tuple(map(float, moved.max(axis=0))),
    )


def left_view(profile: SafetyProfile, base_right_to_base_left) -> SafetyProfile:
    """The same fence, expressed in the left arm's base frame.

    ``base_right_to_base_left`` is the measured 4x4 from ``config.yaml``; it maps
    a point given in the left base frame into the right one, so its inverse is
    what carries the right-framed fence over to the left.
    """
    matrix = np.asarray(base_right_to_base_left, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise SafetyAbort("双臂基座变换必须是 4x4 有限矩阵")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
        raise SafetyAbort("双臂基座变换的旋转部分不是正交阵")
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ matrix[:3, 3]
    return replace(
        profile,
        name=f"{profile.name}__left",
        tcp_workspace=transform_box(profile.tcp_workspace, inverse),
        allowed_tcp_zones=tuple(
            transform_box(zone, inverse) for zone in profile.allowed_tcp_zones
        ),
        keepout_boxes=tuple(
            transform_box(box, inverse) for box in profile.keepout_boxes
        ),
    )


def open_left_arm(cfg, params, profile: SafetyProfile, *, take_control: bool):
    """Start the left arm's own process, wired to the shared tool calibration.

    Both arms carry the same RMG24, so the right arm's measured mount chain
    describes the left tool too -- there is nothing left-specific to calibrate.
    """
    from .arm_worker import ArmProxy

    link7_to_flange, flange_to_tcp = (
        profile.tool_mount_calibration.require_transforms()
    )
    return ArmProxy(
        cfg.connections.left_arm_ip,
        cfg.connections.arm_port,
        params.tcp_z_m,
        model_flange_offset_m=params.moveit_link7_to_controller_flange_m,
        take_control=take_control,
        tcp_transform=flange_to_tcp,
        link7_to_controller_flange=link7_to_flange,
    )


def arrival_error_deg(reached: Sequence[float], target: Sequence[float]) -> float:
    current = np.asarray(reached, dtype=float)
    goal = np.asarray(target, dtype=float)
    if current.shape != (7,) or goal.shape != (7,):
        raise SafetyAbort("左臂关节角必须是 7 个数")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(goal)):
        raise SafetyAbort("左臂关节角包含非有限数")
    return float(np.max(np.abs(current - goal)))
