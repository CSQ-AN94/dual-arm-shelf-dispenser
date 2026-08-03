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


def _assert_rigid(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise SafetyAbort("双臂基座变换必须是 4x4 有限矩阵")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
        raise SafetyAbort("双臂基座变换的旋转部分不是正交阵")
    return matrix


class LeftArmFence:
    """The right arm's fence, applied to left-arm poses without moving it.

    Rewriting each box into the left base frame looked cheaper -- one conversion
    instead of one per trajectory point -- but a rotated box is not axis
    aligned, and bounding it grows the box.  For a keepout that is conservative;
    for an *allowed* zone it hands out space the fence never granted, and a
    point can pass the left-framed zone while landing outside every zone the
    real fence has.  Codex reproduced exactly that.

    So convert the point instead.  It is exact in both directions, and the
    fence stays the single artefact everyone reads.
    """

    def __init__(self, profile: SafetyProfile, base_right_to_base_left):
        matrix = _assert_rigid(base_right_to_base_left)
        self.profile = profile
        self._rotation = matrix[:3, :3]
        self._translation = matrix[:3, 3]

    def to_right_base(self, point) -> np.ndarray:
        """A point named in the left base frame, named in the right one."""
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise SafetyAbort("左臂 TCP 必须是三个有限数")
        return self._rotation @ point + self._translation

    def contains(self, point) -> bool:
        moved = self.to_right_base(point)
        return bool(
            self.profile.tcp_workspace.contains(moved, self.profile.clearance_m)
            and any(
                zone.contains(moved, 0.0) for zone in self.profile.allowed_tcp_zones
            )
            and not any(
                box.contains(moved, 0.0) for box in self.profile.keepout_boxes
            )
        )


def open_left_arm(cfg, params, profile: SafetyProfile, *, take_control: bool):
    """Start the left arm's own process.

    Refuses to reuse the right arm's tool calibration, because that record says
    in its own evidence_id not to:

        NOT metrology: do not transfer to the left arm ... without measuring
        first.

    The two grippers being the same part does not make the right arm's number
    right for the left one -- that number is nominal, with its residual absorbed
    by a stop-short distance tuned on the right arm against a real shelf.  Using
    it here would put a wrong TCP into every fence check silently.  Measure the
    left tool, record it as ``left_tool_mount_calibration``, then this opens.
    """
    from .arm_worker import ArmProxy

    left_calibration = getattr(profile, "left_tool_mount_calibration", None)
    if left_calibration is None:
        raise SafetyAbort(
            f"profile {profile.name} 没有左臂工具标定；"
            "右臂那份的 evidence_id 明确写着不得迁移到左臂"
            "（它是 nominal，残差靠右臂实测的 grasp_stop_short_m 吸收）。"
            "先实测左臂工具链并写入 left_tool_mount_calibration"
        )
    link7_to_flange, flange_to_tcp = left_calibration.require_transforms()
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
