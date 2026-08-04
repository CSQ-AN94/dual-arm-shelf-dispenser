"""Drive the left arm through the same safety chain as the right one.

Two things stood between the left arm and the existing machinery, and neither
was the machinery's fault:

* The RealMan SDK's ``Algo`` is process-global, so a second ``RobotSession``
  overwrites the first one's kinematics.  ``ArmProxy`` gives the left arm its
  own process.
* Every fence box in ``safety_profiles.json`` is authored in the *right* arm's
  base frame, while the left arm's own FK reports poses in the *left* base
  frame.  Checking one against the other silently compares two origins 120 mm
  apart.

``left_view`` solves the second by handing the profile the rigid transform that
names left-arm points in the fence's frame.  Every existing caller of
``assert_tcp_point`` and ``assert_tcp_path`` then works unchanged -- the dense
re-check inside ``SafeMotionPlanner`` included.

An earlier version rewrote each box into the left frame instead.  That is not
equivalent: a rotated box is not axis aligned, and bounding it grows the box.
For a keepout that is conservative, but for an *allowed* zone it grants space
the fence never had, and a point could pass the left-framed check while landing
outside every zone the real fence has.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from .core import SafetyAbort
from .safety import SafetyProfile


def assert_rigid(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise SafetyAbort("双臂基座变换必须是 4x4 有限矩阵")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
        raise SafetyAbort("双臂基座变换的旋转部分不是正交阵")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise SafetyAbort("双臂基座变换的齐次末行无效")
    return matrix


def left_view(profile: SafetyProfile, base_right_to_base_left) -> SafetyProfile:
    """The same fence, told how to read left-arm coordinates.

    ``base_right_to_base_left`` is the measured 4x4 from ``config.yaml``: it
    maps a point given in the left base frame into the right one, which is
    exactly the conversion the fence needs.

    This does not make the profile plannable for the left arm.  ``T_moveit_from
    _profile`` is the bridge from the *right* controller base to the MoveIt
    frame, and composing it with this transform does not produce the left one:
    a real run on 2026-08-04 had MoveIt l_link7 FK and the SDK's left flange
    disagree by 127.2 mm and 179.9 deg at the same joint state, and 180 deg is
    not something a near-identity rotation accounts for.  The two arms'
    controller base frames appear to differ by a half turn that this measured
    transform -- taken between two head-camera eye-to-hand rounds -- does not
    describe.  ``assert_left_bridge_measured`` gates on that separately.
    """
    return replace(
        profile,
        name=f"{profile.name}__left",
        tcp_frame_transform=assert_rigid(base_right_to_base_left),
    )


def assert_left_bridge_measured(profile: SafetyProfile) -> np.ndarray:
    """Refuse left-arm planning until its own MoveIt bridge exists.

    Caught by the runtime FK contract rather than by review, which is the right
    order but an expensive one: the arm was powered, teleop stopped, and a plan
    already computed.  Fail here instead, before anything opens.
    """
    bridge = getattr(profile, "T_moveit_from_left_profile", None)
    if bridge is None:
        raise SafetyAbort(
            "左臂缺少自己的 MoveIt 坐标桥（T_moveit_from_left_profile）。"
            "profile 里那份是右臂控制器基座到 MoveIt 的桥；2026-08-04 实测显示"
            "同关节状态下 MoveIt l_link7 与 SDK 左法兰差 127.2 mm / 179.9°，"
            "而双臂实测变换的旋转接近单位阵，合成消不掉这半圈。"
            "先实测左臂的桥再开左臂规划"
        )
    return assert_rigid(bridge)


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
    left tool with ``scripts/measure_left_tool_mount.py``, and this opens.
    """
    from .arm_worker import ArmProxy

    left_calibration = getattr(profile, "left_tool_mount_calibration", None)
    if left_calibration is None:
        raise SafetyAbort(
            f"profile {profile.name} 没有左臂工具标定；"
            "右臂那份的 evidence_id 明确写着不得迁移到左臂"
            "（它是 nominal，残差靠右臂实测的 grasp_stop_short_m 吸收）。"
            "先跑 scripts/measure_left_tool_mount.py 实测"
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
