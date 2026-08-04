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


def left_view(profile: SafetyProfile) -> SafetyProfile:
    """The right arm's fence and planner, told how the left arm reports itself.

    Three constants separate the two, all measured together by
    ``scripts/solve_left_arm_model.py`` (16 joint states, 0.583 mm / 0.108 deg):

      * joints 2, 4 and 6 run the other way, applied in ``planner``;
      * a bridge from the left controller base to the MoveIt frame, used here;
      * a tool-side offset, already folded into the left arm's
        ``T_link7_controller_flange`` -- it multiplies from the right, which is
        why no base-frame bridge could ever absorb it, and why solving for one
        alone left an 889 mm spread.

    The fence conversion is derived from the two measured bridges rather than
    from ``config.yaml``'s dual-arm transform: that one was taken between two
    head-camera eye-to-hand rounds and does not describe the controller base
    frames, which is what the 179.9 deg surprise was.
    """
    model = getattr(profile, "left_arm_model", None)
    if model is None:
        raise SafetyAbort(
            f"profile {profile.name} 缺少 left_arm_model；"
            "先跑 scripts/solve_left_arm_model.py --write"
        )
    left_bridge = assert_rigid(model["T_moveit_from_left_profile"])
    right_bridge = np.asarray(profile.T_moveit_from_profile, dtype=float)
    return replace(
        profile,
        name=f"{profile.name}__left",
        T_moveit_from_profile=left_bridge,
        tool_mount_calibration=profile.left_tool_mount_calibration,
        # Left controller base -> right controller base, via the MoveIt frame
        # both were measured against.
        tcp_frame_transform=np.linalg.inv(right_bridge) @ left_bridge,
    )


def left_joint_signs(profile: SafetyProfile) -> tuple[int, ...]:
    """Which joints the SDK and the URDF disagree about the direction of."""
    model = getattr(profile, "left_arm_model", None)
    if model is None:
        raise SafetyAbort(f"profile {profile.name} 缺少 left_arm_model")
    signs = tuple(int(s) for s in model["joint_signs"])
    if len(signs) != 7 or any(s not in (1, -1) for s in signs):
        raise SafetyAbort("left_arm_model.joint_signs 必须是 7 个 ±1")
    return signs


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
    _, flange_to_tcp = left_calibration.require_transforms()
    # The session is configured with the nominal chain, not the folded one.
    # ``controller_flange_from_joints`` is defined against whatever link7
    # transform the session holds, so handing it the folded value applies the
    # tool constant twice -- the runtime FK contract reported exactly the
    # unfolded residual, 28.9 mm and 179.9 deg, which is how this surfaced.
    # The folded transform belongs to the planner, which is the thing that has
    # to speak MoveIt's frame.
    return ArmProxy(
        cfg.connections.left_arm_ip,
        cfg.connections.arm_port,
        params.tcp_z_m,
        model_flange_offset_m=params.moveit_link7_to_controller_flange_m,
        take_control=take_control,
        tcp_transform=flange_to_tcp,
    )


def arrival_error_deg(reached: Sequence[float], target: Sequence[float]) -> float:
    current = np.asarray(reached, dtype=float)
    goal = np.asarray(target, dtype=float)
    if current.shape != (7,) or goal.shape != (7,):
        raise SafetyAbort("左臂关节角必须是 7 个数")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(goal)):
        raise SafetyAbort("左臂关节角包含非有限数")
    return float(np.max(np.abs(current - goal)))
