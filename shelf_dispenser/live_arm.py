"""Open one live arm by id, with the fence view that reads its poses.

Two callers now need this pair and they must not diverge: the arm handle and
the ``SafetyProfile`` whose frame that handle reports in.  Getting the pairing
wrong is silent -- the left arm's own FK is 120 mm from the frame every fence
box is authored in, so a left handle checked against the raw profile passes and
means nothing.  Handing both out together is the only way that stays true.
"""

from __future__ import annotations

import threading

from .core import DemoParams, SafetyAbort
from .left_arm import left_view, open_left_arm
from .safety import SafetyProfile

ARMS = ("left_arm", "right_arm")


def open_live_arm(
    cfg, params: DemoParams, profile: SafetyProfile, arm_id: str, *,
    take_control: bool,
):
    """Return ``(robot, view)`` for one arm.  Caller owns ``robot.close()``.

    The left arm comes back as an ``ArmProxy`` (its own process, because the
    SDK's ``Algo`` is process-global) and the right as a ``RobotSession``.
    Both answer the same method names, with one difference the caller must
    respect: ``ArmProxy`` crosses a JSON boundary, so never hand it a
    ``DemoParams`` -- its worker builds the same defaults on the far side.
    """
    from .arm import RobotSession

    if arm_id not in ARMS:
        raise SafetyAbort(f"未知机械臂: {arm_id}")
    if arm_id == "left_arm":
        return open_left_arm(
            cfg, params, profile, take_control=take_control
        ), left_view(profile)

    calibration = profile.tool_mount_calibration
    if calibration is None:
        raise SafetyAbort(f"profile {profile.name} 缺少右臂工具标定")
    link7_to_flange, flange_to_tcp = calibration.require_transforms()
    return RobotSession(
        cfg.connections.right_arm_ip,
        cfg.connections.arm_port,
        threading.Event(),
        params.tcp_z_m,
        params.moveit_link7_to_controller_flange_m,
        take_control=take_control,
        tcp_transform=flange_to_tcp,
        link7_to_controller_flange=link7_to_flange,
    ), profile

