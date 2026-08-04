"""Authored grasp-frame construction shared by rehearsal and execution.

Shelf grasp axes describe the physical gripper-base frame: +Z is the approach
axis, +X is the finger opening axis, and +Y is the palm-height axis. The
installed RMG24 controller TCP differs by the fixed +90° mount about r_link7 Z;
``installed_rmg24_tcp_rotation()`` applies that relationship for the new MTC
chain without changing the legacy public launcher.

There is a legacy collinear-offset helper below for non-executing historical
rehearsals.  It is deliberately not evidence that the installed right gripper
has no mounting rotation.  Hardware execution of a profile with authored
shelf axes requires an explicit, verified ``ToolMountCalibration``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import SafetyAbort


@dataclass(frozen=True)
class GraspFrameSpec:
    """Physical shelf-grasp axes expressed in the profile/base frame."""

    opening_normal_base: tuple[float, float, float]
    finger_axis_base: tuple[float, float, float]
    palm_vertical_base: tuple[float, float, float]


MEASURED = "measured"
NOMINAL_FUNCTIONALLY_VALIDATED = "nominal_functionally_validated"
# The nominal installation with nothing downstream having absorbed its error:
# no metrology, and no task on this arm that succeeded and therefore proved the
# residual small enough.  Representable so an arm can be moved in free space
# under an honest label, never enough to grasp with.
NOMINAL_UNVALIDATED = "nominal_unvalidated"


@dataclass(frozen=True)
class ToolMountCalibration:
    """Transforms spanning MoveIt link7 and the controller TCP.

    ``T_link7_controller_flange`` is the pose of the controller flange in
    ``r_link7`` and ``T_controller_flange_tcp`` is the pose of the physical
    TCP in the controller flange. Both use the usual ``T_parent_child``
    convention. Keeping them separate prevents a tool-installation rotation
    from being hidden in a scalar TCP-Z offset.

    ``provenance`` says how the numbers were obtained, because the two ways
    carry very different guarantees and used to be indistinguishable:

    ``measured``
        Metrology.  Residuals are real and are checked against the admission
        limits.

    ``nominal_functionally_validated``
        The nominal installation from the robot description, kept because a
        downstream working distance was tuned on the real arm until the task
        succeeded and therefore absorbs whatever the true offset is.  There
        are no residuals to report; claiming any would be fiction.  Good
        enough to keep one validated configuration running, not good enough
        to transfer to another arm, another approach direction, or to a
        planner that derives the grasp point geometrically.

    ``nominal_unvalidated``
        The nominal installation, on an arm where nothing has yet absorbed its
        error.  Free-space motion only: a fence check is wrong by whatever the
        true offset is, which generous transit zones tolerate and a grasp does
        not.  ``grasping_allowed`` is False.

    ``verified`` means the record was chosen deliberately either way -- it
    exists so a profile cannot silently fall back to an unexamined identity
    assumption.  It does not by itself mean anything was measured.
    """

    verified: bool
    provenance: str = MEASURED
    evidence_id: str | None = None
    measured_at_utc: str | None = None
    max_position_residual_m: float | None = None
    max_orientation_residual_deg: float | None = None
    T_link7_controller_flange: np.ndarray | None = None
    T_controller_flange_tcp: np.ndarray | None = None

    @property
    def grasping_allowed(self) -> bool:
        """Whether this record is good enough to close fingers on something."""
        return self.provenance in (MEASURED, NOMINAL_FUNCTIONALLY_VALIDATED)

    def require_grasping_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Transforms for a grasp, which an unvalidated nominal cannot back."""
        if not self.grasping_allowed:
            raise SafetyAbort(
                f"工具标定 provenance={self.provenance} 只能用于自由空间运动，"
                "不足以支撑抓取：标称偏差没有任何东西吸收过"
            )
        return self.require_transforms()

    def require_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Return defensive copies only when this calibration is executable."""
        if (
            not self.verified
            or self.T_link7_controller_flange is None
            or self.T_controller_flange_tcp is None
        ):
            raise SafetyAbort(
                "工具安装变换未完成实测/验证，禁止将 link7、控制器法兰与 TCP 混用"
            )
        # Dataclass instances may also be assembled by an embedding caller,
        # bypassing the JSON loader, so validate again at the safety boundary.
        return (
            validate_rigid_transform(
                self.T_link7_controller_flange,
                label="T_link7_controller_flange",
            ),
            validate_rigid_transform(
                self.T_controller_flange_tcp,
                label="T_controller_flange_tcp",
            ),
        )

    @property
    def T_link7_tcp(self) -> np.ndarray:
        link7_to_flange, flange_to_tcp = self.require_transforms()
        return link7_to_flange @ flange_to_tcp


def tool_mount_chain_residual(
    *,
    T_base_link7,
    T_base_tcp,
    T_link7_controller_flange,
    T_controller_flange_tcp,
) -> tuple[float, float]:
    """Compare observed TCP against the composed, same-frame mount chain.

    RealMan Algo's zero-tool FK is the model endpoint ``r_link7``. The
    controller's active tool pose remains controller-flange-relative.
    Therefore a static sample is predicted as
    ``T_base_link7 @ T_link7_controller_flange @ T_controller_flange_tcp``;
    comparing ``inv(T_base_link7) @ T_base_tcp`` directly with only the
    flange→TCP segment would misclassify the known link7→flange offset as a
    calibration residual.

    This calculation reports evidence quality only. It never marks a
    calibration or execution profile verified.
    """

    base_link7 = validate_rigid_transform(
        T_base_link7, label="T_base_link7"
    )
    base_tcp = validate_rigid_transform(T_base_tcp, label="T_base_tcp")
    link7_flange = validate_rigid_transform(
        T_link7_controller_flange,
        label="T_link7_controller_flange",
    )
    flange_tcp = validate_rigid_transform(
        T_controller_flange_tcp,
        label="T_controller_flange_tcp",
    )
    predicted = base_link7 @ link7_flange @ flange_tcp
    position_m = float(
        np.linalg.norm(predicted[:3, 3] - base_tcp[:3, 3])
    )
    relative = predicted[:3, :3].T @ base_tcp[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    orientation_deg = float(np.degrees(np.arccos(cosine)))
    return position_m, orientation_deg


def validate_rigid_transform(value, *, label: str) -> np.ndarray:
    """Validate a finite proper homogeneous rigid transform."""
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise SafetyAbort(f"{label} 必须是 4x4 有限刚体变换")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise SafetyAbort(f"{label} 最后一行必须是 [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise SafetyAbort(f"{label} 旋转部分必须正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise SafetyAbort(f"{label} 旋转部分行列式必须为 +1")
    return transform.copy()


def _unit(vector, *, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise SafetyAbort(f"{label} 必须是 3 个有限数")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise SafetyAbort(f"{label} 不能是零向量")
    return value / norm


def authored_tcp_rotation(spec: GraspFrameSpec) -> np.ndarray:
    """Construct and orthogonalize the one authored physical grasp frame.

    TCP +Z follows the shelf opening normal. TCP +X is the horizontal finger
    axis after removing any component parallel to +Z. The right-handed TCP +Y
    is ``Z × X`` and must point along the authored vertical palm direction.
    """

    z_axis = _unit(spec.opening_normal_base, label="货架开口法向")
    finger = _unit(spec.finger_axis_base, label="物理夹指轴")
    finger = finger - float(np.dot(finger, z_axis)) * z_axis
    finger_norm = float(np.linalg.norm(finger))
    if finger_norm <= 1e-6:
        raise SafetyAbort("物理夹指轴不能与货架开口法向平行")
    x_axis = finger / finger_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= float(np.linalg.norm(y_axis))
    palm = _unit(spec.palm_vertical_base, label="手掌竖直轴")
    if abs(float(np.dot(palm, z_axis))) > 1e-6:
        raise SafetyAbort("手掌竖直轴必须垂直于货架开口法向")
    if float(np.dot(y_axis, palm)) < 0.0:
        x_axis = -x_axis
        y_axis = -y_axis
    if float(np.dot(y_axis, palm)) < 1.0 - 1e-6:
        raise SafetyAbort("夹指轴、开口法向与手掌竖直语义不一致")

    authored = np.column_stack((x_axis, y_axis, z_axis))
    u, _singular, vt = np.linalg.svd(authored)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9)
    ):
        raise SafetyAbort("抓取姿态正交化失败")
    return rotation


def installed_rmg24_tcp_rotation(spec: GraspFrameSpec) -> np.ndarray:
    """Convert the physical gripper frame to the installed controller TCP.

    The mount is +90° about r_link7 Z.  4bc47cc introduced this helper with
    Rz(-90°) instead, which is the same axis 180° the wrong way round: every
    scenario authored since then asked for a grasp rolled upside down about
    the approach axis, and MoveIt could only reach it by driving r_joint3 into
    its position limit.  See
    ``test_installed_rmg24_tcp_matches_demonstrated_grasp``.
    """

    physical = authored_tcp_rotation(spec)
    controller_from_gripper = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return physical @ controller_from_gripper


def link7_to_tcp_fixed_mount(
    *,
    link7_to_controller_flange_m: float,
    controller_flange_to_tcp_m: float,
    T_link7_controller_flange: np.ndarray | None = None,
    T_controller_flange_tcp: np.ndarray | None = None,
) -> np.ndarray:
    """Compose a measured mount, or the legacy offline collinear fallback.

    Supplying exactly one full transform is ambiguous and rejected. With no
    matrices this preserves historic offline fixtures only; hardware execution
    is gated by the safety-profile loader rather than treating the fallback as
    an audited installation fact.
    """

    supplied = (T_link7_controller_flange, T_controller_flange_tcp)
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            raise SafetyAbort(
                "工具安装变换必须同时提供 T_link7_controller_flange 和 "
                "T_controller_flange_tcp"
            )
        return validate_rigid_transform(
            T_link7_controller_flange,
            label="T_link7_controller_flange",
        ) @ validate_rigid_transform(
            T_controller_flange_tcp,
            label="T_controller_flange_tcp",
        )

    offsets = np.asarray(
        [link7_to_controller_flange_m, controller_flange_to_tcp_m],
        dtype=float,
    )
    if offsets.shape != (2,) or not np.all(np.isfinite(offsets)):
        raise SafetyAbort("link7→TCP 历史离线偏移必须是有限数")
    if np.any(offsets < 0.0):
        raise SafetyAbort("link7→TCP 历史离线偏移不能为负")
    transform = np.eye(4)
    transform[2, 3] = float(np.sum(offsets))
    return transform


def resolve_tcp_grasp_rotation(profile, observed_tcp: np.ndarray) -> np.ndarray:
    """Use authored shelf geometry, preserving legacy non-shelf behaviour."""

    observed = np.asarray(observed_tcp, dtype=float)
    if observed.shape != (4, 4) or not np.all(np.isfinite(observed)):
        raise SafetyAbort("当前 TCP 位姿无效，无法构造抓取姿态")
    spec = getattr(profile, "grasp_frame", None)
    if spec is None:
        return observed[:3, :3].copy()
    return authored_tcp_rotation(spec)
