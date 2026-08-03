"""Locked-target visibility supervision for multi-camera grasp motion.

The interface deliberately separates three different facts that the old
``ensure_bottle_visible`` boolean collapsed together:

* the RGB-D stream is alive;
* a wrist detection is associated with the already locked 3-D target;
* an independent head observer can confirm that target when the wrist view
  genuinely loses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .core import BottleDetectionLost, Detection, SafetyAbort


@dataclass(frozen=True)
class GuardResult:
    source: str
    detection: Detection | None = None


class LockedTargetGuard:
    """Verify one locked target, using head confirmation at most once per leg."""

    def __init__(
        self,
        *,
        wrist_check: Callable[[np.ndarray], Detection | None],
        head_confirm: Callable[[np.ndarray], None],
    ):
        self._wrist_check = wrist_check
        self._head_confirm = head_confirm
        self._head_confirmed = False

    def verify(self, target_base: np.ndarray) -> GuardResult:
        target = np.asarray(target_base, dtype=float)
        try:
            detection = self._wrist_check(target)
            return GuardResult("wrist", detection)
        except BottleDetectionLost:
            # CameraFrameUnavailable and every other SafetyAbort deliberately
            # pass through: only a live-frame detector miss is degradable.
            if not self._head_confirmed:
                self._head_confirm(target)
                self._head_confirmed = True
                return GuardResult("head")
            return GuardResult("head_cached")


@dataclass(frozen=True)
class ProjectedTargetAssociation:
    """Associate raw bottle boxes with a locked 3-D target projection."""

    pixel: tuple[float, float]
    in_front: bool
    in_image: bool
    image_size: tuple[int, int]

    @classmethod
    def from_view(
        cls,
        *,
        target_base: np.ndarray,
        T_base_camera: np.ndarray,
        intrinsics: np.ndarray,
        image_shape,
    ) -> "ProjectedTargetAssociation":
        height, width = image_shape[:2]
        target = np.r_[np.asarray(target_base, dtype=float), 1.0]
        point_camera = np.linalg.inv(np.asarray(T_base_camera, dtype=float)) @ target
        z = float(point_camera[2])
        if z <= 1e-6:
            return cls(
                (float("nan"), float("nan")),
                False,
                False,
                (int(width), int(height)),
            )
        K = np.asarray(intrinsics, dtype=float)
        u = float(K[0, 0] * point_camera[0] / z + K[0, 2])
        v = float(K[1, 1] * point_camera[1] / z + K[1, 2])
        return cls(
            (u, v),
            True,
            0 <= u < width and 0 <= v < height,
            (int(width), int(height)),
        )

    @staticmethod
    def _axis_accepts(
        coordinate: float,
        box_minimum: float,
        box_maximum: float,
        image_extent: int,
        box_span: float,
        ordinary_margin: float,
        *,
        border_margin_px: int = 4,
    ) -> bool:
        if 0 <= coordinate < image_extent:
            return (
                box_minimum - ordinary_margin
                <= coordinate
                <= box_maximum + ordinary_margin
            )

        # A locked physical grasp point may leave the frame while a clipped
        # portion of the same bottle remains visible.  Admit that association
        # only at the corresponding touched border and only for a bounded
        # fraction of the visible box/image; an arbitrary off-screen object is
        # never accepted merely because some bottle is present.
        outside_limit = max(
            24.0,
            min(0.25 * image_extent, 0.5 * box_span + 12.0),
        )
        if coordinate < 0:
            return (
                box_minimum <= border_margin_px
                and coordinate >= -outside_limit
            )
        return (
            box_maximum >= image_extent - 1 - border_margin_px
            and coordinate <= image_extent - 1 + outside_limit
        )

    def accepts(self, detection: Detection) -> bool:
        if not self.in_front:
            return False
        x1, y1, x2, y2 = detection.box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        margin_x = max(12.0, 0.18 * width)
        margin_y = max(12.0, 0.18 * height)
        u, v = self.pixel
        image_width, image_height = self.image_size
        return (
            self._axis_accepts(
                u,
                x1,
                x2,
                image_width,
                width,
                margin_x,
            )
            and self._axis_accepts(
                v,
                y1,
                y2,
                image_height,
                height,
                margin_y,
            )
        )

    @staticmethod
    def touches_image_border(
        detection: Detection, image_shape, *, margin_px: int = 4
    ) -> bool:
        """Return whether a detector box is visibly truncated by the frame.

        The 2026-07-18 wrist failure had ``y2=479`` in a 480-pixel image.
        Treating that box as a complete bottle changed the physical meaning of
        its height-dependent grasp pixel.  Border contact is deterministic
        missing-data evidence, not detector noise.
        """
        height, width = image_shape[:2]
        x1, y1, x2, y2 = detection.box
        return bool(
            x1 <= margin_px
            or y1 <= margin_px
            or x2 >= width - 1 - margin_px
            or y2 >= height - 1 - margin_px
        )

    def refine_locked_depth(
        self,
        *,
        detection: Detection,
        target_base: np.ndarray,
        T_base_camera: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float]:
        """Refine a locked target without reinterpreting a truncated box.

        The fixed-head observation owns the target depth and grasp height.
        A wrist detection may refine horizontal centring, but it must not use
        the near surface of a transparent bottle (or a clipped box fraction)
        to overwrite that lock.  The returned tuple is camera point, base
        point, pixel and locked camera depth.
        """
        if not self.accepts(detection):
            raise SafetyAbort("腕部 bottle 检测与头部锁定目标投影不关联")
        target = np.r_[np.asarray(target_base, dtype=float), 1.0]
        transform = np.asarray(T_base_camera, dtype=float)
        point_camera = np.linalg.inv(transform) @ target
        z = float(point_camera[2])
        if z <= 1e-6:
            raise SafetyAbort("头部锁定目标位于腕部相机后方")

        K = np.asarray(intrinsics, dtype=float)
        x1, _y1, x2, _y2 = detection.box
        # The cylinder centre is better represented by the 2-D box centre
        # than by whichever label/surface pixels happened to return depth.
        # A side-clipped box has no horizontal-centre semantics, so preserve
        # the locked projection there.  Top/bottom clipping does not invalidate
        # the visible horizontal centre.  Vertical grasp height always remains
        # owned by the fixed-head lock.
        image_width, _image_height = self.image_size
        side_truncated = x1 <= 4 or x2 >= image_width - 1 - 4
        u = float(self.pixel[0]) if side_truncated else 0.5 * (x1 + x2)
        v = float(self.pixel[1])
        refined_camera = np.array(
            [
                (u - K[0, 2]) * z / K[0, 0],
                (v - K[1, 2]) * z / K[1, 1],
                z,
            ],
            dtype=float,
        )
        refined_base = (transform @ np.r_[refined_camera, 1.0])[:3]
        return refined_camera, refined_base, (float(u), v), z


@dataclass(frozen=True)
class PostLiftTargetAssociation:
    """Associate a lifted target without reusing its stale vertical projection.

    The wrist/gripper moves with the bottle, and the hand commonly hides the
    lower part of the box.  Horizontal bearing, requested class and apparent
    width remain useful identity constraints; the old vertical box projection
    does not get veto authority after lift.
    """

    projected_u: float
    projected_v: float
    expected_depth_m: float
    expected_width_px: float
    image_size: tuple[int, int]
    class_name: str | None = None

    @classmethod
    def from_view(
        cls,
        *,
        locked,
        expected_base: np.ndarray,
        T_base_camera: np.ndarray,
        intrinsics: np.ndarray,
        image_shape,
    ) -> "PostLiftTargetAssociation":
        height, width = image_shape[:2]
        target_camera = np.linalg.inv(np.asarray(T_base_camera, dtype=float)) @ np.r_[
            np.asarray(expected_base, dtype=float), 1.0
        ]
        z = float(target_camera[2])
        if not np.isfinite(z) or z <= 1e-6:
            return cls(
                float("nan"),
                float("nan"),
                z,
                0.0,
                (int(width), int(height)),
                getattr(locked, "class_name", None),
            )
        K = np.asarray(intrinsics, dtype=float)
        u = float(K[0, 0] * target_camera[0] / z + K[0, 2])
        v = float(K[1, 1] * target_camera[1] / z + K[1, 2])
        locked_width = max(1.0, float(locked.box[2] - locked.box[0]))
        locked_depth = float(getattr(locked, "depth_m", z))
        scale = locked_depth / z if np.isfinite(locked_depth) else 1.0
        expected_width = float(
            np.clip(locked_width * scale, 0.35 * locked_width, 3.0 * locked_width)
        )
        return cls(
            u,
            v,
            z,
            expected_width,
            (int(width), int(height)),
            getattr(locked, "class_name", None),
        )

    def accepts(self, detection: Detection) -> bool:
        if (
            not np.isfinite(self.projected_u)
            or not np.isfinite(self.expected_depth_m)
            or self.expected_depth_m <= 0.0
        ):
            return False
        if (
            self.class_name is not None
            and detection.class_name.lower().replace(" ", "_")
            != self.class_name.lower().replace(" ", "_")
        ):
            return False
        x1, _y1, x2, _y2 = detection.box
        width = max(1.0, float(x2 - x1))
        ratio = width / max(1.0, self.expected_width_px)
        if not 0.45 <= ratio <= 2.2:
            return False
        margin = max(12.0, 0.25 * self.expected_width_px)
        return bool(x1 - margin <= self.projected_u <= x2 + margin)

    def occlusion_roi(self) -> tuple[slice, slice]:
        width, height = self.image_size
        half_width = int(max(12.0, 0.55 * self.expected_width_px))
        half_height = int(max(12.0, 0.30 * self.expected_width_px))
        u = int(round(self.projected_u))
        v = int(round(self.projected_v))
        return (
            slice(max(0, v - half_height), min(height, v + half_height + 1)),
            slice(max(0, u - half_width), min(width, u + half_width + 1)),
        )
