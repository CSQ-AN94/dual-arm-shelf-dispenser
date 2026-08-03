"""Typed post-lift evidence; ordinary safety exceptions are not outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .core import Localization


class LiftEvidenceKind(str, Enum):
    VISUAL_CONFIRMED = "visual_confirmed"
    VISUAL_NEGATIVE = "visual_negative"
    OCCLUDED_WITH_FRESH_FRAME = "occluded_with_fresh_frame"
    CAMERA_UNAVAILABLE = "camera_unavailable"
    INSUFFICIENT_DEPTH = "insufficient_depth"


@dataclass(frozen=True)
class LiftVisualEvidence:
    kind: LiftEvidenceKind
    reason: str
    fresh_frames: int
    associated_frames: int
    measurement: Localization | None = None
