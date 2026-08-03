"""BottleDetector target_classes filtering (no hardware, no ultralytics).

Shelf/vending selection needs to pick "the requested product", not just
"any bottle". target_classes, when given, replaces the generic bottle-alias
filter; omitted, detect()/_best() must behave exactly as before (regression
guard for table_demo, which never passes target_classes).
"""

import threading

import numpy as np
import pytest

from bottle_grasp.perception import BottleDetector


class FakeBox:
    def __init__(self, cls_index, confidence, xyxy):
        self._cls_index = cls_index
        self._confidence = confidence
        self._xyxy = xyxy

    @property
    def cls(self):
        return [self._cls_index]

    @property
    def conf(self):
        return [self._confidence]

    @property
    def xyxy(self):
        return [np.asarray(self._xyxy, dtype=float)]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """names maps class index -> label; predict() returns canned boxes."""

    def __init__(self, names, boxes):
        self.names = names
        self._boxes = boxes

    def predict(self, bgr, conf, verbose=False):
        return [FakeResult(self._boxes)]


def _detector(model, fallback_model=None):
    detector = BottleDetector.__new__(BottleDetector)
    detector.model = model
    detector.confidence = 0.25
    detector.lock = threading.Lock()
    detector.aliases = {"bottle", "water", "mineral_water", "矿泉水", "水瓶"}
    detector.fallback_model = fallback_model
    detector.fallback_confidence = 0.05
    return detector


NAMES = {0: "bottle", 1: "coke_can", 2: "sprite_bottle"}


def test_without_target_classes_keeps_generic_bottle_alias_behaviour():
    """table_demo never passes target_classes: behaviour must be unchanged."""
    boxes = [
        FakeBox(0, 0.9, (0, 0, 10, 10)),  # bottle: matches generic alias
        FakeBox(1, 0.95, (0, 0, 10, 10)),  # coke_can: not in generic aliases
    ]
    detector = _detector(FakeModel(NAMES, boxes))
    detection = detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))
    assert detection is not None
    assert detection.class_name == "bottle"


def test_target_classes_selects_matching_product_over_higher_confidence_other():
    boxes = [
        FakeBox(1, 0.95, (0, 0, 10, 10)),  # coke_can: higher confidence
        FakeBox(2, 0.5, (5, 5, 20, 20)),  # sprite_bottle: requested product
    ]
    detector = _detector(FakeModel(NAMES, boxes))
    detection = detector.detect(
        np.zeros((4, 4, 3), dtype=np.uint8),
        target_classes={"sprite_bottle"},
    )
    assert detection is not None
    assert detection.class_name == "sprite_bottle"


def test_target_classes_returns_none_when_product_not_in_frame():
    boxes = [FakeBox(0, 0.9, (0, 0, 10, 10)), FakeBox(1, 0.95, (0, 0, 10, 10))]
    detector = _detector(FakeModel(NAMES, boxes))
    detection = detector.detect(
        np.zeros((4, 4, 3), dtype=np.uint8),
        target_classes={"sprite_bottle"},
    )
    assert detection is None


def test_target_classes_applies_to_fallback_model_too():
    primary = FakeModel(NAMES, [])  # primary finds nothing
    fallback_boxes = [FakeBox(2, 0.3, (0, 0, 10, 10))]
    fallback = FakeModel(NAMES, fallback_boxes)
    detector = _detector(primary, fallback_model=fallback)
    detection = detector.detect(
        np.zeros((4, 4, 3), dtype=np.uint8),
        target_classes={"sprite_bottle"},
    )
    assert detection is not None
    assert detection.class_name == "sprite_bottle"


def test_predicate_still_applies_alongside_target_classes():
    boxes = [FakeBox(2, 0.5, (0, 0, 10, 10))]
    detector = _detector(FakeModel(NAMES, boxes))
    detection = detector.detect(
        np.zeros((4, 4, 3), dtype=np.uint8),
        predicate=lambda det: False,
        target_classes={"sprite_bottle"},
    )
    assert detection is None
