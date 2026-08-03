"""YOLO bottle detection and robust transparent-object depth estimation."""

from __future__ import annotations

import logging
import threading
from typing import Optional, Sequence

import numpy as np

from .core import DemoParams, Detection, InsufficientDepth

LOG = logging.getLogger("bottle_demo")


def robust_near_cluster(values: np.ndarray, params: DemoParams) -> tuple[float, float]:
    """Select the nearest supported depth cluster instead of wall background."""
    values = np.asarray(values, dtype=float)
    values = values[
        np.isfinite(values)
        & (values >= params.min_depth_m)
        & (values <= params.max_depth_m)
    ]
    if values.size < 18:
        raise InsufficientDepth(f"有效深度点太少: {values.size}")

    bins = np.arange(params.min_depth_m, params.max_depth_m + 0.011, 0.01)
    hist, edges = np.histogram(values, bins=bins)
    supported = np.flatnonzero(hist >= max(5, int(values.size * 0.035)))
    if supported.size == 0:
        raise InsufficientDepth("深度没有稳定聚类")

    runs: list[list[int]] = [[int(supported[0])]]
    for idx in supported[1:]:
        if int(idx) <= runs[-1][-1] + 1:
            runs[-1].append(int(idx))
        else:
            runs.append([int(idx)])

    chosen = None
    for run in runs:
        lo, hi = edges[run[0]], edges[run[-1] + 1]
        cluster = values[(values >= lo - 0.006) & (values <= hi + 0.006)]
        if cluster.size >= max(12, int(values.size * 0.10)):
            chosen = cluster
            break
    if chosen is None:
        raise InsufficientDepth("近景深度聚类支持不足")

    median = float(np.median(chosen))
    mad = float(np.median(np.abs(chosen - median)))
    filtered = chosen[np.abs(chosen - median) <= max(0.012, 3.5 * mad)]
    if filtered.size < 10:
        raise InsufficientDepth("深度离群过滤后点数不足")
    median = float(np.median(filtered))
    mad = float(np.median(np.abs(filtered - median)))
    if mad > params.max_depth_mad_m:
        raise InsufficientDepth(f"深度分布过散: MAD={mad * 1000:.1f} mm")
    return median, mad


def _zones(
    box: Sequence[int], shape: Sequence[int]
) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = map(int, box)
    h, w = shape[:2]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    specs = ((0.35, 0.03, 0.65, 0.22), (0.25, 0.38, 0.75, 0.68), (0.34, 0.20, 0.66, 0.86))
    return [
        (
            max(0, int(x1 + ax * bw)),
            max(0, int(y1 + ay * bh)),
            min(w, int(x1 + bx * bw)),
            min(h, int(y1 + by * bh)),
        )
        for ax, ay, bx, by in specs
    ]


def depth_point_for_detection(
    depth: np.ndarray,
    det: Detection,
    K: np.ndarray,
    params: DemoParams,
    *,
    measured_vertical: bool = False,
) -> tuple[np.ndarray, float, float, tuple[float, float]]:
    """Fuse equal-sized cap, label, and body samples into a camera 3-D point.

    Initial grasp localization authors a stable vertical pixel from the full
    detection box.  Post-lift evidence instead requests ``measured_vertical``:
    its vertical coordinate then comes only from fresh depth-support pixels,
    because a bottom-clipped box no longer represents the full bottle.
    """
    all_values, pixels = [], []
    rng = np.random.default_rng(7)
    for x1, y1, x2, y2 in _zones(det.box, depth.shape):
        roi = depth[y1:y2, x1:x2]
        yy, xx = np.nonzero(
            np.isfinite(roi)
            & (roi >= params.min_depth_m)
            & (roi <= params.max_depth_m)
        )
        if yy.size == 0:
            continue
        take = rng.choice(yy.size, size=min(180, yy.size), replace=False)
        z = roi[yy[take], xx[take]]
        all_values.append(z)
        pixels.append(np.column_stack((xx[take] + x1, yy[take] + y1, z)))
    if not all_values:
        raise InsufficientDepth("瓶盖/标签/瓶身区域均无有效深度")

    z, mad = robust_near_cluster(np.concatenate(all_values), params)
    samples = np.concatenate(pixels)
    near = samples[np.abs(samples[:, 2] - z) <= max(0.015, 3.5 * mad)]
    if near.shape[0] < 8:
        raise InsufficientDepth("目标深度附近的像素太少")
    # 横向取深度支持像素的中位数（对松框鲁棒）。初始定位的纵向坐标采用
    # 固定框比例以稳定抓取点；post-lift 则只能采用新鲜深度支持像素，不能
    # 把截断检测框或旧投影解释成实际抬升测量。
    u = float(np.median(near[:, 0]))
    x1, y1, x2, y2 = det.box
    v = (
        float(np.median(near[:, 1]))
        if measured_vertical
        else float(y1 + params.grasp_height_fraction * (y2 - y1))
    )
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    point = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float)
    return point, z, mad, (u, v)


class BottleDetector:
    """A single long-lived YOLO instance restricted to bottle aliases.

    The fine-tuned model is confident when the bottle label faces the camera
    but its confidence collapses (observed <0.1 vs. >0.8) once the bottle is
    turned label-away, since it leans on label/branding features rather than
    silhouette. An optional generic COCO fallback model, tried only when the
    primary model finds nothing, recovers detection in that case.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float,
        fallback_model_path: Optional[str] = None,
        fallback_confidence: float = 0.05,
    ):
        from ultralytics import YOLO

        LOG.info("加载 YOLO: %s", model_path)
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.lock = threading.Lock()
        self.aliases = {"bottle", "water", "mineral_water", "矿泉水", "水瓶"}
        self.fallback_model = None
        self.fallback_confidence = fallback_confidence
        if fallback_model_path is not None:
            LOG.info("加载兜底 YOLO: %s", fallback_model_path)
            self.fallback_model = YOLO(fallback_model_path)

    def _best(
        self,
        model,
        bgr: np.ndarray,
        confidence: float,
        predicate=None,
        target_classes: Optional[set] = None,
    ) -> Optional[Detection]:
        with self.lock:
            result = model.predict(bgr, conf=confidence, verbose=False)[0]
        allowed = target_classes if target_classes is not None else self.aliases
        choices = []
        for box in result.boxes:
            cls = int(box.cls[0])
            name = str(model.names[cls])
            normalized = name.lower().replace(" ", "_")
            if normalized not in allowed and name not in allowed:
                continue
            detection = Detection(
                tuple(map(int, box.xyxy[0].tolist())),
                float(box.conf[0]),
                name,
            )
            if predicate is not None and not predicate(detection):
                continue
            choices.append(detection)
        return max(choices, key=lambda d: d.confidence, default=None)

    def detect(
        self,
        bgr: np.ndarray,
        predicate=None,
        target_classes: Optional[set] = None,
    ) -> Optional[Detection]:
        """Best bottle detection, optionally restricted by a predicate.

        The predicate pre-filters candidate boxes (e.g. a close-range shape
        gate) so a higher-confidence but implausible background bottle cannot
        shadow the real target.

        `target_classes`, when given, replaces the generic bottle-alias
        class filter with a specific set of YOLO class names — shelf/vending
        selection by requested product instead of "any bottle". Omitted, the
        behaviour is unchanged: match against the built-in `self.aliases`.
        """
        detection = self._best(
            self.model, bgr, self.confidence, predicate, target_classes
        )
        if detection is not None or self.fallback_model is None:
            return detection
        return self._best(
            self.fallback_model,
            bgr,
            self.fallback_confidence,
            predicate,
            target_classes,
        )
