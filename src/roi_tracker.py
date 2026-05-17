from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from .base_detector import Detector
from .types import BoundingBox, Pose


class RoiTrackingDetector:
    """Single-target ROI tracker that wraps any `Detector`.

    Policy:
      * If the previous frame produced no detection, run the wrapped detector
        on the full input image.
      * Otherwise, crop the new frame to the previous-frame bbox expanded by
        `padding_ratio` (a multiplier of bbox width/height) and run the
        detector only on that crop. Results are translated back to
        original-frame coordinates before being returned.

    State is internal; call `reset()` between independent videos.

    `padding_ratio` is the multiplier applied to each side of the seed bbox.
    `padding_ratio = 0.5` means the ROI is the bbox grown by 50% of its size on
    each side (so total width/height is 2x the bbox). For a fast-moving small
    object like a football, set this large (e.g. 2.0); for a slow-moving large
    object like a person, a small value (e.g. 0.2) is enough.
    """

    MIN_ROI_SIDE = 8  # px; below this the ROI is treated as degenerate

    def __init__(self, detector: Detector, padding_ratio: float = 0.5):
        if padding_ratio < 0:
            raise ValueError("padding_ratio must be >= 0")
        self.detector = detector
        self.padding_ratio = padding_ratio
        self._last_roi: Optional[Tuple[int, int, int, int]] = None

    def detect(self, frame: np.ndarray) -> List[Any]:
        if self._last_roi is None:
            results = list(self.detector.detect(frame))
        else:
            x1, y1, x2, y2 = self._last_roi
            crop = frame[y1:y2, x1:x2]
            raw = self.detector.detect(crop)
            results = [r.shift(x1, y1) for r in raw]

        self._last_roi = self._next_roi(results, frame.shape)
        return results

    def reset(self) -> None:
        self._last_roi = None

    @property
    def last_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """Current ROI in original-frame pixel coordinates, or None for full-frame."""
        return self._last_roi

    def _next_roi(
        self, results: List[Any], frame_shape: Tuple[int, ...]
    ) -> Optional[Tuple[int, int, int, int]]:
        if not results:
            return None
        best = max(results, key=lambda r: self._seed_bbox(r).confidence)
        return self._expand(self._seed_bbox(best), frame_shape)

    @staticmethod
    def _seed_bbox(result: Any) -> BoundingBox:
        # Pose carries an explicit person bbox; plain BoundingBox is its own seed.
        if isinstance(result, Pose):
            return result.bbox
        if isinstance(result, BoundingBox):
            return result
        raise TypeError(f"Unsupported detection result type: {type(result).__name__}")

    def _expand(
        self, bbox: BoundingBox, frame_shape: Tuple[int, ...]
    ) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame_shape[:2]
        bw = bbox.x2 - bbox.x1
        bh = bbox.y2 - bbox.y1
        pad_x = bw * self.padding_ratio
        pad_y = bh * self.padding_ratio

        x1 = max(0, int(round(bbox.x1 - pad_x)))
        y1 = max(0, int(round(bbox.y1 - pad_y)))
        x2 = min(w, int(round(bbox.x2 + pad_x)))
        y2 = min(h, int(round(bbox.y2 + pad_y)))

        if x2 - x1 < self.MIN_ROI_SIDE or y2 - y1 < self.MIN_ROI_SIDE:
            return None
        return (x1, y1, x2, y2)
