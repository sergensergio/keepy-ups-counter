"""Strategy interfaces for converting raw ONNX outputs into structured detections.

Different YOLO families ship different output tensor layouts:
  * v8 / v11 -> transposed (1, 4 + C [+ 3K], N), needs NMS
  * v10 / v26 (end-to-end) -> top-K filtered (1, max_det, 6 [+ 3K]), no NMS

Detectors stay agnostic of these differences: they only know how to run
ONNX inference and reverse-map letterbox coordinates. All format-specific
logic lives in a `DetectionPostprocessor` strategy.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..preprocessor import LetterboxInfo, YoloPreprocessor
from ..types import BoundingBox


class DetectionPostprocessor(ABC):
    @abstractmethod
    def parse(
        self,
        raw_outputs: List[np.ndarray],
        info: LetterboxInfo,
        preprocessor: YoloPreprocessor,
        conf_threshold: float,
        iou_threshold: float,
        target_class_id: Optional[int] = None,
    ) -> List[BoundingBox]:
        ...
