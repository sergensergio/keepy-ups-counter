"""YOLO26 ONNX postprocessors (end-to-end / NMS-free).

YOLO26 is designed as an end-to-end detector: the ONNX graph emits a fixed
number of top-K predictions, pre-sorted by score, with NMS already baked in.
The architecture-specific postprocessor therefore only needs to:
  * filter by user confidence threshold,
  * optionally restrict to a target class id,
  * reverse-map letterbox coordinates back to the original frame.

Expected output tensor layouts (from the Ultralytics export):

  Detection (yolo26n.onnx):
      shape (1, max_det, 6)
      per row: [x1, y1, x2, y2, conf, class_id]   <-- xyxy already in pixels
"""

from typing import List, Optional

import numpy as np

from ..preprocessor import LetterboxInfo, YoloPreprocessor
from ..types import BoundingBox
from .base import DetectionPostprocessor


class Yolo26DetectionPostprocessor(DetectionPostprocessor):
    def parse(
        self,
        raw_outputs: List[np.ndarray],
        info: LetterboxInfo,
        preprocessor: YoloPreprocessor,
        conf_threshold: float,
        iou_threshold: float,  # unused — kept for interface symmetry
        target_class_id: Optional[int] = None,
    ) -> List[BoundingBox]:
        preds = raw_outputs[0][0]  # (max_det, 6)
        if preds.size == 0:
            return []

        x1 = preds[:, 0]
        y1 = preds[:, 1]
        x2 = preds[:, 2]
        y2 = preds[:, 3]
        conf = preds[:, 4]
        class_ids = preds[:, 5].astype(int)

        mask = conf > conf_threshold
        if target_class_id is not None:
            mask = mask & (class_ids == target_class_id)
        if not np.any(mask):
            return []

        idx = np.where(mask)[0]
        results: List[BoundingBox] = []
        for i in idx:
            x1m, y1m, x2m, y2m = preprocessor.reverse_box(
                (x1[i], y1[i], x2[i], y2[i]), info
            )
            results.append(
                BoundingBox(
                    x1=x1m,
                    y1=y1m,
                    x2=x2m,
                    y2=y2m,
                    confidence=float(conf[i]),
                    class_id=int(class_ids[i]),
                )
            )
        return results
