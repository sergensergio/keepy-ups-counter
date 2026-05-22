"""YOLOv8 / YOLO11 ONNX postprocessor.

Detection output:  (1, 4 + num_classes, num_anchors) -> transposed -> (N, 4+C).
                   Per-row layout: [cx, cy, w, h, score_0, ..., score_{C-1}].
                   Requires argmax over class scores and post-hoc NMS.

"""

from typing import List, Optional

import cv2
import numpy as np

from ..preprocessor import LetterboxInfo, YoloPreprocessor
from ..types import BoundingBox
from .base import DetectionPostprocessor


def _nms_keep(
    x1: np.ndarray,
    y1: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    confidences: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """Run cv2 NMS on top-left + size boxes and return surviving indices."""
    boxes = np.stack([x1, y1, w, h], axis=1).tolist()
    keep = cv2.dnn.NMSBoxes(boxes, confidences.tolist(), conf_threshold, iou_threshold)
    if len(keep) == 0:
        return np.empty(0, dtype=int)
    return np.array(keep).flatten()


class Yolo11DetectionPostprocessor(DetectionPostprocessor):
    def parse(
        self,
        raw_outputs: List[np.ndarray],
        info: LetterboxInfo,
        preprocessor: YoloPreprocessor,
        conf_threshold: float,
        iou_threshold: float,
        target_class_id: Optional[int] = None,
    ) -> List[BoundingBox]:
        preds = raw_outputs[0][0].T  # (N, 4 + C)
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

        if target_class_id is not None:
            mask = (class_ids == target_class_id) & (confidences > conf_threshold)
        else:
            mask = confidences > conf_threshold
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        cx, cy, w, h = boxes_xywh.T
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0

        keep_idx = _nms_keep(x1, y1, w, h, confidences, conf_threshold, iou_threshold)
        if keep_idx.size == 0:
            return []

        results: List[BoundingBox] = []
        for i in keep_idx:
            x1m, y1m, x2m, y2m = preprocessor.reverse_box(
                (x1[i], y1[i], x1[i] + w[i], y1[i] + h[i]), info
            )
            results.append(
                BoundingBox(
                    x1=x1m,
                    y1=y1m,
                    x2=x2m,
                    y2=y2m,
                    confidence=float(confidences[i]),
                    class_id=int(class_ids[i]),
                )
            )
        return results
