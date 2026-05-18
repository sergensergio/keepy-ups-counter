from typing import List

import cv2
import numpy as np

from .base_detector import BaseOnnxDetector
from .preprocessor import LetterboxInfo, YoloPreprocessor
from .types import BoundingBox

# COCO class id 32 = "sports ball".
SPORTS_BALL_CLASS_ID = 32


class BallDetector(BaseOnnxDetector):
    """YOLO11 detector filtered to the COCO `sports ball` class.

    YOLO11 detection ONNX output shape: (1, 4 + num_classes, num_anchors).
    For the default 80-class COCO model that is (1, 84, 8400).
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        input_size: int = 640,
        target_class_id: int = SPORTS_BALL_CLASS_ID,
    ):
        super().__init__(model_path)
        self.preprocessor = YoloPreprocessor(input_size)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_class_id = target_class_id

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        tensor, info = self.preprocessor.preprocess(frame)
        outputs = self._run(tensor)
        return self._postprocess(outputs[0], info)

    def _postprocess(
        self, raw_output: np.ndarray, info: LetterboxInfo
    ) -> List[BoundingBox]:
        # (1, 84, N) -> (N, 84)
        preds = raw_output[0].T

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

        mask = (class_ids == self.target_class_id) & (confidences > self.conf_threshold)
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]

        cx, cy, w, h = boxes_xywh.T
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0

        # cv2.dnn.NMSBoxes expects [x, y, w, h] (top-left + size).
        boxes_for_nms = np.stack([x1, y1, w, h], axis=1).tolist()
        keep = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )
        if len(keep) == 0:
            return []
        keep_idx = np.array(keep).flatten()

        results: List[BoundingBox] = []
        for i in keep_idx:
            x1m, y1m, x2m, y2m = self.preprocessor.reverse_box(
                (x1[i], y1[i], x1[i] + w[i], y1[i] + h[i]), info
            )
            results.append(
                BoundingBox(
                    x1=x1m,
                    y1=y1m,
                    x2=x2m,
                    y2=y2m,
                    confidence=float(confidences[i]),
                    class_id=self.target_class_id,
                )
            )
        return results
