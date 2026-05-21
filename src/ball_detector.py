from typing import List, Optional

import numpy as np

from .base_detector import BaseOnnxDetector
from .postprocessors import (
    DetectionPostprocessor,
    Yolo11DetectionPostprocessor,
)
from .preprocessor import YoloPreprocessor
from .types import BoundingBox

# COCO class id 32 = "sports ball".
SPORTS_BALL_CLASS_ID = 0


class BallDetector(BaseOnnxDetector):
    """ONNX ball detector, agnostic to YOLO architecture.

    Architecture-specific output parsing lives in a `DetectionPostprocessor`
    strategy injected at construction time — the detector itself only owns
    ONNX inference and letterbox pre/reverse-mapping.

    Defaults to the YOLO11 (transposed + NMS) postprocessor for backwards
    compatibility; pass `Yolo26DetectionPostprocessor()` for end-to-end models.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        input_size: int = 640,
        target_class_id: int = SPORTS_BALL_CLASS_ID,
        postprocessor: Optional[DetectionPostprocessor] = None,
    ):
        super().__init__(model_path)
        self.preprocessor = YoloPreprocessor(input_size)
        self.postprocessor = postprocessor or Yolo11DetectionPostprocessor()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_class_id = target_class_id

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        tensor, info = self.preprocessor.preprocess(frame)
        outputs = self._run(tensor)
        return self.postprocessor.parse(
            outputs,
            info,
            self.preprocessor,
            self.conf_threshold,
            self.iou_threshold,
            target_class_id=self.target_class_id,
        )
