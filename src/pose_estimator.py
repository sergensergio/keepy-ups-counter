from typing import List, Optional

import numpy as np

from .base_detector import BaseOnnxDetector
from .postprocessors import PosePostprocessor, Yolo11PosePostprocessor
from .preprocessor import YoloPreprocessor
from .types import Pose


class PoseEstimator(BaseOnnxDetector):
    """ONNX whole-body pose estimator (17 COCO keypoints), architecture-agnostic.

    Architecture-specific output parsing lives in a `PosePostprocessor`
    strategy injected at construction time. Defaults to the YOLO11-pose
    postprocessor; pass `Yolo26PosePostprocessor()` for end-to-end models.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        input_size: int = 640,
        postprocessor: Optional[PosePostprocessor] = None,
    ):
        super().__init__(model_path)
        self.preprocessor = YoloPreprocessor(input_size)
        self.postprocessor = postprocessor or Yolo11PosePostprocessor()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame: np.ndarray) -> List[Pose]:
        tensor, info = self.preprocessor.preprocess(frame)
        outputs = self._run(tensor)
        return self.postprocessor.parse(
            outputs,
            info,
            self.preprocessor,
            self.conf_threshold,
            self.iou_threshold,
        )
