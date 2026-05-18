from typing import List

import cv2
import numpy as np

from .base_detector import BaseOnnxDetector
from .preprocessor import LetterboxInfo, YoloPreprocessor
from .types import BoundingBox, Keypoint, Pose


class PoseEstimator(BaseOnnxDetector):
    """YOLO11-pose: single-class (person) detector with 17 COCO keypoints.

    YOLO11-pose ONNX output shape: (1, 56, num_anchors), where the channel layout
    is [cx, cy, w, h, person_conf, kp0_x, kp0_y, kp0_conf, ..., kp16_x, kp16_y, kp16_conf].
    """

    NUM_KEYPOINTS = 17

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        input_size: int = 640,
    ):
        super().__init__(model_path)
        self.preprocessor = YoloPreprocessor(input_size)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame: np.ndarray) -> List[Pose]:
        tensor, info = self.preprocessor.preprocess(frame)
        outputs = self._run(tensor)
        return self._postprocess(outputs[0], info)

    def _postprocess(
        self, raw_output: np.ndarray, info: LetterboxInfo
    ) -> List[Pose]:
        # (1, 56, N) -> (N, 56)
        preds = raw_output[0].T

        boxes_xywh = preds[:, :4]
        confidences = preds[:, 4]
        keypoints = preds[:, 5:].reshape(-1, self.NUM_KEYPOINTS, 3)

        mask = confidences > self.conf_threshold
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        keypoints = keypoints[mask]

        cx, cy, w, h = boxes_xywh.T
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0

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

        results: List[Pose] = []
        for i in keep_idx:
            x1m, y1m, x2m, y2m = self.preprocessor.reverse_box(
                (x1[i], y1[i], x1[i] + w[i], y1[i] + h[i]), info
            )
            kp_list: List[Keypoint] = []
            for j in range(self.NUM_KEYPOINTS):
                kx, ky, kc = keypoints[i, j]
                kxm, kym = self.preprocessor.reverse_point((kx, ky), info)
                kp_list.append(Keypoint(x=kxm, y=kym, confidence=float(kc)))
            results.append(
                Pose(
                    keypoints=kp_list,
                    bbox=BoundingBox(
                        x1=x1m,
                        y1=y1m,
                        x2=x2m,
                        y2=y2m,
                        confidence=float(confidences[i]),
                    ),
                )
            )
        return results
