"""MediaPipe BlazePose estimator (33 keypoints).

MediaPipe owns its own pre/post-processing (RGB conversion, internal resize,
single-person ROI tracking), so this class is a thin adapter that:
  * converts BGR → RGB before inference,
  * scales normalized landmark coords (0..1) back to pixel space,
  * derives a person bbox from the keypoints,
  * wraps the result in the project's `Pose` type so downstream code
    (pipeline, visualizer, future counter) stays model-agnostic.
"""

from typing import List

import cv2
import mediapipe as mp
import numpy as np

from .types import BoundingBox, Keypoint, Pose


# Indices follow the BlazePose 33-keypoint layout. See:
# https://google.github.io/mediapipe/solutions/pose#pose-landmark-model-blazepose-ghum-3d
MEDIAPIPE_NUM_KEYPOINTS = 33


class MediaPipePoseEstimator:
    """Single-person MediaPipe Pose wrapper.

    Exposes the same `detect(frame) -> List[Pose]` contract as
    `PoseEstimator`, so it's a drop-in replacement in the pipeline.
    """

    NUM_KEYPOINTS = MEDIAPIPE_NUM_KEYPOINTS

    def __init__(
        self,
        conf_threshold: float = 0.25,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=conf_threshold,
            min_tracking_confidence=conf_threshold,
        )

    def detect(self, frame: np.ndarray) -> List[Pose]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # MediaPipe mutates the writeable flag for a perf hint; honor it.
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        if results.pose_landmarks is None:
            return []

        keypoints: List[Keypoint] = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append(
                Keypoint(
                    x=float(landmark.x) * w,
                    y=float(landmark.y) * h,
                    confidence=float(landmark.visibility),
                )
            )

        visible = [
            (kp.x, kp.y) for kp in keypoints if kp.confidence > self.conf_threshold
        ]
        if visible:
            xs = [p[0] for p in visible]
            ys = [p[1] for p in visible]
            bbox = BoundingBox(
                x1=float(min(xs)),
                y1=float(min(ys)),
                x2=float(max(xs)),
                y2=float(max(ys)),
                confidence=1.0,
            )
        else:
            bbox = BoundingBox(x1=0.0, y1=0.0, x2=0.0, y2=0.0, confidence=0.0)

        return [Pose(keypoints=keypoints, bbox=bbox)]

    def close(self) -> None:
        self._pose.close()
