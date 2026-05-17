import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .ball_detector import BallDetector
from .pose_estimator import PoseEstimator
from .types import FrameDetections
from .video_reader import VideoReader
from .visualizer import Visualizer


class KeepyUpsPipeline:
    """Orchestrates the per-frame pipeline:
       read -> ball detect -> pose detect -> visualize -> write/display.

    Counting logic is intentionally out of scope for v1; this pipeline only
    surfaces the structured detections needed by a future counter component.
    """

    def __init__(
        self,
        ball_detector: BallDetector,
        pose_estimator: PoseEstimator,
        visualizer: Visualizer,
    ):
        self.ball_detector = ball_detector
        self.pose_estimator = pose_estimator
        self.visualizer = visualizer

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, FrameDetections]:
        balls = self.ball_detector.detect(frame)
        poses = self.pose_estimator.detect(frame)
        annotated = self.visualizer.draw(frame.copy(), balls, poses)
        return annotated, FrameDetections(balls=balls, poses=poses)

    def run(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        log_every: int = 30,
    ) -> None:
        with VideoReader(video_path) as reader:
            writer: Optional[cv2.VideoWriter] = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    30,
                    (reader.width, reader.height),
                )
                if not writer.isOpened():
                    raise IOError(f"Cannot open output video: {output_path}")

            try:
                t0 = time.time()
                for idx, frame in enumerate(reader):
                    annotated, _ = self.process_frame(frame)

                    if writer is not None:
                        writer.write(annotated)

                    if display:
                        cv2.imshow("keepy-ups", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    if log_every and (idx + 1) % log_every == 0:
                        elapsed = time.time() - t0
                        fps = (idx + 1) / elapsed if elapsed > 0 else 0.0
                        print(f"[{idx + 1} frames] avg {fps:.1f} fps")
            finally:
                if writer is not None:
                    writer.release()
                if display:
                    cv2.destroyAllWindows()
