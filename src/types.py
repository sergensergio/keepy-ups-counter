from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    # Populated by tracking layers (e.g. centroid + Kalman BallTracker) and
    # `None` for raw per-frame detections.
    track_id: Optional[int] = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class Pose:
    keypoints: List[Keypoint]
    bbox: BoundingBox


@dataclass
class FrameDetections:
    balls: List[BoundingBox] = field(default_factory=list)
    poses: List[Pose] = field(default_factory=list)
    # `(track_id, x, y)` Kalman-predicted centroids; empty when the ball
    # component isn't a tracker.
    ball_predictions: List[Tuple[int, float, float]] = field(default_factory=list)
