from dataclasses import dataclass, field
from typing import List


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0

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
