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

    def shift(self, dx: float, dy: float) -> "BoundingBox":
        """Return a new box translated by (dx, dy)."""
        return BoundingBox(
            x1=self.x1 + dx,
            y1=self.y1 + dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
            confidence=self.confidence,
            class_id=self.class_id,
        )


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float

    def shift(self, dx: float, dy: float) -> "Keypoint":
        return Keypoint(x=self.x + dx, y=self.y + dy, confidence=self.confidence)


@dataclass
class Pose:
    keypoints: List[Keypoint]
    bbox: BoundingBox

    def shift(self, dx: float, dy: float) -> "Pose":
        return Pose(
            keypoints=[kp.shift(dx, dy) for kp in self.keypoints],
            bbox=self.bbox.shift(dx, dy),
        )


@dataclass
class FrameDetections:
    balls: List[BoundingBox] = field(default_factory=list)
    poses: List[Pose] = field(default_factory=list)
