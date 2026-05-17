from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .types import BoundingBox, Pose

# COCO 17-keypoint connectivity used by YOLOv8-pose.
# Indices:
#  0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
#  5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist,
#  11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
COCO_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # face
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]


class Visualizer:
    """Draws ball boxes, person bbox, skeleton and keypoints onto a frame."""

    def __init__(
        self,
        keypoint_conf_threshold: float = 0.3,
        ball_color: Tuple[int, int, int] = (0, 165, 255),         # orange
        keypoint_color: Tuple[int, int, int] = (0, 255, 0),       # green
        skeleton_color: Tuple[int, int, int] = (255, 200, 0),     # cyan-ish
        person_bbox_color: Tuple[int, int, int] = (200, 200, 200),
    ):
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.ball_color = ball_color
        self.keypoint_color = keypoint_color
        self.skeleton_color = skeleton_color
        self.person_bbox_color = person_bbox_color

    def draw_ball(self, frame: np.ndarray, ball: BoundingBox) -> None:
        x1, y1, x2, y2 = map(int, (ball.x1, ball.y1, ball.x2, ball.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.ball_color, 2)

        label = f"ball {ball.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), self.ball_color, -1
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def draw_pose(self, frame: np.ndarray, pose: Pose) -> None:
        x1, y1, x2, y2 = map(int, (pose.bbox.x1, pose.bbox.y1, pose.bbox.x2, pose.bbox.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.person_bbox_color, 1)

        for a, b in COCO_SKELETON:
            ka, kb = pose.keypoints[a], pose.keypoints[b]
            if (
                ka.confidence < self.keypoint_conf_threshold
                or kb.confidence < self.keypoint_conf_threshold
            ):
                continue
            cv2.line(
                frame,
                (int(ka.x), int(ka.y)),
                (int(kb.x), int(kb.y)),
                self.skeleton_color,
                2,
                cv2.LINE_AA,
            )

        for kp in pose.keypoints:
            if kp.confidence < self.keypoint_conf_threshold:
                continue
            cv2.circle(
                frame, (int(kp.x), int(kp.y)), 3, self.keypoint_color, -1, cv2.LINE_AA
            )

    def draw(
        self,
        frame: np.ndarray,
        balls: Iterable[BoundingBox],
        poses: Iterable[Pose],
    ) -> np.ndarray:
        for pose in poses:
            self.draw_pose(frame, pose)
        for ball in balls:
            self.draw_ball(frame, ball)
        return frame
