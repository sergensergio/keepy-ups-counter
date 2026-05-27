from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import BoundingBox, Pose


# BlazePose 33-keypoint connectivity used by MediaPipe Pose.
# Indices:
#  0 nose, 1-3 left_eye_*, 4-6 right_eye_*, 7 left_ear, 8 right_ear,
#  9 mouth_left, 10 mouth_right,
#  11 l_shoulder, 12 r_shoulder, 13 l_elbow, 14 r_elbow, 15 l_wrist, 16 r_wrist,
#  17 l_pinky, 18 r_pinky, 19 l_index, 20 r_index, 21 l_thumb, 22 r_thumb,
#  23 l_hip, 24 r_hip, 25 l_knee, 26 r_knee, 27 l_ankle, 28 r_ankle,
#  29 l_heel, 30 r_heel, 31 l_foot_index, 32 r_foot_index
MEDIAPIPE_POSE_SKELETON: List[Tuple[int, int]] = [
    # face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # left arm + hand
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    # right arm + hand
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    # left leg + foot
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # right leg + foot
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


class Visualizer:
    """Draws ball boxes, person bbox, skeleton and keypoints onto a frame."""

    def __init__(
        self,
        keypoint_conf_threshold: float = 0.3,
        ball_color: Tuple[int, int, int] = (0, 165, 255),         # orange
        foot_color: Tuple[int, int, int] = (255, 0, 255),         # magenta
        keypoint_color: Tuple[int, int, int] = (0, 255, 0),       # green
        skeleton_color: Tuple[int, int, int] = (255, 200, 0),     # cyan-ish
        person_bbox_color: Tuple[int, int, int] = (200, 200, 200),
        prediction_color: Tuple[int, int, int] = (0, 255, 255),   # yellow
    ):
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.ball_color = ball_color
        self.foot_color = foot_color
        self.keypoint_color = keypoint_color
        self.skeleton_color = skeleton_color
        self.person_bbox_color = person_bbox_color
        self.prediction_color = prediction_color

    def _draw_box(self, frame: np.ndarray, box: BoundingBox, color: Tuple[int, int, int], label: str) -> None:
        x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

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

    def draw_ball(self, frame: np.ndarray, ball: BoundingBox) -> None:
        if ball.track_id is not None:
            label = f"ball#{ball.track_id} {ball.confidence:.2f}"
        else:
            label = f"ball {ball.confidence:.2f}"
        self._draw_box(frame, ball, self.ball_color, label)

    def draw_pose(self, frame: np.ndarray, pose: Pose) -> None:
        x1, y1, x2, y2 = map(int, (pose.bbox.x1, pose.bbox.y1, pose.bbox.x2, pose.bbox.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.person_bbox_color, 1)

        for a, b in MEDIAPIPE_POSE_SKELETON:
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

    def draw_hud(
        self, frame: np.ndarray, lines: Sequence[str], y0: int = 10
    ) -> None:
        """Render text lines in the upper-left corner with a translucent backdrop.

        `y0` lets callers stack the HUD below other overlays (e.g. the
        keepy-ups counter); defaults to 10 to keep stand-alone use unchanged.
        """
        if not lines:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        pad = 6
        line_gap = 4
        x0 = 10

        sizes = [cv2.getTextSize(s, font, scale, thickness)[0] for s in lines]
        max_w = max(w for w, _ in sizes)
        line_h = max(h for _, h in sizes)
        box_w = max_w + 2 * pad
        box_h = (line_h + line_gap) * len(lines) - line_gap + 2 * pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, text in enumerate(lines):
            ty = y0 + pad + (i + 1) * line_h + i * line_gap
            cv2.putText(
                frame,
                text,
                (x0 + pad, ty),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    def draw_counter(
        self,
        frame: np.ndarray,
        count: int,
        last_contact_part: Optional[str] = None,
    ) -> int:
        """Render a prominent keepy-ups counter in the upper-left.

        Returns the y-pixel just below the counter overlay so other
        overlays (e.g. the diagnostic HUD) can stack underneath without
        overlap.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        count_text = str(count)
        label_text = "keepy-ups"
        if last_contact_part is not None:
            label_text = f"keepy-ups - {last_contact_part}"

        count_scale = 1.6
        count_thickness = 3
        label_scale = 0.5
        label_thickness = 1
        pad = 10
        gap = 6
        x0, y0 = 10, 10

        (cw, ch), _ = cv2.getTextSize(
            count_text, font, count_scale, count_thickness
        )
        (lw, lh), _ = cv2.getTextSize(
            label_text, font, label_scale, label_thickness
        )

        box_w = max(cw, lw) + 2 * pad
        box_h = ch + gap + lh + 2 * pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(
            frame,
            count_text,
            (x0 + pad, y0 + pad + ch),
            font,
            count_scale,
            (255, 255, 255),
            count_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label_text,
            (x0 + pad, y0 + pad + ch + gap + lh),
            font,
            label_scale,
            (200, 200, 200),
            label_thickness,
            cv2.LINE_AA,
        )
        return y0 + box_h + 8

    def draw_prediction(
        self, frame: np.ndarray, track_id: int, x: float, y: float
    ) -> None:
        cx, cy = int(round(x)), int(round(y))
        cv2.drawMarker(
            frame,
            (cx, cy),
            self.prediction_color,
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        label = f"K#{track_id}"
        cv2.putText(
            frame,
            label,
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.prediction_color,
            1,
            cv2.LINE_AA,
        )

    def draw(
        self,
        frame: np.ndarray,
        balls: Iterable[BoundingBox],
        poses: Iterable[Pose],
        ball_predictions: Sequence[Tuple[int, float, float]] = (),
    ) -> np.ndarray:
        for pose in poses:
            self.draw_pose(frame, pose)
        for ball in balls:
            self.draw_ball(frame, ball)
        for tid, x, y in ball_predictions:
            self.draw_prediction(frame, tid, x, y)
        return frame
