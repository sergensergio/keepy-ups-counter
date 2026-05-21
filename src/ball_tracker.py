"""Ball tracker built on centroid association + per-track Kalman filter.

Composition wrapper around a `BallDetector`: same `.detect(frame)` surface,
but each returned `BoundingBox` now carries a persistent `track_id` so
downstream logic (keepy-up counting, motion analysis) can follow the same
physical ball across frames even when per-frame detection confidence dips.

Each track holds a 4-state constant-velocity Kalman filter `(x, y, vx, vy)`
driven by a control input `u = [g]` that injects gravity into the predict
step (`B @ u` adds `0.5 g` to `y` and `g` to `vy` each frame, with image
y-axis pointing down). Horizontal acceleration is assumed zero; impulsive
kicks are absorbed via the velocity process-noise. Per frame:
  1. Predict every existing track's next centroid.
  2. Greedily associate detections to tracks by Euclidean distance between
     detection centroid and predicted centroid (gated by `max_distance`).
  3. Correct matched tracks with the detection centroid; start new tracks
     for unmatched, high-confidence detections; age unmatched tracks and
     drop those past `lost_track_buffer`.
"""


from dataclasses import dataclass
from typing import List, Set, Tuple

import cv2
import numpy as np

from .ball_detector import BallDetector
from .types import BoundingBox


@dataclass
class _Track:
    track_id: int
    kf: cv2.KalmanFilter
    misses: int = 0

    @property
    def predicted_centroid(self) -> np.ndarray:
        s = self.kf.statePre
        return np.array([s[0, 0], s[1, 0]], dtype=np.float32)


def _make_kalman(x: float, y: float) -> cv2.KalmanFilter:
    # 4-state constant-velocity model; gravity injected via control input.
    kf = cv2.KalmanFilter(4, 2, 1)
    kf.transitionMatrix = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    # B @ u with u = [g] adds 0.5 g to y and g to vy each frame.
    kf.controlMatrix = np.array(
        [[0.0], [0.5], [0.0], [1.0]], dtype=np.float32
    )
    # Velocity process noise lets the filter absorb impulsive kicks.
    kf.processNoiseCov = np.diag([0.5, 0.5, 10.0, 10.0]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([5.0, 5.0]).astype(np.float32)
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
    kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
    return kf


class BallTracker:
    """Drop-in replacement for `BallDetector` that adds centroid + Kalman tracking.

    The tracker has internal state (Kalman filters, lost-track buffers),
    so call `reset()` between independent videos.
    """

    def __init__(
        self,
        detector: BallDetector,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        max_distance: float = 120.0,
        gravity: float = 1.5,
    ):
        self.detector = detector
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.max_distance = max_distance
        # Pixels/frame^2, positive = downward in image coords. Scene-dependent
        # (ball-to-camera distance and frame rate); tune per video setup.
        self.gravity = gravity
        self._control = np.array([[gravity]], dtype=np.float32)
        self._tracks: List[_Track] = []
        self._next_id: int = 1

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        raw = self.detector.detect(frame)

        for t in self._tracks:
            t.kf.predict(self._control)

        matched, unmatched_dets, unmatched_tracks = self._associate(raw)

        out: List[BoundingBox] = []

        for di, tj in matched:
            det = raw[di]
            track = self._tracks[tj]
            cx, cy = det.center
            track.kf.correct(np.array([[cx], [cy]], dtype=np.float32))
            track.misses = 0
            out.append(_with_track_id(det, track.track_id))

        for di in unmatched_dets:
            det = raw[di]
            if det.confidence < self.track_activation_threshold:
                continue
            cx, cy = det.center
            new_track = _Track(track_id=self._next_id, kf=_make_kalman(cx, cy))
            self._next_id += 1
            self._tracks.append(new_track)
            out.append(_with_track_id(det, new_track.track_id))

        for tj in unmatched_tracks:
            self._tracks[tj].misses += 1
        self._tracks = [
            t for t in self._tracks if t.misses <= self.lost_track_buffer
        ]

        return out

    def reset(self) -> None:
        """Forget all tracks. Call between independent videos."""
        self._tracks = []
        self._next_id = 1

    def _associate(
        self, dets: List[BoundingBox]
    ) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
        """Greedy nearest-neighbor association gated by `max_distance`."""
        matched: List[Tuple[int, int]] = []
        unmatched_dets: Set[int] = set(range(len(dets)))
        unmatched_tracks: Set[int] = set(range(len(self._tracks)))

        if not dets or not self._tracks:
            return matched, unmatched_dets, unmatched_tracks

        det_centroids = np.array([d.center for d in dets], dtype=np.float32)
        track_centroids = np.stack(
            [t.predicted_centroid for t in self._tracks], axis=0
        )
        dists = np.linalg.norm(
            det_centroids[:, None, :] - track_centroids[None, :, :], axis=2
        )

        while unmatched_dets and unmatched_tracks:
            di_arr = np.array(sorted(unmatched_dets))
            ti_arr = np.array(sorted(unmatched_tracks))
            sub = dists[np.ix_(di_arr, ti_arr)]
            flat = int(np.argmin(sub))
            i, j = np.unravel_index(flat, sub.shape)
            if sub[i, j] > self.max_distance:
                break
            di, tj = int(di_arr[i]), int(ti_arr[j])
            matched.append((di, tj))
            unmatched_dets.discard(di)
            unmatched_tracks.discard(tj)

        return matched, unmatched_dets, unmatched_tracks


def _with_track_id(b: BoundingBox, tid: int) -> BoundingBox:
    return BoundingBox(
        x1=b.x1,
        y1=b.y1,
        x2=b.x2,
        y2=b.y2,
        confidence=b.confidence,
        class_id=b.class_id,
        track_id=tid,
    )
