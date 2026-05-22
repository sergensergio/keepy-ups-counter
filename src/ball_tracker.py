"""Ball tracker built on centroid association + per-track Kalman filter.

Adds a persistent `track_id` to each YOLO detection so downstream logic
(keepy-up counting, motion analysis) can follow the same physical ball
across frames even when per-frame detection confidence dips.

Each track holds a 4-state constant-velocity Kalman filter `(x, y, vx, vy)`
driven by a control input `u = [g]` that injects gravity into the predict
step (`B @ u` adds `0.5 g · dt²` to `y` and `g · dt` to `vy` each frame,
with image y-axis pointing down). Horizontal acceleration is assumed zero;
impulsive kicks are absorbed via the velocity process-noise.

Per-frame association is a **two-stage** greedy match:
  1. Predict every existing track's next centroid via Kalman.
  2. Stage 1: associate detections to tracks by Euclidean distance between
     each detection's centroid and each track's *last matched YOLO detection
     centroid*, gated by `max_distance`. This is the common case — the ball
     moves only a little between adjacent frames.
  3. Stage 2: for detections still unmatched, retry against the *Kalman
     predicted centroid* of still-unmatched tracks. Picks up cases where
     the ball has moved fast or just emerged from an occlusion.
  4. Detections that survive both stages spawn new tracks (subject to
     `track_activation_threshold`). Tracks unmatched past `lost_track_buffer`
     consecutive frames are dropped.
"""


from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from .types import BoundingBox

EARTH_GRAVITY_M_S2 = 9.81


@dataclass
class _Track:
    track_id: int
    kf: cv2.KalmanFilter
    # Centroid of the most recent YOLO detection matched to this track.
    # Used as the stage-1 association anchor (cheap, robust for slow motion).
    last_detection_centroid: Tuple[float, float]
    misses: int = 0

    @property
    def predicted_centroid(self) -> np.ndarray:
        s = self.kf.statePre
        return np.array([s[0, 0], s[1, 0]], dtype=np.float32)


def _make_kalman(x: float, y: float, fps: float) -> cv2.KalmanFilter:
    # 4-state model; state is (x[px], y[px], vx[px/s], vy[px/s]).
    # dt converts between per-second and per-frame quantities.
    dt = 1.0 / fps
    kf = cv2.KalmanFilter(4, 2, 1)
    kf.transitionMatrix = np.array(
        [
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
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
    # u = [g_px_s2]; B scales to per-frame displacement and velocity change.
    # y += 0.5 * g * dt²,  vy += g * dt
    kf.controlMatrix = np.array(
        [[0.0], [0.5 * dt * dt], [0.0], [dt]], dtype=np.float32
    )
    # Velocity process noise lets the filter absorb impulsive kicks.
    kf.processNoiseCov = np.diag([5, 10, 10000, 500000]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([1, 1]).astype(np.float32)
    kf.errorCovPost = np.diag([25, 25, 160000, 160000]).astype(np.float32)
    kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
    return kf


class BallTracker:
    """Tracker that adds centroid + Kalman tracking.

    The tracker has internal state (Kalman filters, lost-track buffers),
    so call `reset()` between independent videos.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        max_distance: float = 120.0,
        gravity: float = 1800.0,
        fps: float = 30.0,  # Default. Call `set_fps()` in `run()` for correct timing.
        ball_diameter_m: Optional[float] = 0.22,
        gravity_estimation_conf: float = 0.9,
        gravity_estimation_smoothing: float = 0.1,
    ):
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.max_distance = max_distance
        # px/s², positive = downward in image coords. B converts to per-frame
        # quantities via dt=1/fps, so this value is fps-independent.
        self._gravity = gravity
        self._fps = fps
        self._control = np.array([[gravity]], dtype=np.float32)
        # Real-world ball diameter (metres) used to estimate the pixels-per-metre
        # scale from a high-confidence detection's apparent size, which in turn
        # converts earth gravity into px/s². `None` or non-positive disables.
        self.ball_diameter_m = ball_diameter_m
        self.gravity_estimation_conf = gravity_estimation_conf
        self.gravity_estimation_smoothing = gravity_estimation_smoothing
        self._tracks: List[_Track] = []
        self._next_id: int = 1
        # Snapshot of `(track_id, x, y)` Kalman predictions from the last
        # `detect()` call, exposed for visualization / debugging. Includes
        # tracks regardless of whether they got a matching detection.
        self._last_predictions: List[Tuple[int, float, float]] = []

    def set_fps(self, fps: float) -> None:
        """Set the video's FPS for correct Kalman timing. Call in `run()`."""
        self._fps = fps

    @property
    def gravity(self) -> float:
        """Current gravity estimate in px/s² (auto-updated each frame)."""
        return self._gravity

    @property
    def track_states(self) -> List[Tuple[int, float, float, float, float, int]]:
        """`(track_id, x, y, vx, vy, misses)` for each active track.

        Values come from `statePost`; `vx`/`vy` are in px/s. `misses` is the
        number of consecutive frames the track has gone without a matching
        detection — useful for picking the freshest track when no detection
        is available this frame.
        """
        out = []
        for t in self._tracks:
            s = t.kf.statePost
            out.append(
                (
                    t.track_id,
                    float(s[0, 0]),
                    float(s[1, 0]),
                    float(s[2, 0]),
                    float(s[3, 0]),
                    t.misses,
                )
            )
        return out

    def detect(
        self, raw: List[BoundingBox]
    ) -> Tuple[List[BoundingBox], List[Tuple[int, float, float]]]:
        self._maybe_update_gravity(raw)

        for t in self._tracks:
            t.kf.predict(self._control)

        # Snapshot pre-correction predictions so visualizers can see where
        # the motion model placed each track before the measurement update.
        last_predictions = [
            (
                t.track_id,
                float(t.predicted_centroid[0]),
                float(t.predicted_centroid[1]),
            )
            for t in self._tracks
        ]

        unmatched_dets: Set[int] = set(range(len(raw)))
        unmatched_tracks: Set[int] = set(range(len(self._tracks)))
        matched: List[Tuple[int, int]] = []

        if raw and self._tracks:
            det_centroids = np.array([d.center for d in raw], dtype=np.float32)

            # Stage 1: anchor on each track's last YOLO detection centroid.
            last_det_centroids = np.array(
                [t.last_detection_centroid for t in self._tracks],
                dtype=np.float32,
            )
            matched.extend(
                self._greedy_match(
                    det_centroids,
                    last_det_centroids,
                    unmatched_dets,
                    unmatched_tracks,
                )
            )

            # Stage 2: fall back to the Kalman-predicted centroid for the
            # detections / tracks that stage 1 couldn't pair up.
            if unmatched_dets and unmatched_tracks:
                pred_centroids = np.stack(
                    [t.predicted_centroid for t in self._tracks], axis=0
                )
                matched.extend(
                    self._greedy_match(
                        det_centroids,
                        pred_centroids,
                        unmatched_dets,
                        unmatched_tracks,
                    )
                )

        out: List[BoundingBox] = []

        for di, tj in matched:
            det = raw[di]
            track = self._tracks[tj]
            cx, cy = det.center
            track.kf.correct(np.array([[cx], [cy]], dtype=np.float32))
            track.last_detection_centroid = (float(cx), float(cy))
            track.misses = 0
            out.append(_with_track_id(det, track.track_id))

        for di in unmatched_dets:
            det = raw[di]
            if det.confidence < self.track_activation_threshold:
                continue
            cx, cy = det.center
            new_track = _Track(
                track_id=self._next_id,
                kf=_make_kalman(cx, cy, self._fps),
                last_detection_centroid=(float(cx), float(cy)),
            )
            self._next_id += 1
            self._tracks.append(new_track)
            out.append(_with_track_id(det, new_track.track_id))

        for tj in unmatched_tracks:
            self._tracks[tj].misses += 1
        self._tracks = [
            t for t in self._tracks if t.misses <= self.lost_track_buffer
        ]

        return out, last_predictions

    def reset(self) -> None:
        """Forget all tracks. Call between independent videos."""
        self._tracks = []
        self._next_id = 1
        self._last_predictions = []

    def _maybe_update_gravity(self, raw: List[BoundingBox]) -> None:
        """Re-estimate `self.gravity` (px/s²) from the apparent ball size.

        Picks the largest high-confidence detection in this frame, treats
        `max(width, height)` as the ball's apparent diameter, and derives
        `px_per_metre = diameter_px / ball_diameter_m`. Earth gravity
        (`9.81 m/s²`) scaled by that ratio gives the per-frame observation;
        an EMA blends it into the current estimate so per-frame bbox jitter
        doesn't whiplash the control input.
        """
        if not self.ball_diameter_m or self.ball_diameter_m <= 0:
            return
        candidates = [
            d for d in raw if d.confidence >= self.gravity_estimation_conf
        ]
        if not candidates:
            return
        best = max(candidates, key=lambda d: max(d.width, d.height))
        diameter_px = max(best.width, best.height)
        if diameter_px <= 0:
            return
        px_per_m = diameter_px / self.ball_diameter_m
        g_obs = EARTH_GRAVITY_M_S2 * px_per_m
        alpha = self.gravity_estimation_smoothing
        self._gravity = alpha * g_obs + (1.0 - alpha) * self._gravity
        self._control = np.array([[self._gravity]], dtype=np.float32)

    def _greedy_match(
        self,
        det_centroids: np.ndarray,
        track_centroids: np.ndarray,
        unmatched_dets: Set[int],
        unmatched_tracks: Set[int],
    ) -> List[Tuple[int, int]]:
        """Greedy nearest-neighbor match between the given subsets.

        Mutates `unmatched_dets` and `unmatched_tracks` by removing pairs
        that get matched. Gated by `max_distance`. `track_centroids` is
        indexed by absolute track index (same axis as `self._tracks`).
        """
        matched: List[Tuple[int, int]] = []
        if not unmatched_dets or not unmatched_tracks:
            return matched

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

        return matched


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
