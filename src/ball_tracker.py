"""Ball tracker built on centroid association + per-track Kalman filter.

Adds a persistent `track_id` to each YOLO detection so downstream logic
(keepy-up counting, motion analysis) can follow the same physical ball
across frames even when per-frame detection confidence dips.

The tracker runs entirely in **metres**. The Kalman state is `(x[m], y[m],
vx[m/s], vy[m/s])` and every covariance / threshold is expressed in
metric units, so the tuning is independent of how far the player is from
the camera. The one piece of pixel↔metre conversion is `px_per_m`,
estimated from the apparent ball diameter (real-world diameter given by
`ball_diameter_m`). Gravity becomes a constant 9.81 m/s² control input;
only the scale is observed each frame.

Per-frame flow:
  1. Update `px_per_m` from the largest high-confidence detection. If the
     scale shifts, rescale every track's stored metric position so that
     the same physical pixel coordinate maps to it (velocities are in
     m/s and stay physically invariant).
  2. Predict each Kalman; the constant gravity control adds `0.5 g · dt²`
     to `y` and `g · dt` to `vy`.
  3. Convert detection centroids from pixels to metres.
  4. **Two-stage greedy match** against existing tracks (in metric space):
     - Stage 1: nearest neighbour to each track's *last matched YOLO
       detection centroid* (the common case — the ball moved only a
       little between adjacent frames). Gated by `max_distance` (metres).
     - Stage 2: for detections still unmatched, retry against the
       *Kalman predicted centroid* of still-unmatched tracks. Picks up
       cases where the ball has moved fast or just emerged from an
       occlusion.
  5. Detections that survive both stages spawn new tracks (subject to
     `track_activation_threshold`). Tracks unmatched past
     `lost_track_buffer` consecutive frames are dropped.

Predictions returned to the visualizer are converted back to pixels so
downstream rendering doesn't need to know about the metric internals.
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
    # Centroid of the most recent YOLO detection matched to this track,
    # stored in metres so it lines up with the Kalman state for stage-1
    # association.
    last_detection_centroid_m: Tuple[float, float]
    misses: int = 0

    @property
    def predicted_centroid_m(self) -> np.ndarray:
        s = self.kf.statePre
        return np.array([s[0, 0], s[1, 0]], dtype=np.float32)


def _make_kalman(x_m: float, y_m: float, fps: float) -> cv2.KalmanFilter:
    # 4-state constant-velocity model in metric units. dt converts between
    # per-second and per-frame quantities.
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
    # u = [g] = 9.81 m/s²; B scales it to per-frame metric displacement /
    # velocity change:  y += 0.5 · g · dt²,  vy += g · dt   (image y-axis
    # points down, so positive g pulls the ball downward).
    kf.controlMatrix = np.array(
        [[0.0], [0.5 * dt * dt], [0.0], [dt]], dtype=np.float32
    )
    # Covariances in metric units. Position jitter is small (sub-centimetre
    # detection-centroid noise); velocity has wide variance — especially on
    # vy — so the filter can absorb the impulsive kicks the gravity model
    # doesn't account for.
    kf.processNoiseCov = np.diag(
        [0.01, 0.01, 4.0, 25.0]
    ).astype(np.float32)
    kf.measurementNoiseCov = np.diag([0.01, 0.01]).astype(np.float32)
    kf.errorCovPost = np.diag(
        [0.1, 0.1, 25.0, 50.0]
    ).astype(np.float32)
    kf.statePost = np.array([[x_m], [y_m], [0.0], [0.0]], dtype=np.float32)
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
        max_distance: float = 0.6,
        px_per_m: float = 200.0,
        fps: float = 30.0,  # Default. Call `set_fps()` in `run()` for correct timing.
        ball_diameter_m: Optional[float] = 0.22,
        scale_estimation_conf: float = 0.9,
        scale_estimation_smoothing: float = 0.1,
    ):
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        # Max distance in metres between a detection centroid and a track's
        # association anchor (last detection / Kalman prediction). Distance-
        # independent because matching happens in metric space.
        self.max_distance = max_distance
        # Pixel-per-metre conversion factor estimated from the apparent ball
        # diameter. The Kalman runs entirely in metres; this scale is the
        # only thing that depends on the player's distance from the camera.
        self._px_per_m = px_per_m
        self._fps = fps
        # Gravity is a constant in the metric world (9.81 m/s² downward).
        self._control = np.array([[EARTH_GRAVITY_M_S2]], dtype=np.float32)
        # Real-world ball diameter (metres). Drives the px-per-metre EMA
        # from each high-confidence detection's apparent diameter. `None`
        # or non-positive disables auto-estimation (scale stays fixed at
        # the initial value).
        self.ball_diameter_m = ball_diameter_m
        self.scale_estimation_conf = scale_estimation_conf
        self.scale_estimation_smoothing = scale_estimation_smoothing
        self._tracks: List[_Track] = []
        self._next_id: int = 1

    def set_fps(self, fps: float) -> None:
        """Set the video's FPS for correct Kalman timing. Call in `run()`."""
        self._fps = fps

    @property
    def px_per_m(self) -> float:
        """Current pixels-per-metre estimate (auto-updated each frame)."""
        return self._px_per_m

    @property
    def track_states(self) -> List[Tuple[int, float, float, float, float, int]]:
        """`(track_id, x_m, y_m, vx_m_s, vy_m_s, misses)` for each active track.

        Values come from `statePost`. `misses` is the number of consecutive
        frames the track has gone without a matching detection — useful for
        picking the freshest track when no detection is available this frame.
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
        self._maybe_update_scale(raw)

        for t in self._tracks:
            t.kf.predict(self._control)

        # Snapshot pre-correction predictions in *pixels* so visualizers can
        # render where the motion model placed each track before the
        # measurement update, without needing to know about the metric state.
        last_predictions = [
            (
                t.track_id,
                float(t.predicted_centroid_m[0]) * self._px_per_m,
                float(t.predicted_centroid_m[1]) * self._px_per_m,
            )
            for t in self._tracks
        ]

        # Convert detection centroids px→m once; matching and the Kalman
        # correction both run in metric space.
        if raw:
            det_centroids_m = (
                np.array([d.center for d in raw], dtype=np.float32)
                / self._px_per_m
            )
        else:
            det_centroids_m = np.empty((0, 2), dtype=np.float32)

        unmatched_dets: Set[int] = set(range(len(raw)))
        unmatched_tracks: Set[int] = set(range(len(self._tracks)))
        matched: List[Tuple[int, int]] = []

        if raw and self._tracks:
            # Stage 1: anchor on each track's last YOLO detection centroid.
            last_det_centroids_m = np.array(
                [t.last_detection_centroid_m for t in self._tracks],
                dtype=np.float32,
            )
            matched.extend(
                self._greedy_match(
                    det_centroids_m,
                    last_det_centroids_m,
                    unmatched_dets,
                    unmatched_tracks,
                )
            )

            # Stage 2: fall back to the Kalman-predicted centroid for the
            # detections / tracks that stage 1 couldn't pair up.
            if unmatched_dets and unmatched_tracks:
                pred_centroids_m = np.stack(
                    [t.predicted_centroid_m for t in self._tracks], axis=0
                )
                matched.extend(
                    self._greedy_match(
                        det_centroids_m,
                        pred_centroids_m,
                        unmatched_dets,
                        unmatched_tracks,
                    )
                )

        out: List[BoundingBox] = []

        for di, tj in matched:
            det = raw[di]
            track = self._tracks[tj]
            cx_m = float(det_centroids_m[di, 0])
            cy_m = float(det_centroids_m[di, 1])
            track.kf.correct(np.array([[cx_m], [cy_m]], dtype=np.float32))
            track.last_detection_centroid_m = (cx_m, cy_m)
            track.misses = 0
            out.append(_with_track_id(det, track.track_id))

        for di in unmatched_dets:
            det = raw[di]
            if det.confidence < self.track_activation_threshold:
                continue
            cx_m = float(det_centroids_m[di, 0])
            cy_m = float(det_centroids_m[di, 1])
            new_track = _Track(
                track_id=self._next_id,
                kf=_make_kalman(cx_m, cy_m, self._fps),
                last_detection_centroid_m=(cx_m, cy_m),
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

    def _maybe_update_scale(self, raw: List[BoundingBox]) -> None:
        """Re-estimate `self._px_per_m` from the apparent ball size.

        Picks the largest high-confidence detection in this frame, treats
        `max(width, height)` as the ball's apparent diameter, and derives
        `px_per_m = diameter_px / ball_diameter_m`. An EMA blends it into
        the current estimate so per-frame bbox jitter doesn't whiplash
        downstream pixel ↔ metre conversions.

        When the scale shifts, every track's metric position is rescaled
        so it continues to represent the same physical pixel — otherwise
        the next correction step would see a spurious position innovation
        and corrupt the velocity estimate.
        """
        if not self.ball_diameter_m or self.ball_diameter_m <= 0:
            return
        candidates = [
            d for d in raw if d.confidence >= self.scale_estimation_conf
        ]
        if not candidates:
            return
        best = max(candidates, key=lambda d: max(d.width, d.height))
        diameter_px = max(best.width, best.height)
        if diameter_px <= 0:
            return
        px_per_m_obs = diameter_px / self.ball_diameter_m
        alpha = self.scale_estimation_smoothing
        old_px_per_m = self._px_per_m
        self._px_per_m = alpha * px_per_m_obs + (1.0 - alpha) * self._px_per_m
        ratio = old_px_per_m / self._px_per_m
        if abs(ratio - 1.0) <= 1e-6:
            return
        for t in self._tracks:
            t.kf.statePost[0, 0] *= ratio
            t.kf.statePost[1, 0] *= ratio
            lx, ly = t.last_detection_centroid_m
            t.last_detection_centroid_m = (lx * ratio, ly * ratio)

    def _greedy_match(
        self,
        det_centroids: np.ndarray,
        track_centroids: np.ndarray,
        unmatched_dets: Set[int],
        unmatched_tracks: Set[int],
    ) -> List[Tuple[int, int]]:
        """Greedy nearest-neighbor match between the given subsets.

        Mutates `unmatched_dets` and `unmatched_tracks` by removing pairs
        that get matched. Gated by `max_distance` (metres). `track_centroids`
        is indexed by absolute track index (same axis as `self._tracks`).
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
