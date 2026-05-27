import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .ball_detector import BallDetector
from .ball_tracker import BallTracker
from .keepy_ups_counter import KeepyUpsCounter
from .mediapipe_pose_estimator import MediaPipePoseEstimator as PoseEstimator
from .signal_visualizer import ScrollingSignalVisualizer, SignalSpec
from .types import FrameDetections
from .video_reader import VideoReader
from .visualizer import Visualizer

# BlazePose foot-index landmarks (the toes).
_LEFT_TOE_IDX = 31
_RIGHT_TOE_IDX = 32
_TOE_CONF_THRESHOLD = 0.3


class KeepyUpsPipeline:
    """Orchestrates the per-frame pipeline:
       read -> ball detect -> pose detect -> track -> count -> visualize.
    """

    def __init__(
        self,
        ball_detector: BallDetector,
        pose_estimator: PoseEstimator,
        visualizer: Visualizer,
        ball_tracker: Optional[BallTracker] = None,
        counter: Optional[KeepyUpsCounter] = None,
    ):
        self.ball_detector = ball_detector
        self.pose_estimator = pose_estimator
        self.visualizer = visualizer
        self.ball_tracker = ball_tracker
        # The counter needs the tracker's metric velocity to detect flicks,
        # so it's silently ignored when no tracker is configured.
        self.counter = counter if ball_tracker is not None else None

    def process_frame(
        self, frame: np.ndarray, frame_idx: int = 0
    ) -> Tuple[np.ndarray, FrameDetections]:
        balls = self.ball_detector.detect(frame)
        poses = self.pose_estimator.detect(frame)
        predictions = []
        if self.ball_tracker:
            balls, predictions = self.ball_tracker.detect(balls)
        if self.counter is not None:
            self._step_counter(poses)
        annotated = self.visualizer.draw(
            frame.copy(), balls, poses, ball_predictions=predictions
        )
        hud_y0 = 10
        if self.counter is not None:
            hud_y0 = self.visualizer.draw_counter(
                annotated, self.counter.count, self.counter.last_contact_part
            )
        if self.ball_tracker is not None:
            self.visualizer.draw_hud(
                annotated,
                self._build_hud_lines(frame_idx, balls, predictions),
                y0=hud_y0,
            )
        return annotated, FrameDetections(
            balls=balls, poses=poses, ball_predictions=list(predictions)
        )

    def _step_counter(self, poses) -> None:
        """Feed the counter the freshest metric ball state for this frame."""
        assert self.counter is not None and self.ball_tracker is not None
        state = self._freshest_track_state_m()
        if state is None:
            self.counter.update(None, None, poses, self.ball_tracker.px_per_m)
            return
        x_m, y_m, _vx, vy = state
        self.counter.update(
            (x_m, y_m), vy, poses, self.ball_tracker.px_per_m
        )

    def _freshest_track_state_m(
        self,
    ) -> Optional[Tuple[float, float, float, float]]:
        """`(x_m, y_m, vx_m_s, vy_m_s)` of the freshest active track, or `None`.

        "Freshest" = lowest `misses`, ties broken by lowest `track_id`. The
        same selection is used for the signal panel so the displayed Kalman
        trace and the counter agree on which track represents "the ball".
        """
        if self.ball_tracker is None:
            return None
        states = self.ball_tracker.track_states
        if not states:
            return None
        best = min(states, key=lambda s: (s[5], s[0]))
        return best[1], best[2], best[3], best[4]

    def _build_hud_lines(
        self,
        frame_idx: int,
        balls,
        predictions,
    ):
        """Diagnostic lines drawn in the upper-left of the output frame.

        Reports global tracker state (frame number, fps, px-per-metre scale)
        and per-track Kalman info: for the largest detected ball this frame
        plus its YOLO centroid, or — when no detection exists — for the
        freshest active track (lowest `misses`) so the operator can still
        follow the motion-model trajectory during occlusions.
        """
        tracker = self.ball_tracker
        assert tracker is not None  # caller checks
        px_per_m = tracker.px_per_m
        lines = [
            f"frame {frame_idx}    fps {tracker._fps:.0f}",
            f"px/m {px_per_m:.0f}",
        ]
        if self.counter is not None:
            lines.append(
                f"phase {self.counter.phase.name}  cd {self.counter.cooldown}"
            )
        preds_by_id = {pid: (px, py) for pid, px, py in predictions}
        # Tracker emits state in metres; HUD shows positions in pixels (so
        # they line up with the frame) and velocities in m/s.
        states_by_id = {
            sid: (sx * px_per_m, sy * px_per_m, svx, svy, miss)
            for sid, sx, sy, svx, svy, miss in tracker.track_states
        }

        if balls:
            largest = max(balls, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
            diameter = max(largest.x2 - largest.x1, largest.y2 - largest.y1)
            cx, cy = largest.center
            tid = largest.track_id
            tid_str = f"track#{tid}" if tid is not None else "track#-"
            lines.append(
                f"det {tid_str} conf {largest.confidence:.2f} d {diameter:.0f}px"
            )
            lines.append(f"det pos  ({cx:7.1f}, {cy:7.1f})")
            self._append_kalman_lines(lines, tid, preds_by_id, states_by_id)
            return lines

        # No detection — fall back to the freshest active track (smallest
        # `misses`, tie-broken by lowest track_id).
        if states_by_id:
            tid = min(states_by_id, key=lambda t: (states_by_id[t][4], t))
            miss = states_by_id[tid][4]
            lines.append(f"no det — track#{tid} (missed {miss}f)")
            self._append_kalman_lines(lines, tid, preds_by_id, states_by_id)
        else:
            lines.append("no ball detection")
        return lines

    @staticmethod
    def _append_kalman_lines(lines, tid, preds_by_id, states_by_id) -> None:
        if tid is None:
            return
        if tid in preds_by_id:
            px, py = preds_by_id[tid]
            lines.append(f"kpred    ({px:7.1f}, {py:7.1f})")
        if tid in states_by_id:
            sx, sy, svx, svy, _ = states_by_id[tid]
            lines.append(f"kcorr    ({sx:7.1f}, {sy:7.1f})")
            lines.append(f"kvel     ({svx:6.2f}, {svy:6.2f}) m/s")

    def run(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        signal_panel_width: Optional[int] = None,
        display: bool = False,
        log_every: int = 30,
    ) -> None:
        """Process the video. When writing or displaying, the signal panel
        is hconcat'd to the right of the annotated frame at matching height.

        `signal_panel_width` controls the panel width in pixels; defaults
        to the input video's width (1:1 split, easy to read on portrait
        sources).
        """
        with VideoReader(video_path) as reader:
            if self.ball_tracker:
                self.ball_tracker.set_fps(reader.fps)

            panel_w = signal_panel_width or reader.width
            signal_visualizer: Optional[ScrollingSignalVisualizer] = None
            if output_path or display:
                signal_visualizer = self._make_signal_visualizer(
                    panel_w, reader.height, reader.width
                )

            writer: Optional[cv2.VideoWriter] = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                out_w = reader.width + (panel_w if signal_visualizer else 0)
                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    reader.fps,
                    (out_w, reader.height),
                )
                if not writer.isOpened():
                    raise IOError(f"Cannot open output video: {output_path}")

            try:
                t0 = time.time()
                for idx, frame in enumerate(reader):
                    annotated, det = self.process_frame(frame, frame_idx=idx)

                    if signal_visualizer is not None:
                        signal_visualizer.push(self._collect_signal_samples(det))
                        composed = np.hstack([annotated, signal_visualizer.render()])
                    else:
                        composed = annotated

                    if writer is not None:
                        writer.write(composed)

                    if display:
                        cv2.imshow("keepy-ups", composed)
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

    def _make_signal_visualizer(
        self, panel_w: int, panel_h: int, source_frame_w: int
    ) -> ScrollingSignalVisualizer:
        """8-channel monitor: Kalman (x, y, vx, vy) + L/R toe (x, y).

        Position channels are displayed in pixels (pinned to the source
        frame's extents so they read consistently across the video — the
        tracker's metric state is converted in `_collect_signal_samples`).
        Velocity channels are in m/s and auto-scale.
        """
        signals = [
            SignalSpec("kx", "ball x", (80, 255, 80),
                       y_range=(0.0, float(source_frame_w)), unit="px"),
            SignalSpec("ky", "ball y", (80, 255, 80),
                       y_range=(0.0, float(panel_h)), unit="px"),
            SignalSpec("kvx", "ball vx", (80, 220, 255), unit="m/s"),
            SignalSpec("kvy", "ball vy", (80, 220, 255), unit="m/s"),
            SignalSpec("lx", "L toe x", (255, 200, 80),
                       y_range=(0.0, float(source_frame_w)), unit="px"),
            SignalSpec("ly", "L toe y", (255, 200, 80),
                       y_range=(0.0, float(panel_h)), unit="px"),
            SignalSpec("rx", "R toe x", (220, 120, 255),
                       y_range=(0.0, float(source_frame_w)), unit="px"),
            SignalSpec("ry", "R toe y", (220, 120, 255),
                       y_range=(0.0, float(panel_h)), unit="px"),
        ]
        return ScrollingSignalVisualizer(
            width=panel_w,
            height=panel_h,
            signals=signals,
            window_size=240,
        )

    def _collect_signal_samples(
        self, det: FrameDetections
    ) -> Dict[str, Optional[float]]:
        """Extract one sample per channel for the current frame.

        Ball channels come from the freshest active Kalman track (lowest
        `misses`, tie-broken by lowest track_id) so the trace stays
        continuous through brief occlusions. Toe channels come from the
        first pose's foot-index landmarks; gated by visibility so dropouts
        render as gaps rather than misleading straight lines.
        """
        samples: Dict[str, Optional[float]] = {
            "kx": None, "ky": None, "kvx": None, "kvy": None,
            "lx": None, "ly": None, "rx": None, "ry": None,
        }
        if self.ball_tracker is not None:
            state = self._freshest_track_state_m()
            if state is not None:
                x_m, y_m, vx, vy = state
                # Tracker emits state in metres; convert position to pixels
                # so it aligns with the frame extents pinned on the panel.
                # Velocity stays in m/s.
                px_per_m = self.ball_tracker.px_per_m
                samples["kx"] = x_m * px_per_m
                samples["ky"] = y_m * px_per_m
                samples["kvx"] = vx
                samples["kvy"] = vy
        if det.poses:
            kps = det.poses[0].keypoints
            if len(kps) > _RIGHT_TOE_IDX:
                lt = kps[_LEFT_TOE_IDX]
                rt = kps[_RIGHT_TOE_IDX]
                if lt.confidence >= _TOE_CONF_THRESHOLD:
                    samples["lx"] = lt.x
                    samples["ly"] = lt.y
                if rt.confidence >= _TOE_CONF_THRESHOLD:
                    samples["rx"] = rt.x
                    samples["ry"] = rt.y
        return samples
