"""CLI entry point for the keepy-ups tool (v1: detect + visualize)."""


import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.ball_detector import BallDetector
from src.ball_tracker import BallTracker
from src.keepy_ups_counter import KeepyUpsCounter
from src.mediapipe_pose_estimator import MediaPipePoseEstimator
from src.pipeline import KeepyUpsPipeline
from src.postprocessors import (
    SUPPORTED_ARCHITECTURES,
    make_detection_postprocessor,
)
from src.visualizer import Visualizer

POSE_MODEL_TYPES = ("mediapipe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Keepy-ups counter (v1): read a portrait video, detect the ball "
        "and the player's pose, render annotated output.",
    )
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument(
        "--output",
        default=None,
        help="Optional directory to write annotated mp4 into "
        "(file is named annotated_<datetime>.mp4)",
    )
    p.add_argument(
        "--ball-model",
        default="models/yolo11n.onnx",
        help="Path to YOLO11 detection ONNX model",
    )
    p.add_argument(
        "--mediapipe-complexity",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="MediaPipe Pose model_complexity: 0=lite, 1=full, 2=heavy. "
        "Only used when --pose-model-type=mediapipe.",
    )
    p.add_argument("--ball-conf", type=float, default=0.25)
    p.add_argument("--pose-detection-conf", type=float, default=0.7)
    p.add_argument("--pose-tracking-conf", type=float, default=0.5)
    p.add_argument("--pose-keypoint-conf", type=float, default=0.5)
    p.add_argument(
        "--ball-arch",
        choices=SUPPORTED_ARCHITECTURES,
        default="yolo11",
        help="Architecture family of the ball-detection ONNX. "
        "'yolo11' = transposed output + NMS. "
        "'yolo26' = end-to-end / NMS-free output.",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Square model input size (must match the exported ONNX)",
    )
    p.add_argument(
        "--no-ball-tracking",
        action="store_true",
        help="Disable centroid+Kalman tracking on the ball "
        "(use raw per-frame detections only).",
    )
    p.add_argument(
        "--track-activation-threshold",
        type=float,
        default=0.25,
        help="Tracker: detection confidence above which a new track is started.",
    )
    p.add_argument(
        "--lost-track-buffer",
        type=int,
        default=30,
        help="Tracker: frames a track is kept alive without a hit before being dropped.",
    )
    p.add_argument(
        "--max-distance",
        type=float,
        default=0.6,
        help="Tracker: max distance in metres between a detection centroid and a "
        "track's association anchor (last detection or Kalman prediction). "
        "Independent of camera distance — matching runs in metric space.",
    )
    p.add_argument(
        "--init-px-per-m",
        type=float,
        default=200.0,
        help="Tracker: initial pixels-per-metre scale, used before the first "
        "high-confidence detection refines it. Gravity is constant 9.81 m/s² in "
        "metric space; this scale is the only thing depending on camera distance.",
    )
    p.add_argument(
        "--ball-diameter-m",
        type=float,
        default=0.22,
        help="Tracker: real-world ball diameter in metres, used to estimate "
        "pixels-per-metre from high-confidence detections (diameter_px / "
        "ball_diameter_m). Set to 0 to disable auto-estimation.",
    )
    p.add_argument(
        "--scale-estimation-conf",
        type=float,
        default=0.9,
        help="Tracker: min detection confidence used as a sample for pixels-per-"
        "metre auto-estimation (avoids learning from noisy low-conf bboxes).",
    )
    p.add_argument(
        "--no-counter",
        action="store_true",
        help="Disable the keepy-ups counter overlay (counter requires the tracker).",
    )
    p.add_argument(
        "--contact-distance-m",
        type=float,
        default=0.3,
        help="Counter: max distance in metres between the ball centroid and a valid "
        "body-part keypoint (head/shoulders/knees/feet) at the moment of a vy "
        "reversal for the flick to count. Arms are excluded.",
    )
    p.add_argument(
        "--display",
        action="store_true",
        help="Show annotated frames in a window (press q to quit)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.ball_model).is_file():
        raise FileNotFoundError(
            f"Model not found: {args.ball_model}. Run `python download_models.py` first."
        )

    ball_detector = BallDetector(
        args.ball_model,
        conf_threshold=args.ball_conf,
        input_size=args.input_size,
        postprocessor=make_detection_postprocessor(args.ball_arch),
    )
    pose_estimator = MediaPipePoseEstimator(
        min_detection_confidence=args.pose_detection_conf,
        min_tracking_confidence=args.pose_tracking_conf,
        kp_conf_threshold=args.pose_keypoint_conf,
        model_complexity=args.mediapipe_complexity,
    )

    if args.no_ball_tracking:
        ball_tracker = None
    else:
        ball_tracker = BallTracker(
            track_activation_threshold=args.track_activation_threshold,
            lost_track_buffer=args.lost_track_buffer,
            max_distance=args.max_distance,
            px_per_m=args.init_px_per_m,
            ball_diameter_m=args.ball_diameter_m,
            scale_estimation_conf=args.scale_estimation_conf,
        )

    counter: Optional[KeepyUpsCounter] = None
    if ball_tracker is not None and not args.no_counter:
        counter = KeepyUpsCounter(contact_distance_m=args.contact_distance_m)

    visualizer = Visualizer()

    pipeline = KeepyUpsPipeline(
        ball_detector,
        pose_estimator,
        visualizer,
        ball_tracker,
        counter=counter,
    )

    output_path = None
    if args.output is not None:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"annotated_{timestamp}.mp4")

    pipeline.run(
        args.video,
        output_path=output_path,
        display=args.display
    )


if __name__ == "__main__":
    main()
