"""CLI entry point for the keepy-ups tool (v1: detect + visualize)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ball_detector import BallDetector
from src.pipeline import KeepyUpsPipeline
from src.pose_estimator import PoseEstimator
from src.roi_tracker import RoiTrackingDetector
from src.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Keepy-ups counter (v1): read a portrait video, detect the ball "
        "and the player's pose, render annotated output.",
    )
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument(
        "--output",
        default=None,
        help="Optional path to write annotated mp4 (e.g. out.mp4)",
    )
    p.add_argument(
        "--ball-model",
        default="models/yolov8n.onnx",
        help="Path to YOLOv8 detection ONNX model",
    )
    p.add_argument(
        "--pose-model",
        default="models/yolov8n-pose.onnx",
        help="Path to YOLOv8-pose ONNX model",
    )
    p.add_argument("--ball-conf", type=float, default=0.25)
    p.add_argument("--pose-conf", type=float, default=0.25)
    p.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Square model input size (must match the exported ONNX)",
    )
    p.add_argument(
        "--ball-roi-padding",
        type=float,
        default=0.5,
        help="ROI extension ratio around the previous ball bbox "
        "(2.0 = grow each side by 200%% of bbox size).",
    )
    p.add_argument(
        "--pose-roi-padding",
        type=float,
        default=0.2,
        help="ROI extension ratio around the previous person bbox.",
    )
    p.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable ROI tracking and always run detectors on the full frame.",
    )
    p.add_argument(
        "--display",
        action="store_true",
        help="Show annotated frames in a window (press q to quit)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for path in (args.ball_model, args.pose_model):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Model not found: {path}. Run `python download_models.py` first."
            )

    ball_detector = BallDetector(
        args.ball_model,
        conf_threshold=args.ball_conf,
        input_size=args.input_size
    )
    pose_estimator = PoseEstimator(
        args.pose_model,
        conf_threshold=args.pose_conf,
        input_size=args.input_size
    )

    if args.no_tracking:
        ball_component = ball_detector
        pose_component = pose_estimator
    else:
        ball_component = RoiTrackingDetector(
            ball_detector,
            padding_ratio=args.ball_roi_padding,
        )
        pose_component = RoiTrackingDetector(
            pose_estimator,
            padding_ratio=args.pose_roi_padding,
        )

    visualizer = Visualizer()

    pipeline = KeepyUpsPipeline(
        ball_component,
        pose_component,
        visualizer
    )
    pipeline.run(
        args.video,
        output_path=args.output, 
        display=args.display
    )


if __name__ == "__main__":
    main()
