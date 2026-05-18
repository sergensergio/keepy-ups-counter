"""CLI entry point for the keepy-ups tool (v1: detect + visualize)."""


import argparse
from pathlib import Path

from src.ball_detector import BallDetector
from src.pipeline import KeepyUpsPipeline
from src.pose_estimator import PoseEstimator
from src.postprocessors import (
    SUPPORTED_ARCHITECTURES,
    make_detection_postprocessor,
    make_pose_postprocessor,
)
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
        default="models/yolo11n.onnx",
        help="Path to YOLO11 detection ONNX model",
    )
    p.add_argument(
        "--pose-model",
        default="models/yolo11n-pose.onnx",
        help="Path to YOLO11-pose ONNX model",
    )
    p.add_argument("--ball-conf", type=float, default=0.25)
    p.add_argument("--pose-conf", type=float, default=0.25)
    p.add_argument(
        "--ball-arch",
        choices=SUPPORTED_ARCHITECTURES,
        default="yolo11",
        help="Architecture family of the ball-detection ONNX. "
        "'yolo11' = transposed output + NMS. "
        "'yolo26' = end-to-end / NMS-free output.",
    )
    p.add_argument(
        "--pose-arch",
        choices=SUPPORTED_ARCHITECTURES,
        default="yolo11",
        help="Architecture family of the pose ONNX (see --ball-arch).",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Square model input size (must match the exported ONNX)",
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
        input_size=args.input_size,
        postprocessor=make_detection_postprocessor(args.ball_arch),
    )
    pose_estimator = PoseEstimator(
        args.pose_model,
        conf_threshold=args.pose_conf,
        input_size=args.input_size,
        postprocessor=make_pose_postprocessor(args.pose_arch),
    )
    visualizer = Visualizer()

    pipeline = KeepyUpsPipeline(
        ball_detector,
        pose_estimator,
        visualizer
    )
    pipeline.run(
        args.video,
        output_path=args.output, 
        display=args.display
    )


if __name__ == "__main__":
    main()
