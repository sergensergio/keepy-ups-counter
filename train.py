"""CLI: fine-tune a YOLO model on a custom dataset and export it to ONNX.

Example (detection):
    python train.py --base-model yolo11n.pt --data datasets/ball/data.yaml \
        --epochs 50 --imgsz 640 --batch 16 --device 0

Example (pose):
    python train.py --base-model yolo11n-pose.pt --data datasets/pose/data.yaml \
        --epochs 80 --imgsz 640 --device 0
"""

from __future__ import annotations

import argparse

from src.trainer import YoloTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a YOLO model on a custom dataset and export to ONNX.",
    )
    p.add_argument(
        "--base-model",
        default="yolo11n.pt",
        help="Starting checkpoint (e.g. yolo11n.pt, yolo11n-pose.pt, yolo26n.pt). "
        "The task (detect vs pose) is inferred from the filename.",
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to the dataset YAML (Ultralytics format).",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--device",
        default="",
        help='Training device: "" (auto), "cpu", "0", "0,1", or "mps".',
    )
    p.add_argument(
        "--project",
        default="runs/finetune",
        help="Output directory for the training run.",
    )
    p.add_argument(
        "--name",
        default="exp",
        help="Experiment subdirectory name inside --project.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience (epochs without val improvement).",
    )
    p.add_argument(
        "--no-export",
        action="store_true",
        help="Skip the ONNX export step at the end.",
    )
    p.add_argument(
        "--onnx-opset",
        type=int,
        default=12,
        help="ONNX opset version for the exported model.",
    )
    p.add_argument(
        "--onnx-dir",
        default="models",
        help="Directory where the final ONNX file is written.",
    )
    p.add_argument(
        "--onnx-name",
        default=None,
        help="Output ONNX filename. Defaults to <run-name>.onnx.",
    )
    p.add_argument(
        "--benchmark-runs",
        type=int,
        default=30,
        help="Number of forward passes used to estimate CPU latency.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trainer = YoloTrainer(
        base_model=args.base_model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        export_onnx=not args.no_export,
        onnx_opset=args.onnx_opset,
        onnx_dir=args.onnx_dir,
        onnx_name=args.onnx_name,
        benchmark_runs=args.benchmark_runs,
    )
    kpis = trainer.run()
    kpis.pretty_print()


if __name__ == "__main__":
    main()
