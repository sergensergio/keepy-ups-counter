"""One-off script: download Ultralytics YOLO weights and export them to ONNX.

Run this once after installing requirements:

    python download_models.py                       # default: yolo11
    python download_models.py --arch yolo26         # end-to-end / NMS-free
    python download_models.py --arch yolo11 yolo26  # both families

Produces (per requested arch):
    models/<arch>n.onnx       (general COCO detector; we use class 32 = sports ball)
    models/<arch>n-pose.onnx  (single-class person + 17 COCO keypoints)

Inference itself does NOT depend on ultralytics; only onnxruntime + opencv + numpy.
"""


import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

SUPPORTED_ARCHS = ("yolo11", "yolo26")


def export(name: str, imgsz: int = 640, opset: int = 12) -> None:
    from ultralytics import YOLO  # imported lazily so inference deps stay slim

    MODELS_DIR.mkdir(exist_ok=True)
    onnx_target = MODELS_DIR / f"{name}.onnx"
    # if onnx_target.exists():
    #     print(f"[skip] {onnx_target} already exists")
    #     return

    cwd = os.getcwd()
    os.chdir(MODELS_DIR)
    try:
        model = YOLO(f"{name}.pt")
        exported = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True)
    finally:
        os.chdir(cwd)

    # Ultralytics returns the path to the produced file; move it into models/ if needed.
    exported_path = Path(exported)
    if not exported_path.is_absolute():
        exported_path = MODELS_DIR / exported_path.name
    if exported_path != onnx_target:
        shutil.move(str(exported_path), str(onnx_target))
    print(f"[ok] exported {onnx_target}")


def main(archs: Iterable[str] = ("yolo11",), imgsz: int = 640) -> None:
    for arch in archs:
        if arch not in SUPPORTED_ARCHS:
            raise ValueError(
                f"Unsupported arch '{arch}'. Choose from {SUPPORTED_ARCHS}."
            )
        export(f"{arch}n", imgsz=imgsz)
        export(f"{arch}n-pose", imgsz=imgsz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and export YOLO ONNX models")
    parser.add_argument(
        "--arch",
        nargs="+",
        default=["yolo11"],
        choices=list(SUPPORTED_ARCHS),
        help="One or more YOLO families to fetch and export "
        "(e.g. --arch yolo11 yolo26).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square input size for the exported ONNX models "
        "(must match inference-time preprocessor)",
    )
    args = parser.parse_args()
    main(archs=args.arch, imgsz=args.imgsz)
