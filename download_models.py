"""One-off script: download Ultralytics YOLO11 weights and export them to ONNX.

Run this once after installing requirements:

    python download_models.py

Produces:
    models/yolo11n.onnx       (general COCO detector; we use class 32 = sports ball)
    models/yolo11n-pose.onnx  (single-class person + 17 COCO keypoints)

Inference itself does NOT depend on ultralytics; only onnxruntime + opencv + numpy.
"""


import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


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


def main(imgsz: int = 640) -> None:
    export("yolo11n", imgsz=imgsz)
    export("yolo11n-pose", imgsz=imgsz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and export YOLO11 ONNX models")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square input size for the exported ONNX models (must match inference-time preprocessor)",
    )
    args = parser.parse_args()
    main(imgsz=args.imgsz)
