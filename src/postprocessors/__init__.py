from .base import DetectionPostprocessor
from .yolo11 import Yolo11DetectionPostprocessor
from .yolo26 import Yolo26DetectionPostprocessor

DETECTION_POSTPROCESSORS = {
    "yolo11": Yolo11DetectionPostprocessor,
    "yolo26": Yolo26DetectionPostprocessor,
}

SUPPORTED_ARCHITECTURES = sorted(set(DETECTION_POSTPROCESSORS))


def make_detection_postprocessor(arch: str) -> DetectionPostprocessor:
    try:
        return DETECTION_POSTPROCESSORS[arch]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown detection arch '{arch}'. "
            f"Available: {sorted(DETECTION_POSTPROCESSORS)}"
        ) from exc


__all__ = [
    "DetectionPostprocessor",
    "Yolo11DetectionPostprocessor",
    "Yolo26DetectionPostprocessor",
    "SUPPORTED_ARCHITECTURES",
    "make_detection_postprocessor",
]
