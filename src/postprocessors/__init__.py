from .base import DetectionPostprocessor, PosePostprocessor
from .yolo11 import Yolo11DetectionPostprocessor, Yolo11PosePostprocessor
from .yolo26 import Yolo26DetectionPostprocessor, Yolo26PosePostprocessor

DETECTION_POSTPROCESSORS = {
    "yolo11": Yolo11DetectionPostprocessor,
    "yolo26": Yolo26DetectionPostprocessor,
}

POSE_POSTPROCESSORS = {
    "yolo11": Yolo11PosePostprocessor,
    "yolo26": Yolo26PosePostprocessor,
}

SUPPORTED_ARCHITECTURES = sorted(set(DETECTION_POSTPROCESSORS) | set(POSE_POSTPROCESSORS))


def make_detection_postprocessor(arch: str) -> DetectionPostprocessor:
    try:
        return DETECTION_POSTPROCESSORS[arch]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown detection arch '{arch}'. "
            f"Available: {sorted(DETECTION_POSTPROCESSORS)}"
        ) from exc


def make_pose_postprocessor(arch: str) -> PosePostprocessor:
    try:
        return POSE_POSTPROCESSORS[arch]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown pose arch '{arch}'. Available: {sorted(POSE_POSTPROCESSORS)}"
        ) from exc


__all__ = [
    "DetectionPostprocessor",
    "PosePostprocessor",
    "Yolo11DetectionPostprocessor",
    "Yolo11PosePostprocessor",
    "Yolo26DetectionPostprocessor",
    "Yolo26PosePostprocessor",
    "SUPPORTED_ARCHITECTURES",
    "make_detection_postprocessor",
    "make_pose_postprocessor",
]
