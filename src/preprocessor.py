from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class LetterboxInfo:
    """Geometry needed to map model-space coordinates back to the original frame."""

    scale: float
    pad_x: int
    pad_y: int
    original_width: int
    original_height: int


class YoloPreprocessor:
    """Letterbox-resize a BGR frame into a YOLO11 input tensor.

    Output tensor: shape (1, 3, S, S), float32, RGB, normalized to [0, 1].
    """

    def __init__(self, input_size: int = 640, pad_value: int = 114):
        self.input_size = input_size
        self.pad_value = pad_value

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, LetterboxInfo]:
        h, w = frame.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2

        canvas = np.full(
            (self.input_size, self.input_size, 3), self.pad_value, dtype=np.uint8
        )
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]  # NCHW
        tensor = np.ascontiguousarray(tensor)

        info = LetterboxInfo(
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            original_width=w,
            original_height=h,
        )
        return tensor, info

    def reverse_box(
        self, box: Tuple[float, float, float, float], info: LetterboxInfo
    ) -> Tuple[float, float, float, float]:
        """Map an (x1, y1, x2, y2) box from letterbox-space back to original frame."""
        x1, y1, x2, y2 = box
        x1 = (x1 - info.pad_x) / info.scale
        y1 = (y1 - info.pad_y) / info.scale
        x2 = (x2 - info.pad_x) / info.scale
        y2 = (y2 - info.pad_y) / info.scale
        return (
            float(np.clip(x1, 0, info.original_width - 1)),
            float(np.clip(y1, 0, info.original_height - 1)),
            float(np.clip(x2, 0, info.original_width - 1)),
            float(np.clip(y2, 0, info.original_height - 1)),
        )

    def reverse_point(
        self, point: Tuple[float, float], info: LetterboxInfo
    ) -> Tuple[float, float]:
        x, y = point
        x = (x - info.pad_x) / info.scale
        y = (y - info.pad_y) / info.scale
        return float(x), float(y)
