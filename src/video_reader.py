from __future__ import annotations

import cv2
import numpy as np


class VideoReader:
    """Iterator/context-manager wrapper around cv2.VideoCapture.

    Yields BGR frames (cv2 native) one by one. Use as a context manager
    so the underlying capture is always released.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> "VideoReader":
        return self

    def __next__(self) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as a context manager")
        ret, frame = self._cap.read()
        if not ret:
            raise StopIteration
        return frame

    def _get(self, prop: int) -> float:
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as a context manager")
        return self._cap.get(prop)

    @property
    def fps(self) -> float:
        return self._get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def width(self) -> int:
        return int(self._get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        return int(self._get(cv2.CAP_PROP_FRAME_COUNT))
