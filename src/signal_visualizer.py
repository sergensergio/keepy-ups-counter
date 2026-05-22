"""Scrolling multi-signal visualizer in hospital-monitor / ECG style.

Each call to `push()` appends one new sample per signal; `render()` produces
a BGR frame with one subplot per signal stacked vertically. The signal
"moves" because the buffer is a fixed-length deque — the oldest sample
drops off the left as the newest arrives on the right. A single vertical
cursor at the right edge marks the current frame across every subplot.

NaN samples are allowed and rendered as gaps (e.g. frames where the
tracker has no active track, or a keypoint is below the confidence
threshold).
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SignalSpec:
    """Per-channel configuration."""

    key: str
    label: str
    color: Tuple[int, int, int]  # BGR
    # Fixed display range. When None, the subplot auto-scales each frame to
    # the visible buffer's min/max with light padding.
    y_range: Optional[Tuple[float, float]] = None
    unit: str = ""


class ScrollingSignalVisualizer:
    """Vertical stack of scrolling time-series subplots."""

    def __init__(
        self,
        width: int,
        height: int,
        signals: List[SignalSpec],
        window_size: int = 240,
        bg_color: Tuple[int, int, int] = (10, 10, 10),
        grid_color: Tuple[int, int, int] = (40, 40, 40),
        text_color: Tuple[int, int, int] = (200, 200, 200),
        cursor_color: Tuple[int, int, int] = (60, 60, 230),  # red, BGR
        label_pad: int = 110,
        margin: int = 10,
    ):
        if not signals:
            raise ValueError("ScrollingSignalVisualizer needs at least one signal")
        self.width = width
        self.height = height
        self.signals = signals
        self.window_size = window_size
        self.bg_color = bg_color
        self.grid_color = grid_color
        self.text_color = text_color
        self.cursor_color = cursor_color
        self.label_pad = label_pad
        self.margin = margin

        self.row_h = (height - 2 * margin) // len(signals)
        self.buffers: Dict[str, Deque[float]] = {
            sig.key: deque([float("nan")] * window_size, maxlen=window_size)
            for sig in signals
        }

    def push(self, samples: Dict[str, Optional[float]]) -> None:
        for sig in self.signals:
            v = samples.get(sig.key)
            self.buffers[sig.key].append(
                float("nan") if v is None else float(v)
            )

    def render(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

        plot_x0 = self.label_pad
        plot_x1 = self.width - self.margin
        plot_w = plot_x1 - plot_x0
        if plot_w < 16:
            return img

        # Newest sample sits at the right edge; oldest at the left edge.
        denom_x = max(1, self.window_size - 1)
        xs = plot_x0 + (np.arange(self.window_size) / denom_x) * plot_w
        xs = xs.astype(np.int32)

        for i, sig in enumerate(self.signals):
            row_y0 = self.margin + i * self.row_h
            row_y1 = row_y0 + self.row_h - 4
            self._draw_subplot(img, sig, row_y0, row_y1, plot_x0, plot_x1, xs)

        # One vertical cursor at the current-frame edge, spanning all subplots.
        cursor_x = plot_x1
        cv2.line(
            img,
            (cursor_x, self.margin),
            (cursor_x, self.margin + len(self.signals) * self.row_h),
            self.cursor_color,
            1,
            cv2.LINE_AA,
        )
        return img

    def _draw_subplot(
        self,
        img: np.ndarray,
        sig: SignalSpec,
        row_y0: int,
        row_y1: int,
        plot_x0: int,
        plot_x1: int,
        xs: np.ndarray,
    ) -> None:
        cv2.rectangle(img, (plot_x0, row_y0), (plot_x1, row_y1), self.grid_color, 1)
        mid_y = (row_y0 + row_y1) // 2
        cv2.line(img, (plot_x0, mid_y), (plot_x1, mid_y), self.grid_color, 1)
        plot_w = plot_x1 - plot_x0
        for k in (1, 2, 3):
            gx = plot_x0 + (plot_w * k) // 4
            cv2.line(img, (gx, row_y0), (gx, row_y1), self.grid_color, 1)

        buf = np.asarray(self.buffers[sig.key], dtype=np.float32)
        valid = ~np.isnan(buf)

        if sig.y_range is not None:
            ymin, ymax = sig.y_range
        elif valid.any():
            ymin = float(np.nanmin(buf))
            ymax = float(np.nanmax(buf))
            if ymax - ymin < 1e-6:
                ymin -= 1.0
                ymax += 1.0
            else:
                pad = (ymax - ymin) * 0.1
                ymin -= pad
                ymax += pad
        else:
            ymin, ymax = -1.0, 1.0

        denom_y = max(ymax - ymin, 1e-9)
        row_span = row_y1 - row_y0
        ys_pix = row_y1 - ((buf - ymin) / denom_y) * row_span
        ys_pix = np.clip(np.nan_to_num(ys_pix, nan=row_y1), row_y0, row_y1).astype(
            np.int32
        )
        pts = np.stack([xs, ys_pix], axis=1)

        # Polyline, broken at NaN gaps so missing data shows as a gap rather
        # than a misleading straight line across the discontinuity.
        seg_start: Optional[int] = None
        for j in range(self.window_size):
            if valid[j]:
                if seg_start is None:
                    seg_start = j
            elif seg_start is not None:
                if j - seg_start >= 2:
                    cv2.polylines(
                        img, [pts[seg_start:j]], False, sig.color, 1, cv2.LINE_AA
                    )
                seg_start = None
        if seg_start is not None and self.window_size - seg_start >= 2:
            cv2.polylines(
                img, [pts[seg_start:]], False, sig.color, 1, cv2.LINE_AA
            )

        cv2.putText(
            img,
            sig.label,
            (self.margin, row_y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            sig.color,
            1,
            cv2.LINE_AA,
        )
        if valid[-1]:
            cur_text = f"{buf[-1]:7.1f}"
            if sig.unit:
                cur_text = f"{cur_text} {sig.unit}"
            cv2.putText(
                img,
                cur_text,
                (self.margin, mid_y + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                self.text_color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            img,
            f"{ymax:.0f}",
            (plot_x0 - 36, row_y0 + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            self.text_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"{ymin:.0f}",
            (plot_x0 - 36, row_y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            self.text_color,
            1,
            cv2.LINE_AA,
        )
