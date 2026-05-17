from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np
import onnxruntime as ort


@runtime_checkable
class Detector(Protocol):
    """Structural interface every detection component honours.

    Both `BaseOnnxDetector` subclasses and `RoiTrackingDetector` satisfy this
    by exposing `detect(frame)` and a `reset()` hook for stateful trackers.
    """

    def detect(self, frame: np.ndarray): ...

    def reset(self) -> None: ...


class BaseOnnxDetector(ABC):
    """Shared ONNX-runtime plumbing for all CPU-bound detectors.

    Subclasses implement `detect(frame)` and own their own pre/post-processing.
    """

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=self.providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _run(self, tensor: np.ndarray) -> List[np.ndarray]:
        return self.session.run(self.output_names, {self.input_name: tensor})

    def reset(self) -> None:
        """No-op: raw detectors are stateless. Trackers override this."""
        return None

    @abstractmethod
    def detect(self, frame: np.ndarray):
        """Run detection on a single BGR frame and return structured results."""
        raise NotImplementedError
