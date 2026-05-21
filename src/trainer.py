"""Fine-tune a YOLO model on a custom dataset and export it to ONNX.

Wraps Ultralytics' training API with this project's conventions:

  * The base-model name implies the task (`*-pose.pt` -> pose, otherwise detect).
  * Training metrics are pulled into a typed `TrainingKpis` dataclass.
  * Best weights are exported to ONNX matching the inference preprocessor
    (`imgsz`, `opset`).
  * A short CPU latency benchmark is run on the exported ONNX so the operator
    can immediately see whether the finetuned model is fast enough.

The module imports `ultralytics` lazily so the inference pipeline does not
pay its import cost.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TrainingKpis:
    base_model: str
    task: str
    data: str
    imgsz: int
    duration_seconds: float
    metrics: Dict[str, float] = field(default_factory=dict)
    best_weights_path: Optional[str] = None
    onnx_path: Optional[str] = None
    onnx_latency_ms: Optional[float] = None

    def pretty_print(self) -> None:
        line = "=" * 64
        print()
        print(line)
        print("  Training summary")
        print(line)
        print(f"  Task                {self.task}")
        print(f"  Base model          {self.base_model}")
        print(f"  Dataset             {self.data}")
        print(f"  Image size          {self.imgsz}")
        print(f"  Duration            {self.duration_seconds / 60:.1f} min")
        if self.metrics:
            print(f"  {'-' * 60}")
            for k, v in self.metrics.items():
                print(f"  {k:<20s}{v:.4f}")
        print(f"  {'-' * 60}")
        if self.best_weights_path:
            print(f"  Best weights        {self.best_weights_path}")
        if self.onnx_path:
            print(f"  Exported ONNX       {self.onnx_path}")
        if self.onnx_latency_ms is not None:
            print(f"  CPU latency (med)   {self.onnx_latency_ms:.1f} ms / frame")
        print(line)
        print()


class YoloTrainer:
    """Fine-tunes a YOLO model on a custom dataset and exports to ONNX.

    Detection (`yolo11n.pt`, `yolo26n.pt`, ...) and pose
    (`yolo11n-pose.pt`, `yolo26n-pose.pt`, ...) checkpoints are both
    supported — the task is inferred from the base-model filename.
    """

    def __init__(
        self,
        base_model: str,
        data: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "",
        project: str = "runs/finetune",
        name: str = "exp",
        patience: int = 20,
        export_onnx: bool = True,
        onnx_opset: int = 12,
        onnx_dir: str = "models",
        onnx_name: Optional[str] = None,
        benchmark_runs: int = 30,
    ):
        self.base_model = base_model
        self.data = data
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.project = project
        self.name = name
        self.patience = patience
        self.export_onnx_enabled = export_onnx
        self.onnx_opset = onnx_opset
        self.onnx_dir = Path(onnx_dir)
        self.onnx_name = onnx_name
        self.benchmark_runs = benchmark_runs

    @property
    def task(self) -> str:
        return "pose" if "pose" in Path(self.base_model).stem.lower() else "detect"

    def run(self) -> TrainingKpis:
        from ultralytics import YOLO  # lazy: keeps inference deps slim

        if not Path(self.data).is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {self.data}")

        print(
            f"[train] task={self.task}  base={self.base_model}  data={self.data}  "
            f"epochs={self.epochs}  imgsz={self.imgsz}  batch={self.batch}"
        )

        t0 = time.time()
        model = YOLO(self.base_model)
        train_results = model.train(
            data=self.data,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            project=self.project,
            name=self.name,
            patience=self.patience,
            exist_ok=True,
        )
        duration = time.time() - t0

        save_dir = Path(train_results.save_dir)
        best_weights = save_dir / "weights" / "best.pt"
        if not best_weights.is_file():
            raise FileNotFoundError(
                f"Expected best weights at {best_weights}, but file is missing."
            )

        # Re-run validation on the best weights so the KPIs reflect a clean,
        # explicit evaluation rather than just the last-epoch training stats.
        print(f"[val] re-evaluating best weights on {self.data}")
        val_model = YOLO(str(best_weights))
        val_results = val_model.val(
            data=self.data,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            project=self.project,
            name=f"{self.name}_val",
            exist_ok=True,
        )
        metrics = self._extract_metrics(val_results)

        onnx_path: Optional[Path] = None
        latency_ms: Optional[float] = None
        if self.export_onnx_enabled:
            onnx_path = self._export_onnx(best_weights)
            try:
                latency_ms = self._benchmark_onnx(onnx_path)
            except Exception as exc:  # benchmark is best-effort
                print(f"[bench] skipped ({exc})")

        return TrainingKpis(
            base_model=self.base_model,
            task=self.task,
            data=self.data,
            imgsz=self.imgsz,
            duration_seconds=duration,
            metrics=metrics,
            best_weights_path=str(best_weights),
            onnx_path=str(onnx_path) if onnx_path else None,
            onnx_latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_metrics(results_obj: Any) -> Dict[str, float]:
        """Pull the main KPIs from an Ultralytics validation results object.

        Detection results expose only the `box` block; pose results expose
        both `box` (person bbox) and `pose` (keypoint AP).
        """
        out: Dict[str, float] = {}
        for prefix in ("box", "pose"):
            block = getattr(results_obj, prefix, None)
            if block is None:
                continue
            for label, attr in (
                ("mAP@0.5", "map50"),
                ("mAP@0.5-0.95", "map"),
                ("precision", "mp"),
                ("recall", "mr"),
            ):
                value = getattr(block, attr, None)
                if value is None:
                    continue
                try:
                    out[f"{prefix} {label}"] = float(value)
                except (TypeError, ValueError):
                    continue
        return out

    def _export_onnx(self, weights_path: Path) -> Path:
        from ultralytics import YOLO

        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        # Default ONNX name follows the experiment, e.g.
        # runs/finetune/exp/weights/best.pt -> models/exp.onnx
        target_name = self.onnx_name or f"{weights_path.parent.parent.name}.onnx"
        target_path = self.onnx_dir / target_name

        print(f"[export] {weights_path} -> {target_path}")
        model = YOLO(str(weights_path))
        exported = model.export(
            format="onnx",
            imgsz=self.imgsz,
            opset=self.onnx_opset,
            simplify=True,
        )
        exported_path = Path(exported)
        if exported_path != target_path:
            shutil.move(str(exported_path), str(target_path))
        return target_path

    def _benchmark_onnx(self, onnx_path: Path) -> float:
        """Median per-frame CPU inference time in milliseconds."""
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        meta = sess.get_inputs()[0]

        # Replace dynamic dimensions in the declared shape with concrete values
        # matching the inference preprocessor: batch=1, C=3, H=W=imgsz.
        expected = [1, 3, self.imgsz, self.imgsz]
        shape = [
            int(d) if isinstance(d, int) and d > 0 else expected[i]
            for i, d in enumerate(meta.shape)
        ]
        dummy = np.random.rand(*shape).astype(np.float32)

        for _ in range(3):  # warm-up
            sess.run(None, {meta.name: dummy})

        timings = []
        for _ in range(self.benchmark_runs):
            t0 = time.perf_counter()
            sess.run(None, {meta.name: dummy})
            timings.append((time.perf_counter() - t0) * 1000.0)
        timings.sort()
        return timings[len(timings) // 2]
