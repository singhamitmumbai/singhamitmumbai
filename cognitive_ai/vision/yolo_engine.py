"""YOLO inference engine — Ultralytics when available, deterministic stub otherwise.

CNN detectors are allowed as specialist perception (blueprint §5C). This module
never loads a general-purpose LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


# COCO class names used by Ultralytics pretrained YOLO
COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "cat",
    16: "dog",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    73: "book",
}


VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
PERSON_CLASS = "person"


class YOLOEngine:
    """Thin wrapper around Ultralytics YOLO with offline stub mode."""

    def __init__(self, weights: str = "yolo11n.pt") -> None:
        self.weights = weights
        self._model = None
        self.engine = "stub"
        self.error: Optional[str] = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.weights)
            self.engine = "ultralytics"
            self.error = None
        except Exception as exc:  # noqa: BLE001 — optional dep
            self._model = None
            self.engine = "stub"
            self.error = f"ultralytics_unavailable:{type(exc).__name__}:{exc}"

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(
        self,
        image: Any,
        *,
        conf: float = 0.35,
        classes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Run detection on image path / ndarray / PIL image.

        Returns structured detections compatible with VisionSystem.
        """
        if self._model is not None:
            return self._detect_ultralytics(image, conf=conf, classes=classes)
        return self._detect_stub(image, classes=classes)

    def _detect_ultralytics(
        self,
        image: Any,
        *,
        conf: float,
        classes: Optional[list[str]],
    ) -> dict[str, Any]:
        results = self._model(image, conf=conf, verbose=False)
        detections: list[dict[str, Any]] = []
        for r in results:
            names = r.names or {}
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = names.get(cls_id, COCO_NAMES.get(cls_id, str(cls_id)))
                if classes and label not in classes:
                    continue
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "label": label,
                        "confidence": float(box.conf[0]),
                        "bbox_xyxy": xyxy,
                        "class_id": cls_id,
                    }
                )
        return {
            "engine": "ultralytics",
            "weights": self.weights,
            "detections": detections,
            "count": len(detections),
            "counts_by_label": _count_labels(detections),
            "error": None,
        }

    def _detect_stub(
        self,
        image: Any,
        *,
        classes: Optional[list[str]],
    ) -> dict[str, Any]:
        """Deterministic offline detections for CI / missing weights.

        Heuristic: if path name hints at content, emit matching labels;
        otherwise report empty detections with engine=stub.
        """
        hint = ""
        if isinstance(image, (str, Path)):
            hint = Path(image).stem.lower()
        elif hasattr(image, "filename"):
            hint = str(getattr(image, "filename") or "").lower()

        synthetic: list[dict[str, Any]] = []
        if any(k in hint for k in ("car", "vehicle", "truck", "bus", "traffic")):
            synthetic.append(
                {
                    "label": "car",
                    "confidence": 0.91,
                    "bbox_xyxy": [10.0, 20.0, 200.0, 150.0],
                    "class_id": 2,
                    "stub": True,
                }
            )
            synthetic.append(
                {
                    "label": "truck",
                    "confidence": 0.84,
                    "bbox_xyxy": [220.0, 30.0, 400.0, 180.0],
                    "class_id": 7,
                    "stub": True,
                }
            )
        if any(k in hint for k in ("people", "person", "crowd", "queue", "human")):
            for i in range(3):
                synthetic.append(
                    {
                        "label": "person",
                        "confidence": 0.88 - i * 0.05,
                        "bbox_xyxy": [50.0 + i * 80, 40.0, 120.0 + i * 80, 220.0],
                        "class_id": 0,
                        "stub": True,
                    }
                )
        if "fire" in hint or "smoke" in hint:
            synthetic.append(
                {
                    "label": "fire",
                    "confidence": 0.79,
                    "bbox_xyxy": [100.0, 100.0, 300.0, 300.0],
                    "class_id": -1,
                    "stub": True,
                }
            )
        if not synthetic and hint:
            # Generic object stub for any named image
            synthetic.append(
                {
                    "label": "object",
                    "confidence": 0.55,
                    "bbox_xyxy": [0.0, 0.0, 64.0, 64.0],
                    "class_id": -1,
                    "stub": True,
                }
            )

        if classes:
            synthetic = [d for d in synthetic if d["label"] in classes]

        return {
            "engine": "stub",
            "weights": self.weights,
            "detections": synthetic,
            "count": len(synthetic),
            "counts_by_label": _count_labels(synthetic),
            "error": self.error,
            "note": "Install ultralytics + weights for real CNN inference",
        }


def _count_labels(detections: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in detections:
        lab = str(d.get("label", "?"))
        out[lab] = out.get(lab, 0) + 1
    return out
