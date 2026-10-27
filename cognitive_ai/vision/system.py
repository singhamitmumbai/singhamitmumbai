"""VisionSystem — AI-facing computer vision gateway."""

from __future__ import annotations

from typing import Any, Optional

from ..core.records import Observation
from ..core.working_state import WorkingState
from .adapters import VisionAdapters
from .catalog import list_projects
from .yolo_engine import YOLOEngine


class VisionSystem:
    """Specialist perception for images/video frames (YOLO + classical CV)."""

    def __init__(self, engine: Optional[YOLOEngine] = None) -> None:
        self.engine = engine or YOLOEngine()
        self.adapters = VisionAdapters(self.engine)

    def analyze(
        self,
        image: Any,
        *,
        task: str = "detect_objects",
        state: Optional[WorkingState] = None,
        conf: float = 0.35,
        source: str = "user",
    ) -> dict[str, Any]:
        result = self.adapters.run(task, image, conf=conf)
        result.setdefault("llm", False)
        result.setdefault("human_level_agi", False)

        obs = Observation(
            source=source,
            modality="vision",
            raw_evidence=str(image) if not isinstance(image, (bytes, bytearray)) else None,
            extracted_value={
                "task": result.get("task"),
                "count": result.get("count"),
                "counts_by_label": result.get("counts_by_label"),
                "people_count": result.get("people_count"),
                "vehicle_count": result.get("vehicle_count"),
                "alert": result.get("alert"),
                "text": result.get("text"),
            },
            recognition_version=f"vision_{result.get('engine', 'unknown')}",
            uncertainty=0.15 if result.get("engine") == "ultralytics" else 0.45,
            context={"vision_meta": {k: result.get(k) for k in ("engine", "error", "project", "note", "weights")}},
            alternatives=[],
        )
        if state is not None:
            state.add_observation(obs)
        result["observation_id"] = obs.observation_id
        return result

    def status(self) -> dict[str, Any]:
        return self.adapters.status()

    def projects(self) -> dict[str, Any]:
        return list_projects()
