"""VisionSystem — AI-facing computer vision gateway with understand loop."""

from __future__ import annotations

from typing import Any, Optional

from ..core.records import Observation
from ..core.working_state import WorkingState
from .adapters import VisionAdapters
from .catalog import list_projects
from .knowledge import ObjectKnowledge
from .understand import VisionUnderstanding
from .yolo_engine import YOLOEngine


class VisionSystem:
    """Specialist perception: see → analyze → understand (+ search if unknown)."""

    def __init__(self, engine: Optional[YOLOEngine] = None) -> None:
        self.engine = engine or YOLOEngine()
        self.adapters = VisionAdapters(self.engine)
        self.knowledge = ObjectKnowledge()
        self.understanding = VisionUnderstanding(self.engine, self.knowledge)

    def analyze(
        self,
        image: Any,
        *,
        task: str = "detect_objects",
        state: Optional[WorkingState] = None,
        conf: float = 0.35,
        source: str = "user",
        search_unknown: bool = True,
    ) -> dict[str, Any]:
        # Understanding pipeline tasks
        if task in ("understand_scene", "see_analyze_understand", "perceive", "understand"):
            result = self.understanding.perceive(image, conf=conf, search_unknown=search_unknown)
        elif task in ("see", "vision_see"):
            result = {**self.understanding.see(image, conf=conf), "task": "see", "llm": False}
        elif task in ("analyze_scene", "vision_analyze"):
            seen = self.understanding.see(image, conf=conf)
            result = {**self.understanding.analyze(seen, image=image), "task": "analyze_scene", "llm": False}
        elif task in ("explain_object", "understand_object"):
            # image may be unused; label expected as string image? prefer kwargs via adapters
            label = str(image) if not _looks_like_path(image) else "object"
            result = self.understanding.explain_object(label)
        else:
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
                "understood_count": (result.get("understand") or {}).get("understood_count"),
                "unknown_count": (result.get("understand") or {}).get("unknown_count"),
                "summary": result.get("text") or (result.get("understand") or {}).get("summary"),
            },
            recognition_version=f"vision_{result.get('engine', result.get('task', 'unknown'))}",
            uncertainty=0.15 if result.get("engine") == "ultralytics" else 0.35,
            context={
                "vision_meta": {
                    k: result.get(k)
                    for k in ("engine", "error", "project", "note", "weights", "pipeline")
                }
            },
            alternatives=[],
        )
        if state is not None:
            state.add_observation(obs)
        result["observation_id"] = obs.observation_id
        return result

    def see(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        return self.analyze(image, task="see", conf=conf)

    def analyze_scene(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        return self.analyze(image, task="analyze_scene", conf=conf)

    def understand(
        self,
        image: Any,
        *,
        conf: float = 0.35,
        search_unknown: bool = True,
        state: Optional[WorkingState] = None,
    ) -> dict[str, Any]:
        """See → analyze → understand; search objects that are not understood."""
        return self.analyze(
            image,
            task="understand_scene",
            conf=conf,
            search_unknown=search_unknown,
            state=state,
        )

    def explain_object(self, label: str, *, image: Any = None) -> dict[str, Any]:
        return self.understanding.explain_object(label, image=image)

    def status(self) -> dict[str, Any]:
        st = self.adapters.status()
        st["understanding"] = {
            "pipeline": ["see", "analyze", "understand", "search_if_needed"],
            "knowledge": self.knowledge.stats(),
            "search_history": len(self.understanding.search.history),
        }
        return st

    def projects(self) -> dict[str, Any]:
        return list_projects()


def _looks_like_path(value: Any) -> bool:
    if not isinstance(value, str):
        return isinstance(value, (bytes, bytearray)) or hasattr(value, "__fspath__")
    s = value.lower()
    return "/" in value or "\\" in value or s.endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    )
