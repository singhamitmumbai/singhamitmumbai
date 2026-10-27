"""VisionSystem — smart multi-modality gateway (RGB, thermal, IR, night, …)."""

from __future__ import annotations

from typing import Any, Optional

from ..core.records import Observation
from ..core.working_state import WorkingState
from .adapters import VisionAdapters
from .catalog import list_projects
from .knowledge import ObjectKnowledge
from .modalities import MODALITIES, detect_modality
from .thermal import ThermalAnalyzer
from .understand import VisionUnderstanding
from .yolo_engine import YOLOEngine


class VisionSystem:
    """See → analyze → understand (+ search); thermal and other modalities supported."""

    def __init__(self, engine: Optional[YOLOEngine] = None) -> None:
        self.engine = engine or YOLOEngine()
        self.adapters = VisionAdapters(self.engine)
        self.knowledge = ObjectKnowledge()
        self.understanding = VisionUnderstanding(self.engine, self.knowledge)
        self.thermal = ThermalAnalyzer()

    def analyze(
        self,
        image: Any,
        *,
        task: str = "detect_objects",
        state: Optional[WorkingState] = None,
        conf: float = 0.35,
        source: str = "user",
        search_unknown: bool = True,
        modality: Optional[str] = None,
    ) -> dict[str, Any]:
        if task in ("understand_scene", "see_analyze_understand", "perceive", "understand", "smart_understand"):
            result = self.understanding.perceive(
                image, conf=conf, search_unknown=search_unknown, modality_force=modality
            )
        elif task in ("see", "vision_see"):
            mod = detect_modality(image, force=modality)
            result = {**self.understanding.see(image, conf=conf, modality=mod), "task": "see", "llm": False}
        elif task in ("analyze_scene", "vision_analyze"):
            mod = detect_modality(image, force=modality)
            seen = self.understanding.see(image, conf=conf, modality=mod)
            result = {**self.understanding.analyze(seen, image=image), "task": "analyze_scene", "llm": False}
        elif task in ("thermal_analyze", "thermal", "analyze_thermal"):
            result = self.thermal.analyze(image, modality=modality or "thermal")
            result.setdefault("task", "thermal_analyze")
        elif task in ("detect_modality", "modality"):
            result = {"task": "detect_modality", **detect_modality(image, force=modality), "llm": False}
        elif task in ("explain_object", "understand_object"):
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
                "image_modality": (result.get("modality") or {}).get("modality")
                if isinstance(result.get("modality"), dict)
                else result.get("modality"),
                "understood_count": (result.get("understand") or {}).get("understood_count"),
                "unknown_count": (result.get("understand") or {}).get("unknown_count"),
                "scene_type": (result.get("smart") or {}).get("scene_type"),
                "summary": result.get("text") or (result.get("understand") or {}).get("summary"),
            },
            recognition_version=f"vision_{result.get('engine', result.get('task', 'unknown'))}",
            uncertainty=0.15 if result.get("engine") == "ultralytics" else 0.35,
            context={
                "vision_meta": {
                    k: result.get(k)
                    for k in ("engine", "error", "project", "note", "weights", "pipeline", "smart")
                }
            },
            alternatives=[],
        )
        if state is not None:
            state.add_observation(obs)
        result["observation_id"] = obs.observation_id
        return result

    def see(self, image: Any, *, conf: float = 0.35, modality: Optional[str] = None) -> dict[str, Any]:
        return self.analyze(image, task="see", conf=conf, modality=modality)

    def analyze_scene(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        return self.analyze(image, task="analyze_scene", conf=conf)

    def understand(
        self,
        image: Any,
        *,
        conf: float = 0.35,
        search_unknown: bool = True,
        state: Optional[WorkingState] = None,
        modality: Optional[str] = None,
    ) -> dict[str, Any]:
        """Smart see→analyze→understand; auto-detects thermal and other modalities."""
        return self.analyze(
            image,
            task="understand_scene",
            conf=conf,
            search_unknown=search_unknown,
            state=state,
            modality=modality,
        )

    def analyze_thermal(self, image: Any, *, modality: str = "thermal") -> dict[str, Any]:
        return self.analyze(image, task="thermal_analyze", modality=modality)

    def explain_object(self, label: str, *, image: Any = None) -> dict[str, Any]:
        return self.understanding.explain_object(label, image=image)

    def status(self) -> dict[str, Any]:
        st = self.adapters.status()
        st["understanding"] = {
            "pipeline": ["modality_detect", "see", "analyze", "understand", "search_if_needed", "smart"],
            "knowledge": self.knowledge.stats(),
            "search_history": len(self.understanding.search.history),
            "modalities": list(MODALITIES),
            "thermal": True,
            "smart_reasoner": True,
        }
        return st

    def projects(self) -> dict[str, Any]:
        return list_projects()


def _looks_like_path(value: Any) -> bool:
    if not isinstance(value, str):
        return isinstance(value, (bytes, bytearray)) or hasattr(value, "__fspath__")
    s = value.lower()
    return "/" in value or "\\" in value or s.endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dcm")
    )
