"""See → Analyze → Understand → (Search if unknown) → Understand (blueprint vision loop).

Non-LLM. Specialist detectors + structured knowledge + search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapters import VisionAdapters
from .knowledge import ObjectKnowledge
from .modalities import detect_modality
from .search import ObjectSearch
from .smart import SmartVisionReasoner
from .thermal import ThermalAnalyzer
from .yolo_engine import YOLOEngine


class VisionUnderstanding:
    """
    0. MODALITY — rgb / thermal / IR / night / xray / depth / document / …
    1. SEE      — detect objects (modality-aware; thermal hotspots when IR)
    2. ANALYZE  — size, position, confidence, relations, OCR, thermal stats
    3. UNDERSTAND — map each object to knowledge
    4. If not understood — SEARCH, then re-understand with new facts
    5. SMART    — scene type, risks, recommendations
    """

    def __init__(
        self,
        engine: Optional[YOLOEngine] = None,
        knowledge: Optional[ObjectKnowledge] = None,
    ) -> None:
        self.engine = engine or YOLOEngine()
        self.adapters = VisionAdapters(self.engine)
        self.knowledge = knowledge or ObjectKnowledge()
        self.search = ObjectSearch(self.knowledge)
        self.thermal = ThermalAnalyzer()
        self.smart = SmartVisionReasoner()

    # --- stages ---------------------------------------------------------------

    def detect_modality(self, image: Any, *, force: Optional[str] = None) -> dict[str, Any]:
        return detect_modality(image, force=force)

    def see(
        self,
        image: Any,
        *,
        conf: float = 0.35,
        modality: Optional[dict[str, Any]] = None,
        thermal: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        mod = modality or detect_modality(image)
        mod_name = mod.get("modality") or "rgb"

        # Document modality prefers scanner path
        if mod_name == "document":
            scanned = self.adapters.scan_document(image)
            detections = []
            if scanned.get("has_document_quad") or scanned.get("text"):
                detections.append(
                    {
                        "label": "document",
                        "confidence": 0.8,
                        "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
                        "class_id": -20,
                        "modality": "document",
                    }
                )
            return {
                "stage": "see",
                "image": str(image) if isinstance(image, (str, Path)) else "<buffer>",
                "engine": scanned.get("engine") or "document",
                "detections": detections,
                "count": len(detections),
                "counts_by_label": {"document": len(detections)} if detections else {},
                "document": scanned,
                "modality": mod_name,
                "error": scanned.get("error"),
                "note": "Document modality — OCR/scan prioritized",
            }

        detected = self.adapters.detect_objects(image, conf=conf)
        detections = list(detected.get("detections") or [])

        # Merge thermal detections for thermal/IR
        if mod_name in ("thermal", "infrared") and thermal:
            for d in thermal.get("detections") or []:
                detections.append(d)
        elif mod_name in ("thermal", "infrared") and thermal is None:
            th = self.thermal.analyze(image, modality=mod_name)
            for d in th.get("detections") or []:
                detections.append(d)
            thermal = th

        counts: dict[str, int] = {}
        for d in detections:
            lab = str(d.get("label") or "?")
            counts[lab] = counts.get(lab, 0) + 1

        return {
            "stage": "see",
            "image": str(image) if isinstance(image, (str, Path)) else "<buffer>",
            "engine": detected.get("engine"),
            "detections": detections,
            "count": len(detections),
            "counts_by_label": counts,
            "modality": mod_name,
            "thermal": thermal,
            "error": detected.get("error"),
            "note": detected.get("note"),
        }

    def analyze(self, see_result: dict[str, Any], *, image: Any = None) -> dict[str, Any]:
        detections = see_result.get("detections") or []
        analyzed = []
        for i, det in enumerate(detections):
            bbox = det.get("bbox_xyxy") or [0, 0, 0, 0]
            x1, y1, x2, y2 = (float(bbox[j]) if j < len(bbox) else 0.0 for j in range(4))
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            area = w * h
            analyzed.append(
                {
                    "id": f"obj_{i}",
                    "label": det.get("label") or "object",
                    "confidence": float(det.get("confidence") or 0),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "width": w,
                    "height": h,
                    "area": area,
                    "aspect_ratio": (w / h) if h else None,
                    "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    "size_class": _size_class(area, detections),
                    "position": _position_label(x1, y1, x2, y2, detections),
                }
            )
        relations = _spatial_relations(analyzed)
        ocr = None
        if image is not None:
            ocr = self._try_ocr(image)
        return {
            "stage": "analyze",
            "objects": analyzed,
            "relations": relations,
            "object_count": len(analyzed),
            "labels": sorted({o["label"] for o in analyzed}),
            "ocr": ocr,
            "dominant_label": _dominant(see_result.get("counts_by_label") or {}),
        }

    def understand_objects(
        self,
        analysis: dict[str, Any],
        *,
        image: Any = None,
        search_unknown: bool = True,
        max_searches: int = 8,
    ) -> dict[str, Any]:
        context = {
            "image_path": str(image) if isinstance(image, (str, Path)) else None,
            "filename": Path(str(image)).name if isinstance(image, (str, Path)) else None,
            "ocr_text": (analysis.get("ocr") or {}).get("text") if analysis.get("ocr") else None,
            "hints": analysis.get("labels") or [],
        }
        understood = []
        unknown = []
        searches = []
        search_budget = max_searches

        for obj in analysis.get("objects") or []:
            label = obj["label"]
            fact = self.knowledge.lookup(label)
            entry = {
                **obj,
                "knowledge": fact,
                "understood": bool(fact and fact.get("understood")),
                "understanding": _describe(fact, obj) if fact else None,
            }
            if entry["understood"]:
                understood.append(entry)
            else:
                unknown.append(entry)

        # Search loop for unknowns
        if search_unknown:
            still_unknown = []
            for entry in unknown:
                if search_budget <= 0:
                    still_unknown.append(entry)
                    continue
                search_budget -= 1
                sr = self.search.search(entry["label"], context=context)
                searches.append(sr)
                if sr.get("found") and sr.get("fact"):
                    fact = sr["fact"]
                    entry["knowledge"] = fact
                    entry["understood"] = bool(fact.get("understood"))
                    entry["understanding"] = _describe(fact, entry)
                    entry["searched"] = True
                    entry["search_method"] = sr.get("method")
                    if entry["understood"]:
                        understood.append(entry)
                    else:
                        still_unknown.append(entry)
                else:
                    entry["searched"] = True
                    entry["search_method"] = sr.get("method")
                    entry["understanding"] = sr.get("message") or "Not understood after search"
                    still_unknown.append(entry)
            unknown = still_unknown

        summary = _scene_summary(understood, unknown, analysis)
        return {
            "stage": "understand",
            "understood": understood,
            "unknown": unknown,
            "searches": searches,
            "understood_count": len(understood),
            "unknown_count": len(unknown),
            "search_count": len(searches),
            "all_understood": len(unknown) == 0 and len(understood) > 0,
            "summary": summary,
            "knowledge_stats": self.knowledge.stats(),
        }

    def perceive(
        self,
        image: Any,
        *,
        conf: float = 0.35,
        search_unknown: bool = True,
        modality_force: Optional[str] = None,
    ) -> dict[str, Any]:
        """Smart pipeline: modality → (thermal) → see → analyze → understand → smart."""
        modality = detect_modality(image, force=modality_force)
        mod_name = modality.get("modality") or "rgb"
        pipeline = list(modality.get("supported_pipeline") or ["see", "analyze", "understand"])

        thermal_result = None
        if mod_name in ("thermal", "infrared"):
            thermal_result = self.thermal.analyze(image, modality=mod_name)

        seen = self.see(image, conf=conf, modality=modality, thermal=thermal_result)
        analysis = self.analyze(seen, image=image)
        # Attach relative heat onto analyzed thermal objects
        if thermal_result:
            heat_by_box = {
                tuple(d.get("bbox_xyxy") or []): d.get("relative_heat")
                for d in (thermal_result.get("detections") or [])
            }
            for obj in analysis.get("objects") or []:
                key = tuple(obj.get("bbox_xyxy") or [])
                if key in heat_by_box:
                    obj["relative_heat"] = heat_by_box[key]

        understanding = self.understand_objects(
            analysis, image=image, search_unknown=search_unknown
        )
        smart = self.smart.reason(
            detections=seen.get("detections") or [],
            understood=understanding.get("understood") or [],
            unknown=understanding.get("unknown") or [],
            modality=modality,
            thermal=thermal_result,
            analysis=analysis,
        )

        text_parts = [understanding["summary"], smart.get("smart_summary") or ""]
        if thermal_result and thermal_result.get("summary"):
            text_parts.append(thermal_result["summary"])
        text = " ".join(p for p in text_parts if p).strip()

        return {
            "task": "understand_scene",
            "pipeline": pipeline,
            "modality": modality,
            "thermal": thermal_result,
            "see": seen,
            "analyze": analysis,
            "understand": understanding,
            "smart": smart,
            "text": text,
            "count": seen.get("count") or 0,
            "counts_by_label": seen.get("counts_by_label") or {},
            "detections": seen.get("detections") or [],
            "alert": bool((thermal_result or {}).get("alert") or smart.get("max_risk") in ("critical", "high")),
            "engine": seen.get("engine"),
            "llm": False,
            "human_level_agi": False,
            "ok": True,
        }

    def explain_object(self, label: str, *, image: Any = None) -> dict[str, Any]:
        """Understand a single named object; search if unknown."""
        fact = self.knowledge.lookup(label)
        searched = None
        if not fact:
            searched = self.search.search(
                label,
                context={"image_path": str(image) if image else None, "hints": [label]},
            )
            fact = searched.get("fact") if searched.get("found") else searched.get("fact")
        return {
            "task": "explain_object",
            "label": label,
            "understood": bool(fact and fact.get("understood")),
            "fact": fact,
            "search": searched,
            "text": _describe(fact, {"label": label}) if fact else f"Do not yet understand '{label}'.",
            "llm": False,
        }

    def _try_ocr(self, image: Any) -> Optional[dict[str, Any]]:
        try:
            from ..perception.ocr import OCRPipeline

            path = str(image) if isinstance(image, (str, Path)) else None
            if not path:
                return None
            out = OCRPipeline().recognize(path, context={})
            return {
                "text": out.get("text") or out.get("transcript") or "",
                "engine": out.get("engine") or "ocr",
            }
        except Exception:  # noqa: BLE001
            return None


def _size_class(area: float, detections: list[dict[str, Any]]) -> str:
    areas = []
    for d in detections:
        b = d.get("bbox_xyxy") or [0, 0, 0, 0]
        areas.append(max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1])))
    if not areas:
        return "unknown"
    mx = max(areas) or 1.0
    ratio = area / mx
    if ratio >= 0.6:
        return "large"
    if ratio >= 0.25:
        return "medium"
    return "small"


def _position_label(x1, y1, x2, y2, detections) -> str:
    # Relative to union of all boxes
    xs, ys = [], []
    for d in detections:
        b = d.get("bbox_xyxy") or [0, 0, 0, 0]
        xs.extend([float(b[0]), float(b[2])])
        ys.extend([float(b[1]), float(b[3])])
    if not xs:
        return "center"
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    hx = "left" if cx < minx + (maxx - minx) / 3 else "right" if cx > minx + 2 * (maxx - minx) / 3 else "center"
    vy = "top" if cy < miny + (maxy - miny) / 3 else "bottom" if cy > miny + 2 * (maxy - miny) / 3 else "middle"
    if hx == "center" and vy == "middle":
        return "center"
    return f"{vy}-{hx}"


def _spatial_relations(objects: list[dict[str, Any]]) -> list[dict[str, str]]:
    rels = []
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            if a["center"][0] < b["center"][0] - 5:
                rels.append({"subject": a["id"], "relation": "left_of", "object": b["id"], "labels": f"{a['label']}|{b['label']}"})
            if a["center"][1] < b["center"][1] - 5:
                rels.append({"subject": a["id"], "relation": "above", "object": b["id"], "labels": f"{a['label']}|{b['label']}"})
            if a.get("size_class") == "large" and b.get("size_class") == "small":
                rels.append({"subject": a["id"], "relation": "larger_than", "object": b["id"], "labels": f"{a['label']}|{b['label']}"})
    return rels[:20]


def _dominant(counts: dict[str, int]) -> Optional[str]:
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _describe(fact: Optional[dict[str, Any]], obj: dict[str, Any]) -> str:
    if not fact:
        return f"Unknown object '{obj.get('label')}'."
    label = fact.get("label") or obj.get("label")
    cat = fact.get("category") or "thing"
    summary = fact.get("summary") or fact.get("site_note") or fact.get("note") or ""
    uses = ", ".join(fact.get("uses") or []) or "unspecified"
    safety = fact.get("safety")
    bits = [f"{label} ({cat})"]
    if summary:
        bits.append(summary)
    bits.append(f"Typical uses: {uses}.")
    if safety:
        bits.append(f"Safety: {safety}.")
    if obj.get("size_class"):
        bits.append(f"In scene: {obj.get('size_class')} at {obj.get('position', 'unknown')}.")
    return " ".join(bits)


def _scene_summary(understood: list, unknown: list, analysis: dict) -> str:
    if not understood and not unknown:
        return "I see no objects in this image."
    parts = []
    parts.append(f"I see {analysis.get('object_count', 0)} object(s).")
    if understood:
        labels = sorted({u['label'] for u in understood})
        parts.append("Understood: " + ", ".join(labels) + ".")
        # brief on first few
        for u in understood[:3]:
            if u.get("understanding"):
                parts.append(u["understanding"])
    if unknown:
        labels = sorted({u['label'] for u in unknown})
        parts.append(
            "Not fully understood after analysis/search: "
            + ", ".join(labels)
            + ". Marked for review."
        )
    if analysis.get("relations"):
        parts.append(f"Spatial relations noted: {len(analysis['relations'])}.")
    return " ".join(parts)
