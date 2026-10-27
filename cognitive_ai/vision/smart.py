"""Smarter vision reasoning — scene type, risks, stories, modality-aware advice."""

from __future__ import annotations

from typing import Any, Optional


SCENE_RULES: list[tuple[str, tuple[str, ...], str]] = [
    ("construction_site", ("person", "truck", "hardhat", "helmet", "steel", "cement", "crane"), "Active construction / civil site cues"),
    ("traffic", ("car", "truck", "bus", "motorcycle", "traffic light"), "Road / traffic scene"),
    ("office", ("laptop", "keyboard", "mouse", "chair", "dining table", "tv"), "Indoor workplace / office"),
    ("crowd", ("person",), "People-dense scene"),
    ("hazard_fire", ("fire", "smoke", "thermal_hotspot"), "Possible fire / heat hazard"),
    ("warehouse", ("truck", "suitcase", "backpack", "person"), "Logistics / storage-like mix"),
    ("domestic", ("couch", "bed", "tv", "cat", "dog", "bottle", "cup"), "Indoor domestic scene"),
    ("thermal_inspection", ("thermal_hotspot", "person_thermal", "cold_region"), "Thermal inspection view"),
]


RISK_RULES: list[tuple[str, str, str]] = [
    ("fire", "critical", "Fire/smoke visible — treat as emergency until cleared"),
    ("smoke", "critical", "Smoke present — ventilate and investigate source"),
    ("thermal_hotspot", "high", "Hot region on thermal — check equipment/overheating"),
    ("person_thermal", "info", "Warm human-shaped region on thermal"),
    ("knife", "medium", "Sharp tool visible — handle/safety awareness"),
    ("scissors", "low", "Cutting tool visible"),
]


class SmartVisionReasoner:
    """Compose a smarter narrative from detections + modality + thermal."""

    def reason(
        self,
        *,
        detections: list[dict[str, Any]],
        understood: list[dict[str, Any]],
        unknown: list[dict[str, Any]],
        modality: dict[str, Any],
        thermal: Optional[dict[str, Any]] = None,
        analysis: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        labels = [str(d.get("label") or "") for d in detections]
        label_set = set(labels)
        counts: dict[str, int] = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1

        scene = self._classify_scene(label_set, counts, modality)
        risks = self._risks(label_set, thermal)
        insights = self._insights(understood, unknown, modality, thermal, analysis)
        story = self._story(scene, risks, insights, modality, thermal)
        questions = self._followups(unknown, modality, thermal)

        return {
            "scene_type": scene["type"],
            "scene_confidence": scene["confidence"],
            "scene_evidence": scene["evidence"],
            "risks": risks,
            "max_risk": risks[0]["level"] if risks else "none",
            "insights": insights,
            "followup_questions": questions,
            "smart_summary": story,
            "modality": modality.get("modality"),
            "recommendations": self._recommendations(scene, risks, modality, thermal),
        }

    def _classify_scene(self, labels: set[str], counts: dict[str, int], modality: dict) -> dict[str, Any]:
        mod = modality.get("modality") or "rgb"
        if mod == "thermal":
            return {
                "type": "thermal_inspection",
                "confidence": 0.9,
                "evidence": ["modality:thermal"],
                "description": "Thermal inspection view",
            }
        best = ("general", 0.0, (), "General visual scene")
        for name, keys, desc in SCENE_RULES:
            hit = tuple(sorted(labels & set(keys)))
            if not hit:
                continue
            score = len(hit) / max(1, len(keys))
            if name == "crowd" and counts.get("person", 0) >= 3:
                score = max(score, 0.8)
            if score > best[1]:
                best = (name, score, hit, desc)
        return {
            "type": best[0],
            "confidence": round(min(0.95, 0.4 + best[1]), 3),
            "evidence": list(best[2]),
            "description": best[3],
        }

    def _risks(self, labels: set[str], thermal: Optional[dict]) -> list[dict[str, Any]]:
        risks = []
        for lab, level, msg in RISK_RULES:
            if lab in labels:
                risks.append({"label": lab, "level": level, "message": msg})
        if thermal and thermal.get("alert"):
            risks.append(
                {
                    "label": "thermal_alert",
                    "level": "high",
                    "message": thermal.get("summary") or "Thermal alert raised",
                }
            )
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        risks.sort(key=lambda r: order.get(r["level"], 9))
        # dedupe by label
        seen = set()
        out = []
        for r in risks:
            if r["label"] in seen:
                continue
            seen.add(r["label"])
            out.append(r)
        return out

    def _insights(
        self,
        understood: list,
        unknown: list,
        modality: dict,
        thermal: Optional[dict],
        analysis: Optional[dict],
    ) -> list[str]:
        tips = []
        mod = modality.get("modality") or "rgb"
        tips.append(f"Image modality: {mod} (confidence={modality.get('confidence')}).")
        if understood:
            cats = sorted({(u.get("knowledge") or {}).get("category") for u in understood if u.get("knowledge")})
            cats = [c for c in cats if c]
            if cats:
                tips.append("Categories present: " + ", ".join(cats) + ".")
        if unknown:
            tips.append(f"{len(unknown)} object(s) still uncertain after search.")
        if thermal:
            rel = (thermal.get("map") or {}).get("relative") or {}
            if rel:
                tips.append(
                    f"Thermal relative heat mean={rel.get('mean')}, "
                    f"hot≥{rel.get('hot_threshold')} (not °C unless calibrated)."
                )
            tips.append(f"Hotspots={len(thermal.get('hotspots') or [])}, coldspots={len(thermal.get('coldspots') or [])}.")
        if analysis and analysis.get("relations"):
            tips.append(f"{len(analysis['relations'])} spatial relation(s) inferred.")
        if mod == "night_vision":
            tips.append("Night-vision: expect lower color fidelity; rely on shape/motion cues.")
        if mod == "xray":
            tips.append("X-ray: treat dense bright regions as denser materials; not a medical diagnosis.")
        if mod == "document":
            tips.append("Document modality: prefer OCR + field extraction over object detection.")
        return tips

    def _story(
        self,
        scene: dict,
        risks: list,
        insights: list,
        modality: dict,
        thermal: Optional[dict],
    ) -> str:
        parts = [
            f"Smart read: this looks like a {scene['type'].replace('_', ' ')} "
            f"(confidence={scene['confidence']})."
        ]
        if scene.get("description"):
            parts.append(scene["description"] + ".")
        if risks:
            top = risks[0]
            parts.append(f"Top risk [{top['level']}]: {top['message']}.")
        elif modality.get("modality") == "thermal":
            parts.append("No critical thermal alert in current thresholds.")
        if insights:
            parts.append(insights[0])
            if len(insights) > 1:
                parts.append(insights[1])
        return " ".join(parts)

    def _followups(self, unknown: list, modality: dict, thermal: Optional[dict]) -> list[str]:
        qs = []
        for u in unknown[:3]:
            qs.append(f"Can you confirm what '{u.get('label')}' is?")
        if (modality.get("modality") == "thermal") and not thermal.get("calibrated_celsius"):
            qs.append("Do you have camera calibration so heat can be read in °C?")
        if modality.get("modality") in ("xray", "ultrasound"):
            qs.append("Is this for industrial inspection or medical context? (system will not diagnose).")
        return qs

    def _recommendations(
        self,
        scene: dict,
        risks: list,
        modality: dict,
        thermal: Optional[dict],
    ) -> list[str]:
        recs = []
        if any(r["level"] in ("critical", "high") for r in risks):
            recs.append("Prioritize safety: verify the hazard on-site before continuing work.")
        if scene["type"] == "construction_site":
            recs.append("Cross-check PPE and material shortage workflows in Finderbay.")
        if modality.get("modality") == "thermal":
            recs.append("Compare hotspot locations against equipment layout / last inspection.")
        if scene["type"] == "traffic":
            recs.append("If needed, run detect_vehicles for a vehicle-only count.")
        if not recs:
            recs.append("Scene understood at a basic level; ask to explain a specific object for more detail.")
        return recs
