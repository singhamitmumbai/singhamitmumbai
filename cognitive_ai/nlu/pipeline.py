"""End-to-end non-LLM language understanding pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .intents import best_intent, extract_project_id, rank_intents
from .normalize import detect_language, normalize_text
from .slots import extract_slots


@dataclass
class SemanticFrame:
    """Inspectable meaning representation produced from language."""

    operation: str
    intent_scores: list[dict[str, Any]] = field(default_factory=list)
    project_id: Optional[str] = None
    conditions: list[dict[str, Any]] = field(default_factory=list)
    unknown_policies: dict[str, str] = field(default_factory=dict)
    prohibitions: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    output: str = "structured"
    language: str = "en"
    original_text: str = ""
    normalized_text: str = ""
    needs_clarification: list[str] = field(default_factory=list)
    confidence: float = 0.0
    supported: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "intent_scores": self.intent_scores,
            "project_id": self.project_id,
            "conditions": self.conditions,
            "unknown_policies": self.unknown_policies,
            "prohibitions": self.prohibitions,
            "exclusions": self.exclusions,
            "output": self.output,
            "language": self.language,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "needs_clarification": self.needs_clarification,
            "confidence": self.confidence,
            "supported": self.supported,
        }


class LanguageUnderstanding:
    """Stages: preserve → detect language → normalize → extract → validate."""

    SUPPORTED_OPS = {
        "query_invoices",
        "query_deliveries",
        "query_materials",
        "manage_task",
        "document_qa",
        "prepare_followup",
        "status",
        "cancel",
        "correct",
        "clarify",
    }

    def parse(self, text: str, dialogue_params: Optional[dict] = None) -> SemanticFrame:
        original = text
        normalized = normalize_text(text)
        language = detect_language(normalized)
        ranked = rank_intents(normalized)
        hit = best_intent(normalized, threshold=1.0)
        project_id = extract_project_id(normalized)
        if dialogue_params and not project_id:
            project_id = dialogue_params.get("project_id")

        slots = extract_slots(normalized, project_id=project_id)

        if hit is None:
            frame = SemanticFrame(
                operation="unsupported",
                intent_scores=[{"intent": h.intent, "score": h.score} for h in ranked],
                language=language,
                original_text=original,
                normalized_text=normalized,
                supported=False,
                confidence=0.0,
                needs_clarification=["I could not map this request to a supported capability."],
            )
            return frame

        operation = hit.intent
        # Compose prepare_followup with deliveries when both cues present
        intents = {h.intent for h in ranked}
        if "prepare_followup" in intents and "query_deliveries" in intents:
            operation = "prepare_followup"
        if "query_deliveries" in intents and "prepare_followup" not in intents:
            if "pending" in normalized.lower() and "project" in normalized.lower():
                operation = "query_deliveries"

        needs: list[str] = []
        if slots.unknown_policies.get("time") == "clarify_kal":
            needs.append(
                "'kal' can mean yesterday or tomorrow — please clarify which you mean."
            )

        # Merge dialogue confirmed parameters
        conditions = [c.to_dict() for c in slots.conditions]
        if dialogue_params:
            for k, v in dialogue_params.items():
                if k.endswith("_id") or k in ("time_window",):
                    continue
                # corrections already applied upstream

        confidence = min(1.0, hit.score / 5.0)
        frame = SemanticFrame(
            operation=operation,
            intent_scores=[{"intent": h.intent, "score": h.score, "evidence": h.evidence} for h in ranked],
            project_id=slots.project_id or project_id,
            conditions=conditions,
            unknown_policies=slots.unknown_policies,
            prohibitions=slots.prohibitions,
            exclusions=slots.exclusions,
            output=slots.output_hint or "table_with_sources",
            language=language,
            original_text=original,
            normalized_text=normalized,
            needs_clarification=needs,
            confidence=confidence,
            supported=operation in self.SUPPORTED_OPS,
        )
        return frame
