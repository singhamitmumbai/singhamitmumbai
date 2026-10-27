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
    entities: list[dict[str, Any]] = field(default_factory=list)
    output: str = "structured"
    language: str = "en"
    original_text: str = ""
    normalized_text: str = ""
    needs_clarification: list[str] = field(default_factory=list)
    confidence: float = 0.0
    supported: bool = True
    intent_type: str = "request"  # request|statement|question|quotation|hypothetical

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "intent_scores": self.intent_scores,
            "project_id": self.project_id,
            "conditions": self.conditions,
            "unknown_policies": self.unknown_policies,
            "prohibitions": self.prohibitions,
            "exclusions": self.exclusions,
            "entities": self.entities,
            "output": self.output,
            "language": self.language,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "needs_clarification": self.needs_clarification,
            "confidence": self.confidence,
            "supported": self.supported,
            "intent_type": self.intent_type,
        }


class LanguageUnderstanding:
    """Stages: preserve → detect language → normalize → extract → validate."""

    SUPPORTED_OPS = {
        "query_invoices",
        "query_deliveries",
        "query_materials",
        "reserve_stock",
        "manage_task",
        "document_qa",
        "prepare_followup",
        "learn_alias",
        "query_payments",
        "record_payment",
        "query_contracts",
        "login",
        "query_alerts",
        "query_purchase_orders",
        "create_purchase_order",
        "query_vendors",
        "export_report",
        "status",
        "cancel",
        "correct",
        "clarify",
    }

    def __init__(self, entity_linker=None) -> None:
        self.entity_linker = entity_linker

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
        entities: list[dict[str, Any]] = []
        if self.entity_linker is not None:
            entities = [e.__dict__ for e in self.entity_linker.link(normalized)]

        intent_type = "request"
        if normalized.strip().endswith("?"):
            intent_type = "question"
        if re_search_hypothetical(normalized):
            intent_type = "hypothetical"

        if hit is None:
            frame = SemanticFrame(
                operation="unsupported",
                intent_scores=[{"intent": h.intent, "score": h.score} for h in ranked],
                language=language,
                original_text=original,
                normalized_text=normalized,
                supported=False,
                confidence=0.0,
                entities=entities,
                intent_type=intent_type,
                needs_clarification=["I could not map this request to a supported capability."],
            )
            return frame

        operation = hit.intent
        intents = {h.intent for h in ranked}
        if "prepare_followup" in intents and "query_deliveries" in intents:
            operation = "prepare_followup"
        if "reserve_stock" in intents:
            operation = "reserve_stock"
        if "manage_task" in intents and (
            "create task" in normalized.lower()
            or "add task" in normalized.lower()
            or "list tasks" in normalized.lower()
            or "naya kaam" in normalized.lower()
        ):
            operation = "manage_task"
        if "document_qa" in intents and any(
            x in normalized.lower() for x in ("amount due", "invoice scan", "document", "ocr")
        ):
            operation = "document_qa"
        if "create_purchase_order" in intents:
            operation = "create_purchase_order"
        if "export_report" in intents:
            operation = "export_report"
        if "query_alerts" in intents and any(
            x in normalized.lower()
            for x in ("alert", "alerts", "low stock", "needs attention", "what needs attention", "expiring contract")
        ):
            # Prefer invoice query when "overdue invoices" is explicit
            if not (("invoice" in normalized.lower() or "invoices" in normalized.lower()) and "overdue" in normalized.lower()):
                operation = "query_alerts"

        needs: list[str] = []
        if slots.unknown_policies.get("time") == "clarify_kal":
            needs.append(
                "'kal' can mean yesterday or tomorrow — please clarify which you mean."
            )

        conditions = [c.to_dict() for c in slots.conditions]
        confidence = min(1.0, hit.score / 5.0)
        frame = SemanticFrame(
            operation=operation,
            intent_scores=[{"intent": h.intent, "score": h.score, "evidence": h.evidence} for h in ranked],
            project_id=slots.project_id or project_id,
            conditions=conditions,
            unknown_policies=slots.unknown_policies,
            prohibitions=slots.prohibitions,
            exclusions=slots.exclusions,
            entities=entities,
            output=slots.output_hint or "table_with_sources",
            language=language,
            original_text=original,
            normalized_text=normalized,
            needs_clarification=needs,
            confidence=confidence,
            supported=operation in self.SUPPORTED_OPS,
            intent_type=intent_type,
        )
        return frame


def re_search_hypothetical(text: str) -> bool:
    import re

    return bool(re.search(r"\b(?:what if|suppose|agar)\b", text, flags=re.I))
