"""End-to-end non-LLM language understanding pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .intents import best_intent, extract_project_id, rank_intents
from .normalize import detect_language, normalize_text
from .questions import QuestionAnalysis, QuestionAnalyzer
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
    question: Optional[dict[str, Any]] = None
    clauses: list[str] = field(default_factory=list)

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
            "question": self.question,
            "clauses": list(self.clauses),
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
        self.questions = QuestionAnalyzer()

    def parse(self, text: str, dialogue_params: Optional[dict] = None) -> SemanticFrame:
        original = text
        normalized = normalize_text(text)
        language = detect_language(normalized)
        qinfo: QuestionAnalysis = self.questions.analyze(normalized)
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
        if qinfo.is_question:
            intent_type = "question"
        if qinfo.is_hypothetical or re_search_hypothetical(normalized):
            intent_type = "hypothetical"

        # Map WH / ask-form focus onto Finderbay ops when intents are weak
        if hit is None and qinfo.is_question:
            mapped = self._map_question_to_op(normalized, qinfo)
            if mapped:
                hit_intent = mapped
                ranked_scores = [{"intent": mapped, "score": 2.0, "evidence": ["question_map"]}]
            else:
                hit_intent = None
                ranked_scores = [{"intent": h.intent, "score": h.score} for h in ranked]
        else:
            hit_intent = hit.intent if hit else None
            ranked_scores = [
                {"intent": h.intent, "score": h.score, "evidence": h.evidence} for h in ranked
            ]

        if hit_intent is None:
            frame = SemanticFrame(
                operation="unsupported",
                intent_scores=ranked_scores,
                language=language,
                original_text=original,
                normalized_text=normalized,
                supported=False,
                confidence=qinfo.confidence * 0.5,
                entities=entities,
                intent_type=intent_type,
                question=qinfo.to_dict(),
                clauses=list(qinfo.clauses),
                needs_clarification=["I could not map this request to a supported capability."],
            )
            return frame

        operation = hit_intent
        intents = {h.intent for h in ranked} | {operation}
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

        # Question-form refinements for domain nouns
        mapped = self._map_question_to_op(normalized, qinfo)
        if mapped and (hit is None or (hit and hit.score < 2.5 and mapped != operation)):
            # Only override when question map is clearer for WH questions
            if qinfo.wh_type in ("how_many", "how_much", "which", "what") and mapped in self.SUPPORTED_OPS:
                if any(n in normalized.lower() for n in qinfo.focus):
                    operation = mapped

        needs: list[str] = []
        if slots.unknown_policies.get("time") == "clarify_kal":
            needs.append(
                "'kal' can mean yesterday or tomorrow — please clarify which you mean."
            )
        if qinfo.is_multi_clause and len(qinfo.clauses) > 2:
            needs.append(
                "That looks like several questions at once — I will answer the primary clause first; "
                "ask the others separately if needed."
            )

        conditions = [c.to_dict() for c in slots.conditions]
        # Attach aggregation / ask-form as explicit conditions for downstream reasoners
        if qinfo.aggregation:
            conditions.append({"field": "aggregation", "operator": "eq", "value": qinfo.aggregation})
        if qinfo.ask_form in ("count", "sum", "exists", "list"):
            conditions.append({"field": "ask_form", "operator": "eq", "value": qinfo.ask_form})

        # Output shape from question analysis
        output = slots.output_hint or "table_with_sources"
        if qinfo.expected_answer == "number":
            output = "numeric_answer"
        elif qinfo.expected_answer == "boolean":
            output = "yes_no"
        elif qinfo.expected_answer == "explanation":
            output = "explanation"

        base_conf = min(1.0, (hit.score / 5.0) if hit else 0.4)
        confidence = max(base_conf, qinfo.confidence * 0.7) if qinfo.is_question else base_conf
        frame = SemanticFrame(
            operation=operation,
            intent_scores=ranked_scores,
            project_id=slots.project_id or project_id,
            conditions=conditions,
            unknown_policies=slots.unknown_policies,
            prohibitions=slots.prohibitions,
            exclusions=slots.exclusions,
            entities=entities,
            output=output,
            language=language,
            original_text=original,
            normalized_text=normalized,
            needs_clarification=needs,
            confidence=confidence,
            supported=operation in self.SUPPORTED_OPS,
            intent_type=intent_type,
            question=qinfo.to_dict(),
            clauses=list(qinfo.clauses),
        )
        return frame

    def _map_question_to_op(self, text: str, qinfo: QuestionAnalysis) -> Optional[str]:
        lower = text.lower()
        focus = " ".join(qinfo.focus).lower()
        blob = f"{lower} {focus}"
        mapping = [
            (("invoice", "invoices", "bill"), "query_invoices"),
            (("delivery", "deliveries", "shipment"), "query_deliveries"),
            (("payment", "payments", "settlement"), "query_payments"),
            (("contract", "contracts", "agreement"), "query_contracts"),
            (("material", "materials", "stock", "inventory"), "query_materials"),
            (("task", "tasks", "todo"), "manage_task"),
            (("alert", "alerts", "attention"), "query_alerts"),
            (("vendor", "vendors", "supplier"), "query_vendors"),
            (("purchase order", "purchase orders", " po "), "query_purchase_orders"),
        ]
        for keys, op in mapping:
            if any(k in blob for k in keys):
                return op
        return None


def re_search_hypothetical(text: str) -> bool:
    import re

    return bool(re.search(r"\b(?:what if|suppose|agar)\b", text, flags=re.I))
