"""Intent ranking with n-gram / keyword features — no LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IntentHit:
    intent: str
    score: float
    evidence: list[str] = field(default_factory=list)


# Pattern libraries: (intent, weighted keyword/phrase cues)
INTENT_CUES: list[tuple[str, list[tuple[str, float]]]] = [
    (
        "query_invoices",
        [
            ("invoice", 2.0),
            ("invoices", 2.2),
            ("unpaid", 1.5),
            ("overdue", 1.8),
            ("payment", 1.0),
            ("bill", 1.2),
            ("dikhao", 0.8),
            ("show", 0.5),
        ],
    ),
    (
        "query_deliveries",
        [
            ("delivery", 2.0),
            ("deliveries", 2.2),
            ("pending", 1.5),
            ("shipment", 1.5),
            ("received", 1.2),
            ("follow-up", 1.0),
            ("followup", 1.0),
        ],
    ),
    (
        "query_materials",
        [
            ("material", 2.0),
            ("stock", 1.8),
            ("inventory", 1.8),
            ("warehouse", 1.2),
            ("available", 0.8),
            ("kitna stock", 2.0),
        ],
    ),
    (
        "reserve_stock",
        [
            ("reserve", 2.5),
            ("book stock", 2.0),
            ("hold", 1.0),
            ("allocate", 1.5),
        ],
    ),
    (
        "manage_task",
        [
            ("task", 2.0),
            ("todo", 1.8),
            ("reminder", 1.5),
            ("deadline", 1.2),
            ("assign", 1.0),
            ("create task", 2.5),
            ("add task", 2.5),
            ("list tasks", 2.2),
            ("naya kaam", 2.0),
        ],
    ),
    (
        "document_qa",
        [
            ("document", 1.5),
            ("pdf", 1.5),
            ("scan", 1.2),
            ("ocr", 1.5),
            ("from this", 1.0),
            ("amount due", 2.0),
            ("invoice scan", 2.0),
            ("what is the", 0.8),
            ("field", 1.0),
        ],
    ),
    (
        "prepare_followup",
        [
            ("prepare", 1.5),
            ("draft", 1.8),
            ("follow-up", 2.0),
            ("followup", 2.0),
            ("do not send", 2.5),
            ("don't send", 2.5),
            ("mat bhejo", 2.5),
        ],
    ),
    (
        "clarify",
        [
            ("what do you mean", 2.0),
            ("clarify", 1.5),
            ("which one", 1.5),
        ],
    ),
    (
        "cancel",
        [
            ("cancel", 2.5),
            ("stop", 1.5),
            ("abort", 2.0),
            ("never mind", 2.0),
        ],
    ),
    (
        "correct",
        [
            ("make that", 2.0),
            ("actually", 1.2),
            ("correction", 2.0),
            ("i meant", 2.0),
            ("change to", 1.8),
        ],
    ),
    (
        "status",
        [
            ("status", 1.5),
            ("what can you", 1.5),
            ("capabilities", 2.0),
            ("help", 1.0),
        ],
    ),
    (
        "query_payments",
        [
            ("payment", 2.0),
            ("payments", 2.2),
            ("paid amount", 1.5),
            ("settlement", 1.8),
            ("neft", 1.2),
        ],
    ),
    (
        "record_payment",
        [
            ("record payment", 4.5),
            ("post payment", 4.5),
            ("mark paid", 3.5),
            ("receive payment", 3.5),
            ("log payment", 4.0),
            ("enter payment", 4.0),
            ("payment of", 2.0),
        ],
    ),
    (
        "query_contracts",
        [
            ("contract", 2.0),
            ("contracts", 2.2),
            ("agreement", 1.8),
            ("supplier agreement", 2.0),
        ],
    ),
    (
        "login",
        [
            ("login", 2.5),
            ("log in", 2.5),
            ("sign in", 2.0),
            ("authenticate", 2.0),
        ],
    ),
    (
        "query_alerts",
        [
            ("alert", 2.0),
            ("alerts", 2.2),
            ("overdue", 1.5),
            ("low stock", 2.5),
            ("expiring", 1.8),
            ("what needs attention", 2.5),
            ("proactive", 1.5),
            ("monitor", 1.2),
        ],
    ),
    (
        "query_purchase_orders",
        [
            ("purchase order", 2.5),
            ("purchase orders", 2.7),
            ("po list", 2.2),
            ("show pos", 2.0),
            ("open pos", 2.0),
        ],
    ),
    (
        "create_purchase_order",
        [
            ("create purchase order", 4.5),
            ("create po", 4.0),
            ("new purchase order", 4.0),
            ("raise po", 3.5),
            ("raise purchase order", 4.0),
        ],
    ),
    (
        "query_vendors",
        [
            ("vendor", 2.0),
            ("vendors", 2.2),
            ("supplier list", 2.0),
            ("show suppliers", 2.0),
        ],
    ),
    (
        "export_report",
        [
            ("export report", 3.5),
            ("export csv", 3.5),
            ("download report", 3.0),
            ("generate report", 2.5),
            ("csv report", 3.0),
        ],
    ),
    (
        "detect_objects",
        [
            ("detect objects", 4.5),
            ("object detection", 4.0),
            ("what objects", 3.0),
            ("yolo detect", 4.0),
            ("run vision", 3.5),
            ("analyze image", 3.5),
            ("computer vision", 3.5),
        ],
    ),
    (
        "detect_vehicles",
        [
            ("detect vehicles", 4.5),
            ("vehicle detection", 4.0),
            ("count cars", 3.5),
            ("count vehicles", 4.0),
            ("track vehicles", 3.5),
            ("cars in image", 3.0),
            ("traffic detection", 3.5),
        ],
    ),
    (
        "count_people",
        [
            ("count people", 4.5),
            ("people counter", 4.0),
            ("how many people", 4.0),
            ("person count", 3.5),
            ("crowd count", 3.5),
            ("count persons", 4.0),
        ],
    ),
    (
        "detect_fire",
        [
            ("detect fire", 4.5),
            ("fire detection", 4.0),
            ("smoke detection", 4.0),
            ("is there fire", 3.5),
            ("fire alert", 3.5),
        ],
    ),
    (
        "scan_document_vision",
        [
            ("document scanner", 4.5),
            ("scan this page", 4.0),
            ("perspective scan", 4.0),
            ("warp document", 4.0),
            ("scan this document with opencv", 5.0),
            ("cv document scan", 4.5),
            ("deskew document", 4.0),
            ("scan page edges", 4.0),
        ],
    ),
    (
        "vision_status",
        [
            ("vision status", 4.5),
            ("yolo status", 4.0),
            ("vision projects", 4.0),
            ("computer vision status", 4.5),
            ("list vision", 3.5),
            ("cv capabilities", 3.5),
        ],
    ),
    (
        "ppe_compliance",
        [
            ("ppe compliance", 4.5),
            ("hard hat", 3.0),
            ("safety vest", 3.0),
            ("ppe check", 4.0),
            ("safety compliance", 3.5),
        ],
    ),
    (
        "estimate_queue",
        [
            ("estimate queue", 4.5),
            ("queue wait", 4.0),
            ("wait time", 3.0),
            ("queue length", 3.5),
        ],
    ),
]


def rank_intents(text: str, top_k: int = 3) -> list[IntentHit]:
    lower = text.lower()
    hits: list[IntentHit] = []
    for intent, cues in INTENT_CUES:
        score = 0.0
        evidence: list[str] = []
        for cue, weight in cues:
            if cue in lower:
                score += weight
                evidence.append(cue)
        if score > 0:
            hits.append(IntentHit(intent=intent, score=score, evidence=evidence))
    # Prefer write actions over read queries when explicit record/post verbs appear
    if re.search(r"\b(record|post|log|enter)\s+payment\b|\bmark\s+paid\b", lower):
        for h in hits:
            if h.intent == "record_payment":
                h.score += 3.0
                h.evidence.append("write_payment_boost")
            elif h.intent in {"query_invoices", "query_payments"}:
                h.score -= 1.5
                h.evidence.append("write_payment_penalty")
    hits.sort(key=lambda h: -h.score)
    return hits[:top_k]


def best_intent(text: str, threshold: float = 1.0) -> Optional[IntentHit]:
    ranked = rank_intents(text, top_k=1)
    if not ranked or ranked[0].score < threshold:
        return None
    return ranked[0]


PROJECT_RE = re.compile(
    r"\b(?:project|proj)\s*([A-Za-z0-9_-]+)\b|"
    r"\b(?:for|of)\s+project\s+([A-Za-z0-9_-]+)\b|"
    r"\bproject\s+([A-Za-z])\b",
    re.I,
)


def extract_project_id(text: str) -> Optional[str]:
    m = PROJECT_RE.search(text)
    if not m:
        # Hinglish: "Project A" already covered; try bare "Project A"
        m2 = re.search(r"\b([Pp]roject)\s+([A-Za-z0-9_-]+)\b", text)
        if m2:
            return f"project_{m2.group(2).lower()}"
        return None
    token = next(g for g in m.groups() if g)
    return f"project_{token.lower()}"
