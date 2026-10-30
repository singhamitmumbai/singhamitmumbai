"""Non-LLM intent + slot extraction (blueprint §5).

Baseline: token/char n-gram features + linear scoring over a defined command set.
No general conversational LLM in the decision path.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


INTENT_EXAMPLES: dict[str, list[str]] = {
    "inventory.shortages": [
        "prepare material shortage report",
        "tomorrow material shortage for site",
        "show material balance shortages",
        "shortage report selected site",
        "material shortage kal",
    ],
    "files.search": [
        "find the latest project report",
        "search files for invoice",
        "locate document project a",
    ],
    "documents.extract": [
        "read this invoice",
        "extract fields from pdf",
        "parse document",
    ],
    "schedule.create": [
        "remind me tomorrow morning",
        "kal subah yaad dilana",
        "schedule tender review at 10",
        "yaad dilana tender review",
    ],
    "jobs.cancel": [
        "stop this task",
        "cancel the job",
        "abort this",
    ],
    "memory.query": [
        "what do you remember",
        "recall project deadline",
        "memory query",
    ],
    "math.calculate": [
        "calculate 12 + 30",
        "compute 5 * 8",
        "how much is 100 - 40",
    ],
    "system.health": [
        "system health",
        "are you ready",
        "status check",
    ],
    "web.search": [
        "search the web for",
        "look up online",
        "fetch url",
    ],
}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    words = re.findall(r"[a-z0-9_]+", text)
    chars = [text[i : i + 3] for i in range(max(0, len(text) - 2))]
    return words + [f"c:{c}" for c in chars if c.strip()]


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    n = max(1, len(tokens))
    return {t: (c / n) * idf.get(t, 1.0) for t, c in tf.items()}


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _norm(a: dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in a.values())) or 1.0


@dataclass
class IntentResult:
    intent: str
    confidence: float
    slots: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    raw_text: str = ""
    supported: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "slots": self.slots,
            "missing": self.missing,
            "raw_text": self.raw_text,
            "supported": self.supported,
        }


class IntentClassifier:
    """Character + word TF-IDF cosine classifier over supported intents."""

    def __init__(self, examples: Optional[dict[str, list[str]]] = None) -> None:
        self.examples = examples or INTENT_EXAMPLES
        self._fit()

    def _fit(self) -> None:
        docs = [(intent, ex) for intent, exs in self.examples.items() for ex in exs]
        df: Counter[str] = Counter()
        tokenized: list[tuple[str, list[str]]] = []
        for intent, ex in docs:
            toks = _tokenize(ex)
            tokenized.append((intent, toks))
            for t in set(toks):
                df[t] += 1
        n = len(docs)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        # centroid per intent
        self.centroids: dict[str, dict[str, float]] = {}
        buckets: dict[str, list[dict[str, float]]] = {}
        for intent, toks in tokenized:
            buckets.setdefault(intent, []).append(_tfidf_vector(toks, self.idf))
        for intent, vecs in buckets.items():
            acc: Counter[str] = Counter()
            for v in vecs:
                acc.update(v)
            self.centroids[intent] = {k: v / len(vecs) for k, v in acc.items()}

    def predict(self, text: str, *, threshold: float = 0.12) -> IntentResult:
        toks = _tokenize(text)
        vec = _tfidf_vector(toks, self.idf)
        best_intent = "unsupported"
        best_score = -1.0
        for intent, centroid in self.centroids.items():
            score = _dot(vec, centroid) / (_norm(vec) * _norm(centroid))
            if score > best_score:
                best_score = score
                best_intent = intent
        if best_score < threshold:
            return IntentResult(
                intent="unsupported",
                confidence=max(0.0, best_score),
                raw_text=text,
                supported=False,
            )
        slots, missing = extract_slots(text, best_intent)
        return IntentResult(
            intent=best_intent,
            confidence=best_score,
            slots=slots,
            missing=missing,
            raw_text=text,
            supported=True,
        )


def extract_slots(text: str, intent: str) -> tuple[dict[str, Any], list[str]]:
    slots: dict[str, Any] = {}
    missing: list[str] = []
    if re.search(r"\btomorrow\b|\bkal\b", text, re.I):
        slots["relative_date"] = "tomorrow"
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(baje|am|pm)?\b", text, re.I)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = (m.group(3) or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        slots["local_time"] = f"{hour:02d}:{minute:02d}"
        slots["timezone"] = "Asia/Kolkata"
        if ampm in ("", "baje") and "subah" not in text.lower() and "am" not in text.lower() and "pm" not in text.lower():
            slots["ampm_ambiguous"] = True
    m = re.search(r"(?:for|site|project)\s+([A-Za-z0-9_\- ]+?)(?:\s*$|\s+report|\s+tomorrow)", text, re.I)
    if m:
        slots["site"] = m.group(1).strip()
    elif re.search(r"selected site|this site", text, re.I):
        slots["site"] = "selected_site"
    elif intent == "inventory.shortages":
        missing.append("site")

    if intent == "schedule.create":
        m = re.search(r"(?:remind me(?: to)?)\s+(.+?)(?:\s+kal|\s+at|\s+tomorrow|\s+\d|$)", text, re.I)
        if m:
            slots["task"] = m.group(1).strip(" .,")
        else:
            m2 = re.search(r"(.+?)\s+yaad dilana", text, re.I)
            slots["task"] = m2.group(1).strip() if m2 else "reminder"
            # strip leading kal/subah/time from hinglish task
            slots["task"] = re.sub(
                r"^(kal|subah|\d{1,2}(?::\d{2})?\s*(?:baje|am|pm)?)\s+",
                "",
                slots["task"],
                flags=re.I,
            ).strip() or "reminder"
        when_parts = []
        if slots.get("relative_date"):
            when_parts.append(slots["relative_date"])
        if slots.get("local_time"):
            when_parts.append(slots["local_time"])
        slots["when"] = " ".join(when_parts) or "unspecified"

    if intent == "math.calculate":
        m = re.search(r"([\d\.\s\+\-\*/\(\)]+)", text)
        if m:
            slots["expression"] = m.group(1).strip()
        else:
            missing.append("expression")
    if intent == "files.search":
        slots["query"] = text
    if intent == "documents.extract":
        m = re.search(r"(/[\w\-./]+\.\w+|[\w\-]+\.(?:pdf|docx|xlsx))", text, re.I)
        slots["path"] = m.group(1) if m else "document.pdf"
    if intent == "jobs.cancel":
        slots["job_id"] = "current"
    if intent == "memory.query":
        slots["query"] = text
    if intent == "web.search":
        slots["query"] = re.sub(r"^(search the web for|look up online|fetch)\s*", "", text, flags=re.I).strip()
    return slots, missing
