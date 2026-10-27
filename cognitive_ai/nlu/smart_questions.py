"""Smarter question understanding layer on top of QuestionAnalyzer.

Adds paraphrase normalization, intent frames, domain tagging, and answer policy
for the non-LLM brain.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .questions import QuestionAnalysis, QuestionAnalyzer


PARAPHRASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcould you (?:please )?tell me\b", re.I), "what is"),
    (re.compile(r"\bcan you (?:please )?explain\b", re.I), "explain"),
    (re.compile(r"\bi (?:want to|wanna) know\b", re.I), "what is"),
    (re.compile(r"\bdo you know (?:if|whether)\b", re.I), "is"),
    (re.compile(r"\bplease (?:search|look up|find)\b", re.I), "search"),
    (re.compile(r"\bmujhe batao\b", re.I), "tell me"),
    (re.compile(r"\bkya hota hai\b", re.I), "what is"),
    (re.compile(r"\bkya hai\b", re.I), "what is"),
]


DOMAIN_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("finderbay", ("invoice", "payment", "delivery", "material", "warehouse", "purchase order", "vendor", "project")),
    ("vision", ("image", "photo", "detect", "thermal", "camera", "ocr", "yolo")),
    ("science", ("gravity", "photosynthesis", "atom", "planet", "speed of light", "boiling")),
    ("computing", ("python", "sqlite", "algorithm", "internet", "ocr", "api")),
    ("web", ("search the web", "look up", "google", "online", "internet search", "wikipedia")),
]


@dataclass
class SmartQuestionFrame:
    raw: str
    normalized: str
    analysis: dict[str, Any]
    domain: str = "general"
    needs_search: bool = False
    needs_tools: bool = False
    answer_policy: str = "local_first"  # local_first|search|tool|clarify
    entities: list[str] = field(default_factory=list)
    paraphrase_applied: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartQuestionUnderstanding:
    """Non-LLM smarter Q→intent framing."""

    def __init__(self) -> None:
        self.analyzer = QuestionAnalyzer()

    def normalize(self, text: str) -> tuple[str, list[str]]:
        s = (text or "").strip()
        applied = []
        for pat, repl in PARAPHRASES:
            if pat.search(s):
                s2 = pat.sub(repl, s)
                if s2 != s:
                    applied.append(f"{pat.pattern}->{repl}")
                    s = s2
        s = re.sub(r"\s+", " ", s).strip()
        return s, applied

    def understand(self, text: str) -> SmartQuestionFrame:
        normalized, applied = self.normalize(text)
        qa = self.analyzer.analyze(normalized)
        # Also analyze raw for focus if normalized lost nouns
        if not qa.focus:
            qa2 = self.analyzer.analyze(text)
            if qa2.focus:
                qa.focus = qa2.focus

        domain = self._domain(normalized)
        entities = self._entities(normalized, qa)
        needs_search = self._needs_search(normalized, qa, domain)
        needs_tools = domain in ("finderbay", "vision") or any(
            n in normalized.lower()
            for n in ("invoice", "detect", "thermal", "shortage", "reserve")
        )
        # Prefer local common-sense / tools first; search is escalation or explicit
        if needs_tools and domain in ("finderbay", "vision"):
            policy = "tool"
        elif needs_search and (
            qa.ask_form == "search"
            or domain == "web"
            or re.search(r"\b(?:search|look up|google|wikipedia|online)\b", normalized, re.I)
        ):
            policy = "search"
        else:
            policy = "local_first"
        if qa.ask_form == "unknown" and not qa.is_question and not needs_search:
            policy = "clarify" if len(normalized.split()) < 3 else "local_first"

        conf = qa.confidence
        if applied:
            conf = min(1.0, conf + 0.05)
        if domain != "general":
            conf = min(1.0, conf + 0.05)

        return SmartQuestionFrame(
            raw=text,
            normalized=normalized,
            analysis=qa.to_dict(),
            domain=domain,
            needs_search=needs_search,
            needs_tools=needs_tools,
            answer_policy=policy,
            entities=entities,
            paraphrase_applied=applied,
            confidence=round(conf, 3),
        )

    def _domain(self, text: str) -> str:
        lower = text.lower()
        best, score = "general", 0
        for name, cues in DOMAIN_CUES:
            s = sum(1 for c in cues if c in lower)
            if s > score:
                best, score = name, s
        return best if score else "general"

    def _entities(self, text: str, qa: QuestionAnalysis) -> list[str]:
        ents: list[str] = []
        for f in qa.focus:
            ents.append(f)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text):
            ents.append(m.group(1))
        # quoted
        for m in re.finditer(r"[\"']([^\"']{2,40})[\"']", text):
            ents.append(m.group(1))
        # lowercase content nouns from focus already
        seen = set()
        out = []
        for e in ents:
            k = e.lower()
            if k not in seen:
                seen.add(k)
                out.append(e)
        return out[:8]

    def _needs_search(self, text: str, qa: QuestionAnalysis, domain: str) -> bool:
        lower = text.lower()
        if domain == "web":
            return True
        if re.search(r"\b(?:search|look up|google|wikipedia|online|internet)\b", lower):
            return True
        # Science/computing defines may escalate later after KB miss — not forced here
        if domain in ("science", "computing") and qa.ask_form in ("define", "causal"):
            return True
        return False
