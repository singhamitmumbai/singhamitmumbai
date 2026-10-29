"""Universal input gateway — every utterance becomes a typed interpretation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


HINGLISH_GREET = re.compile(
    r"^(?:hii+|hi+|hello|hey|yo|namaste|namaskar|kaise\s+ho|kaisa\s+hai|kya\s+haal|"
    r"good\s+(?:morning|afternoon|evening)|sup)\b",
    re.I,
)
HINGLISH_CAP = re.compile(
    r"(?:tum\s+kya\s+kar\s+sakte|kya\s+kar\s+sakte\s+ho|what\s+can\s+you\s+do|"
    r"who\s+are\s+you|what\s+are\s+you|capabilities|status)",
    re.I,
)


@dataclass
class InputInterpretation:
    raw: str
    normalized: str
    language: str = "en"  # en|hi|hinglish|mixed
    dialogue_act: str = "unknown"
    intent: str = "conversation"
    entities: list[str] = field(default_factory=list)
    confidence: float = 0.5
    needs_clarification: bool = False
    objective_hint: str = ""
    route_hint: str = "mother"
    notes: list[str] = field(default_factory=list)
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InputGateway:
    """Normalize + classify input without requiring exact CLI commands."""

    COMMAND_TOKENS = {
        "help",
        "quit",
        "exit",
        "q",
        "domain",
        "assert",
        "rule",
        "goal",
        "think",
        "reason",
        "prove",
        "ask",
        "why",
        "query",
        "dump",
        "save",
        "load",
        "lessons",
        "thoughts",
        "status",
        "diagnose",
        "benchmark",
        "mother",
    }

    def interpret(self, text: str, *, context: Optional[dict[str, Any]] = None) -> InputInterpretation:
        raw = (text or "").strip()
        normalized = re.sub(r"\s+", " ", raw)
        lower = normalized.lower()
        lang = self._language(lower)
        act, intent, conf, notes = self._classify(lower, normalized)
        entities = self._entities(normalized)
        objective = ""
        if intent in ("task", "planning", "coding", "inspect"):
            objective = normalized
        elif intent == "question":
            objective = f"answer:{normalized}"
        needs = intent == "unknown" and conf < 0.35
        route = self._route(intent)
        return InputInterpretation(
            raw=raw,
            normalized=normalized,
            language=lang,
            dialogue_act=act,
            intent=intent,
            entities=entities,
            confidence=round(conf, 3),
            needs_clarification=needs,
            objective_hint=objective,
            route_hint=route,
            notes=notes,
        )

    def is_explicit_command(self, text: str) -> bool:
        parts = (text or "").strip().split(None, 1)
        if not parts:
            return False
        return parts[0].lower() in self.COMMAND_TOKENS

    def _language(self, lower: str) -> str:
        hi_cues = ("kya", "hai", "ho", "tum", "mera", "isko", "kyu", "kaise", "dhundo", "dikhao", "batao")
        hits = sum(1 for c in hi_cues if re.search(rf"\b{c}\b", lower))
        has_latin = bool(re.search(r"[a-z]", lower))
        if hits >= 2 and has_latin:
            return "hinglish"
        if hits >= 2:
            return "hi"
        return "en"

    def _classify(self, lower: str, normalized: str) -> tuple[str, str, float, list[str]]:
        notes: list[str] = []
        if not lower:
            return "unknown", "unknown", 0.1, ["empty"]

        if HINGLISH_GREET.search(lower) or lower in ("hii", "hi", "hello", "hey"):
            return "greet", "conversation", 0.95, ["greeting"]
        if lower in ("thanks", "thank you", "thx", "dhanyavaad", "shukriya"):
            return "thanks", "conversation", 0.9, []
        if lower in ("bye", "goodbye", "exit", "quit"):
            return "bye", "system_control", 0.9, []
        if lower in ("help", "?", "commands") or "what can you do" in lower or HINGLISH_CAP.search(lower):
            return "ask", "meta", 0.9, ["capabilities"]

        if re.search(r"\b(?:how are you|kaise ho|kaisa hai|how r u)\b", lower):
            return "greet", "conversation", 0.92, ["wellbeing"]

        if re.search(
            r"\b(?:inspect|audit|diagnose|why.*(fail|broken|error)|find why|debug|"
            r"isko check|kyu fail|repository|codebase)\b",
            lower,
        ):
            return "request", "inspect", 0.88, ["repo_task"]

        if re.search(r"\b(?:generate|implement|refactor|write (?:a )?function|fix code|pytest)\b", lower):
            return "request", "coding", 0.85, []

        if re.search(r"\b(?:plan|multi-step|then verify|break down|steps to)\b", lower):
            return "request", "planning", 0.8, []

        if re.search(r"\b(?:remember|recall|what did we|mera pichla goal|previous goal)\b", lower):
            return "request", "memory", 0.82, []

        if re.search(r"\b(?:learn|train|improve yourself|self learning)\b", lower):
            return "request", "learning", 0.75, []

        if re.search(r"\b(?:open |screenshot|list files|computer )\b", lower):
            return "request", "tool", 0.8, ["computer"]

        if re.search(r"\b(?:image|photo|ocr|thermal|detect|vision)\b", lower):
            return "request", "vision", 0.8, []

        if "?" in normalized or re.match(
            r"^(?:what|why|how|when|where|who|which|can|does|is|are|do)\b", lower
        ):
            return "ask", "question", 0.78, []

        if re.search(r"\b(?:show|list|find|create|reserve|login|export)\b", lower):
            return "request", "task", 0.72, ["finderbay_or_action"]

        # Free-form statement / conversation fallback — NEVER unknown-command
        notes.append("nl_fallback_conversation")
        return "tell", "conversation", 0.55, notes

    def _entities(self, text: str) -> list[str]:
        ents = []
        for m in re.finditer(r"\b(?:project\s+[a-z0-9_-]+|[A-Z][a-zA-Z0-9_-]{2,})\b", text):
            ents.append(m.group(0))
        return ents[:12]

    def _route(self, intent: str) -> str:
        return {
            "conversation": "dialogue",
            "meta": "dialogue",
            "question": "complete_ai",
            "task": "complete_ai",
            "inspect": "inspect",
            "coding": "coding",
            "planning": "planning",
            "memory": "memory",
            "learning": "learning",
            "tool": "computer",
            "vision": "vision",
            "system_control": "system",
        }.get(intent, "complete_ai")
