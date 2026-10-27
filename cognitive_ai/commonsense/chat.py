"""Open-domain chat router — broad conversation without an LLM.

Combines: greetings, FAQ, common-sense QA, analogies, topic memory,
and graceful unsupported handling with capability pointers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .kb import CommonSenseKB
from .reasoner import CommonSenseReasoner


def _singularize(word: str) -> str:
    """Very small English plural stripper for KB lookup."""
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes") or w.endswith("zes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


FAQ = [
    (
        re.compile(r"what (?:is|are) you|who are you|what can you do", re.I),
        "I am a complete non-LLM cognitive system: Finderbay task AI plus open-domain "
        "common-sense chat, speech (Vosk) and OCR (Tesseract). I use explicit knowledge, "
        "rules, planning and verification — not a large language model.",
    ),
    (
        re.compile(r"are you (?:chatgpt|an? llm|a language model)", re.I),
        "No. I do not run a pretrained general-purpose LLM. My answers come from "
        "structured knowledge, rules, search/planning and specialist models (speech/OCR).",
    ),
    (
        re.compile(r"what is (?:ai|artificial intelligence)", re.I),
        "Artificial intelligence here means software that perceives, understands supported "
        "language, remembers, reasons, plans, acts and verifies — under declared limits.",
    ),
    (
        re.compile(r"how (?:do you|does .+ ) think", re.I),
        "I run a perceive → understand → update state → reason → plan → act → verify → "
        "respond cycle, with a separate reviewed learning loop.",
    ),
]


@dataclass
class ChatTurn:
    role: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


class OpenDomainChat:
    def __init__(self, reasoner: Optional[CommonSenseReasoner] = None) -> None:
        self.reasoner = reasoner or CommonSenseReasoner()
        self.history: list[ChatTurn] = []
        self.topic: Optional[str] = None

    def respond(self, message: str) -> dict[str, Any]:
        msg = message.strip()
        self.history.append(ChatTurn("user", msg))
        lower = msg.lower()

        # FAQ / meta
        for pat, ans in FAQ:
            if pat.search(msg):
                return self._emit(ans, method="faq", confidence=0.9)

        if re.search(r"^(hi|hello|hey|namaste|good (morning|evening))\b", lower):
            return self._emit(
                "Hello. Ask a Finderbay task, a common-sense question "
                "(e.g. 'can penguins fly?'), or an analogy ('dog:bark :: cat:?').",
                method="greeting",
                confidence=1.0,
            )

        if lower in ("thanks", "thank you", "thx"):
            return self._emit("You're welcome.", method="social", confidence=1.0)

        # Analogy: a:b :: c:?   or  a is to b as c is to
        m = re.search(
            r"([a-zA-Z_]+)\s*:\s*([a-zA-Z_]+)\s*::\s*([a-zA-Z_]+)\s*:\s*\??",
            msg,
        ) or re.search(
            r"([a-zA-Z_]+)\s+is to\s+([a-zA-Z_]+)\s+as\s+([a-zA-Z_]+)\s+is to\s*\??",
            msg,
            re.I,
        )
        if m:
            ans = self.reasoner.analogy(m.group(1), m.group(2), m.group(3))
            return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples)

        # can X Y? / does X Y? / is X a Y? / what does X used for?
        m = re.search(r"^can\s+(\w+)\s+(\w+)\??$", lower)
        if m:
            subj = _singularize(m.group(1))
            ans = self.reasoner.reason(subj, "can", m.group(2))
            self.topic = subj
            return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples)

        m = re.search(r"^is\s+(\w+)\s+(?:a|an)\s+(\w+)\??$", lower)
        if m:
            subj = _singularize(m.group(1))
            ans = self.reasoner.reason(subj, "is_a", _singularize(m.group(2)))
            self.topic = subj
            return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples)

        m = re.search(r"^does\s+(\w+)\s+(\w+)\??$", lower)
        if m:
            subj = _singularize(m.group(1))
            # try as can / has
            for pred in ("can", "has", "needs"):
                ans = self.reasoner.reason(subj, pred, m.group(2))
                if ans.success:
                    self.topic = subj
                    return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples)
            ans = self.reasoner.reason(subj, "*", m.group(2))
            return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples)

        m = re.search(r"what(?:'s| is)?\s+(\w+)\s+used for\??", lower)
        if m:
            ans = self.reasoner.reason(m.group(1), "used_for", "*")
            self.topic = m.group(1)
            return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples)

        m = re.search(r"tell me about\s+(\w+)|what (?:do you know|is known) about\s+(\w+)", lower)
        if m:
            ent = m.group(1) or m.group(2)
            rel = self.reasoner.cs.related(ent)
            self.topic = ent
            if not rel:
                return self._emit(f"I do not yet have common-sense facts about '{ent}'.", method="lookup", confidence=0.2)
            text = f"About {ent}: " + "; ".join(str(t) for t in rel[:12]) + "."
            return self._emit(text, method="describe", confidence=0.8, triples=[str(t) for t in rel[:12]])

        # why can / why is / why <entity>
        m = re.search(r"why\s+(?:can|does)\s+(\w+)\s+(\w+)\??", lower)
        if m:
            ans = self.reasoner.why(m.group(1), "can", m.group(2))
            self.topic = m.group(1)
            return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples)
        m = re.search(r"why\s+(?:is|are)\s+(?:a\s+|an\s+)?(\w+)\??", lower)
        if m:
            ans = self.reasoner.why(m.group(1))
            self.topic = m.group(1)
            return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples)
        m = re.search(r"why\s+(\w+)\??$", lower)
        if m:
            ans = self.reasoner.why(m.group(1))
            self.topic = m.group(1)
            return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples)
        # Hindi-ish can/is patterns
        m = re.search(r"(\w+)\s+(?:ud\s*sakta|fly\s*kar\s*sakta)\s*(?:hai)?\??", lower)
        if m:
            ans = self.reasoner.reason(_singularize(m.group(1)), "can", "fly")
            self.topic = _singularize(m.group(1))
            return self._emit(ans.text, method="hi_can", confidence=ans.confidence, triples=ans.triples)
        m = re.search(r"(\w+)\s+kya\s+(?:hai|hota\s+hai)\??", lower)
        if m:
            rel = self.reasoner.cs.related(_singularize(m.group(1)))
            self.topic = _singularize(m.group(1))
            if rel:
                text = f"About {self.topic}: " + "; ".join(str(t) for t in rel[:10]) + "."
                return self._emit(text, method="hi_describe", confidence=0.7, triples=[str(t) for t in rel[:10]])

        # Topic follow-up: "and cats?" after talking about dogs
        m = re.search(r"^and\s+(\w+)\??$", lower)
        if m and self.topic:
            # reuse last predicate if possible
            last = self.history[-2].meta if len(self.history) >= 2 else {}
            pred = last.get("last_predicate", "is_a")
            ans = self.reasoner.reason(m.group(1), pred if pred != "*" else "is_a", "*")
            # better: dump related
            rel = self.reasoner.cs.related(m.group(1))
            if rel:
                text = f"About {m.group(1)}: " + "; ".join(str(t) for t in rel[:10]) + "."
                return self._emit(text, method="followup", confidence=0.7, triples=[str(t) for t in rel[:10]])

        # Entity dump for short messages
        tokens = re.findall(r"[a-zA-Z_]{3,}", lower)
        for tok in tokens:
            rel = self.reasoner.cs.related(tok)
            if rel and len(tokens) <= 4:
                self.topic = tok
                text = f"About {tok}: " + "; ".join(str(t) for t in rel[:10]) + "."
                return self._emit(text, method="entity_mention", confidence=0.55, triples=[str(t) for t in rel[:10]])

        return self._emit(
            "I could not map that to Finderbay tools or my common-sense knowledge. "
            "Try a task ('show unpaid invoices for Project A'), "
            "a fact question ('can penguins fly?'), "
            "or an analogy ('knife:cutting :: umbrella:?').",
            method="fallback",
            confidence=0.2,
            supported=False,
        )

    def _emit(
        self,
        text: str,
        *,
        method: str,
        confidence: float,
        triples: Optional[list[str]] = None,
        supported: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "text": text,
            "method": method,
            "confidence": confidence,
            "triples": triples or [],
            "supported": supported,
            "topic": self.topic,
            "llm_dependency": False,
            "open_domain": True,
        }
        self.history.append(ChatTurn("assistant", text, payload))
        return payload
