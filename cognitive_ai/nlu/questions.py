"""Advanced non-LLM question understanding.

Classifies WH-type / ask-form, detects polarity & aggregation, splits
multi-clause questions, and extracts focus phrases — without an LLM.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


WH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("how_many", re.compile(r"^\s*(?:how\s+many|kitne|kitni|kitna)\b", re.I)),
    ("how_much", re.compile(r"^\s*(?:how\s+much|kitna\s+(?:paisa|amount))\b", re.I)),
    ("how", re.compile(r"^\s*how\b", re.I)),
    ("what", re.compile(r"^\s*(?:what|what's|whats|kya)\b", re.I)),
    ("which", re.compile(r"^\s*(?:which|kaunsa|kaunsi)\b", re.I)),
    ("who", re.compile(r"^\s*(?:who|kisne|kaun)\b", re.I)),
    ("where", re.compile(r"^\s*(?:where|kahan|kahaan)\b", re.I)),
    ("when", re.compile(r"^\s*(?:when|kab)\b", re.I)),
    ("why", re.compile(r"^\s*(?:why|kyun|kyu|kyon)\b", re.I)),
    ("whose", re.compile(r"^\s*whose\b", re.I)),
]

# Mid-sentence WH (not only line-start) — for "Please tell me what gravity is"
WH_INLINE = re.compile(
    r"\b(?:how\s+many|how\s+much|what|which|who|where|when|why|whose|kya|kitne|kitni|kyun)\b",
    re.I,
)

YESNO_STARTERS = re.compile(
    r"^\s*(?:can|could|does|do|did|is|are|was|were|will|would|should|has|have|had|"
    r"kya)\b",
    re.I,
)

ASK_FORM_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("count", re.compile(r"\b(?:how\s+many|count(?:\s+of)?|number\s+of|kitne|kitni)\b", re.I)),
    ("sum", re.compile(r"\b(?:how\s+much|total(?:\s+of)?|sum(?:\s+of)?|grand\s+total)\b", re.I)),
    ("exists", re.compile(r"\b(?:is\s+there|are\s+there|any\s+\w+|do\s+we\s+have|kya\s+hai)\b", re.I)),
    ("compare", re.compile(r"\b(?:compare|versus|vs\.?|difference\s+between|more\s+than|less\s+than|bigger|smaller)\b", re.I)),
    ("causal", re.compile(r"\b(?:why|because|cause|kyun|kyon|reason)\b", re.I)),
    ("howto", re.compile(r"\b(?:how\s+(?:do|to|can|should)|steps?\s+to|tarika)\b", re.I)),
    ("define", re.compile(r"\b(?:what\s+(?:is|are)|define|meaning\s+of|kya\s+(?:hai|hota)|tell\s+me\s+about)\b", re.I)),
    ("search", re.compile(r"\b(?:search(?:\s+the\s+web)?|look\s+up|google|wikipedia|find\s+online)\b", re.I)),
    ("list", re.compile(r"\b(?:list|show|which|dikhao|batao|find|get)\b", re.I)),
    ("yesno", re.compile(r"^\s*(?:can|does|do|is|are|will|should|has|have|kya)\b", re.I)),
]

FOLLOWUP_MARKERS = re.compile(
    r"^\s*(?:and\b|also\b|what about\b|how about\b|same for\b|for\s+that\b|"
    r"those\b|them\b|that one\b|(?:how\s+many\s+)?of those\b|the same\b|aur\b|"
    r"how\s+many\s+(?:of\s+)?(?:those|them|that)\b)",
    re.I,
)

CLAUSE_SPLIT = re.compile(
    r"\s*(?:,\s*(?:and|or)\s+|\s+and\s+(?=how\b|what\b|which\b|who\b|where\b|when\b|why\b|is\b|are\b|can\b)|;\s*)",
    re.I,
)


@dataclass
class QuestionAnalysis:
    is_question: bool = False
    wh_type: str = "none"  # what|who|where|when|why|how|how_many|how_much|which|yesno|none
    ask_form: str = "unknown"  # count|sum|exists|list|compare|define|causal|howto|yesno|unknown
    polarity: str = "positive"  # positive|negative
    aggregation: Optional[str] = None  # count|sum|avg|none
    focus: list[str] = field(default_factory=list)
    clauses: list[str] = field(default_factory=list)
    is_multi_clause: bool = False
    is_followup: bool = False
    is_comparative: bool = False
    is_hypothetical: bool = False
    expected_answer: str = "text"  # text|number|boolean|list|table|explanation
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QuestionAnalyzer:
    """Parse question structure from text — inspectable, non-LLM."""

    def analyze(self, text: str) -> QuestionAnalysis:
        raw = (text or "").strip()
        lower = raw.lower()
        qa = QuestionAnalysis()
        if not raw:
            return qa

        has_qmark = raw.endswith("?")
        wh = self._wh_type(lower)
        if wh == "none" and WH_INLINE.search(lower):
            # Inline WH: "Please tell me what gravity is"
            for name, pat in WH_PATTERNS:
                # match mid-string by dropping start anchor semantics
                inline = re.compile(pat.pattern.lstrip("^\\s*"), pat.flags)
                if inline.search(lower):
                    wh = name
                    break
        looks_yesno = bool(YESNO_STARTERS.match(lower))
        is_q = has_qmark or wh != "none" or looks_yesno or bool(
            re.search(
                r"\b(?:tell me|explain|describe|search|look up|define|meaning of)\b",
                lower,
            )
        )
        qa.is_question = is_q
        qa.wh_type = wh if wh != "none" else ("yesno" if looks_yesno and is_q else "none")

        if re.search(r"\b(?:not|n't|never|no\b|nahi|mat)\b", lower):
            # "how many unpaid" should stay positive; negation of ability/existence
            if re.search(r"\b(?:can(?:not|'t)|does not|don't|do not|is not|aren't|won't|nahi)\b", lower):
                qa.polarity = "negative"

        qa.ask_form = self._ask_form(lower, qa.wh_type)
        qa.aggregation = self._aggregation(lower, qa.ask_form)
        qa.focus = self._focus_phrases(raw, lower)
        qa.clauses = self._split_clauses(raw)
        qa.is_multi_clause = len(qa.clauses) > 1
        qa.is_followup = bool(FOLLOWUP_MARKERS.match(lower))
        qa.is_comparative = bool(
            re.search(r"\b(?:more|less|greater|fewer|vs\.?|versus|compared\s+to|bigger|smaller)\b", lower)
        )
        qa.is_hypothetical = bool(re.search(r"\b(?:what if|suppose|agar|if we)\b", lower))
        qa.expected_answer = self._expected_answer(qa)
        qa.confidence = self._confidence(qa, has_qmark)
        if qa.is_multi_clause:
            qa.notes.append(f"split_into_{len(qa.clauses)}_clauses")
        if qa.is_followup:
            qa.notes.append("dialogue_followup")
        return qa

    def _wh_type(self, lower: str) -> str:
        for name, pat in WH_PATTERNS:
            if pat.search(lower):
                return name
        return "none"

    def _ask_form(self, lower: str, wh: str) -> str:
        for name, pat in ASK_FORM_RULES:
            if pat.search(lower):
                # "how are you" is not howto
                if name == "howto" and re.search(r"^\s*how\s+(?:are|is|old|long|far)\b", lower):
                    continue
                return name
        if wh == "why":
            return "causal"
        if wh in ("how_many",):
            return "count"
        if wh in ("how_much",):
            return "sum"
        if wh == "which":
            return "list"
        if wh == "yesno":
            return "yesno"
        if wh == "what":
            return "define"
        if re.search(r"\b(?:search|look\s+up|google)\b", lower):
            return "search"
        return "unknown"

    def _aggregation(self, lower: str, ask_form: str) -> Optional[str]:
        if ask_form == "count" or re.search(r"\b(?:how\s+many|count|number\s+of)\b", lower):
            return "count"
        if ask_form == "sum" or re.search(r"\b(?:total|sum|how\s+much)\b", lower):
            return "sum"
        if re.search(r"\b(?:average|avg|mean)\b", lower):
            return "avg"
        return None

    def _focus_phrases(self, raw: str, lower: str) -> list[str]:
        focus: list[str] = []
        # after WH word
        m = re.search(
            r"(?:how\s+many|how\s+much|what|which|who|where|when|why|kitne|kitni|kitna|kya)\s+(.+?)(?:\?|$)",
            lower,
        )
        if m:
            phrase = m.group(1).strip(" .")
            phrase = re.sub(
                r"^(?:is|are|was|were|do|does|did|the|a|an|of)\s+",
                "",
                phrase,
            )
            # truncate at project / for / in
            phrase = re.split(r"\b(?:for|in|on|from|with|above|below|excluding)\b", phrase)[0].strip()
            if 1 < len(phrase) <= 60:
                focus.append(phrase)
        # domain nouns
        for noun in (
            "invoices",
            "invoice",
            "deliveries",
            "delivery",
            "payments",
            "payment",
            "contracts",
            "contract",
            "materials",
            "material",
            "tasks",
            "task",
            "alerts",
            "vendors",
            "purchase orders",
            "stock",
        ):
            if re.search(rf"\b{re.escape(noun)}\b", lower) and noun not in focus:
                focus.append(noun)
        # keep unique order
        seen = set()
        out = []
        for f in focus:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out[:6]

    def _split_clauses(self, raw: str) -> list[str]:
        # Prefer splitting when second clause also looks interrogative
        parts = [p.strip(" ,;") for p in CLAUSE_SPLIT.split(raw) if p and p.strip(" ,;")]
        if len(parts) <= 1:
            return [raw.strip()]
        # Validate: keep only if at least two look like questions/requests
        scored = []
        for p in parts:
            if self._looks_clause(p):
                scored.append(p)
        return scored if len(scored) >= 2 else [raw.strip()]

    def _looks_clause(self, text: str) -> bool:
        lower = text.lower().strip()
        if not lower:
            return False
        if lower.endswith("?"):
            return True
        if self._wh_type(lower) != "none" or YESNO_STARTERS.match(lower):
            return True
        if re.search(r"^\s*(?:show|list|find|get|dikhao|batao)\b", lower):
            return True
        return len(lower.split()) >= 3

    def _expected_answer(self, qa: QuestionAnalysis) -> str:
        if qa.ask_form == "count" or qa.aggregation == "count":
            return "number"
        if qa.ask_form == "sum" or qa.aggregation == "sum":
            return "number"
        if qa.ask_form == "yesno" or qa.wh_type == "yesno":
            return "boolean"
        if qa.ask_form in ("list",) or qa.wh_type == "which":
            return "list"
        if qa.ask_form == "causal" or qa.wh_type == "why":
            return "explanation"
        if qa.ask_form == "exists":
            return "boolean"
        if qa.ask_form == "search":
            return "text"
        if qa.ask_form == "define":
            return "explanation"
        return "text"

    def _confidence(self, qa: QuestionAnalysis, has_qmark: bool) -> float:
        if not qa.is_question:
            return 0.2
        score = 0.45
        if has_qmark:
            score += 0.2
        if qa.wh_type != "none":
            score += 0.2
        if qa.ask_form != "unknown":
            score += 0.1
        if qa.focus:
            score += 0.05
        return min(1.0, score)


def analyze_question(text: str) -> QuestionAnalysis:
    return QuestionAnalyzer().analyze(text)
