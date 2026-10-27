"""Smarter ChatGPT-level question understanding (non-LLM).

Adds paraphrase normalization, dialogue acts, domain tagging, style preference
(eli5/detailed/concise), multi-intent hints, and answer policy routing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .questions import QuestionAnalysis, QuestionAnalyzer


PARAPHRASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcould you (?:please )?tell me\b", re.I), "what is"),
    (re.compile(r"\bcan you (?:please )?explain\b", re.I), "explain"),
    (re.compile(r"\bcan you (?:please )?(?:help me|help)\b", re.I), "help"),
    (re.compile(r"\bi (?:want to|wanna) know\b", re.I), "what is"),
    (re.compile(r"\bi(?:'d| would) like to know\b", re.I), "what is"),
    (re.compile(r"\bdo you know (?:if|whether)\b", re.I), "is"),
    (re.compile(r"\bdo you know (?:about|what)\b", re.I), "what is"),
    (re.compile(r"\bplease (?:search|look up|find)\b", re.I), "search"),
    (re.compile(r"\bgive me (?:a |an )?(?:quick |brief )?(?:overview|summary) of\b", re.I), "summarize"),
    (re.compile(r"\bin (?:simple|plain|layman(?:'s)?) terms\b", re.I), "eli5"),
    (re.compile(r"\bexplain (?:like i(?:'m| am) five|eli5)\b", re.I), "eli5"),
    (re.compile(r"\bhelp me understand\b", re.I), "explain"),
    (re.compile(r"\bwalk me through\b", re.I), "how to"),
    (re.compile(r"\bwhat(?:'s| is) the difference between\b", re.I), "compare"),
    (re.compile(r"\bmujhe batao\b", re.I), "tell me"),
    # Hinglish: "gravity kya hai" / "photosynthesis kya hota hai" → "what is gravity"
    (re.compile(r"\b([a-z][a-z0-9_\-]{1,40})\s+kya\s+hota\s+hai\b", re.I), r"what is \1"),
    (re.compile(r"\b([a-z][a-z0-9_\-]{1,40})\s+kya\s+hai\b", re.I), r"what is \1"),
    (re.compile(r"\bkya hota hai\b", re.I), "what is"),
    (re.compile(r"\bkya hai\b", re.I), "what is"),
    (re.compile(r"\bsamjha(?:o| digiye)\b", re.I), "explain"),
]


DOMAIN_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("finderbay", ("invoice", "payment", "delivery", "material", "warehouse", "purchase order", "vendor", "project", "shortage", "reserve")),
    ("vision", ("image", "photo", "detect", "thermal", "camera", "ocr", "yolo", "scene")),
    ("science", ("gravity", "photosynthesis", "atom", "planet", "speed of light", "boiling", "physics", "chemistry", "biology", "dna", "molecule", "newton")),
    ("computing", ("python", "sqlite", "algorithm", "internet", "ocr", "api", "javascript", "database", "programming", "code", "linux")),
    ("math", ("calculate", "compute", "plus", "minus", "multiply", "divide", "equation", "percent", "%")),
    ("writing", ("poem", "essay", "email", "letter", "story", "rewrite", "draft", "paragraph")),
    ("web", ("search the web", "look up", "google", "online", "internet search", "wikipedia")),
]


DIALOGUE_ACTS: list[tuple[str, re.Pattern[str]]] = [
    ("greet", re.compile(r"^(?:hi|hello|hey|namaste|good (?:morning|evening|afternoon))\b", re.I)),
    ("thanks", re.compile(r"^(?:thanks|thank you|thx)\b", re.I)),
    ("bye", re.compile(r"^(?:bye|goodbye|see you|good night)\b", re.I)),
    ("affirm", re.compile(r"^(?:yes|yeah|yep|ok|okay|sure|haan)\b", re.I)),
    ("deny", re.compile(r"^(?:no|nope|nah|nahi)\b", re.I)),
    ("clarify", re.compile(r"\b(?:what do you mean|can you clarify|confused)\b", re.I)),
]


STYLE_CUES: list[tuple[str, re.Pattern[str]]] = [
    ("eli5", re.compile(r"\b(?:eli5|simple terms|simply|like i(?:'m| am) five|beginner|layman)\b", re.I)),
    ("detailed", re.compile(r"\b(?:in detail|detailed|deep dive|thorough|comprehensive|exhaustively)\b", re.I)),
    ("concise", re.compile(r"\b(?:briefly|in short|tldr|tl;dr|quick answer|one sentence|concise)\b", re.I)),
    ("steps", re.compile(r"\b(?:step by step|steps|walkthrough|checklist)\b", re.I)),
]


CREATIVE_KINDS: list[tuple[str, re.Pattern[str]]] = [
    ("poem", re.compile(r"\b(?:write|compose|make)\b.*\bpoem\b|\bpoem about\b", re.I)),
    ("email", re.compile(r"\b(?:write|draft)\b.*\bemail\b|\bemail (?:to|about|for)\b", re.I)),
    ("outline", re.compile(r"\b(?:outline|structure)\b.*\b(?:for|of)\b|\bmake an outline\b", re.I)),
    ("brainstorm", re.compile(r"\b(?:brainstorm|ideas for|suggest ideas)\b", re.I)),
    ("pros_cons", re.compile(r"\b(?:pros?\s*(?:and|&)\s*cons?|advantages?\s+and\s+disadvantages?)\b", re.I)),
    ("summarize", re.compile(r"\b(?:summarize|summary of|tldr|tl;dr|overview of)\b", re.I)),
]


PERCENT_OF = re.compile(
    r"(?:calculate|compute|what(?:'s| is))?\s*"
    r"(-?\d+(?:\.\d+)?)\s*%\s*of\s*(-?\d+(?:\.\d+)?)",
    re.I,
)

# Full arithmetic with nested parentheses (extract, don't eval yet)
MATH_SLICE = re.compile(
    r"(?:"
    r"calculate|compute|evaluate|what(?:'s| is)"
    r")?\s*"
    r"("
    r"(?:\([^()]*\)|-?\d+(?:\.\d+)?)(?:\s*[+\-*/x×÷^%]\s*(?:\([^()]*\)|-?\d+(?:\.\d+)?))+"
    r"|"
    r"-?\d+(?:\.\d+)?\s*[+\-*/x×÷^%]\s*-?\d+(?:\.\d+)?(?:\s*[+\-*/x×÷^%]\s*-?\d+(?:\.\d+)?)*"
    r")",
    re.I,
)


@dataclass
class SmartQuestionFrame:
    raw: str
    normalized: str
    analysis: dict[str, Any]
    domain: str = "general"
    dialogue_act: str = "ask"  # ask|greet|thanks|bye|affirm|deny|clarify
    style: str = "helpful"  # helpful|eli5|detailed|concise|steps
    creative_kind: Optional[str] = None  # poem|email|outline|brainstorm|pros_cons|summarize
    math_expr: Optional[str] = None
    needs_search: bool = False
    needs_tools: bool = False
    answer_policy: str = "local_first"  # local_first|search|tool|clarify|creative|math
    entities: list[str] = field(default_factory=list)
    paraphrase_applied: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    confidence: float = 0.0
    chatgpt_level: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartQuestionUnderstanding:
    """Non-LLM ChatGPT-level Q→intent framing."""

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
        if not qa.focus:
            qa2 = self.analyzer.analyze(text)
            if qa2.focus:
                qa.focus = qa2.focus

        domain = self._domain(normalized)
        style = self._style(text)
        dialogue_act = self._dialogue_act(text)
        creative = self._creative(text)
        math_expr = self._math(text)
        entities = self._entities(normalized, qa)
        intents = self._intents(qa, domain, creative, math_expr, style)
        needs_search = self._needs_search(normalized, qa, domain)
        needs_tools = domain in ("finderbay", "vision") or any(
            n in normalized.lower()
            for n in ("invoice", "detect", "thermal", "shortage", "reserve")
        )

        policy = self._policy(
            qa=qa,
            domain=domain,
            dialogue_act=dialogue_act,
            creative=creative,
            math_expr=math_expr,
            needs_search=needs_search,
            needs_tools=needs_tools,
            normalized=normalized,
        )

        conf = qa.confidence
        if applied:
            conf = min(1.0, conf + 0.05)
        if domain != "general":
            conf = min(1.0, conf + 0.05)
        if creative or math_expr:
            conf = min(1.0, conf + 0.1)
        if dialogue_act != "ask":
            conf = min(1.0, conf + 0.15)

        return SmartQuestionFrame(
            raw=text,
            normalized=normalized,
            analysis=qa.to_dict(),
            domain=domain,
            dialogue_act=dialogue_act,
            style=style,
            creative_kind=creative,
            math_expr=math_expr,
            needs_search=needs_search,
            needs_tools=needs_tools,
            answer_policy=policy,
            entities=entities,
            paraphrase_applied=applied,
            intents=intents,
            confidence=round(conf, 3),
            chatgpt_level=True,
        )

    def _policy(
        self,
        *,
        qa: QuestionAnalysis,
        domain: str,
        dialogue_act: str,
        creative: Optional[str],
        math_expr: Optional[str],
        needs_search: bool,
        needs_tools: bool,
        normalized: str,
    ) -> str:
        if dialogue_act in ("greet", "thanks", "bye", "affirm", "deny"):
            return "social"
        if math_expr:
            return "math"
        if creative:
            return "creative"
        if needs_tools and domain in ("finderbay", "vision"):
            return "tool"
        if needs_search and (
            qa.ask_form == "search"
            or domain == "web"
            or re.search(r"\b(?:search|look up|google|wikipedia|online)\b", normalized, re.I)
        ):
            return "search"
        if qa.ask_form == "unknown" and not qa.is_question and not needs_search:
            return "clarify" if len(normalized.split()) < 3 else "local_first"
        return "local_first"

    def _domain(self, text: str) -> str:
        lower = text.lower()
        best, score = "general", 0
        for name, cues in DOMAIN_CUES:
            s = sum(1 for c in cues if c in lower)
            if s > score:
                best, score = name, s
        return best if score else "general"

    def _style(self, text: str) -> str:
        for name, pat in STYLE_CUES:
            if pat.search(text):
                return name
        return "helpful"

    def _dialogue_act(self, text: str) -> str:
        for name, pat in DIALOGUE_ACTS:
            if pat.search(text.strip()):
                return name
        return "ask"

    def _creative(self, text: str) -> Optional[str]:
        for name, pat in CREATIVE_KINDS:
            if pat.search(text):
                return name
        return None

    def _math(self, text: str) -> Optional[str]:
        lower = text.lower().strip()
        m_pct = PERCENT_OF.search(text)
        if m_pct:
            return f"({m_pct.group(1)}/100)*{m_pct.group(2)}"
        looks_math = bool(
            re.search(r"\b(?:calculate|compute|evaluate)\b", lower)
            or re.search(r"what(?:'s| is)\s+\d", lower)
            or re.search(r"^\s*-?\d+(?:\.\d+)?\s*[+\-*/x×÷(]", lower)
        )
        if not looks_math:
            return None
        m = MATH_SLICE.search(text)
        if m:
            expr = m.group(1).strip()
            expr = expr.replace("×", "*").replace("÷", "/")
            # lone letter x as multiply (keep algebraic x out by requiring digit contexts — already numeric)
            expr = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", expr, flags=re.I)
            return expr
        return None

    def _intents(
        self,
        qa: QuestionAnalysis,
        domain: str,
        creative: Optional[str],
        math_expr: Optional[str],
        style: str,
    ) -> list[str]:
        intents: list[str] = []
        if qa.ask_form and qa.ask_form != "unknown":
            intents.append(qa.ask_form)
        if domain != "general":
            intents.append(f"domain:{domain}")
        if creative:
            intents.append(f"creative:{creative}")
        if math_expr:
            intents.append("math")
        if style != "helpful":
            intents.append(f"style:{style}")
        if qa.is_followup:
            intents.append("followup")
        if qa.is_multi_clause:
            intents.append("multi_clause")
        return intents[:8]

    def _entities(self, text: str, qa: QuestionAnalysis) -> list[str]:
        ents: list[str] = []
        for f in qa.focus:
            ents.append(f)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text):
            ents.append(m.group(1))
        for m in re.finditer(r"[\"']([^\"']{2,40})[\"']", text):
            ents.append(m.group(1))
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
        if domain in ("science", "computing") and qa.ask_form in ("define", "causal"):
            return True
        # Current events / people often need search
        if re.search(r"\b(?:latest|current|today|news|who is|ceo of)\b", lower):
            return True
        return False
