"""Open-domain chat router — ChatGPT-level conversation without requiring an LLM.

Combines: greetings, FAQ, smart question frames, common-sense QA, analogies,
creative templates, safe math, intelligent web search, and ChatGPT-style prose.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..nlu.questions import QuestionAnalyzer
from ..nlu.smart_questions import SmartQuestionUnderstanding
from ..response.chatgpt_style import ChatGPTStyleComposer
from ..search import IntelligentWebSearch
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


_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval_math(expr: str) -> tuple[bool, Any]:
    """Evaluate a tiny arithmetic AST — no names/calls/attributes."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False, None

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and (abs(left) > 1e6 or abs(right) > 20):
                raise ValueError("pow too large")
            return _SAFE_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY:
            return _SAFE_UNARY[type(node.op)](_eval(node.operand))
        raise ValueError("unsafe expression")

    try:
        return True, _eval(tree)
    except Exception:  # noqa: BLE001
        return False, None


FAQ = [
    (
        re.compile(r"what (?:is|are) you|who are you|what can you do", re.I),
        "I am **Complete AI** — a ChatGPT-level assistant stack: "
        "natural Q&A, intelligent web search, Finderbay business tools, "
        "computer vision, coding, and an autonomous agent. "
        "With `OPENAI_API_KEY` I also use ChatGPT-class models for open chat. "
        "I do not claim to be human-level AGI.",
    ),
    (
        re.compile(r"are you (?:chatgpt|an? llm|a language model)", re.I),
        "I am **Complete AI**. My non-LLM brain answers with smart question "
        "understanding, common-sense knowledge, and intelligent search — "
        "composed in a ChatGPT-style helpful format. "
        "When `OPENAI_API_KEY` is set I also call ChatGPT-class models. "
        "I do not claim to be AGI.",
    ),
    (
        re.compile(r"what is (?:ai|artificial intelligence)", re.I),
        "Artificial intelligence means software that can chat, perceive, remember, "
        "reason, plan, act and verify — Complete AI packages those capabilities "
        "together (and keeps `human_level_agi=false`).",
    ),
    (
        re.compile(r"how (?:do you|does .+ ) think", re.I),
        "I understand your question (paraphrase → intent → domain → policy), "
        "answer from local knowledge or tools when possible, escalate to "
        "intelligent web search when needed, and compose a clear ChatGPT-style reply.",
    ),
    (
        re.compile(r"are you (?:complete ai|a complete ai)", re.I),
        "Yes. I am Complete AI.",
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
        self.last_predicate: str = "is_a"
        self.last_object: str = "*"
        self.questions = QuestionAnalyzer()
        self.smart_q = SmartQuestionUnderstanding()
        self.web = IntelligentWebSearch(allow_live=True)
        self.composer = ChatGPTStyleComposer()
        self.user_name: Optional[str] = None
        self.preferences: dict[str, str] = {}
        self._world = None  # lazy WorldKnowledgeHub for open-world framing

    @property
    def world(self):
        if self._world is None:
            try:
                from ..agi_gap.world_knowledge import WorldKnowledgeHub

                self._world = WorldKnowledgeHub(expand=True)
            except Exception:  # noqa: BLE001
                self._world = False
        return self._world if self._world is not False else None

    def respond(self, message: str, *, allow_live_search: bool = True) -> dict[str, Any]:
        msg = message.strip()
        self.history.append(ChatTurn("user", msg))
        lower = msg.lower()
        frame = self.smart_q.understand(msg)
        qinfo = self.questions.analyze(frame.normalized)
        if frame.analysis.get("focus") and not qinfo.focus:
            qinfo.focus = list(frame.analysis.get("focus") or [])
        style = frame.style or "helpful"

        # Remember user name
        m_name = re.search(
            r"(?:my name is|i am|i'm|call me)\s+([A-Z][a-zA-Z]{1,30})\b",
            msg,
        ) or re.search(r"(?:my name is|i am|i'm|call me)\s+(\w+)\b", lower)
        if m_name:
            self.user_name = m_name.group(1).strip().title()
            return self._emit(
                f"Nice to meet you, **{self.user_name}**. How can Complete AI help you today?",
                method="name_memory",
                confidence=0.95,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
            )

        # Social / dialogue acts
        if frame.dialogue_act == "greet" or re.search(
            r"^(hi|hello|hey|namaste|good (morning|evening))\b", lower
        ):
            who = f" {self.user_name}" if self.user_name else ""
            return self._emit(
                f"Hello{who}. I'm **Complete AI** — ask me anything:\n"
                "- Facts & explanations (`what is gravity?`, `explain photosynthesis simply`)\n"
                "- How-to steps, comparisons, pros/cons\n"
                "- Web research (`search the web for …`)\n"
                "- Finderbay tasks, vision, or coding\n"
                "- Quick math (`calculate 12 * (3 + 4)`)\n"
                "- Drafts (poem / email / outline / brainstorm)",
                method="greeting",
                confidence=1.0,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
            )

        if frame.dialogue_act == "thanks" or lower in (
            "thanks",
            "thank you",
            "thx",
            "thank you so much",
        ):
            return self._emit(
                "You're welcome — happy to help anytime.",
                method="social",
                confidence=1.0,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
            )

        if frame.dialogue_act == "bye" or re.search(
            r"\b(bye|goodbye|see you|good night)\b", lower
        ):
            bye = f"Goodbye{', ' + self.user_name if self.user_name else ''}."
            return self._emit(
                bye + " Come back anytime.",
                method="social",
                confidence=1.0,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
            )

        if re.search(r"\b(how are you|how's it going|how r u)\b", lower):
            return self._emit(
                "I'm running normally — ready for questions, research, Finderbay tasks, "
                "vision, coding, or creative drafts.",
                method="social",
                confidence=0.9,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
            )

        # FAQ / meta
        for pat, ans in FAQ:
            if pat.search(msg):
                return self._emit(
                    ans, method="faq", confidence=0.9, question=qinfo.to_dict(), smart=frame.to_dict()
                )

        # Math
        if frame.answer_policy == "math" and frame.math_expr:
            ok, val = _safe_eval_math(frame.math_expr)
            ans = self.composer.compose_math(frame.math_expr, val, ok=ok)
            return self._emit(
                ans.text,
                method=ans.method,
                confidence=ans.confidence,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
                style_meta=ans.to_dict(),
            )

        # Creative writing templates
        if frame.answer_policy == "creative" and frame.creative_kind:
            topic = self._creative_topic(msg, frame.creative_kind)
            ans = self.composer.compose_writing(kind=frame.creative_kind, topic=topic)
            return self._emit(
                ans.text,
                method=ans.method,
                confidence=ans.confidence,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
                style_meta=ans.to_dict(),
            )

        # Explicit web / internet search
        m_search = re.search(
            r"(?:search(?:\s+the\s+web)?(?:\s+for)?|look\s+up|google|find\s+online)\s+(.+)$",
            lower,
        )
        if m_search or frame.answer_policy == "search" or qinfo.ask_form == "search":
            topic = (m_search.group(1) if m_search else frame.normalized).strip(" ?.")
            return self._web_answer(
                topic or msg, qinfo, frame, allow_live=allow_live_search, style=style
            )

        # Explain / ELI5
        m_ex = re.search(
            r"(?:explain|describe|eli5|what does .+ mean(?: by)?)\s+(.+?)(?:\s+simply)?\??$",
            lower,
        ) or re.search(r"explain\s+(.+?)\s+(?:to me|simply|in simple terms)\??$", lower)
        if m_ex or re.search(r"^(?:explain|eli5)\s+", lower):
            topic = (m_ex.group(1) if m_ex else re.sub(r"^(?:explain|eli5)\s+", "", lower)).strip(
                " ?."
            )
            topic = re.sub(r"\s+(simply|in simple terms|to me)$", "", topic).strip()
            use_style = "eli5" if ("eli5" in lower or style == "eli5") else style
            ent = _singularize(topic.split()[0]) if topic else ""
            rel = self.reasoner.cs.related(ent) if ent else []
            if rel:
                self.topic = ent
                composed = self.composer.compose(
                    topic=topic or ent,
                    ask_form="define",
                    triples=rel,
                    style=use_style,
                    confidence=0.82,
                )
                return self._emit(
                    composed.text,
                    method="explain_chatgpt",
                    confidence=composed.confidence,
                    triples=[str(t) for t in rel[:8]],
                    question=qinfo.to_dict(),
                    smart=frame.to_dict(),
                    style_meta=composed.to_dict(),
                )
            return self._web_answer(
                topic or msg, qinfo, frame, allow_live=allow_live_search, style=use_style
            )

        # Compare X and Y
        m_cmp = re.search(r"compare\s+(\w+)\s+(?:and|vs\.?|versus)\s+(\w+)", lower)
        if m_cmp or qinfo.ask_form == "compare":
            if m_cmp:
                a, b = _singularize(m_cmp.group(1)), _singularize(m_cmp.group(2))
            else:
                # fallback: try focus words
                toks = re.findall(r"[a-zA-Z]{3,}", lower)
                stop = {"compare", "versus", "and", "the", "with", "difference", "between"}
                nouns = [t for t in toks if t not in stop][:2]
                if len(nouns) < 2:
                    nouns = ["item_a", "item_b"]
                a, b = _singularize(nouns[0]), _singularize(nouns[1])
            ra, rb = self.reasoner.cs.related(a), self.reasoner.cs.related(b)
            if ra or rb:
                self.topic = a
                bullets = [
                    f"{a}: " + "; ".join(str(t) for t in ra[:4]),
                    f"{b}: " + "; ".join(str(t) for t in rb[:4]),
                ]
                composed = self.composer.compose(
                    topic=f"{a} vs {b}",
                    ask_form="compare",
                    extra_bullets=bullets,
                    summary=f"Here is a structured comparison of **{a}** and **{b}** from local common-sense knowledge:",
                    confidence=0.78,
                )
                return self._emit(
                    composed.text,
                    method="compare_chatgpt",
                    confidence=composed.confidence,
                    question=qinfo.to_dict(),
                    smart=frame.to_dict(),
                    style_meta=composed.to_dict(),
                )

        # How to / advice
        m_how = re.search(r"how (?:do i|to|can i)\s+(.+?)\??$", lower)
        if m_how or qinfo.ask_form == "howto":
            goal = (m_how.group(1) if m_how else frame.normalized).strip()
            goal = re.sub(r"^(?:how (?:do i|to|can i)\s+)", "", goal, flags=re.I)
            composed = self.composer.compose(
                topic=goal,
                ask_form="howto",
                confidence=0.62,
                style="steps",
            )
            return self._emit(
                composed.text,
                method="howto_chatgpt",
                confidence=composed.confidence,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
                style_meta=composed.to_dict(),
            )

        if re.search(r"\b(?:joke|tell me a joke)\b", lower):
            return self._emit(
                "Why did the symbolic reasoner break up with the neural net?\n"
                "It needed more **explicit commitments** — and clearer citations.",
                method="joke",
                confidence=0.7,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
            )

        # Analogy
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
            text = (
                f"**Analogy**\n\n{ans.text}\n\n"
                "Want another analogy in the same pattern? Give me three more terms."
            )
            return self._emit(
                text,
                method=ans.method,
                confidence=ans.confidence,
                triples=ans.triples,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
            )

        # Multi-clause: answer primary clause first
        clause = qinfo.clauses[0] if qinfo.clauses else frame.normalized
        answered = self._answer_clause(clause, qinfo, frame, style=style)
        if answered is not None:
            if str(answered.get("text") or "").startswith("__SEARCH__:"):
                topic = answered["text"].split(":", 1)[1]
                return self._web_answer(
                    topic, qinfo, frame, allow_live=allow_live_search, style=style
                )
            answered["smart"] = frame.to_dict()
            if qinfo.is_multi_clause and len(qinfo.clauses) > 1:
                answered["text"] = (
                    answered["text"]
                    + f"\n\n_(Also noted {len(qinfo.clauses) - 1} more clause(s); "
                    "ask separately for each for the best answer.)_"
                )
                answered["question"] = qinfo.to_dict()
            return answered

        if frame.needs_search or frame.answer_policy == "search":
            return self._web_answer(
                frame.normalized, qinfo, frame, allow_live=allow_live_search, style=style
            )

        # Topic follow-up
        m = re.search(r"^(?:and|what about|how about)\s+(\w+)\??$", lower)
        if m and self.topic:
            ent = _singularize(m.group(1))
            ans = self.reasoner.reason(ent, self.last_predicate, self.last_object)
            if ans.success:
                self.topic = ent
                composed = self.composer.compose(
                    topic=ent,
                    ask_form="yesno" if self.last_predicate in ("can", "is_a") else "define",
                    triples=ans.triples,
                    summary=ans.text,
                    confidence=ans.confidence,
                    style=style,
                )
                return self._emit(
                    composed.text,
                    method="followup_predicate",
                    confidence=ans.confidence,
                    triples=ans.triples,
                    question=qinfo.to_dict(),
                    smart=frame.to_dict(),
                    last_predicate=self.last_predicate,
                    last_object=self.last_object,
                    style_meta=composed.to_dict(),
                )
            rel = self.reasoner.cs.related(ent)
            if rel:
                self.topic = ent
                composed = self.composer.compose(
                    topic=ent, ask_form="define", triples=rel, style=style, confidence=0.72
                )
                return self._emit(
                    composed.text,
                    method="followup",
                    confidence=0.72,
                    triples=[str(t) for t in rel[:10]],
                    question=qinfo.to_dict(),
                    smart=frame.to_dict(),
                    style_meta=composed.to_dict(),
                )

        # Entity dump for short messages
        tokens = re.findall(r"[a-zA-Z_]{3,}", lower)
        stop = {
            "what",
            "which",
            "where",
            "when",
            "why",
            "how",
            "the",
            "and",
            "for",
            "about",
            "tell",
            "know",
            "please",
            "explain",
        }
        for tok in tokens:
            if tok in stop:
                continue
            rel = self.reasoner.cs.related(tok)
            if rel and len(tokens) <= 5:
                self.topic = tok
                composed = self.composer.compose(
                    topic=tok, ask_form="define", triples=rel, style=style, confidence=0.6
                )
                return self._emit(
                    composed.text,
                    method="entity_mention",
                    confidence=0.6,
                    triples=[str(t) for t in rel[:10]],
                    question=qinfo.to_dict(),
                    smart=frame.to_dict(),
                    style_meta=composed.to_dict(),
                )

        # Open-world framing: surface known facts / hypotheses before giving up
        if self.world is not None and (qinfo.is_question or frame.confidence >= 0.45):
            ow = self.world.understand(frame.normalized or msg)
            if ow.known_facts:
                composed = self.composer.compose(
                    topic=" ".join(ow.entities[:2]) or "topic",
                    ask_form=qinfo.ask_form if qinfo.ask_form != "unknown" else "define",
                    extra_bullets=list(ow.known_facts[:8]),
                    summary="Open-world read from expanded world knowledge:",
                    style=style,
                    confidence=ow.confidence,
                )
                if ow.hypotheses:
                    composed_text = composed.text + "\n\nHypotheses:\n" + "\n".join(
                        f"- {h}" for h in ow.hypotheses[:4]
                    )
                else:
                    composed_text = composed.text
                return self._emit(
                    composed_text,
                    method="open_world_chatgpt",
                    confidence=ow.confidence,
                    question=qinfo.to_dict(),
                    smart=frame.to_dict(),
                    style_meta={**composed.to_dict(), "open_world": ow.to_dict()},
                )

        # Last resort: intelligent search for question-like input
        if qinfo.is_question or frame.confidence >= 0.5:
            web = self._web_answer(
                frame.normalized, qinfo, frame, allow_live=allow_live_search, style=style
            )
            if web.get("supported") and web.get("confidence", 0) >= 0.45:
                return web

        return self._emit(
            "I could not map that confidently yet — but I can still help.\n\n"
            "Try one of these ChatGPT-style prompts:\n"
            "- A fact question: `Can penguins fly?` / `What is gravity?`\n"
            "- An explanation: `Explain OCR simply`\n"
            "- Web research: `Search the web for photosynthesis`\n"
            "- A how-to: `How do I reserve cement?`\n"
            "- Creative: `Write a poem about Mars` / `Brainstorm ideas for a study plan`\n"
            "- Math: `Calculate 15% of 240`\n"
            "- Finderbay: `Show unpaid invoices for Project A`\n\n"
            "For unrestricted open chat like ChatGPT cloud models, set "
            "`OPENAI_API_KEY` (Complete AI auto-enables the LLM bridge).",
            method="fallback",
            confidence=0.25,
            supported=False,
            question=qinfo.to_dict(),
            smart=frame.to_dict(),
        )

    def _creative_topic(self, msg: str, kind: str) -> str:
        lower = msg.lower()
        patterns = [
            rf"{kind}\s+(?:about|on|for|regarding)\s+(.+)$",
            r"(?:write|draft|compose|make|brainstorm)\s+(?:a |an |me )?(?:short |quick )?\w+\s+(?:about|on|for)\s+(.+)$",
            r"(?:outline|summary|summarize|pros?\s*(?:and|&)\s*cons?)\s+(?:of |for |about )?(.+)$",
            r"ideas for\s+(.+)$",
        ]
        for pat in patterns:
            m = re.search(pat, lower)
            if m:
                return m.group(1).strip(" ?.!")
        # strip leading verbs
        t = re.sub(
            r"^(?:please\s+)?(?:write|draft|compose|make|brainstorm|outline|summarize)\s+",
            "",
            lower,
        )
        t = re.sub(r"^(?:a |an |me )?(?:short |quick )?(?:poem|email|essay|story|outline)\s*", "", t)
        t = re.sub(r"^(?:about|on|for)\s+", "", t)
        return t.strip(" ?.!") or "your topic"

    def _answer_clause(
        self, msg: str, qinfo, frame, *, style: str = "helpful"
    ) -> Optional[dict[str, Any]]:
        lower = msg.lower().strip()

        if qinfo.ask_form == "causal" or qinfo.wh_type == "why":
            m = re.search(r"why\s+(?:can|does)\s+(\w+)\s+(\w+)\??", lower)
            if m:
                ans = self.reasoner.why(m.group(1), "can", m.group(2))
                self.topic = m.group(1)
                composed = self.composer.compose(
                    topic=f"{m.group(1)} {m.group(2)}",
                    ask_form="causal",
                    triples=ans.triples,
                    summary=ans.text,
                    style=style,
                    confidence=ans.confidence,
                )
                return self._emit(
                    composed.text,
                    method="causal_chatgpt",
                    confidence=ans.confidence,
                    triples=ans.triples,
                    question=qinfo.to_dict(),
                    smart=frame.to_dict(),
                    style_meta=composed.to_dict(),
                )
            m = re.search(r"why\s+(?:is|are)\s+(?:a\s+|an\s+)?(\w+)\??", lower)
            if m:
                ans = self.reasoner.why(m.group(1))
                self.topic = m.group(1)
                composed = self.composer.compose(
                    topic=m.group(1),
                    ask_form="causal",
                    triples=ans.triples,
                    summary=ans.text,
                    style=style,
                    confidence=ans.confidence,
                )
                return self._emit(
                    composed.text,
                    method="causal_chatgpt",
                    confidence=ans.confidence,
                    triples=ans.triples,
                    question=qinfo.to_dict(),
                    smart=frame.to_dict(),
                    style_meta=composed.to_dict(),
                )
            m = re.search(r"why\s+(\w+)\??$", lower)
            if m:
                ans = self.reasoner.why(m.group(1))
                self.topic = m.group(1)
                composed = self.composer.compose(
                    topic=m.group(1),
                    ask_form="causal",
                    triples=ans.triples,
                    summary=ans.text,
                    style=style,
                    confidence=ans.confidence,
                )
                return self._emit(
                    composed.text,
                    method="causal_chatgpt",
                    confidence=ans.confidence,
                    triples=ans.triples,
                    question=qinfo.to_dict(),
                    smart=frame.to_dict(),
                    style_meta=composed.to_dict(),
                )

        m = re.search(r"^can\s+(\w+)\s+(\w+)\??$", lower)
        if m:
            subj = _singularize(m.group(1))
            ans = self.reasoner.reason(subj, "can", m.group(2))
            self.topic = subj
            self.last_predicate, self.last_object = "can", m.group(2)
            yesno = None
            low = ans.text.lower()
            if ans.success:
                yesno = not (
                    "no —" in low or "cannot" in low or low.startswith("no")
                )
            composed = self.composer.compose(
                topic=f"can {subj} {m.group(2)}",
                ask_form="yesno",
                triples=ans.triples,
                summary=ans.text,
                yesno=yesno if yesno is not None else ans.success,
                style=style,
                confidence=ans.confidence,
            )
            return self._emit(
                composed.text,
                method="yesno_chatgpt",
                confidence=ans.confidence,
                triples=ans.triples,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
                last_predicate="can",
                last_object=m.group(2),
                style_meta=composed.to_dict(),
            )

        m = re.search(r"^is\s+(\w+)\s+(?:a|an)\s+(\w+)\??$", lower)
        if m:
            subj = _singularize(m.group(1))
            obj = _singularize(m.group(2))
            ans = self.reasoner.reason(subj, "is_a", obj)
            self.topic = subj
            self.last_predicate, self.last_object = "is_a", obj
            composed = self.composer.compose(
                topic=f"is {subj} a {obj}",
                ask_form="yesno",
                triples=ans.triples,
                summary=ans.text,
                yesno=ans.success,
                style=style,
                confidence=ans.confidence,
            )
            return self._emit(
                composed.text,
                method="yesno_chatgpt",
                confidence=ans.confidence,
                triples=ans.triples,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
                last_predicate="is_a",
                last_object=obj,
                style_meta=composed.to_dict(),
            )

        m = re.search(r"^does\s+(\w+)\s+(\w+)\??$", lower)
        if m:
            subj = _singularize(m.group(1))
            for pred in ("can", "has", "needs"):
                ans = self.reasoner.reason(subj, pred, m.group(2))
                if ans.success:
                    self.topic = subj
                    self.last_predicate, self.last_object = pred, m.group(2)
                    composed = self.composer.compose(
                        topic=f"does {subj} {m.group(2)}",
                        ask_form="yesno",
                        triples=ans.triples,
                        summary=ans.text,
                        yesno=True,
                        style=style,
                        confidence=ans.confidence,
                    )
                    return self._emit(
                        composed.text,
                        method="yesno_chatgpt",
                        confidence=ans.confidence,
                        triples=ans.triples,
                        question=qinfo.to_dict(),
                        smart=frame.to_dict(),
                        last_predicate=pred,
                        last_object=m.group(2),
                        style_meta=composed.to_dict(),
                    )
            ans = self.reasoner.reason(subj, "*", m.group(2))
            composed = self.composer.compose(
                topic=subj,
                ask_form="define",
                triples=ans.triples,
                summary=ans.text,
                style=style,
                confidence=ans.confidence,
            )
            return self._emit(
                composed.text,
                method=ans.method,
                confidence=ans.confidence,
                triples=ans.triples,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
                style_meta=composed.to_dict(),
            )

        m = re.search(r"where\s+(?:does|do|is|are)\s+(\w+)\s*(?:live|lives)?\??", lower)
        if m:
            subj = _singularize(m.group(1))
            for pred in ("lives_in", "is_in", "is_in_continent"):
                ans = self.reasoner.reason(subj, pred, "*")
                if ans.success:
                    self.topic = subj
                    composed = self.composer.compose(
                        topic=subj,
                        ask_form="define",
                        triples=ans.triples,
                        summary=ans.text,
                        style=style,
                        confidence=ans.confidence,
                    )
                    return self._emit(
                        composed.text,
                        method="where_chatgpt",
                        confidence=ans.confidence,
                        triples=ans.triples,
                        question=qinfo.to_dict(),
                        smart=frame.to_dict(),
                        style_meta=composed.to_dict(),
                    )

        m = re.search(r"who\s+(?:is|are)\s+(?:a\s+|an\s+)?(\w+)\??", lower)
        if m:
            ans = self.reasoner.reason(_singularize(m.group(1)), "is_a", "*")
            self.topic = _singularize(m.group(1))
            composed = self.composer.compose(
                topic=self.topic,
                ask_form="define",
                triples=ans.triples,
                summary=ans.text,
                style=style,
                confidence=ans.confidence,
            )
            return self._emit(
                composed.text,
                method="who_chatgpt",
                confidence=ans.confidence,
                triples=ans.triples,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
                style_meta=composed.to_dict(),
            )

        m = re.search(r"what(?:'s| is)?\s+(\w+)\s+used for\??", lower)
        if m:
            ans = self.reasoner.reason(m.group(1), "used_for", "*")
            self.topic = m.group(1)
            self.last_predicate, self.last_object = "used_for", "*"
            composed = self.composer.compose(
                topic=m.group(1),
                ask_form="define",
                triples=ans.triples,
                summary=ans.text,
                style=style,
                confidence=ans.confidence,
            )
            return self._emit(
                composed.text,
                method="used_for_chatgpt",
                confidence=ans.confidence,
                triples=ans.triples,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
                style_meta=composed.to_dict(),
            )

        if qinfo.ask_form == "define" or qinfo.wh_type == "what":
            m = re.search(
                r"what(?:'s| is| are)?\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$",
                lower,
            )
            if m:
                phrase = m.group(1).strip()
                if phrase.split()[0] not in ("you", "this", "that", "your"):
                    ent = _singularize(phrase.split()[0])
                    rel = self.reasoner.cs.related(ent)
                    self.topic = ent
                    if rel:
                        composed = self.composer.compose(
                            topic=ent,
                            ask_form="define",
                            triples=rel,
                            style=style,
                            confidence=0.78,
                        )
                        return self._emit(
                            composed.text,
                            method="define_chatgpt",
                            confidence=0.78,
                            triples=[str(t) for t in rel[:12]],
                            question=qinfo.to_dict(),
                            smart=frame.to_dict(),
                            style_meta=composed.to_dict(),
                        )
                    return self._emit(
                        f"__SEARCH__:{phrase}",
                        method="define_escalate_search",
                        confidence=0.4,
                        question=qinfo.to_dict(),
                        supported=True,
                        smart=frame.to_dict(),
                    )

        m = re.search(r"tell me about\s+(.+)|what (?:do you know|is known) about\s+(.+)", lower)
        if m:
            ent_raw = (m.group(1) or m.group(2) or "").strip(" ?.")
            ent = _singularize(ent_raw.split()[0]) if ent_raw else ""
            rel = self.reasoner.cs.related(ent) if ent else []
            self.topic = ent
            if not rel:
                return self._emit(
                    f"__SEARCH__:{ent_raw or ent}",
                    method="lookup_escalate_search",
                    confidence=0.35,
                    question=qinfo.to_dict(),
                    smart=frame.to_dict(),
                )
            composed = self.composer.compose(
                topic=ent, ask_form="define", triples=rel, style=style, confidence=0.8
            )
            return self._emit(
                composed.text,
                method="describe_chatgpt",
                confidence=0.8,
                triples=[str(t) for t in rel[:12]],
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
                style_meta=composed.to_dict(),
            )

        # Hindi-ish
        m = re.search(r"(\w+)\s+(?:ud\s*sakta|fly\s*kar\s*sakta)\s*(?:hai)?\??", lower)
        if m:
            ans = self.reasoner.reason(_singularize(m.group(1)), "can", "fly")
            self.topic = _singularize(m.group(1))
            composed = self.composer.compose(
                topic=self.topic,
                ask_form="yesno",
                triples=ans.triples,
                summary=ans.text,
                yesno=ans.success and "cannot" not in ans.text.lower(),
                style=style,
                confidence=ans.confidence,
            )
            return self._emit(
                composed.text,
                method="hi_can",
                confidence=ans.confidence,
                triples=ans.triples,
                question=qinfo.to_dict(),
                smart=frame.to_dict(),
                style_meta=composed.to_dict(),
            )
        m = re.search(r"(\w+)\s+kya\s+(?:hai|hota\s+hai)\??", lower)
        if m:
            rel = self.reasoner.cs.related(_singularize(m.group(1)))
            self.topic = _singularize(m.group(1))
            if rel:
                composed = self.composer.compose(
                    topic=self.topic, ask_form="define", triples=rel, style=style, confidence=0.72
                )
                return self._emit(
                    composed.text,
                    method="hi_describe",
                    confidence=0.72,
                    triples=[str(t) for t in rel[:10]],
                    question=qinfo.to_dict(),
                    smart=frame.to_dict(),
                    style_meta=composed.to_dict(),
                )

        return None

    def _web_answer(
        self,
        topic: str,
        qinfo,
        frame,
        *,
        allow_live: bool = True,
        style: str = "helpful",
    ) -> dict[str, Any]:
        result = self.web.search(
            topic,
            question=qinfo.to_dict() if hasattr(qinfo, "to_dict") else (qinfo or {}),
            allow_live=allow_live,
            style=style,
        )
        return self._emit(
            result.answer,
            method=f"web_{result.method}",
            confidence=result.confidence,
            supported=result.ok,
            question=qinfo.to_dict() if hasattr(qinfo, "to_dict") else qinfo,
            smart=frame.to_dict() if hasattr(frame, "to_dict") else frame,
            search=result.to_dict(),
        )

    def _emit(
        self,
        text: str,
        *,
        method: str,
        confidence: float,
        triples: Optional[list[str]] = None,
        supported: bool = True,
        question: Optional[dict] = None,
        last_predicate: Optional[str] = None,
        last_object: Optional[str] = None,
        smart: Optional[dict] = None,
        search: Optional[dict] = None,
        style_meta: Optional[dict] = None,
    ) -> dict[str, Any]:
        if last_predicate:
            self.last_predicate = last_predicate
        if last_object is not None:
            self.last_object = last_object
        payload = {
            "text": text,
            "method": method,
            "confidence": confidence,
            "triples": triples or [],
            "supported": supported,
            "topic": self.topic,
            "llm_dependency": False,
            "open_domain": True,
            "chatgpt_level": True,
            "question": question,
            "smart": smart,
            "search": search,
            "style": style_meta,
            "last_predicate": self.last_predicate,
            "last_object": self.last_object,
            "human_level_agi": False,
        }
        self.history.append(ChatTurn("assistant", text, payload))
        return payload
