"""Open-domain chat router — broad conversation without an LLM.

Combines: greetings, FAQ, advanced question analysis, common-sense QA,
analogies, topic memory, and graceful unsupported handling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..nlu.questions import QuestionAnalyzer
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
        "I am Complete AI — general chat (ChatGPT-class when an API key is set), "
        "Finderbay business tasks, computer vision, coding, and an autonomous agent. "
        "Ask me anything.",
    ),
    (
        re.compile(r"are you (?:chatgpt|an? llm|a language model)", re.I),
        "I am Complete AI. With OPENAI_API_KEY set I use ChatGPT-class models for general "
        "chat; I also run local task, vision, and coding engines. I do not claim to be AGI.",
    ),
    (
        re.compile(r"what is (?:ai|artificial intelligence)", re.I),
        "Artificial intelligence means software that can chat, perceive, remember, reason, "
        "plan, act and verify — Complete AI packages those capabilities together.",
    ),
    (
        re.compile(r"how (?:do you|does .+ ) think", re.I),
        "For open chat I use a general LLM when configured; for business/vision/coding I "
        "use dedicated engines (perceive → understand → reason → plan → act → verify).",
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
        self.user_name: Optional[str] = None
        self.preferences: dict[str, str] = {}

    def respond(self, message: str) -> dict[str, Any]:
        msg = message.strip()
        self.history.append(ChatTurn("user", msg))
        lower = msg.lower()
        qinfo = self.questions.analyze(msg)

        # Remember user name
        m_name = re.search(
            r"(?:my name is|i am|i'm|call me)\s+([A-Z][a-zA-Z]{1,30})\b",
            msg,
        ) or re.search(r"(?:my name is|i am|i'm|call me)\s+(\w+)\b", lower)
        if m_name:
            self.user_name = m_name.group(1).strip().title()
            return self._emit(
                f"Nice to meet you, {self.user_name}. How can Complete AI help?",
                method="name_memory",
                confidence=0.95,
                question=qinfo.to_dict(),
            )

        # FAQ / meta
        for pat, ans in FAQ:
            if pat.search(msg):
                return self._emit(ans, method="faq", confidence=0.9, question=qinfo.to_dict())

        if re.search(r"^(hi|hello|hey|namaste|good (morning|evening))\b", lower):
            who = f" {self.user_name}" if self.user_name else ""
            return self._emit(
                f"Hello{who}. I am Complete AI — ask a Finderbay task, a common-sense "
                "question (e.g. 'can penguins fly?'), 'explain gravity', "
                "or an analogy ('dog:bark :: cat:?').",
                method="greeting",
                confidence=1.0,
                question=qinfo.to_dict(),
            )

        if lower in ("thanks", "thank you", "thx", "thank you so much"):
            return self._emit("You're welcome.", method="social", confidence=1.0, question=qinfo.to_dict())

        if re.search(r"\b(how are you|how's it going|how r u)\b", lower):
            return self._emit(
                "I'm running normally — ready for tasks, vision, coding, or questions.",
                method="social",
                confidence=0.9,
                question=qinfo.to_dict(),
            )

        if re.search(r"\b(bye|goodbye|see you|good night)\b", lower):
            bye = f"Goodbye{', ' + self.user_name if self.user_name else ''}."
            return self._emit(bye + " Come back anytime.", method="social", confidence=1.0, question=qinfo.to_dict())

        # Explain / ELI5
        m_ex = re.search(
            r"(?:explain|describe|eli5|what does .+ mean(?: by)?)\s+(.+?)(?:\s+simply)?\??$",
            lower,
        ) or re.search(r"explain\s+(.+?)\s+(?:to me|simply|in simple terms)\??$", lower)
        if m_ex or re.search(r"^(?:explain|eli5)\s+", lower):
            topic = (m_ex.group(1) if m_ex else re.sub(r"^(?:explain|eli5)\s+", "", lower)).strip(" ?.")
            topic = re.sub(r"\s+(simply|in simple terms|to me)$", "", topic).strip()
            ent = _singularize(topic.split()[0]) if topic else ""
            rel = self.reasoner.cs.related(ent) if ent else []
            if rel:
                self.topic = ent
                text = (
                    f"Simple take on '{topic}': "
                    + "; ".join(str(t) for t in rel[:8])
                    + ". (From Complete AI common-sense KB — not an LLM essay.)"
                )
                return self._emit(text, method="explain", confidence=0.8, triples=[str(t) for t in rel[:8]], question=qinfo.to_dict())
            return self._emit(
                f"I don't have a deep essay on '{topic}' in my curated KB. "
                "Try a shorter entity ('explain penguin') or enable LLM chat for open essays.",
                method="explain_miss",
                confidence=0.35,
                supported=False,
                question=qinfo.to_dict(),
            )

        # Compare X and Y
        m_cmp = re.search(r"compare\s+(\w+)\s+(?:and|vs\.?|versus)\s+(\w+)", lower)
        if m_cmp:
            a, b = _singularize(m_cmp.group(1)), _singularize(m_cmp.group(2))
            ra, rb = self.reasoner.cs.related(a), self.reasoner.cs.related(b)
            if ra or rb:
                self.topic = a
                text = (
                    f"Compare {a} vs {b}: "
                    f"{a} → " + "; ".join(str(t) for t in ra[:5])
                    + f" || {b} → " + "; ".join(str(t) for t in rb[:5]) + "."
                )
                return self._emit(text, method="compare", confidence=0.75, question=qinfo.to_dict())

        # How to / advice templates (honest, non-LLM)
        m_how = re.search(r"how (?:do i|to|can i)\s+(.+?)\??$", lower)
        if m_how:
            goal = m_how.group(1).strip()
            return self._emit(
                f"For '{goal}': break it into steps — (1) clarify the goal and constraints, "
                "(2) list required materials/info, (3) do the smallest safe first action, "
                "(4) check the result, (5) iterate. "
                "Ask me a Finderbay/vision/coding task if it maps to those tools.",
                method="howto",
                confidence=0.55,
                question=qinfo.to_dict(),
            )

        if re.search(r"\b(?:joke|tell me a joke)\b", lower):
            return self._emit(
                "Why did the symbolic reasoner break up with the neural net? "
                "It needed more explicit commitments.",
                method="joke",
                confidence=0.7,
                question=qinfo.to_dict(),
            )

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
            return self._emit(
                ans.text,
                method=ans.method,
                confidence=ans.confidence,
                triples=ans.triples,
                question=qinfo.to_dict(),
            )

        # Multi-clause: answer primary clause first
        clause = qinfo.clauses[0] if qinfo.clauses else msg
        answered = self._answer_clause(clause, qinfo)
        if answered is not None:
            if qinfo.is_multi_clause and len(qinfo.clauses) > 1:
                answered["text"] = (
                    answered["text"]
                    + f" (Also noted {len(qinfo.clauses) - 1} more clause(s); ask separately for each.)"
                )
                answered["question"] = qinfo.to_dict()
            return answered

        # Topic follow-up: "and cats?" / "what about dogs?"
        m = re.search(r"^(?:and|what about|how about)\s+(\w+)\??$", lower)
        if m and self.topic:
            ent = _singularize(m.group(1))
            ans = self.reasoner.reason(ent, self.last_predicate, self.last_object)
            if ans.success:
                self.topic = ent
                return self._emit(
                    ans.text,
                    method="followup_predicate",
                    confidence=ans.confidence,
                    triples=ans.triples,
                    question=qinfo.to_dict(),
                    last_predicate=self.last_predicate,
                    last_object=self.last_object,
                )
            rel = self.reasoner.cs.related(ent)
            if rel:
                self.topic = ent
                text = f"About {ent}: " + "; ".join(str(t) for t in rel[:10]) + "."
                return self._emit(
                    text,
                    method="followup",
                    confidence=0.7,
                    triples=[str(t) for t in rel[:10]],
                    question=qinfo.to_dict(),
                )

        # Entity dump for short messages
        tokens = re.findall(r"[a-zA-Z_]{3,}", lower)
        stop = {"what", "which", "where", "when", "why", "how", "the", "and", "for", "about", "tell", "know"}
        for tok in tokens:
            if tok in stop:
                continue
            rel = self.reasoner.cs.related(tok)
            if rel and len(tokens) <= 5:
                self.topic = tok
                text = f"About {tok}: " + "; ".join(str(t) for t in rel[:10]) + "."
                return self._emit(
                    text,
                    method="entity_mention",
                    confidence=0.55,
                    triples=[str(t) for t in rel[:10]],
                    question=qinfo.to_dict(),
                )

        return self._emit(
            "I could not map that to Finderbay tools or my common-sense knowledge. "
            "Try a task ('show unpaid invoices for Project A'), "
            "a fact question ('can penguins fly?'), "
            "an explanation ('explain gravity'), "
            "or an analogy ('knife:cutting :: umbrella:?'). "
            "For open unrestricted chat like ChatGPT, enable COMPLETE_AI_ENABLE_LLM=1 "
            "with OPENAI_API_KEY.",
            method="fallback",
            confidence=0.2,
            supported=False,
            question=qinfo.to_dict(),
        )

    def _answer_clause(self, msg: str, qinfo) -> Optional[dict[str, Any]]:
        lower = msg.lower().strip()

        # Route by ask_form / wh_type from analyzer
        if qinfo.ask_form == "causal" or qinfo.wh_type == "why":
            m = re.search(r"why\s+(?:can|does)\s+(\w+)\s+(\w+)\??", lower)
            if m:
                ans = self.reasoner.why(m.group(1), "can", m.group(2))
                self.topic = m.group(1)
                return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples, question=qinfo.to_dict())
            m = re.search(r"why\s+(?:is|are)\s+(?:a\s+|an\s+)?(\w+)\??", lower)
            if m:
                ans = self.reasoner.why(m.group(1))
                self.topic = m.group(1)
                return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples, question=qinfo.to_dict())
            m = re.search(r"why\s+(\w+)\??$", lower)
            if m:
                ans = self.reasoner.why(m.group(1))
                self.topic = m.group(1)
                return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples, question=qinfo.to_dict())

        # can X Y? / does X Y? / is X a Y?
        m = re.search(r"^can\s+(\w+)\s+(\w+)\??$", lower)
        if m:
            subj = _singularize(m.group(1))
            ans = self.reasoner.reason(subj, "can", m.group(2))
            self.topic = subj
            self.last_predicate, self.last_object = "can", m.group(2)
            return self._emit(
                ans.text,
                method=ans.method,
                confidence=ans.confidence,
                triples=ans.triples,
                question=qinfo.to_dict(),
                last_predicate="can",
                last_object=m.group(2),
            )

        m = re.search(r"^is\s+(\w+)\s+(?:a|an)\s+(\w+)\??$", lower)
        if m:
            subj = _singularize(m.group(1))
            obj = _singularize(m.group(2))
            ans = self.reasoner.reason(subj, "is_a", obj)
            self.topic = subj
            self.last_predicate, self.last_object = "is_a", obj
            return self._emit(
                ans.text,
                method=ans.method,
                confidence=ans.confidence,
                triples=ans.triples,
                question=qinfo.to_dict(),
                last_predicate="is_a",
                last_object=obj,
            )

        m = re.search(r"^does\s+(\w+)\s+(\w+)\??$", lower)
        if m:
            subj = _singularize(m.group(1))
            for pred in ("can", "has", "needs"):
                ans = self.reasoner.reason(subj, pred, m.group(2))
                if ans.success:
                    self.topic = subj
                    self.last_predicate, self.last_object = pred, m.group(2)
                    return self._emit(
                        ans.text,
                        method=ans.method,
                        confidence=ans.confidence,
                        triples=ans.triples,
                        question=qinfo.to_dict(),
                        last_predicate=pred,
                        last_object=m.group(2),
                    )
            ans = self.reasoner.reason(subj, "*", m.group(2))
            return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples, question=qinfo.to_dict())

        # where does X live / where is X
        m = re.search(r"where\s+(?:does|do|is|are)\s+(\w+)\s*(?:live|lives)?\??", lower)
        if m:
            subj = _singularize(m.group(1))
            for pred in ("lives_in", "is_in", "is_in_continent"):
                ans = self.reasoner.reason(subj, pred, "*")
                if ans.success:
                    self.topic = subj
                    return self._emit(ans.text, method="where", confidence=ans.confidence, triples=ans.triples, question=qinfo.to_dict())

        # who helps / who is
        m = re.search(r"who\s+(?:is|are)\s+(?:a\s+|an\s+)?(\w+)\??", lower)
        if m:
            ans = self.reasoner.reason(_singularize(m.group(1)), "is_a", "*")
            self.topic = _singularize(m.group(1))
            return self._emit(ans.text, method="who", confidence=ans.confidence, triples=ans.triples, question=qinfo.to_dict())

        m = re.search(r"what(?:'s| is)?\s+(\w+)\s+used for\??", lower)
        if m:
            ans = self.reasoner.reason(m.group(1), "used_for", "*")
            self.topic = m.group(1)
            self.last_predicate, self.last_object = "used_for", "*"
            return self._emit(ans.text, method=ans.method, confidence=ans.confidence, triples=ans.triples, question=qinfo.to_dict())

        # define / what is X
        if qinfo.ask_form == "define" or qinfo.wh_type == "what":
            m = re.search(r"what(?:'s| is| are)?\s+(?:a\s+|an\s+|the\s+)?(\w+)\??$", lower)
            if m and m.group(1) not in ("you", "ai", "this", "that"):
                ent = _singularize(m.group(1))
                rel = self.reasoner.cs.related(ent)
                self.topic = ent
                if rel:
                    text = f"About {ent}: " + "; ".join(str(t) for t in rel[:12]) + "."
                    return self._emit(text, method="define", confidence=0.75, triples=[str(t) for t in rel[:12]], question=qinfo.to_dict())

        m = re.search(r"tell me about\s+(\w+)|what (?:do you know|is known) about\s+(\w+)", lower)
        if m:
            ent = m.group(1) or m.group(2)
            rel = self.reasoner.cs.related(ent)
            self.topic = ent
            if not rel:
                return self._emit(
                    f"I do not yet have common-sense facts about '{ent}'.",
                    method="lookup",
                    confidence=0.2,
                    question=qinfo.to_dict(),
                )
            text = f"About {ent}: " + "; ".join(str(t) for t in rel[:12]) + "."
            return self._emit(text, method="describe", confidence=0.8, triples=[str(t) for t in rel[:12]], question=qinfo.to_dict())

        # Hindi-ish can/is patterns
        m = re.search(r"(\w+)\s+(?:ud\s*sakta|fly\s*kar\s*sakta)\s*(?:hai)?\??", lower)
        if m:
            ans = self.reasoner.reason(_singularize(m.group(1)), "can", "fly")
            self.topic = _singularize(m.group(1))
            return self._emit(ans.text, method="hi_can", confidence=ans.confidence, triples=ans.triples, question=qinfo.to_dict())
        m = re.search(r"(\w+)\s+kya\s+(?:hai|hota\s+hai)\??", lower)
        if m:
            rel = self.reasoner.cs.related(_singularize(m.group(1)))
            self.topic = _singularize(m.group(1))
            if rel:
                text = f"About {self.topic}: " + "; ".join(str(t) for t in rel[:10]) + "."
                return self._emit(text, method="hi_describe", confidence=0.7, triples=[str(t) for t in rel[:10]], question=qinfo.to_dict())

        return None

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
            "question": question,
            "last_predicate": self.last_predicate,
            "last_object": self.last_object,
        }
        self.history.append(ChatTurn("assistant", text, payload))
        return payload
