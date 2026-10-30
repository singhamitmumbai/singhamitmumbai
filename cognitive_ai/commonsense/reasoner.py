"""Common-sense reasoning: inheritance, affordances, simple analogies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..inference import InferenceEngine
from ..knowledge import Triple
from .kb import CommonSenseKB


@dataclass
class CSAnswer:
    success: bool
    text: str
    triples: list[str] = field(default_factory=list)
    method: str = "lookup"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "triples": self.triples,
            "method": self.method,
            "confidence": self.confidence,
        }


class CommonSenseReasoner:
    def __init__(self, kb: Optional[CommonSenseKB] = None) -> None:
        self.cs = kb or CommonSenseKB()
        self.engine = InferenceEngine(self.cs.kb)

    def reason(self, subject: str, predicate: str, obj: str = "*") -> CSAnswer:
        subject, predicate, obj = subject.lower(), predicate.lower(), obj.lower()

        # Exceptions first: cannot overrides inherited can
        if predicate == "can" and obj != "*" and self.cs.kb.has_fact(subject, "cannot", obj):
            return CSAnswer(
                True,
                f"No — ({subject} cannot {obj}) is known (exception to category defaults).",
                [f"({subject} cannot {obj})"],
                "exception",
                0.95,
            )

        # Direct lookup
        hits = self.cs.kb.query(subject, predicate, obj if obj != "*" else "*")
        if hits:
            # Filter inherited can if cannot exists
            if predicate == "can" and obj == "*":
                hits = [h for h in hits if not self.cs.kb.has_fact(subject, "cannot", h.object)]
            if hits:
                return CSAnswer(
                    True,
                    f"Yes — known: {hits[0]}.",
                    [str(h) for h in hits[:10]],
                    "lookup",
                    0.95,
                )

        # Forward chain then re-query
        self.engine.forward_chain(write_back=True)
        hits = self.cs.kb.query(subject, predicate, obj if obj != "*" else "*")
        if hits and predicate == "can" and obj != "*" and self.cs.kb.has_fact(subject, "cannot", obj):
            return CSAnswer(
                True,
                f"No — ({subject} cannot {obj}) overrides inherited ability.",
                [f"({subject} cannot {obj})"],
                "exception",
                0.95,
            )
        if hits:
            if predicate == "can" and obj == "*":
                hits = [h for h in hits if not self.cs.kb.has_fact(subject, "cannot", h.object)]
            if hits:
                return CSAnswer(
                    True,
                    f"Derived: {hits[0]}.",
                    [str(h) for h in hits[:10]],
                    "forward_chain",
                    0.85,
                )

        # Backward proof for grounded goals
        if obj != "*" and not obj.startswith("?"):
            if predicate == "can" and self.cs.kb.has_fact(subject, "cannot", obj):
                return CSAnswer(
                    True,
                    f"No — ({subject} cannot {obj}) is known (exception).",
                    [f"({subject} cannot {obj})"],
                    "exception",
                    0.9,
                )
            trace = self.engine.backward_chain(Triple(subject, predicate, obj))
            if trace.success:
                return CSAnswer(
                    True,
                    f"Proven: ({subject} {predicate} {obj}). {trace.explanation}",
                    [str(t) for t in trace.derived[:10]],
                    "backward_chain",
                    0.8,
                )

        related = self.cs.related(subject)
        if related:
            lines = "; ".join(str(t) for t in related[:8])
            return CSAnswer(
                False,
                f"I do not have ({subject} {predicate} {obj}), but I know: {lines}.",
                [str(t) for t in related[:8]],
                "related",
                0.4,
            )
        return CSAnswer(
            False,
            f"I lack common-sense knowledge for ({subject} {predicate} {obj}).",
            [],
            "unknown",
            0.1,
        )

    def why(self, subject: str, predicate: str = "*", obj: str = "*") -> CSAnswer:
        """Causal / explanatory answer from can_cause / because / needs chains."""
        subject = subject.lower()
        predicate = predicate.lower()
        obj = obj.lower()
        causes = self.cs.kb.query(subject, "can_cause", "*")
        needs = self.cs.kb.query(subject, "needs", "*")
        because = self.cs.kb.query(subject, "because", "*")
        reduces = list(self.cs.kb.query(subject, "can_reduce", "*")) + list(
            self.cs.kb.query(subject, "reduces", "*")
        )
        # also inverse: what causes subject
        caused_by = self.cs.kb.query("*", "can_cause", subject)
        parts: list[str] = []
        triples: list[str] = []
        for t in list(because)[:4]:
            parts.append(f"because {t.object}")
            triples.append(str(t))
        for t in list(causes)[:4]:
            parts.append(f"it can cause {t.object}")
            triples.append(str(t))
        for t in list(caused_by)[:4]:
            parts.append(f"it can be caused by {t.subject}")
            triples.append(str(t))
        for t in list(needs)[:3]:
            parts.append(f"it needs {t.object}")
            triples.append(str(t))
        for t in list(reduces)[:3]:
            parts.append(f"it can reduce {t.object}")
            triples.append(str(t))
        if predicate != "*" and obj != "*":
            direct = self.reason(subject, predicate, obj)
            if direct.success:
                parts.insert(0, f"({subject} {predicate} {obj}) holds")
                triples = list(direct.triples) + triples
        if not parts:
            related = self.cs.related(subject)
            if related:
                return CSAnswer(
                    False,
                    f"No explicit causal chain for {subject}; related facts: "
                    + "; ".join(str(t) for t in related[:6]),
                    [str(t) for t in related[:6]],
                    "why_related",
                    0.35,
                )
            return CSAnswer(False, f"No causal explanation for {subject}.", [], "why_unknown", 0.1)
        text = f"Why {subject}: " + "; ".join(parts) + "."
        return CSAnswer(True, text, triples[:12], "causal", 0.8)

    def analogy(self, a: str, b: str, c: str) -> CSAnswer:
        """Simple relational analogy: a:b :: c:? using shared predicates."""
        a, b, c = a.lower(), b.lower(), c.lower()
        rels = self.cs.kb.query(a, "*", b)
        if not rels:
            # try reverse
            rels = self.cs.kb.query(b, "*", a)
            if rels:
                pred = rels[0].predicate
                hits = self.cs.kb.query("*", pred, c) or self.cs.kb.query(c, pred, "*")
                if hits:
                    ans = hits[0].subject if hits[0].object == c else hits[0].object
                    return CSAnswer(True, f"{a}:{b} :: {c}:{ans} via '{pred}'", [str(hits[0])], "analogy", 0.7)
            return CSAnswer(False, f"No relation found between {a} and {b}.", [], "analogy", 0.2)
        pred = rels[0].predicate
        hits = self.cs.kb.query(c, pred, "*")
        if hits:
            return CSAnswer(
                True,
                f"{a}:{b} :: {c}:{hits[0].object} (shared relation '{pred}')",
                [str(hits[0])],
                "analogy",
                0.75,
            )
        return CSAnswer(False, f"No '{pred}' fact for {c}.", [], "analogy", 0.3)
