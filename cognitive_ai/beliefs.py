"""Belief / working memory: current situation, open questions, confidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from .knowledge import Triple


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Belief:
    """A working-memory proposition with confidence and provenance."""

    content: Triple
    confidence: float = 1.0
    source: str = "perception"  # perception | inference | self | user
    justification: str = ""
    belief_id: str = field(default_factory=lambda: uuid4().hex[:10])
    created_at: str = field(default_factory=_now)
    active: bool = True

    def decay(self, factor: float = 0.95) -> None:
        self.confidence = max(0.0, self.confidence * factor)

    def to_dict(self) -> dict:
        return {
            "belief_id": self.belief_id,
            "content": {
                "subject": self.content.subject,
                "predicate": self.content.predicate,
                "object": self.content.object,
            },
            "confidence": self.confidence,
            "source": self.source,
            "justification": self.justification,
            "created_at": self.created_at,
            "active": self.active,
        }


@dataclass
class OpenQuestion:
    """Something the agent wants to resolve."""

    text: str
    pattern: Optional[Triple] = None
    priority: float = 0.5
    question_id: str = field(default_factory=lambda: uuid4().hex[:10])
    created_at: str = field(default_factory=_now)
    resolved: bool = False
    resolution: str = ""

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "pattern": (
                {
                    "subject": self.pattern.subject,
                    "predicate": self.pattern.predicate,
                    "object": self.pattern.object,
                }
                if self.pattern
                else None
            ),
            "priority": self.priority,
            "created_at": self.created_at,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


@dataclass
class Situation:
    """Snapshot of the current situational context."""

    description: str = "idle"
    entities: list[str] = field(default_factory=list)
    focus: Optional[str] = None
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "entities": list(self.entities),
            "focus": self.focus,
            "updated_at": self.updated_at,
        }


class WorkingMemory:
    """Short-term beliefs, open questions, and situational frame."""

    def __init__(self) -> None:
        self.beliefs: dict[str, Belief] = {}
        self.questions: dict[str, OpenQuestion] = {}
        self.situation = Situation()
        self._index: dict[tuple[str, str, str], str] = {}

    def perceive(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "perception",
        justification: str = "",
    ) -> Belief:
        key = (subject, predicate, obj)
        if key in self._index:
            existing = self.beliefs[self._index[key]]
            existing.confidence = max(existing.confidence, confidence)
            existing.justification = justification or existing.justification
            existing.source = source
            existing.active = True
            return existing
        belief = Belief(
            content=Triple(subject, predicate, obj),
            confidence=confidence,
            source=source,
            justification=justification,
        )
        self.beliefs[belief.belief_id] = belief
        self._index[key] = belief.belief_id
        return belief

    def add_question(
        self,
        text: str,
        pattern: Optional[Triple] = None,
        priority: float = 0.5,
    ) -> OpenQuestion:
        q = OpenQuestion(text=text, pattern=pattern, priority=priority)
        self.questions[q.question_id] = q
        return q

    def resolve_question(self, question_id: str, resolution: str) -> bool:
        q = self.questions.get(question_id)
        if not q:
            return False
        q.resolved = True
        q.resolution = resolution
        return True

    def open_questions(self) -> list[OpenQuestion]:
        return sorted(
            (q for q in self.questions.values() if not q.resolved),
            key=lambda q: -q.priority,
        )

    def active_beliefs(self, min_confidence: float = 0.0) -> list[Belief]:
        return [
            b
            for b in self.beliefs.values()
            if b.active and b.confidence >= min_confidence
        ]

    def belief_triples(self, min_confidence: float = 0.0) -> list[Triple]:
        return [b.content for b in self.active_beliefs(min_confidence)]

    def update_situation(
        self,
        description: str,
        entities: Optional[list[str]] = None,
        focus: Optional[str] = None,
    ) -> Situation:
        self.situation.description = description
        if entities is not None:
            self.situation.entities = list(entities)
        if focus is not None:
            self.situation.focus = focus
        self.situation.updated_at = _now()
        return self.situation

    def find_belief(self, subject: str, predicate: str, obj: str) -> Optional[Belief]:
        bid = self._index.get((subject, predicate, obj))
        return self.beliefs.get(bid) if bid else None

    def decay_all(self, factor: float = 0.98) -> None:
        for b in self.beliefs.values():
            if b.source != "user":
                b.decay(factor)

    def clear(self) -> None:
        self.beliefs.clear()
        self.questions.clear()
        self._index.clear()
        self.situation = Situation()

    def dump(self) -> dict:
        return {
            "situation": self.situation.to_dict(),
            "beliefs": [b.to_dict() for b in self.active_beliefs()],
            "open_questions": [q.to_dict() for q in self.open_questions()],
            "resolved_questions": [
                q.to_dict() for q in self.questions.values() if q.resolved
            ],
        }
