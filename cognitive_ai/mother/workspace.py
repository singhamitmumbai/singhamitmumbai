"""Bounded global workspace / working memory for Mother AI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..core.records import utc_now


@dataclass
class WorkspaceItem:
    kind: str
    content: str
    score: float = 0.5
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GlobalWorkspace:
    """Capacity-limited active context — not a full DB dump."""

    def __init__(self, *, capacity: int = 24) -> None:
        self.capacity = capacity
        self.items: list[WorkspaceItem] = []
        self.current_input: str = ""
        self.current_objective: str = ""
        self.uncertainties: list[str] = []
        self.open_questions: list[str] = []
        self.last_prediction: str = ""
        self.last_observation: str = ""
        self.prediction_error: float = 0.0

    def set_input(self, text: str) -> None:
        self.current_input = text
        self.push("input", text, score=1.0, source="user")

    def set_objective(self, text: str) -> None:
        self.current_objective = text
        self.push("objective", text, score=0.95, source="goal_manager")

    def push(
        self,
        kind: str,
        content: str,
        *,
        score: float = 0.5,
        source: str = "",
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        self.items.append(
            WorkspaceItem(kind=kind, content=content[:500], score=score, source=source, meta=dict(meta or {}))
        )
        self.items.sort(key=lambda x: -x.score)
        if len(self.items) > self.capacity:
            self.items = self.items[: self.capacity]

    def note_uncertainty(self, text: str) -> None:
        if text and text not in self.uncertainties:
            self.uncertainties.append(text)
            self.uncertainties = self.uncertainties[-12:]
            self.push("uncertainty", text, score=0.7, source="meta")

    def record_prediction(self, predicted: str, observed: str) -> float:
        self.last_prediction = predicted
        self.last_observation = observed
        # crude lexical error
        p = set((predicted or "").lower().split())
        o = set((observed or "").lower().split())
        if not p and not o:
            err = 0.0
        elif not p or not o:
            err = 1.0
        else:
            err = 1.0 - (len(p & o) / max(1, len(p | o)))
        self.prediction_error = round(err, 3)
        self.push(
            "prediction_error",
            f"err={self.prediction_error}",
            score=0.8,
            source="world_model",
            meta={"predicted": predicted[:120], "observed": observed[:120]},
        )
        return self.prediction_error

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_input": self.current_input,
            "current_objective": self.current_objective,
            "items": [i.to_dict() for i in self.items],
            "uncertainties": list(self.uncertainties),
            "open_questions": list(self.open_questions),
            "prediction_error": self.prediction_error,
            "capacity": self.capacity,
            "human_level_agi": False,
        }
