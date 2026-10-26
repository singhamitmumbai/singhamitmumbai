"""Bounded attention / context selection scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.working_state import WorkingState
from .store import MemorySystem


@dataclass
class AttentionItem:
    kind: str
    ref: str
    score: float
    reason: str


class AttentionScheduler:
    """Select what deserves computation now given goals and budgets."""

    def select(
        self,
        state: WorkingState,
        memory: MemorySystem,
        limit: int = 20,
    ) -> list[AttentionItem]:
        items: list[AttentionItem] = []
        goal = state.active_goal()
        goal_text = (goal.description if goal else "") + " " + str(state.semantic_frame or "")

        # Active goal always high priority
        if goal:
            items.append(
                AttentionItem("goal", goal.goal_id, 10.0 + goal.priority, "active goal")
            )

        # Conflicting evidence
        for note in state.uncertainties:
            boost = 8.0 if note.kind.value == "conflicting_evidence" else 5.0
            items.append(AttentionItem("uncertainty", note.note_id, boost, note.message))

        # Relevant records via lexical search
        hits = memory.search(goal_text) if goal_text.strip() else []
        for h in hits[:limit]:
            freshness = 1.0
            items.append(
                AttentionItem(
                    "record",
                    str(h.get("id")),
                    float(h.get("_score", 1)) + freshness,
                    f"relevant {h.get('kind')}",
                )
            )

        # Unfinished dialogue questions
        for i, q in enumerate(state.dialogue.unresolved_questions):
            items.append(AttentionItem("clarify", f"q{i}", 7.0 - i * 0.1, q))

        # Budget gate
        remaining = state.resource_budget.get("steps", 50)
        items.sort(key=lambda x: -x.score)
        return items[: min(limit, max(1, remaining))]

    def context_pack(
        self,
        state: WorkingState,
        memory: MemorySystem,
        limit: int = 20,
    ) -> dict[str, Any]:
        selected = self.select(state, memory, limit=limit)
        record_ids = {i.ref for i in selected if i.kind == "record"}
        records = []
        for kind, table in memory.records.items():
            for rid, row in table.items():
                if rid in record_ids:
                    records.append({"kind": kind, **row})
        return {
            "attention": [i.__dict__ for i in selected],
            "records": records,
            "claims": [c.to_dict() for c in list(state.claims.values())[:limit]],
            "dialogue": state.dialogue.to_dict(),
        }
