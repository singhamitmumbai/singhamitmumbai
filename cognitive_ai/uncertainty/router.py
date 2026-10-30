"""Route uncertainty kinds to appropriate system responses (blueprint §15)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.records import UncertaintyKind
from ..core.working_state import UncertaintyNote, WorkingState
from .policies import respond_to_uncertainty


@dataclass
class UncertaintyAction:
    kind: str
    action: str  # clarify|reprocess|refresh|compare|abstain|qualify
    message: str
    related_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "action": self.action,
            "message": self.message,
            "related_ids": list(self.related_ids),
        }


ACTION_MAP = {
    UncertaintyKind.RECOGNITION: "reprocess",
    UncertaintyKind.MISSING_INFORMATION: "clarify",
    UncertaintyKind.CONFLICTING_EVIDENCE: "compare",
    UncertaintyKind.STALE_INFORMATION: "refresh",
    UncertaintyKind.UNSUPPORTED_TASK: "abstain",
    UncertaintyKind.FORECAST: "qualify",
}


class UncertaintyRouter:
    def note(
        self,
        state: WorkingState,
        kind: UncertaintyKind | str,
        message: str,
        related_ids: Optional[list[str]] = None,
    ) -> UncertaintyNote:
        if isinstance(kind, str):
            kind = UncertaintyKind(kind)
        note = UncertaintyNote(
            kind=kind,
            message=message,
            related_ids=list(related_ids or []),
        )
        state.uncertainties.append(note)
        return note

    def route(self, state: WorkingState) -> list[UncertaintyAction]:
        actions: list[UncertaintyAction] = []
        for note in state.uncertainties:
            actions.append(
                UncertaintyAction(
                    kind=note.kind.value,
                    action=ACTION_MAP.get(note.kind, "qualify"),
                    message=f"{note.message} → {respond_to_uncertainty(note.kind)}",
                    related_ids=list(note.related_ids),
                )
            )
        return actions

    def primary_response(self, state: WorkingState) -> Optional[str]:
        actions = self.route(state)
        if not actions:
            return None
        # Prefer clarify / abstain over qualify
        priority = {"abstain": 0, "clarify": 1, "compare": 2, "reprocess": 3, "refresh": 4, "qualify": 5}
        actions.sort(key=lambda a: priority.get(a.action, 9))
        return actions[0].message

    def report(self, state: WorkingState) -> dict[str, Any]:
        actions = self.route(state)
        by_kind: dict[str, int] = {}
        for a in actions:
            by_kind[a.kind] = by_kind.get(a.kind, 0) + 1
        return {
            "count": len(actions),
            "by_kind": by_kind,
            "actions": [a.to_dict() for a in actions],
            "primary": self.primary_response(state),
        }
