"""Dialogue / working session state (blueprint §5C)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class DialogueState:
    active_project: Optional[str] = None
    selected_document: Optional[str] = None
    current_task: Optional[str] = None
    unresolved_question: Optional[str] = None
    last_intent: Optional[str] = None
    last_slots: dict[str, Any] = field(default_factory=dict)
    turn: int = 0
    referents: dict[str, str] = field(default_factory=dict)  # "that report" → path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DialogueManager:
    FOLLOWUPS = (
        "use the other",
        "change the date",
        "show only",
        "do the same",
        "other site",
        "other file",
        "other project",
    )

    def __init__(self) -> None:
        self.state = DialogueState()

    def update_from_result(self, intent: str, slots: dict[str, Any], *, missing: Optional[list[str]] = None) -> DialogueState:
        self.state.turn += 1
        self.state.last_intent = intent
        self.state.last_slots = dict(slots)
        if slots.get("site"):
            self.state.active_project = str(slots["site"])
            self.state.referents["that project"] = str(slots["site"])
            self.state.referents["the site"] = str(slots["site"])
        if slots.get("path"):
            self.state.selected_document = str(slots["path"])
            self.state.referents["that report"] = str(slots["path"])
            self.state.referents["the file"] = str(slots["path"])
        if slots.get("task"):
            self.state.current_task = str(slots["task"])
        if missing:
            self.state.unresolved_question = f"missing:{','.join(missing)}"
        else:
            self.state.unresolved_question = None
        return self.state

    def resolve_followup(self, text: str) -> dict[str, Any]:
        lower = text.lower().strip()
        out: dict[str, Any] = {"is_followup": False, "overrides": {}}
        if any(f in lower for f in self.FOLLOWUPS) or lower.startswith(("use ", "change ", "show only", "do the same")):
            out["is_followup"] = True
            out["base_intent"] = self.state.last_intent
            out["base_slots"] = dict(self.state.last_slots)
        # "use the other file/project"
        if "other file" in lower or "other document" in lower:
            out["overrides"]["path"] = None
            out["needs"] = ["path"]
        if "other site" in lower or "other project" in lower:
            out["overrides"]["site"] = None
            out["needs"] = ["site"]
        m_date = None
        import re

        m = re.search(r"change the date to\s+(\w+)", lower)
        if m:
            out["overrides"]["relative_date"] = m.group(1)
            out["is_followup"] = True
        if "show only pending" in lower:
            out["overrides"]["filter"] = "pending"
            out["is_followup"] = True
        # resolve pronouns from referents
        for phrase, value in self.state.referents.items():
            if phrase in lower:
                if "project" in phrase or "site" in phrase:
                    out["overrides"]["site"] = value
                if "report" in phrase or "file" in phrase:
                    out["overrides"]["path"] = value
                out["is_followup"] = True
        return out

    def snapshot(self) -> dict[str, Any]:
        return self.state.to_dict()
