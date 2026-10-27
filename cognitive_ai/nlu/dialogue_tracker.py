"""Explicit dialogue-state tracking without an LLM."""

from __future__ import annotations

import re
from typing import Any, Optional

from ..core.working_state import WorkingState
from .clarification import ClarificationManager
from .pipeline import LanguageUnderstanding, SemanticFrame


class DialogueTracker:
    """Maintain active topic, entities, corrections, cancellations and slot-fill."""

    def __init__(self, nlu: Optional[LanguageUnderstanding] = None) -> None:
        self.nlu = nlu or LanguageUnderstanding()
        self.clarifier = ClarificationManager()

    def process(self, text: str, state: WorkingState) -> SemanticFrame:
        # Correction patterns
        corr = re.search(
            r"(?:make that|change (?:that|it) to|i meant|actually)\s+(.+)$",
            text.strip(),
            flags=re.I,
        )
        if corr:
            new_val = corr.group(1).strip().rstrip(".")
            field = "last_quantity"
            old = state.dialogue.confirmed_parameters.get(field)
            state.apply_correction(field, old, new_val)
            frame = SemanticFrame(
                operation="correct",
                original_text=text,
                normalized_text=text,
                confidence=0.9,
                supported=True,
            )
            frame.conditions = [{"field": field, "operator": "eq", "value": new_val}]
            state.set_semantic_frame(frame.to_dict())
            return frame

        if re.search(r"^\s*(cancel|stop|abort|never mind)\b", text, flags=re.I):
            state.dialogue.cancellations.append(text)
            state.dialogue.current_proposal = None
            state.dialogue.unresolved_questions.clear()
            if state.active_goal_id and state.active_goal_id in state.goals:
                state.goals[state.active_goal_id].status = "abandoned"
            frame = SemanticFrame(
                operation="cancel",
                original_text=text,
                confidence=1.0,
                supported=True,
            )
            state.set_semantic_frame(frame.to_dict())
            return frame

        # Fact / alias acquisition: "X is now called Y"
        alias = re.search(
            r"^(.+?)\s+is now called\s+(.+?)\.?$",
            text.strip(),
            flags=re.I,
        )
        if alias:
            old_name = alias.group(1).strip()
            new_name = alias.group(2).strip()
            frame = SemanticFrame(
                operation="learn_alias",
                original_text=text,
                normalized_text=text,
                confidence=0.95,
                supported=True,
                conditions=[
                    {"field": "alias_of", "operator": "eq", "value": old_name},
                    {"field": "alias", "operator": "eq", "value": new_name},
                ],
            )
            state.set_semantic_frame(frame.to_dict())
            return frame

        # Resume pending slot-fill from a short reply
        pending = state.dialogue.current_proposal
        if pending and pending.get("type") == "slot_fill":
            resumed = self.clarifier.try_resume(text, pending)
            if resumed is not None:
                still_raw = resumed.unknown_policies.get("still_missing") or ""
                still = [s for s in still_raw.split(",") if s]
                if still:
                    filled = dict(pending.get("filled") or {})
                    for c in resumed.conditions:
                        filled[c["field"]] = c["value"]
                    pending = {
                        **pending,
                        "project_id": resumed.project_id,
                        "conditions": list(resumed.conditions),
                        "missing_slots": still,
                        "filled": filled,
                        "questions": self.clarifier.questions_for(still),
                    }
                    state.dialogue.current_proposal = pending
                    state.dialogue.unresolved_questions = list(pending["questions"])
                    resumed.needs_clarification = list(pending["questions"])
                    state.set_semantic_frame(resumed.to_dict())
                    return resumed
                # Complete — clear pending and continue with filled frame
                state.dialogue.current_proposal = None
                state.dialogue.unresolved_questions.clear()
                resumed.needs_clarification = []
                if resumed.project_id:
                    state.dialogue.confirmed_parameters["project_id"] = resumed.project_id
                for c in resumed.conditions:
                    state.dialogue.confirmed_parameters[c["field"]] = c["value"]
                state.set_semantic_frame(resumed.to_dict())
                return resumed

        frame = self.nlu.parse(text, dialogue_params=state.dialogue.confirmed_parameters)

        # Slot-fill gate for supported ops
        missing = self.clarifier.missing_slots(frame)
        if missing and frame.supported and frame.operation not in ("unsupported", "status", "cancel", "correct", "clarify"):
            pending = self.clarifier.make_pending(frame, missing)
            state.dialogue.current_proposal = pending
            state.dialogue.unresolved_questions = list(pending["questions"])
            frame.needs_clarification = list(pending["questions"])
            state.set_semantic_frame(frame.to_dict())
            return frame

        if frame.project_id:
            state.dialogue.confirmed_parameters["project_id"] = frame.project_id
            state.dialogue.selected_entities.append(frame.project_id)
        for c in frame.conditions:
            state.dialogue.confirmed_parameters[c["field"]] = c["value"]
        # Clear stale pending if a complete new request arrived
        if state.dialogue.current_proposal and not frame.needs_clarification:
            state.dialogue.current_proposal = None
        state.set_semantic_frame(frame.to_dict())
        for q in frame.needs_clarification:
            if q not in state.dialogue.unresolved_questions:
                state.dialogue.unresolved_questions.append(q)
        return frame
