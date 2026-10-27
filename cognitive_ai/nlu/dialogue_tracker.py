"""Explicit dialogue-state tracking without an LLM."""

from __future__ import annotations

import copy
import re
from typing import Any, Optional

from ..core.working_state import WorkingState
from .clarification import ClarificationManager
from .intents import extract_project_id
from .pipeline import LanguageUnderstanding, SemanticFrame
from .questions import QuestionAnalyzer


class DialogueTracker:
    """Maintain active topic, entities, corrections, cancellations and slot-fill."""

    def __init__(self, nlu: Optional[LanguageUnderstanding] = None) -> None:
        self.nlu = nlu or LanguageUnderstanding()
        self.clarifier = ClarificationManager()
        self.questions = QuestionAnalyzer()

    def process(self, text: str, state: WorkingState) -> SemanticFrame:
        # Question follow-ups that reuse the last Finderbay frame
        follow = self._try_question_followup(text, state)
        if follow is not None:
            return follow

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
        # Remember last task frame for follow-up questions
        if frame.supported and frame.operation not in (
            "unsupported",
            "cancel",
            "correct",
            "status",
            "clarify",
            "login",
        ):
            state.dialogue.confirmed_parameters["_last_operation"] = frame.operation
            state.dialogue.confirmed_parameters["_last_frame"] = frame.to_dict()
        # Clear stale pending if a complete new request arrived
        if state.dialogue.current_proposal and not frame.needs_clarification:
            state.dialogue.current_proposal = None
        state.set_semantic_frame(frame.to_dict())
        for q in frame.needs_clarification:
            if q not in state.dialogue.unresolved_questions:
                state.dialogue.unresolved_questions.append(q)
        return frame

    def _try_question_followup(self, text: str, state: WorkingState) -> Optional[SemanticFrame]:
        """Reuse prior operation for short follow-ups: 'what about Project B?', 'how many of those?'."""
        qinfo = self.questions.analyze(text)
        if not qinfo.is_followup and not re.search(
            r"^\s*(?:what about|how about|and\s+(?:for\s+)?project|same for|"
            r"(?:how\s+many\s+)?(?:of\s+)?(?:those|them)\b|those\??$)",
            text.strip(),
            flags=re.I,
        ):
            return None
        last = state.dialogue.confirmed_parameters.get("_last_frame")
        last_op = state.dialogue.confirmed_parameters.get("_last_operation")
        if not isinstance(last, dict) or not last_op:
            return None
        if last_op in ("unsupported", "cancel", "correct", "status", "login"):
            return None

        frame = SemanticFrame(
            operation=str(last_op),
            project_id=last.get("project_id"),
            conditions=copy.deepcopy(last.get("conditions") or []),
            prohibitions=list(last.get("prohibitions") or []),
            exclusions=list(last.get("exclusions") or []),
            unknown_policies=dict(last.get("unknown_policies") or {}),
            original_text=text,
            normalized_text=text,
            confidence=0.8,
            supported=True,
            intent_type="question" if qinfo.is_question else "request",
            question=qinfo.to_dict(),
            output=last.get("output") or "table_with_sources",
        )

        # Project switch: "what about Project B?"
        pid = extract_project_id(text)
        if pid:
            frame.project_id = pid
            state.dialogue.confirmed_parameters["project_id"] = pid

        # Aggregation follow-up: "how many of those?"
        if qinfo.aggregation == "count" or re.search(r"\bhow\s+many\b", text, flags=re.I):
            frame.conditions = [c for c in frame.conditions if c.get("field") not in ("aggregation", "ask_form")]
            frame.conditions.append({"field": "aggregation", "operator": "eq", "value": "count"})
            frame.conditions.append({"field": "ask_form", "operator": "eq", "value": "count"})
            frame.output = "numeric_answer"

        # Additive filter: "and the overdue ones?" / "only unpaid"
        lower = text.lower()
        if re.search(r"\boverdue\b", lower) and not any(c.get("field") == "overdue" for c in frame.conditions):
            frame.conditions.append({"field": "overdue", "operator": "eq", "value": True})
        if re.search(r"\bunpaid\b", lower) and not any(
            c.get("field") == "payment_status" for c in frame.conditions
        ):
            frame.conditions.append({"field": "payment_status", "operator": "eq", "value": "unpaid"})

        state.dialogue.confirmed_parameters["_last_operation"] = frame.operation
        state.dialogue.confirmed_parameters["_last_frame"] = frame.to_dict()
        state.set_semantic_frame(frame.to_dict())
        return frame
