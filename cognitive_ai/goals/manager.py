"""Define and prioritize desired outcomes."""

from __future__ import annotations

from typing import Any, Optional

from ..core.records import GoalRecord
from ..core.working_state import WorkingState
from ..nlu.pipeline import SemanticFrame


class GoalManager:
    def create_from_frame(self, frame: SemanticFrame, state: WorkingState) -> GoalRecord:
        permitted = []
        if "send" not in frame.prohibitions:
            permitted.append("send_message")
        permitted.extend(["query", "draft", "calculate", "read"])

        completion = ["results_verified", "sources_linked"]
        if frame.operation == "prepare_followup":
            completion.append("draft_prepared")
            completion.append("not_sent")

        goal = GoalRecord(
            description=frame.original_text or frame.operation,
            desired_conditions=[
                {"operation": frame.operation},
                *frame.conditions,
            ],
            scope={"project_id": frame.project_id},
            priority=1.0,
            permitted_operations=permitted,
            completion_criteria=completion,
            cancellation_conditions=["user_cancel"],
            status="open",
        )
        # Carry prohibitions into budget/meta via scope
        goal.scope["prohibitions"] = list(frame.prohibitions)
        goal.scope["output"] = frame.output
        state.set_goal(goal)
        goal.status = "active"
        return goal

    def mark(self, state: WorkingState, status: str, goal_id: Optional[str] = None) -> None:
        gid = goal_id or state.active_goal_id
        if gid and gid in state.goals:
            state.goals[gid].status = status
