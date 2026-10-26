"""Operator catalogue and hierarchical / workflow planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..core.records import GoalRecord, PlanRecord, PlanStepRecord
from ..core.working_state import WorkingState
from ..reasoning.engine import ReasoningResult


@dataclass
class Operator:
    name: str
    description: str
    preconditions: list[str] = field(default_factory=list)
    expected_effects: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    cost: float = 1.0
    builder: Optional[Callable[[WorkingState, dict], dict]] = None


class OperatorPlanner:
    """Build plans from semantic frames + reasoning conclusions."""

    def __init__(self) -> None:
        self.operators: dict[str, Operator] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for op in [
            Operator(
                "retrieve_records",
                "Retrieve domain records matching filters",
                preconditions=["filters_known"],
                expected_effects=["records_loaded"],
                verification=["record_count_available"],
            ),
            Operator(
                "reason_over_records",
                "Apply rules / comparisons to loaded records",
                preconditions=["records_loaded"],
                expected_effects=["conclusions_ready"],
                verification=["conclusion_present"],
            ),
            Operator(
                "build_table",
                "Build source-linked result table",
                preconditions=["conclusions_ready"],
                expected_effects=["table_ready"],
                verification=["table_row_count"],
            ),
            Operator(
                "draft_followup",
                "Prepare follow-up text without sending",
                preconditions=["table_ready"],
                expected_effects=["draft_ready"],
                verification=["draft_exists", "send_not_performed"],
                side_effects=[],
            ),
            Operator(
                "reserve_stock",
                "Reserve inventory quantity",
                preconditions=[
                    "item_identified",
                    "sufficient_available_stock",
                    "destination_accepts",
                    "authorized",
                ],
                expected_effects=["available_decreased", "reserved_increased"],
                verification=["inventory_reservation_receipt"],
            ),
            Operator(
                "respond",
                "Generate validated natural-language response",
                preconditions=["conclusions_ready"],
                expected_effects=["answer_ready"],
                verification=["claims_match_structure"],
            ),
            Operator(
                "clarify",
                "Ask user to resolve ambiguity",
                preconditions=["ambiguity_present"],
                expected_effects=["clarification_requested"],
                verification=["question_emitted"],
            ),
        ]:
            self.register(op)

    def register(self, operator: Operator) -> None:
        self.operators[operator.name] = operator

    def plan(
        self,
        goal: GoalRecord,
        state: WorkingState,
        reasoning: Optional[ReasoningResult] = None,
    ) -> PlanRecord:
        frame = state.semantic_frame or {}
        op_name = frame.get("operation", "")

        # Clarification needed?
        if state.dialogue.unresolved_questions and frame.get("needs_clarification"):
            step = PlanStepRecord(
                operator="clarify",
                args={"questions": list(state.dialogue.unresolved_questions)},
                verification=["question_emitted"],
            )
            return PlanRecord(
                goal_id=goal.goal_id,
                steps=[step],
                outcome="missing_info",
                explanation="Ambiguity must be resolved before acting.",
                cost=1.0,
            )

        if op_name == "unsupported":
            return PlanRecord(
                goal_id=goal.goal_id,
                steps=[],
                outcome="unsupported",
                explanation="Operation not in supported capability set.",
            )

        steps: list[PlanStepRecord] = []
        steps.append(
            PlanStepRecord(
                operator="retrieve_records",
                args={"project_id": frame.get("project_id"), "operation": op_name},
                preconditions=[{"filters_known": True}],
                expected_effects=[{"records_loaded": True}],
                verification=["record_count_available"],
            )
        )
        steps.append(
            PlanStepRecord(
                operator="reason_over_records",
                args={},
                depends_on=[steps[0].step_id],
                expected_effects=[{"conclusions_ready": True}],
                verification=["conclusion_present"],
            )
        )
        steps.append(
            PlanStepRecord(
                operator="build_table",
                args={"output": frame.get("output", "table_with_sources")},
                depends_on=[steps[1].step_id],
                expected_effects=[{"table_ready": True}],
                verification=["table_row_count"],
            )
        )
        if op_name == "prepare_followup":
            # Honour prohibition on send
            steps.append(
                PlanStepRecord(
                    operator="draft_followup",
                    args={"send": False, "prohibitions": frame.get("prohibitions", [])},
                    depends_on=[steps[-1].step_id],
                    expected_effects=[{"draft_ready": True}],
                    verification=["draft_exists", "send_not_performed"],
                )
            )
        steps.append(
            PlanStepRecord(
                operator="respond",
                args={},
                depends_on=[steps[-1].step_id],
                expected_effects=[{"answer_ready": True}],
                verification=["claims_match_structure"],
            )
        )

        # If reasoning already shows empty feasible path
        if reasoning and not reasoning.success:
            return PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="no_solution",
                explanation=reasoning.message,
                cost=sum(self.operators.get(s.operator, Operator(s.operator, "")).cost for s in steps),
            )

        plan = PlanRecord(
            goal_id=goal.goal_id,
            steps=steps,
            outcome="feasible",
            explanation=f"Feasible plan with {len(steps)} steps for {op_name}.",
            cost=float(len(steps)),
        )
        state.set_plan(plan)
        return plan
