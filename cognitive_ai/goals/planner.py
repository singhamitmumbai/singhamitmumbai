"""Operator catalogue and hierarchical / workflow planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..core.records import GoalRecord, PlanRecord, PlanStepRecord
from ..core.working_state import WorkingState
from ..reasoning.engine import ReasoningResult
from .htn import HTNLibrary


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
        self.htn = HTNLibrary()
        self._register_defaults()

    def _register_defaults(self) -> None:
        for op in [
            Operator("retrieve_records", "Retrieve domain records matching filters",
                     preconditions=["filters_known"], expected_effects=["records_loaded"],
                     verification=["record_count_available"]),
            Operator("reason_over_records", "Apply rules / comparisons to loaded records",
                     preconditions=["records_loaded"], expected_effects=["conclusions_ready"],
                     verification=["conclusion_present"]),
            Operator("build_table", "Build source-linked result table",
                     preconditions=["conclusions_ready"], expected_effects=["table_ready"],
                     verification=["table_row_count"]),
            Operator("draft_followup", "Prepare follow-up text without sending",
                     preconditions=["table_ready"], expected_effects=["draft_ready"],
                     verification=["draft_exists", "send_not_performed"]),
            Operator("reserve_stock", "Reserve inventory quantity",
                     preconditions=["item_identified", "sufficient_available_stock", "authorized"],
                     expected_effects=["available_decreased", "reserved_increased"],
                     verification=["inventory_reservation_receipt"]),
            Operator("create_task", "Persist a local task",
                     preconditions=["title_known"], expected_effects=["task_created"],
                     verification=["task_persisted"]),
            Operator("list_tasks", "List tasks for a project",
                     expected_effects=["tasks_listed"], verification=["record_count_available"]),
            Operator("ocr_extract", "Extract and validate document fields",
                     expected_effects=["fields_ready"], verification=["fields_present"]),
            Operator("vision_analyze", "Run YOLO/classical CV on an image",
                     preconditions=["image_path_known"], expected_effects=["detections_ready"],
                     verification=["vision_result_present"]),
            Operator("respond", "Generate validated natural-language response",
                     preconditions=["conclusions_ready"], expected_effects=["answer_ready"],
                     verification=["claims_match_structure"]),
            Operator("clarify", "Ask user to resolve ambiguity",
                     preconditions=["ambiguity_present"], expected_effects=["clarification_requested"],
                     verification=["question_emitted"]),
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

        if op_name in ("cancel", "correct", "status", "learn_alias"):
            steps = [
                PlanStepRecord(
                    operator="respond" if op_name != "learn_alias" else "learn_alias",
                    args={},
                    expected_effects=[{"answer_ready": True}],
                    verification=["claims_match_structure"] if op_name != "learn_alias" else ["alias_stored"],
                )
            ]
            if op_name == "learn_alias":
                steps.append(
                    PlanStepRecord(
                        operator="respond",
                        args={},
                        verification=["claims_match_structure"],
                    )
                )
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation=f"Direct response plan for {op_name}.",
                cost=float(len(steps)),
            )
            state.set_plan(plan)
            return plan

        if op_name == "reserve_stock":
            qty = None
            name = None
            for c in frame.get("conditions") or []:
                if c.get("field") == "reserve_qty":
                    qty = c.get("value")
                if c.get("field") == "material_name":
                    name = c.get("value")
            steps = [
                PlanStepRecord(
                    operator="reserve_stock",
                    args={"quantity": qty, "material_name": name, "item": None},
                    verification=["inventory_reservation_receipt"],
                ),
                PlanStepRecord(
                    operator="respond",
                    args={},
                    depends_on=[],
                    verification=["claims_match_structure"],
                ),
            ]
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation="Reserve stock then report.",
                cost=2.0,
            )
            state.set_plan(plan)
            return plan

        if op_name == "manage_task":
            title = None
            for c in frame.get("conditions") or []:
                if c.get("field") == "task_title":
                    title = c.get("value")
            lower = (frame.get("normalized_text") or "").lower()
            if "list task" in lower:
                steps = [
                    PlanStepRecord(
                        operator="list_tasks",
                        args={"project_id": frame.get("project_id")},
                        verification=["record_count_available"],
                    ),
                    PlanStepRecord(operator="respond", args={}, verification=["claims_match_structure"]),
                ]
            else:
                if not title:
                    return PlanRecord(
                        goal_id=goal.goal_id,
                        steps=[
                            PlanStepRecord(
                                operator="clarify",
                                args={"questions": ["What is the task title?"]},
                                verification=["question_emitted"],
                            )
                        ],
                        outcome="missing_info",
                        explanation="Task title missing.",
                        cost=1.0,
                    )
                steps = [
                    PlanStepRecord(
                        operator="create_task",
                        args={"title": title, "project_id": frame.get("project_id")},
                        verification=["task_persisted"],
                    ),
                    PlanStepRecord(operator="respond", args={}, verification=["claims_match_structure"]),
                ]
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation=f"Task management plan ({len(steps)} steps).",
                cost=float(len(steps)),
            )
            state.set_plan(plan)
            return plan

        if op_name == "document_qa":
            field = None
            for c in frame.get("conditions") or []:
                if c.get("field") == "document_field":
                    field = c.get("value")
            entities = frame.get("entities") or []
            doc_id = entities[0].get("entity_id") if entities else None
            steps = [
                PlanStepRecord(
                    operator="ocr_extract",
                    args={"field": field, "document_id": doc_id},
                    verification=["fields_present"],
                ),
                PlanStepRecord(operator="respond", args={}, verification=["claims_match_structure"]),
            ]
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation="Document field extraction plan.",
                cost=2.0,
            )
            state.set_plan(plan)
            return plan

        if op_name == "login":
            user = pwd = None
            for c in frame.get("conditions") or []:
                if c.get("field") == "username":
                    user = c.get("value")
                if c.get("field") == "password":
                    pwd = c.get("value")
            steps = [
                PlanStepRecord(
                    operator="login",
                    args={"username": user, "password": pwd},
                    verification=["login_ok"],
                ),
                PlanStepRecord(operator="respond", args={}, verification=["claims_match_structure"]),
            ]
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation="Authentication plan.",
                cost=2.0,
            )
            state.set_plan(plan)
            return plan

        if op_name == "record_payment":
            invoice_id = amount = None
            for c in frame.get("conditions") or []:
                if c.get("field") == "invoice_id":
                    invoice_id = c.get("value")
                if c.get("field") == "payment_amount_inr":
                    amount = c.get("value")
            steps = [
                PlanStepRecord(
                    operator="record_payment",
                    args={"invoice_id": invoice_id, "amount_inr": amount},
                    verification=["payment_posted"],
                ),
                PlanStepRecord(operator="respond", args={}, verification=["claims_match_structure"]),
            ]
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation="Record payment plan.",
                cost=2.0,
            )
            state.set_plan(plan)
            return plan

        if op_name in ("query_payments", "query_contracts"):
            steps = [
                PlanStepRecord(
                    operator="retrieve_records",
                    args={"project_id": frame.get("project_id"), "operation": op_name},
                    verification=["record_count_available"],
                ),
                PlanStepRecord(
                    operator="reason_over_records",
                    args={},
                    verification=["conclusion_present"],
                ),
                PlanStepRecord(operator="build_table", args={}, verification=["table_row_count"]),
                PlanStepRecord(operator="respond", args={}, verification=["claims_match_structure"]),
            ]
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation=f"Query plan for {op_name}.",
                cost=float(len(steps)),
            )
            state.set_plan(plan)
            return plan


        if op_name == "query_alerts":
            steps = [
                PlanStepRecord(
                    operator="scan_alerts",
                    args={"project_id": frame.get("project_id")},
                    verification=["alerts_ready"],
                ),
                PlanStepRecord(operator="respond", args={}, verification=["claims_match_structure"]),
            ]
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation="Proactive alerts scan.",
                cost=2.0,
            )
            state.set_plan(plan)
            return plan

        if op_name == "query_vendors":
            steps = [
                PlanStepRecord(
                    operator="retrieve_records",
                    args={"project_id": frame.get("project_id"), "operation": op_name},
                    verification=["record_count_available"],
                ),
                PlanStepRecord(operator="reason_over_records", args={}, verification=["conclusion_present"]),
                PlanStepRecord(operator="build_table", args={}, verification=["table_row_count"]),
                PlanStepRecord(operator="respond", args={}, verification=["claims_match_structure"]),
            ]
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation="Vendor list plan.",
                cost=4.0,
            )
            state.set_plan(plan)
            return plan

        if op_name == "query_purchase_orders":
            steps = [
                PlanStepRecord(
                    operator="retrieve_records",
                    args={"project_id": frame.get("project_id"), "operation": op_name},
                    verification=["record_count_available"],
                ),
                PlanStepRecord(operator="reason_over_records", args={}, verification=["conclusion_present"]),
                PlanStepRecord(operator="build_table", args={}, verification=["table_row_count"]),
                PlanStepRecord(operator="respond", args={}, verification=["claims_match_structure"]),
            ]
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation="Purchase order query plan.",
                cost=4.0,
            )
            state.set_plan(plan)
            return plan

        if op_name == "create_purchase_order":
            args = {"project_id": frame.get("project_id")}
            for c in frame.get("conditions") or []:
                if c.get("field") in ("vendor_id", "item", "qty", "unit_price_inr", "notes"):
                    args[c["field"]] = c.get("value")
            steps = [
                PlanStepRecord(
                    operator="create_purchase_order",
                    args=args,
                    verification=["po_created"],
                ),
                PlanStepRecord(operator="respond", args={}, verification=["claims_match_structure"]),
            ]
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation="Create purchase order plan.",
                cost=2.0,
            )
            state.set_plan(plan)
            return plan

        if op_name == "export_report":
            kind = None
            for c in frame.get("conditions") or []:
                if c.get("field") == "report_kind":
                    kind = c.get("value")
            steps = [
                PlanStepRecord(
                    operator="export_report",
                    args={"kind": kind, "project_id": frame.get("project_id"), "format": "csv"},
                    verification=["report_ready"],
                ),
                PlanStepRecord(operator="respond", args={}, verification=["claims_match_structure"]),
            ]
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation="Report export plan.",
                cost=2.0,
            )
            state.set_plan(plan)
            return plan

        vision_ops = {
            "detect_objects",
            "detect_vehicles",
            "count_people",
            "detect_fire",
            "scan_document_vision",
            "vision_status",
            "ppe_compliance",
            "estimate_queue",
        }
        if op_name in vision_ops:
            image_path = None
            vision_task = op_name
            for c in frame.get("conditions") or []:
                if c.get("field") == "image_path":
                    image_path = c.get("value")
                if c.get("field") == "vision_task":
                    vision_task = c.get("value") or vision_task
            if op_name == "scan_document_vision":
                vision_task = "scan_document"
            elif op_name == "vision_status":
                vision_task = "vision_status"
            steps = [
                PlanStepRecord(
                    operator="vision_analyze",
                    args={"image_path": image_path, "task": vision_task, "operation": op_name},
                    verification=["vision_result_present"],
                ),
                PlanStepRecord(operator="respond", args={}, verification=["claims_match_structure"]),
            ]
            plan = PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="feasible",
                explanation=f"Computer vision plan ({vision_task}).",
                cost=2.0,
            )
            state.set_plan(plan)
            return plan

        # Default query workflow
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

        if reasoning and not reasoning.success:
            return PlanRecord(
                goal_id=goal.goal_id,
                steps=steps,
                outcome="no_solution",
                explanation=reasoning.message,
                cost=float(len(steps)),
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
