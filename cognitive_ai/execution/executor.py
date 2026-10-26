"""Perform authorized operations with idempotent attempt ids."""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from ..core.records import ActionResult, ActionStatus, PlanRecord, PlanStepRecord, utc_now
from ..core.working_state import WorkingState
from ..memory.store import MemorySystem
from ..reasoning.engine import ReasoningEngine, ReasoningResult
from ..world.model import WorldModel
from .verifier import Verifier


class Executor:
    """Durable-ish in-process executor with duplicate suppression."""

    def __init__(
        self,
        memory: MemorySystem,
        reasoner: ReasoningEngine,
        world: Optional[WorldModel] = None,
        verifier: Optional[Verifier] = None,
    ) -> None:
        self.memory = memory
        self.reasoner = reasoner
        self.world = world or WorldModel()
        self.verifier = verifier or Verifier(memory)
        self._attempts: dict[str, ActionResult] = {}
        self._drafts: dict[str, dict[str, Any]] = {}
        self.last_reasoning: Optional[ReasoningResult] = None
        self.last_table: Optional[dict[str, Any]] = None

    def run_plan(self, plan: PlanRecord, state: WorkingState) -> list[ActionResult]:
        results: list[ActionResult] = []
        # Recheck conditions before executing
        frame = state.semantic_frame or {}
        prohibitions = set(frame.get("prohibitions") or [])

        for step in plan.steps:
            if step.status == "invalidated":
                continue
            if step.operator == "draft_followup" and "send" in prohibitions:
                step.args["send"] = False
            result = self.execute_step(step, state, client_attempt_id=step.step_id)
            results.append(result)
            state.record_action(result)
            if result.status in (ActionStatus.FAILED, ActionStatus.OUTCOME_UNKNOWN):
                break
            step.status = "done" if result.status == ActionStatus.CONFIRMED else result.status.value
        return results

    def execute_step(
        self,
        step: PlanStepRecord,
        state: WorkingState,
        client_attempt_id: Optional[str] = None,
    ) -> ActionResult:
        attempt_id = client_attempt_id or uuid4().hex
        # Duplicate-action suppression
        if attempt_id in self._attempts:
            return self._attempts[attempt_id]

        result = ActionResult(
            attempt_id=attempt_id,
            operator=step.operator,
            status=ActionStatus.RUNNING,
            args=dict(step.args),
        )
        self._attempts[attempt_id] = result

        try:
            effects = self._dispatch(step, state, result)
            result.observed_effects = effects
            # Verify independently
            verified = self.verifier.verify(step, result, state)
            if verified is True:
                result.status = ActionStatus.CONFIRMED
            elif verified is None:
                result.status = ActionStatus.OUTCOME_UNKNOWN
            else:
                result.status = ActionStatus.FAILED
                result.error = "verification_failed"
        except Exception as exc:  # noqa: BLE001 — surface as failed action
            result.status = ActionStatus.FAILED
            result.error = str(exc)

        result.finished_at = utc_now()
        self._attempts[attempt_id] = result
        return result

    def _dispatch(
        self,
        step: PlanStepRecord,
        state: WorkingState,
        result: ActionResult,
    ) -> list[dict[str, Any]]:
        op = step.operator
        if op == "retrieve_records":
            frame = state.semantic_frame or {}
            operation = frame.get("operation")
            kind_map = {
                "query_invoices": "invoice",
                "query_deliveries": "delivery",
                "prepare_followup": "delivery",
                "query_materials": "material",
                "manage_task": "task",
                "document_qa": "document",
            }
            kind = kind_map.get(operation, "invoice")
            rows = self.memory.get_records(kind)
            pid = step.args.get("project_id") or frame.get("project_id")
            if pid:
                rows = [r for r in rows if r.get("project_id") == pid]
            result.external_receipt = f"retrieved:{kind}:{len(rows)}"
            return [{"records_loaded": True, "kind": kind, "count": len(rows), "rows": rows}]

        if op == "reason_over_records":
            self.last_reasoning = self.reasoner.reason(state)
            result.external_receipt = self.last_reasoning.result_id
            return [{"conclusions_ready": True, "reasoning": self.last_reasoning.to_dict()}]

        if op == "build_table":
            reasoning = self.last_reasoning or self.reasoner.reason(state)
            self.last_reasoning = reasoning
            table = self._table_from_reasoning(reasoning)
            self.last_table = table
            result.external_receipt = f"table:{table.get('row_count', 0)}"
            return [{"table_ready": True, "table": table}]

        if op == "draft_followup":
            table = self.last_table or {"rows": [], "row_count": 0}
            draft_id = f"draft_{result.attempt_id}"
            body = self._compose_followup(table, state)
            send = bool(step.args.get("send"))
            if send:
                raise PermissionError("Sending is prohibited by the active goal.")
            draft = {
                "draft_id": draft_id,
                "body": body,
                "sent": False,
                "table": table,
            }
            self._drafts[draft_id] = draft
            result.external_receipt = draft_id
            return [{"draft_ready": True, "draft": draft, "sent": False}]

        if op == "clarify":
            qs = step.args.get("questions") or state.dialogue.unresolved_questions
            return [{"clarification_requested": True, "questions": qs}]

        if op == "respond":
            return [{"answer_ready": True}]

        if op == "reserve_stock":
            item_id = step.args.get("item")
            qty = float(step.args.get("quantity") or 0)
            mats = self.memory.get_records("material", id=item_id) if item_id else []
            if not mats:
                # try by name
                mats = [m for m in self.memory.get_records("material") if m.get("id") == item_id]
            if not mats:
                raise ValueError("item not found")
            mat = mats[0]
            predicted = self.world.predict(mat, "reserve_stock", {"quantity": qty})
            if predicted.get("error"):
                raise ValueError(predicted["error"])
            # Simulate external call — may leave OUTCOME_UNKNOWN if flagged
            if step.args.get("simulate_timeout"):
                result.status = ActionStatus.OUTCOME_UNKNOWN
                result.external_receipt = None
                return [{"attempted": True, "confirmed": False}]
            new_state = predicted["state"]
            self.memory.upsert_record("material", mat["id"], new_state)
            result.external_receipt = f"reservation:{mat['id']}:{qty}"
            return [{"reserved": qty, "item": mat["id"], **new_state}]

        raise ValueError(f"Unknown operator: {op}")

    def _table_from_reasoning(self, reasoning: ReasoningResult) -> dict[str, Any]:
        if not reasoning.conclusions:
            return {"columns": [], "rows": [], "row_count": 0, "meta": {}}
        conc = reasoning.conclusions[0]
        ctype = conc.get("type")
        if ctype == "invoice_set":
            rows = []
            for inv in conc.get("invoices") or []:
                rows.append(
                    {
                        "invoice_id": inv.get("id"),
                        "project_id": inv.get("project_id"),
                        "amount_inr": inv.get("amount_inr"),
                        "payment_status": inv.get("payment_status"),
                        "dispute_status": inv.get("dispute_status"),
                        "due_date": inv.get("due_date"),
                        "source": inv.get("source"),
                        "as_of": inv.get("status_as_of") or inv.get("updated_at"),
                    }
                )
            return {
                "columns": list(rows[0].keys()) if rows else [],
                "rows": rows,
                "row_count": len(rows),
                "meta": {
                    "total_inr": conc.get("total_inr"),
                    "unresolved": conc.get("unresolved") or [],
                },
            }
        if ctype == "pending_deliveries":
            rows = []
            for d in conc.get("pending") or []:
                rows.append(
                    {
                        "delivery_id": d.get("id"),
                        "item": d.get("item"),
                        "project_id": d.get("project_id"),
                        "pending_qty": d.get("pending_qty"),
                        "expected_qty": d.get("expected_qty"),
                        "received_qty": d.get("received_qty"),
                        "source": d.get("source"),
                    }
                )
            return {
                "columns": list(rows[0].keys()) if rows else [],
                "rows": rows,
                "row_count": len(rows),
                "meta": {
                    "uncertain_receipts": conc.get("uncertain_receipts") or [],
                    "count_uncertain": conc.get("count_uncertain", 0),
                },
            }
        if ctype == "materials":
            rows = conc.get("items") or []
            return {
                "columns": list(rows[0].keys()) if rows else [],
                "rows": rows,
                "row_count": len(rows),
                "meta": {},
            }
        return {"columns": [], "rows": [], "row_count": 0, "meta": {"raw": conc}}

    def _compose_followup(self, table: dict[str, Any], state: WorkingState) -> str:
        pid = (state.semantic_frame or {}).get("project_id") or "the project"
        lines = [
            f"Follow-up regarding pending deliveries for {pid}:",
            "",
        ]
        if not table.get("rows"):
            lines.append("No confirmed pending items were identified.")
        else:
            lines.append("Confirmed pending items:")
            for row in table["rows"]:
                lines.append(
                    f"- {row.get('item')}: pending qty {row.get('pending_qty')} "
                    f"(expected {row.get('expected_qty')}, received {row.get('received_qty')})"
                )
        uncertain = (table.get("meta") or {}).get("uncertain_receipts") or []
        if uncertain:
            lines.append("")
            lines.append("Unresolved receipt status (not treated as confirmed pending):")
            for u in uncertain:
                lines.append(f"- {u.get('item') or u.get('id')}: receipt status unknown")
        lines.append("")
        lines.append("This draft was prepared and not sent.")
        return "\n".join(lines)
