"""Perform authorized operations with idempotent attempt ids."""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from ..core.ids import new_id
from ..core.records import ActionResult, ActionStatus, PlanRecord, PlanStepRecord, utc_now
from ..core.working_state import WorkingState
from ..memory.store import MemorySystem
from ..reasoning.engine import ReasoningEngine, ReasoningResult
from ..world.model import WorldModel
from .recovery import RecoveryManager
from .tools import ToolRegistry
from .verifier import Verifier


class Executor:
    """Durable-ish in-process executor with duplicate suppression."""

    def __init__(
        self,
        memory: MemorySystem,
        reasoner: ReasoningEngine,
        world: Optional[WorldModel] = None,
        verifier: Optional[Verifier] = None,
        tools: Optional[ToolRegistry] = None,
        recovery: Optional[RecoveryManager] = None,
        *,
        allow_send: bool = False,
        allow_inventory_write: bool = True,
    ) -> None:
        self.memory = memory
        self.reasoner = reasoner
        self.world = world or WorldModel()
        self.verifier = verifier or Verifier(memory)
        self.tools = tools or ToolRegistry()
        self.recovery = recovery or RecoveryManager()
        self.allow_send = allow_send
        self.allow_inventory_write = allow_inventory_write
        self._attempts: dict[str, ActionResult] = {}
        self._drafts: dict[str, dict[str, Any]] = {}
        self._checkpoints: list[dict[str, Any]] = []
        self.last_reasoning: Optional[ReasoningResult] = None
        self.last_table: Optional[dict[str, Any]] = None
        self.last_document: Optional[dict[str, Any]] = None
        self.last_task: Optional[dict[str, Any]] = None
        self.last_reservation: Optional[dict[str, Any]] = None
        self.last_alias: Optional[dict[str, Any]] = None

    def run_plan(self, plan: PlanRecord, state: WorkingState) -> list[ActionResult]:
        results: list[ActionResult] = []
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
            self._checkpoints.append(
                {"plan_id": plan.plan_id, "step_id": step.step_id, "result": result.to_dict()}
            )
            self.recovery.checkpoint(plan.plan_id, step.step_id, result)
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
        if attempt_id in self._attempts:
            return self._attempts[attempt_id]

        # Tool permission / schema gate when registered
        tool = self.tools.get(step.operator)
        if tool:
            if tool.requires_auth:
                if step.operator == "send_message" and not self.allow_send:
                    result = ActionResult(
                        attempt_id=attempt_id,
                        operator=step.operator,
                        status=ActionStatus.FAILED,
                        args=dict(step.args),
                        error="permission_denied:send",
                        finished_at=utc_now(),
                    )
                    self._attempts[attempt_id] = result
                    return result
                if step.operator == "reserve_stock" and not self.allow_inventory_write:
                    result = ActionResult(
                        attempt_id=attempt_id,
                        operator=step.operator,
                        status=ActionStatus.FAILED,
                        args=dict(step.args),
                        error="permission_denied:inventory_write",
                        finished_at=utc_now(),
                    )
                    self._attempts[attempt_id] = result
                    return result

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
            verified = self.verifier.verify(step, result, state)
            if verified is True:
                result.status = ActionStatus.CONFIRMED
            elif verified is None:
                result.status = ActionStatus.OUTCOME_UNKNOWN
            else:
                result.status = ActionStatus.FAILED
                result.error = result.error or "verification_failed"
        except Exception as exc:  # noqa: BLE001
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
            if pid and kind != "material":
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
            draft = {"draft_id": draft_id, "body": body, "sent": False, "table": table}
            self._drafts[draft_id] = draft
            result.external_receipt = draft_id
            return [{"draft_ready": True, "draft": draft, "sent": False}]

        if op == "clarify":
            qs = step.args.get("questions") or state.dialogue.unresolved_questions
            return [{"clarification_requested": True, "questions": qs}]

        if op == "respond":
            return [{"answer_ready": True}]

        if op == "create_task":
            title = step.args.get("title") or ""
            pid = step.args.get("project_id")
            tid = new_id("task")
            row = {
                "id": tid,
                "title": title,
                "project_id": pid,
                "status": "open",
                "source": "user",
            }
            self.memory.upsert_record("task", tid, row)
            self.last_task = row
            result.external_receipt = tid
            return [{"task_created": True, "task": row}]

        if op == "list_tasks":
            pid = step.args.get("project_id")
            rows = self.memory.get_records("task")
            if pid:
                rows = [r for r in rows if r.get("project_id") == pid]
            self.last_table = {
                "columns": list(rows[0].keys()) if rows else ["id", "title", "status"],
                "rows": rows,
                "row_count": len(rows),
                "meta": {"kind": "task"},
            }
            result.external_receipt = f"tasks:{len(rows)}"
            return [{"tasks_listed": True, "count": len(rows), "rows": rows}]

        if op == "ocr_extract":
            docs = self.memory.get_records("document")
            doc = docs[0] if docs else None
            # Prefer document attached via observation/context
            for obs in reversed(state.observations):
                if obs.modality in ("document", "image") and isinstance(obs.extracted_value, dict):
                    doc = {"id": "obs_doc", **obs.extracted_value}
                    break
            if not doc:
                raise ValueError("no_document_available")
            from ..perception.ocr import DocumentPipeline

            parsed = DocumentPipeline().recognize(doc)
            field = step.args.get("field")
            value = None
            if field:
                value = (parsed.get("fields") or {}).get(field)
            self.last_document = {"document": doc, "parsed": parsed, "field": field, "value": value}
            result.external_receipt = f"ocr:{doc.get('id')}"
            effects = {"fields_ready": True, "parsed": parsed, "field": field, "value": value}
            if parsed.get("issues"):
                effects["issues"] = parsed["issues"]
            return [effects]

        if op == "reserve_stock":
            return self._reserve(step, state, result)

        if op == "learn_alias":
            frame = state.semantic_frame or {}
            old_name = None
            new_name = None
            for c in frame.get("conditions") or []:
                if c.get("field") == "alias_of":
                    old_name = c.get("value")
                if c.get("field") == "alias":
                    new_name = c.get("value")
            if not old_name or not new_name:
                raise ValueError("alias fields missing")
            # Map old warehouse name → canonical entity if known, else store both ways
            canon = self.memory.ontology.resolve(str(old_name)) or str(old_name).lower().replace(" ", "_")
            self.memory.learn_alias(str(new_name), canon, source="user")
            # Also allow looking up by old name
            self.memory.learn_alias(str(old_name), canon, source="user")
            payload = {"alias": new_name, "canonical": canon, "former": old_name}
            self.last_alias = payload
            result.external_receipt = f"alias:{canon}:{new_name}"
            return [{"alias_stored": True, **payload}]

        raise ValueError(f"Unknown operator: {op}")

    def _reserve(
        self,
        step: PlanStepRecord,
        state: WorkingState,
        result: ActionResult,
    ) -> list[dict[str, Any]]:
        item_id = step.args.get("item")
        qty = step.args.get("quantity")
        material_name = step.args.get("material_name")
        frame = state.semantic_frame or {}
        if qty is None or material_name is None:
            for c in frame.get("conditions") or []:
                if c.get("field") == "reserve_qty" and qty is None:
                    qty = c.get("value")
                if c.get("field") == "material_name" and material_name is None:
                    material_name = c.get("value")
        qty = float(qty or 0)
        mats = []
        if item_id:
            mats = [m for m in self.memory.get_records("material") if m.get("id") == item_id]
        if not mats and material_name:
            key = str(material_name).lower().strip()
            for m in self.memory.get_records("material"):
                name = str(m.get("name") or "").lower()
                if key in name or name in key:
                    mats.append(m)
        if not mats:
            raise ValueError("item not found")
        mat = mats[0]
        predicted = self.world.predict(mat, "reserve_stock", {"quantity": qty})
        if predicted.get("error"):
            raise ValueError(predicted["error"])
        if step.args.get("simulate_timeout"):
            result.status = ActionStatus.OUTCOME_UNKNOWN
            result.external_receipt = None
            return [{"attempted": True, "confirmed": False}]
        new_state = predicted["state"]
        new_state["last_attempt"] = result.attempt_id
        self.memory.upsert_record("material", mat["id"], new_state)
        payload = {
            **new_state,
            "item": mat["id"],
            "name": mat.get("name"),
            "qty_reserved": qty,
            "available": new_state.get("available"),
            "reserved_balance": new_state.get("reserved"),
        }
        self.last_reservation = payload
        result.external_receipt = f"reservation:{mat['id']}:{qty}"
        return [payload]

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
        if ctype == "tasks":
            rows = conc.get("items") or []
            return {
                "columns": list(rows[0].keys()) if rows else [],
                "rows": rows,
                "row_count": len(rows),
                "meta": {"kind": "task"},
            }
        return {"columns": [], "rows": [], "row_count": 0, "meta": {"raw": conc}}

    def _compose_followup(self, table: dict[str, Any], state: WorkingState) -> str:
        pid = (state.semantic_frame or {}).get("project_id") or "the project"
        lines = [f"Follow-up regarding pending deliveries for {pid}:", ""]
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
