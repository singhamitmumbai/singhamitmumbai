"""Choose a reasoning mechanism according to the problem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import Claim, EvidenceStatus
from ..core.working_state import WorkingState
from ..memory.store import MemorySystem


@dataclass
class Explanation:
    label: str
    likelihood: float
    evidence: list[str] = field(default_factory=list)
    status: str = "open"  # open|supported|ruled_out


@dataclass
class ReasoningResult:
    method: str
    conclusions: list[dict[str, Any]] = field(default_factory=list)
    explanations: list[Explanation] = field(default_factory=list)
    rule_version: str = "finderbay_rules_v1"
    success: bool = True
    message: str = ""
    result_id: str = field(default_factory=lambda: new_id("reason"))

    def to_dict(self) -> dict:
        return {
            "result_id": self.result_id,
            "method": self.method,
            "conclusions": self.conclusions,
            "explanations": [e.__dict__ for e in self.explanations],
            "rule_version": self.rule_version,
            "success": self.success,
            "message": self.message,
        }


class ReasoningEngine:
    """Production rules, typed arithmetic, and belief-style alternatives."""

    def __init__(self, memory: MemorySystem) -> None:
        self.memory = memory

    def reason(self, state: WorkingState, frame: Optional[dict] = None) -> ReasoningResult:
        frame = frame or state.semantic_frame or {}
        op = frame.get("operation")
        if op == "query_invoices":
            return self._reason_invoices(state, frame)
        if op in ("query_deliveries", "prepare_followup"):
            return self._reason_deliveries(state, frame)
        if op == "query_materials":
            return self._reason_materials(state, frame)
        if op == "manage_task":
            return self._reason_tasks(state, frame)
        if op == "document_qa":
            return ReasoningResult(
                method="document_passthrough",
                success=True,
                message="Document fields handled by OCR extract operator.",
            )
        if op == "reserve_stock":
            return ReasoningResult(
                method="world_model",
                success=True,
                message="Reservation handled by world-model transition.",
            )
        if op == "query_payments":
            rows = self.memory.get_records("payment")
            pid = frame.get("project_id")
            if pid:
                rows = [r for r in rows if r.get("project_id") == pid]
            return ReasoningResult(
                method="lookup",
                conclusions=[{"type": "payments", "items": rows, "count": len(rows)}],
                message=f"{len(rows)} payment(s).",
            )
        if op == "query_contracts":
            rows = self.memory.get_records("contract")
            pid = frame.get("project_id")
            if pid:
                rows = [r for r in rows if r.get("project_id") == pid]
            total = sum(float(r.get("value_inr") or 0) for r in rows if r.get("status") == "active")
            return ReasoningResult(
                method="lookup",
                conclusions=[{
                    "type": "contracts",
                    "items": rows,
                    "count": len(rows),
                    "active_value_inr": total,
                }],
                message=f"{len(rows)} contract(s); active value INR {total:.0f}.",
            )
        if op == "query_purchase_orders":
            rows = self.memory.get_records("purchase_order")
            pid = frame.get("project_id")
            if pid:
                rows = [r for r in rows if r.get("project_id") == pid]
            open_value = sum(float(r.get("total_inr") or 0) for r in rows if r.get("status") == "open")
            return ReasoningResult(
                method="lookup",
                conclusions=[{
                    "type": "purchase_orders",
                    "items": rows,
                    "count": len(rows),
                    "open_value_inr": open_value,
                }],
                message=f"{len(rows)} purchase order(s); open value INR {open_value:.0f}.",
            )
        if op == "query_vendors":
            rows = self.memory.get_records("vendor")
            return ReasoningResult(
                method="lookup",
                conclusions=[{"type": "vendors", "items": rows, "count": len(rows)}],
                message=f"{len(rows)} vendor(s).",
            )
        if op == "query_alerts":
            return ReasoningResult(
                method="proactive_monitor",
                success=True,
                message="Alerts produced by proactive monitor operator.",
            )
        if op == "export_report":
            return ReasoningResult(
                method="report_export",
                success=True,
                message="Report export handled by exporter operator.",
            )
        if op == "create_purchase_order":
            return ReasoningResult(
                method="purchase_order_write",
                success=True,
                message="Purchase order creation handled by PO service.",
            )
        return ReasoningResult(
            method="passthrough",
            success=True,
            message="No specialised reasoning required for this operation.",
        )

    # --- invoice rules --------------------------------------------------------

    def _reason_invoices(self, state: WorkingState, frame: dict) -> ReasoningResult:
        project_id = frame.get("project_id")
        invoices = self.memory.get_records("invoice")
        if project_id:
            invoices = [i for i in invoices if i.get("project_id") == project_id]

        conditions = frame.get("conditions") or []
        matched = []
        unresolved = []
        today = date.today().isoformat()

        META_FIELDS = {
            "aggregation",
            "ask_form",
            "report_kind",
            "output",
            "clause_index",
        }
        for inv in invoices:
            ok = True
            for cond in conditions:
                field = cond["field"]
                op = cond["operator"]
                val = cond["value"]
                if field in META_FIELDS:
                    continue
                if field == "overdue":
                    due = inv.get("due_date")
                    status = inv.get("payment_status")
                    freshness = inv.get("status_as_of") or inv.get("updated_at")
                    if not due or status != "unpaid":
                        ok = False
                        break
                    if due >= today:
                        ok = False
                        break
                    # Retain freshness qualification
                    inv = {
                        **inv,
                        "_overdue_rule": {
                            "due_date": due,
                            "reference_date": today,
                            "payment_status": status,
                            "freshness": freshness,
                            "rule_version": "overdue_v1",
                        },
                    }
                    continue
                actual = inv.get(field)
                if actual is None and field in ("dispute_status", "payment_status"):
                    unresolved.append(inv)
                    policy = (frame.get("unknown_policies") or {}).get(field, "exclude")
                    if policy == "report_separately":
                        ok = False
                    continue
                if not self._compare(actual, op, val):
                    ok = False
                    break
            if ok:
                matched.append(inv)

        total = sum(float(i.get("amount_inr") or 0) for i in matched)
        conclusions = [
            {
                "type": "invoice_set",
                "count": len(matched),
                "total_inr": total,
                "invoices": matched,
                "unresolved": unresolved,
            }
        ]
        # Write claims
        for inv in matched:
            state.add_claim(
                Claim(
                    subject=inv["id"],
                    relationship="selected_for_query",
                    value=True,
                    source="reasoning",
                    evidence_status=EvidenceStatus.INFERRED,
                    project_id=inv.get("project_id"),
                    confidence=0.9,
                )
            )

        return ReasoningResult(
            method="production_rules+arithmetic",
            conclusions=conclusions,
            message=f"Matched {len(matched)} invoices totaling INR {total:.2f}.",
        )

    # --- deliveries -----------------------------------------------------------

    def _reason_deliveries(self, state: WorkingState, frame: dict) -> ReasoningResult:
        project_id = frame.get("project_id")
        rows = self.memory.get_records("delivery")
        if project_id:
            rows = [r for r in rows if r.get("project_id") == project_id]

        # Time window filter
        window = None
        for c in frame.get("conditions") or []:
            if c.get("field") == "time_window":
                window = c.get("value")
            if c.get("field") == "delivery_status":
                pass
        if window:
            rows = [r for r in rows if r.get("time_window") == window or r.get("week") == window]

        pending = []
        uncertain = []
        for r in rows:
            expected = float(r.get("expected_qty") or 0)
            received = r.get("received_qty")
            receipt_status = r.get("receipt_status", "known")
            if receipt_status == "missing":
                # Competing explanations — do NOT treat as confirmed pending alone
                uncertain.append(r)
                continue
            if received is None:
                uncertain.append(r)
                continue
            received_f = float(received)
            if received_f < expected:
                pending.append({**r, "pending_qty": expected - received_f})

        explanations = [
            Explanation(
                "not_delivered",
                0.4,
                ["no confirmed receipt for expected quantity"],
            ),
            Explanation(
                "delivered_unrecorded",
                0.3,
                ["receipt may use different reference"],
            ),
            Explanation(
                "data_incomplete",
                0.3,
                ["available records incomplete"],
            ),
        ]
        for u in uncertain:
            state.add_claim(
                Claim(
                    subject=u["id"],
                    relationship="receipt_status",
                    value="unknown",
                    source="reasoning",
                    evidence_status=EvidenceStatus.UNKNOWN,
                    project_id=u.get("project_id"),
                    notes="Kept separate from confirmed pending items",
                )
            )

        conclusions = [
            {
                "type": "pending_deliveries",
                "pending": pending,
                "uncertain_receipts": uncertain,
                "count_pending": len(pending),
                "count_uncertain": len(uncertain),
            }
        ]
        return ReasoningResult(
            method="comparison+competing_explanations",
            conclusions=conclusions,
            explanations=explanations if uncertain else [],
            message=(
                f"{len(pending)} confirmed pending item(s); "
                f"{len(uncertain)} with unresolved receipt status."
            ),
        )

    def _reason_materials(self, state: WorkingState, frame: dict) -> ReasoningResult:
        rows = self.memory.get_records("material")
        conditions = frame.get("conditions") or []
        matched = []
        for mat in rows:
            ok = True
            for cond in conditions:
                field = cond.get("field")
                op = cond.get("operator")
                val = cond.get("value")
                if field in ("ask_form", "aggregation", "output", "document_field"):
                    continue
                if field == "stock_qty":
                    actual = mat.get("available")
                elif field == "material_tag":
                    tags = mat.get("tags") or []
                    name = str(mat.get("name") or "").lower()
                    if op == "ne":
                        # Exclude materials whose name/tags contain the tag
                        needle = str(val).lower()
                        if needle in name or needle in [str(t).lower() for t in tags]:
                            ok = False
                            break
                        continue
                    actual = tags
                elif field == "material_name":
                    actual = str(mat.get("name") or "").lower()
                    val = str(val).lower()
                else:
                    actual = mat.get(field)
                if field != "material_tag" and not self._compare(actual, op, val):
                    ok = False
                    break
            if ok:
                matched.append(mat)
        return ReasoningResult(
            method="lookup+filters",
            conclusions=[{"type": "materials", "items": matched}],
            message=f"{len(matched)} material record(s).",
        )

    def _reason_tasks(self, state: WorkingState, frame: dict) -> ReasoningResult:
        rows = self.memory.get_records("task")
        pid = frame.get("project_id")
        if pid:
            rows = [r for r in rows if r.get("project_id") == pid]
        return ReasoningResult(
            method="lookup",
            conclusions=[{"type": "tasks", "items": rows, "count": len(rows)}],
            message=f"{len(rows)} task(s).",
        )

    @staticmethod
    def _compare(actual: Any, op: str, expected: Any) -> bool:
        if op == "eq":
            return actual == expected
        if op == "ne":
            return actual != expected
        try:
            a = float(actual)
            e = float(expected)
        except (TypeError, ValueError):
            return False
        if op == "gt":
            return a > e
        if op == "gte":
            return a >= e
        if op == "lt":
            return a < e
        if op == "lte":
            return a <= e
        return False
