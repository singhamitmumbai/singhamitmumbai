"""Payments domain: record settlements against invoices with auth checks."""

from __future__ import annotations

from typing import Any, Optional

from ...core.ids import new_id
from ...core.records import Claim, EvidenceStatus, utc_now
from ...memory.store import MemorySystem
from .auth import AuthService


class PaymentService:
    def __init__(self, memory: MemorySystem, auth: AuthService) -> None:
        self.memory = memory
        self.auth = auth

    def list_payments(self, *, project_id: Optional[str] = None, invoice_id: Optional[str] = None) -> list[dict]:
        rows = self.memory.get_records("payment")
        if project_id:
            rows = [r for r in rows if r.get("project_id") == project_id]
        if invoice_id:
            rows = [r for r in rows if r.get("invoice_id") == invoice_id]
        return rows

    def record_payment(
        self,
        *,
        token: Optional[str],
        invoice_id: str,
        amount_inr: float,
        method: str = "bank_transfer",
        reference: str = "",
    ) -> dict[str, Any]:
        self.auth.require(token, "payment_write")
        invoices = self.memory.get_records("invoice", id=invoice_id)
        if not invoices:
            raise ValueError(f"invoice_not_found:{invoice_id}")
        inv = invoices[0]
        pid = new_id("pay")
        row = {
            "id": pid,
            "invoice_id": invoice_id,
            "project_id": inv.get("project_id"),
            "amount_inr": float(amount_inr),
            "method": method,
            "reference": reference,
            "paid_at": utc_now(),
            "status": "posted",
        }
        self.memory.upsert_record("payment", pid, row)

        # Settle invoice when payment covers amount
        inv_amount = float(inv.get("amount_inr") or 0)
        paid_total = sum(float(p.get("amount_inr") or 0) for p in self.list_payments(invoice_id=invoice_id))
        if paid_total + 1e-6 >= inv_amount:
            self.memory.upsert_record(
                "invoice",
                invoice_id,
                {**inv, "payment_status": "paid", "paid_total": paid_total},
            )
            self.memory.remember_claim(
                Claim(
                    subject=invoice_id,
                    relationship="payment_status",
                    value="paid",
                    source="payment_service",
                    evidence_status=EvidenceStatus.OBSERVED,
                    project_id=inv.get("project_id"),
                )
            )
        else:
            self.memory.upsert_record(
                "invoice",
                invoice_id,
                {**inv, "payment_status": "partial", "paid_total": paid_total},
            )
        return {"payment": row, "invoice_paid_total": paid_total, "invoice_amount": inv_amount}
