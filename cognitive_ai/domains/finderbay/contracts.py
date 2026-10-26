"""Contracts domain: agreements linked to projects, invoices and deliveries."""

from __future__ import annotations

from typing import Any, Optional

from ...core.ids import new_id
from ...core.records import utc_now
from ...memory.store import MemorySystem
from .auth import AuthService


class ContractService:
    def __init__(self, memory: MemorySystem, auth: AuthService) -> None:
        self.memory = memory
        self.auth = auth

    def list_contracts(self, *, project_id: Optional[str] = None, status: Optional[str] = None) -> list[dict]:
        rows = self.memory.get_records("contract")
        if project_id:
            rows = [r for r in rows if r.get("project_id") == project_id]
        if status:
            rows = [r for r in rows if r.get("status") == status]
        return rows

    def create_contract(
        self,
        *,
        token: Optional[str],
        project_id: str,
        title: str,
        supplier: str,
        value_inr: float,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        self.auth.require(token, "contract_write")
        cid = new_id("ctr")
        row = {
            "id": cid,
            "project_id": project_id,
            "title": title,
            "supplier": supplier,
            "value_inr": float(value_inr),
            "start_date": start_date,
            "end_date": end_date,
            "status": "active",
            "created_at": utc_now(),
        }
        self.memory.upsert_record("contract", cid, row)
        return row

    def link_invoice(self, *, token: Optional[str], contract_id: str, invoice_id: str) -> dict[str, Any]:
        self.auth.require(token, "contract_write")
        contracts = self.memory.get_records("contract", id=contract_id)
        invoices = self.memory.get_records("invoice", id=invoice_id)
        if not contracts:
            raise ValueError("contract_not_found")
        if not invoices:
            raise ValueError("invoice_not_found")
        inv = invoices[0]
        self.memory.upsert_record("invoice", invoice_id, {**inv, "contract_id": contract_id})
        return {"contract_id": contract_id, "invoice_id": invoice_id, "linked": True}

    def summarize(self, project_id: Optional[str] = None) -> dict[str, Any]:
        rows = self.list_contracts(project_id=project_id)
        active = [r for r in rows if r.get("status") == "active"]
        total = sum(float(r.get("value_inr") or 0) for r in active)
        return {"count": len(rows), "active": len(active), "active_value_inr": total, "contracts": rows}
