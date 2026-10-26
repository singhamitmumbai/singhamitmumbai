"""Functional capability / limitation model — not a claim of consciousness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CapabilityEntry:
    name: str
    available: bool = True
    tools: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    last_eval_accuracy: Optional[float] = None
    notes: str = ""


class CapabilityModel:
    """Supports statements like: I can read the printed table, but handwriting is ambiguous."""

    def __init__(self) -> None:
        self.entries: dict[str, CapabilityEntry] = {}
        self.access: dict[str, bool] = {
            "invoice_register": True,
            "delivery_register": True,
            "inventory": True,
            "send_email": False,  # forbidden by default in Finderbay demo
            "acoustic_model": False,
            "ocr_engine": False,
        }
        self.resource_budgets: dict[str, Any] = {
            "time_ms": 10000,
            "steps": 50,
            "external_calls": 20,
        }
        self.unresolved_failures: list[str] = []
        self._seed()

    def _seed(self) -> None:
        for name, tools, limits in [
            (
                "query_invoices",
                ["retrieve_records", "reason_over_records", "build_table", "respond"],
                ["typed text only unless speech model loaded", "known project ids"],
            ),
            (
                "query_deliveries",
                ["retrieve_records", "reason_over_records", "build_table", "respond"],
                ["missing receipts stay UNKNOWN"],
            ),
            (
                "prepare_followup",
                ["draft_followup"],
                ["will not send unless explicitly authorized and capability enabled"],
            ),
            (
                "query_materials",
                ["retrieve_records", "respond"],
                ["requires authorized inventory access"],
            ),
            (
                "reserve_stock",
                ["reserve_stock"],
                ["requires authorized inventory write access", "verified reservation receipt"],
            ),
            (
                "manage_task",
                ["create_task", "list_tasks"],
                ["local store only"],
            ),
            (
                "document_qa",
                ["ocr_extract", "field_validate"],
                ["handwritten amounts may be ambiguous", "full OCR engine optional"],
            ),
            (
                "speech",
                ["speech_recognize"],
                ["acoustic model not loaded by default — accept transcript dicts"],
            ),
            (
                "learn_alias",
                ["learn_alias"],
                ["requires reviewed user assertion; scoped aliases"],
            ),
            (
                "query_payments",
                ["retrieve_records", "respond"],
                ["requires read access"],
            ),
            (
                "record_payment",
                ["record_payment"],
                ["requires accountant/manager auth token"],
            ),
            (
                "query_contracts",
                ["retrieve_records", "respond"],
                ["requires read access"],
            ),
            (
                "login",
                ["login"],
                ["multi-user local auth; passwords are demo-only"],
            ),
        ]:
            self.entries[name] = CapabilityEntry(name=name, tools=tools, limitations=limits)

    def update_eval(self, name: str, accuracy: float) -> None:
        if name in self.entries:
            self.entries[name].last_eval_accuracy = accuracy

    def describe(self, name: str) -> str:
        e = self.entries.get(name)
        if not e:
            return f"Capability '{name}' is not registered (unsupported task)."
        if not e.available:
            return f"Capability '{name}' is currently unavailable."
        lim = "; ".join(e.limitations) if e.limitations else "no special limitations recorded"
        acc = (
            f" Last eval accuracy={e.last_eval_accuracy:.0%}."
            if e.last_eval_accuracy is not None
            else ""
        )
        return f"I can perform '{name}' using {', '.join(e.tools)}. Limits: {lim}.{acc}"

    def can(self, name: str) -> bool:
        e = self.entries.get(name)
        return bool(e and e.available)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capabilities": {
                k: {
                    "available": v.available,
                    "tools": v.tools,
                    "limitations": v.limitations,
                    "last_eval_accuracy": v.last_eval_accuracy,
                }
                for k, v in self.entries.items()
            },
            "access": dict(self.access),
            "resource_budgets": dict(self.resource_budgets),
            "unresolved_failures": list(self.unresolved_failures),
        }
