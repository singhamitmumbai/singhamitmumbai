"""Capability specification records (blueprint §3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CapabilitySpec:
    name: str
    task: str
    inputs: list[str]
    operating_conditions: list[str]
    output: str
    knowledge_required: list[str]
    permitted_actions: list[str]
    success_condition: str
    failure_behavior: str
    resource_limits: dict[str, Any] = field(default_factory=dict)
    learning_policy: str = "reviewed_examples_only"

    def to_dict(self) -> dict:
        return asdict(self)


FINDERBAY_CAPABILITIES: list[CapabilitySpec] = [
    CapabilitySpec(
        name="query_invoices",
        task="Given an invoice register and a supported query, identify requested invoices, calculate totals and show source records.",
        inputs=["text", "structured"],
        operating_conditions=["English or Hinglish typed text", "known project ids", "INR amounts"],
        output="table_with_sources + natural language summary",
        knowledge_required=["invoice register", "project ontology", "payment/dispute fields"],
        permitted_actions=["read_invoices", "calculate", "report"],
        success_condition="Matched set and total equal independent audit of the register under the same filters; sources linked.",
        failure_behavior="Clarify ambiguous time/status; report unresolved statuses separately; never treat unknown as unpaid.",
        resource_limits={"time_ms": 3000, "external_calls": 5},
    ),
    CapabilitySpec(
        name="pending_deliveries_followup",
        task="From delivery records for a time window and project, find confirmed pending items and prepare a follow-up draft without sending.",
        inputs=["text", "audio(optional)", "documents(optional)"],
        operating_conditions=["Project A-style ids", "this_week window", "missing receipts possible"],
        output="source-linked table + follow-up draft (not sent)",
        knowledge_required=["orders", "delivery records", "receipts"],
        permitted_actions=["read_deliveries", "draft", "report"],
        success_condition="Confirmed pending separated from unknown receipts; draft exists; send==false verified.",
        failure_behavior="Keep competing explanations for missing receipts; ask to clarify ambiguous 'kal'.",
        resource_limits={"time_ms": 5000, "external_calls": 10},
    ),
    CapabilitySpec(
        name="material_lookup",
        task="Report available/reserved stock for known materials.",
        inputs=["text", "structured"],
        operating_conditions=["known SKUs", "warehouse aliases"],
        output="material table",
        knowledge_required=["material records", "warehouse aliases"],
        permitted_actions=["read_materials"],
        success_condition="Figures match inventory service snapshot as_of timestamp.",
        failure_behavior="Report missing SKU; do not invent quantities.",
        resource_limits={"time_ms": 2000},
    ),
    CapabilitySpec(
        name="local_task_management",
        task="Create and list local tasks tied to a project.",
        inputs=["text"],
        operating_conditions=["single user", "local store"],
        output="task status list",
        knowledge_required=["task records"],
        permitted_actions=["read_tasks", "write_tasks"],
        success_condition="Task persisted and retrievable after restart.",
        failure_behavior="Reject unauthorized deletes; clarify missing deadline.",
        resource_limits={"time_ms": 1000},
    ),
]
