"""Hierarchical task network style decompositions for Finderbay workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class HTNMethod:
    name: str
    task: str
    preconditions: list[str] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    description: str = ""


class HTNLibrary:
    """Reusable decompositions selected by operation name."""

    def __init__(self) -> None:
        self.methods: dict[str, list[HTNMethod]] = {}
        self._seed()

    def _seed(self) -> None:
        self.add(
            HTNMethod(
                "invoice_query_standard",
                task="query_invoices",
                preconditions=["filters_known"],
                subtasks=["retrieve_records", "reason_over_records", "build_table", "respond"],
                description="Standard invoice filter → calculate → report",
            )
        )
        self.add(
            HTNMethod(
                "delivery_followup",
                task="prepare_followup",
                preconditions=["project_known", "time_window_known"],
                subtasks=[
                    "retrieve_records",
                    "reason_over_records",
                    "build_table",
                    "draft_followup",
                    "respond",
                ],
                description="Pending deliveries + unsent follow-up",
            )
        )
        self.add(
            HTNMethod(
                "reserve_then_confirm",
                task="reserve_stock",
                preconditions=["item_identified", "quantity_known", "authorized"],
                subtasks=["reserve_stock", "respond"],
                description="World-model reservation with verification",
            )
        )
        self.add(
            HTNMethod(
                "task_create",
                task="manage_task",
                preconditions=["title_known"],
                subtasks=["create_task", "respond"],
            )
        )
        self.add(
            HTNMethod(
                "document_field",
                task="document_qa",
                preconditions=["document_available"],
                subtasks=["ocr_extract", "respond"],
            )
        )

    def add(self, method: HTNMethod) -> None:
        self.methods.setdefault(method.task, []).append(method)

    def select(self, task: str, facts: Optional[dict[str, Any]] = None) -> Optional[HTNMethod]:
        facts = facts or {}
        for method in self.methods.get(task, []):
            if all(facts.get(p, True) for p in method.preconditions):
                return method
        return None

    def decompose(self, task: str, facts: Optional[dict[str, Any]] = None) -> list[str]:
        method = self.select(task, facts)
        return list(method.subtasks) if method else []
