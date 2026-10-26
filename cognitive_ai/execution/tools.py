"""Typed tool registry: arguments, scope, timeout, side-effects, success predicates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolSpec:
    name: str
    description: str
    arg_schema: dict[str, str] = field(default_factory=dict)  # name → type
    permitted_scope: list[str] = field(default_factory=list)
    timeout_ms: int = 3000
    retry: int = 0
    side_effects: list[str] = field(default_factory=list)
    cancellation: str = "best_effort"
    success_predicate: str = "external_receipt_present"
    requires_auth: bool = False
    handler: Optional[Callable[..., Any]] = None

    def validate_args(self, args: dict[str, Any]) -> list[str]:
        errors = []
        for key, typ in self.arg_schema.items():
            if key not in args:
                errors.append(f"missing_arg:{key}")
                continue
            val = args[key]
            if typ == "str" and not isinstance(val, str):
                errors.append(f"type:{key}:expected_str")
            elif typ == "number" and not isinstance(val, (int, float)):
                errors.append(f"type:{key}:expected_number")
            elif typ == "bool" and not isinstance(val, bool):
                errors.append(f"type:{key}:expected_bool")
        return errors


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}
        self._register_defaults()

    def register(self, tool: ToolSpec) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolSpec]:
        return self.tools.get(name)

    def _register_defaults(self) -> None:
        specs = [
            ToolSpec(
                "retrieve_records",
                "Read transactional records",
                arg_schema={"kind": "str"},
                permitted_scope=["read"],
                side_effects=[],
                success_predicate="record_count_available",
            ),
            ToolSpec(
                "draft_followup",
                "Create a local draft message",
                arg_schema={"send": "bool"},
                permitted_scope=["draft"],
                side_effects=["writes_draft"],
                success_predicate="draft_exists",
            ),
            ToolSpec(
                "send_message",
                "Send an outbound message",
                arg_schema={"draft_id": "str"},
                permitted_scope=["send"],
                side_effects=["external_send"],
                requires_auth=True,
                success_predicate="send_receipt",
            ),
            ToolSpec(
                "reserve_stock",
                "Reserve inventory",
                arg_schema={"item": "str", "quantity": "number"},
                permitted_scope=["inventory_write"],
                side_effects=["mutates_inventory"],
                requires_auth=True,
                retry=1,
                success_predicate="inventory_reservation_receipt",
            ),
            ToolSpec(
                "create_task",
                "Create a local task",
                arg_schema={"title": "str", "project_id": "str"},
                permitted_scope=["task_write"],
                side_effects=["writes_task"],
                success_predicate="task_persisted",
            ),
            ToolSpec(
                "ocr_extract",
                "Extract fields from a document structure",
                arg_schema={},
                permitted_scope=["read"],
                side_effects=[],
                success_predicate="fields_present",
            ),
        ]
        for s in specs:
            self.register(s)
