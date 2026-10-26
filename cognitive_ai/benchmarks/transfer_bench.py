"""Skill-transfer benchmark: reuse a procedure on a renamed project/quantities."""

from __future__ import annotations

from typing import Any

from ..general.transfer import SkillTransfer
from ..memory.store import MemorySystem, Procedure
from ..domains.finderbay.seed import seed_memory


def run_transfer_benchmark(memory: MemorySystem | None = None) -> dict[str, Any]:
    memory = memory or seed_memory()
    memory.add_procedure(
        Procedure(
            name="pending_followup_template",
            steps=[
                "retrieve deliveries for project_a this_week",
                "compare expected vs received",
                "draft followup qty=10",
                "verify send=false",
            ],
            applicability=["project:project_a", "reviewed"],
            version="1",
        )
    )
    xfer = SkillTransfer(memory)
    result = xfer.transfer(
        "pending_followup_template",
        from_project="project_a",
        to_project="project_b",
        quantity_scale=2.5,
    )
    checks = {
        "transfer_success": result.success,
        "renamed_project": all("project_a" not in s for s in result.mapped_steps)
        and any("project_b" in s for s in result.mapped_steps),
        "scaled_qty": any("qty=25" in s for s in result.mapped_steps),
        "procedure_stored": "pending_followup_template__on__project_b" in memory.procedures,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "mapped_steps": result.mapped_steps,
        "message": result.message,
    }
