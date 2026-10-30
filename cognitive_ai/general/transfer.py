"""Skill transfer experiments — reuse procedures across renamed projects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..memory.store import MemorySystem, Procedure


@dataclass
class TransferResult:
    success: bool
    message: str
    mapped_steps: list[str]
    details: dict[str, Any]


class SkillTransfer:
    """Apply a reviewed procedure to a new project with different names/quantities."""

    def __init__(self, memory: MemorySystem) -> None:
        self.memory = memory

    def transfer(
        self,
        procedure_name: str,
        *,
        from_project: str,
        to_project: str,
        quantity_scale: float = 1.0,
    ) -> TransferResult:
        proc = self.memory.procedures.get(procedure_name)
        if not proc:
            return TransferResult(False, f"Unknown procedure '{procedure_name}'", [], {})
        mapped = []
        for step in proc.steps:
            s = step.replace(from_project, to_project)
            # naive quantity scaling marker: qty=N
            if "qty=" in s and quantity_scale != 1.0:
                import re

                def scale(m):
                    return f"qty={float(m.group(1)) * quantity_scale:g}"

                s = re.sub(r"qty=([0-9.]+)", scale, s)
            mapped.append(s)
        # Record transferred skill under new name
        new_name = f"{procedure_name}__on__{to_project}"
        self.memory.add_procedure(
            Procedure(
                name=new_name,
                steps=mapped,
                applicability=[f"project:{to_project}", "transferred"],
                version="transfer_v1",
            )
        )
        return TransferResult(
            True,
            f"Transferred '{procedure_name}' → '{new_name}' for {to_project}.",
            mapped,
            {"from": from_project, "to": to_project, "scale": quantity_scale},
        )
