"""Entity linking against ontology catalogues and memory records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ..memory.store import MemorySystem


@dataclass
class LinkedEntity:
    mention: str
    entity_id: str
    entity_type: str
    confidence: float
    method: str


class EntityLinker:
    def __init__(self, memory: MemorySystem) -> None:
        self.memory = memory

    def link(self, text: str) -> list[LinkedEntity]:
        lower = text.lower()
        found: list[LinkedEntity] = []
        seen: set[str] = set()

        # Projects
        for m in re.finditer(r"\bproject\s+([a-z0-9_-]+)\b", lower):
            pid = f"project_{m.group(1)}"
            if pid not in seen:
                found.append(
                    LinkedEntity(m.group(0), pid, "project", 0.95, "pattern")
                )
                seen.add(pid)

        # Catalogue aliases / ontology
        for alias, canon in self.memory.ontology.aliases.items():
            if alias in lower and canon not in seen:
                found.append(LinkedEntity(alias, canon, "alias", 0.85, "alias"))
                seen.add(canon)

        # Invoice ids
        for m in re.finditer(r"\binvoice\s+(\d+)\b", lower):
            iid = f"inv_{m.group(1)}"
            if iid not in seen:
                found.append(LinkedEntity(m.group(0), iid, "invoice", 0.9, "pattern"))
                seen.add(iid)

        # Material names from records
        for mat in self.memory.get_records("material"):
            name = str(mat.get("name") or "").lower()
            mid = mat.get("id")
            if name and name in lower and mid and mid not in seen:
                found.append(LinkedEntity(name, mid, "material", 0.9, "record"))
                seen.add(mid)

        # Warehouse mentions
        if "warehouse" in lower or "central store" in lower:
            wid = self.memory.ontology.resolve("central store") or "central_store"
            if wid not in seen:
                found.append(LinkedEntity("warehouse", wid, "warehouse", 0.8, "alias"))
                seen.add(wid)

        return found
