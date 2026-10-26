"""Data adapters: import transactional registers from JSON/CSV without an LLM."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from cognitive_ai.memory.store import MemorySystem


SUPPORTED_KINDS = ("invoice", "delivery", "material", "task", "document", "project")


def load_json_records(memory: MemorySystem, path: Path | str, kind: str) -> int:
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"Unsupported kind: {kind}")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        rows = data["records"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError("JSON must be a list or {records: [...]}")
    n = 0
    for row in rows:
        rid = str(row.get("id") or row.get(f"{kind}_id") or f"{kind}_{n}")
        memory.upsert_record(kind, rid, dict(row))
        n += 1
    return n


def load_csv_records(memory: MemorySystem, path: Path | str, kind: str) -> int:
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"Unsupported kind: {kind}")
    n = 0
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            clean: dict[str, Any] = {}
            for k, v in row.items():
                if v is None or v == "":
                    clean[k] = None
                    continue
                # numeric coercion for common fields
                if k.endswith("_qty") or k.endswith("_inr") or k in ("available", "reserved", "amount_inr"):
                    try:
                        clean[k] = float(v)
                        continue
                    except ValueError:
                        pass
                clean[k] = v
            rid = str(clean.get("id") or f"{kind}_{n}")
            memory.upsert_record(kind, rid, clean)
            n += 1
    return n


def export_json(memory: MemorySystem, kind: str, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = memory.get_records(kind)
    target.write_text(json.dumps({"kind": kind, "records": rows}, indent=2, default=str), encoding="utf-8")
    return target
