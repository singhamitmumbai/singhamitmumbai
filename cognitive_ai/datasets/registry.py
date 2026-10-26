"""Dataset registry: IDs, hashes, splits, reviewers — no train/test leakage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from ..core.ids import new_id
from ..core.records import utc_now


@dataclass
class DatasetRecord:
    example_id: str
    task: str
    input_data: Any
    verified_target: Any
    split: str  # train|dev|calibration|test
    source_group: str = "default"
    language: str = "en"
    reviewer: str = "human"
    label_schema_version: str = "v1"
    content_hash: str = ""
    collected_at: str = field(default_factory=utc_now)
    permitted_use: str = "train_eval"
    notes: str = ""
    transformation_history: list[str] = field(default_factory=list)

    def ensure_hash(self) -> str:
        if not self.content_hash:
            blob = json.dumps(
                {"input": self.input_data, "target": self.verified_target, "task": self.task},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
            self.content_hash = hashlib.sha256(blob).hexdigest()[:16]
        return self.content_hash

    def to_dict(self) -> dict:
        self.ensure_hash()
        return asdict(self)


class DatasetRegistry:
    """Stores examples with fixed splits assigned before any fitting."""

    VALID_SPLITS = ("train", "dev", "calibration", "test")

    def __init__(self, root: Path | str = "data/finderbay/examples") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._records: dict[str, DatasetRecord] = {}
        self._source_group_split: dict[str, str] = {}
        self.load()

    def add(self, record: DatasetRecord, *, lock_source_group_split: bool = True) -> DatasetRecord:
        if record.split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split {record.split}")
        if record.reviewer in ("model", "self", "unverified"):
            raise ValueError("Unverified model labels cannot enter the dataset.")
        record.ensure_hash()
        # Prevent leakage: same source_group stays in one split
        if lock_source_group_split and record.source_group in self._source_group_split:
            locked = self._source_group_split[record.source_group]
            if locked != record.split:
                raise ValueError(
                    f"Source group '{record.source_group}' already locked to split '{locked}'"
                )
        self._source_group_split[record.source_group] = record.split
        self._records[record.example_id] = record
        self._write_one(record)
        self.save_manifest()
        return record

    def make(
        self,
        task: str,
        input_data: Any,
        verified_target: Any,
        *,
        split: str,
        source_group: str = "default",
        language: str = "en",
        reviewer: str = "human",
    ) -> DatasetRecord:
        rec = DatasetRecord(
            example_id=new_id("ex"),
            task=task,
            input_data=input_data,
            verified_target=verified_target,
            split=split,
            source_group=source_group,
            language=language,
            reviewer=reviewer,
        )
        return self.add(rec)

    def iter_split(self, split: str) -> Iterator[DatasetRecord]:
        for r in self._records.values():
            if r.split == split:
                yield r

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {s: 0 for s in self.VALID_SPLITS}
        for r in self._records.values():
            counts[r.split] = counts.get(r.split, 0) + 1
        return {
            "total": len(self._records),
            "by_split": counts,
            "source_groups": len(self._source_group_split),
            "root": str(self.root),
        }

    def _write_one(self, record: DatasetRecord) -> None:
        path = self.root / f"{record.example_id}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2, default=str), encoding="utf-8")

    def save_manifest(self) -> Path:
        payload = {
            "version": "1",
            "updated_at": utc_now(),
            "source_group_split": self._source_group_split,
            "records": [r.to_dict() for r in self._records.values()],
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return self.manifest_path

    def load(self) -> None:
        if not self.manifest_path.exists():
            return
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._source_group_split = dict(data.get("source_group_split") or {})
        self._records = {}
        for rd in data.get("records") or []:
            rec = DatasetRecord(**rd)
            self._records[rec.example_id] = rec
