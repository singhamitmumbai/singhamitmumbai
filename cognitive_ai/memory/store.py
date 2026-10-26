"""Multi-memory store: semantic, episodic, procedural, preference."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import Claim, EvidenceStatus, utc_now
from .ontology import Ontology, finderbay_ontology


@dataclass
class Episode:
    summary: str
    observations: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    outcome: str = ""
    episode_id: str = field(default_factory=lambda: new_id("ep"))
    created_at: str = field(default_factory=utc_now)


@dataclass
class Procedure:
    name: str
    steps: list[str]
    applicability: list[str] = field(default_factory=list)
    version: str = "1"
    success_count: int = 0
    failure_count: int = 0


class MemorySystem:
    """Versioned facts with temporal validity, aliases and retrieval."""

    def __init__(self, ontology: Optional[Ontology] = None, path: Optional[Path | str] = None) -> None:
        self.ontology = ontology or finderbay_ontology()
        self.claims: dict[str, Claim] = {}
        self.episodes: list[Episode] = []
        self.procedures: dict[str, Procedure] = {}
        self.preferences: dict[str, Any] = {}
        self.records: dict[str, dict[str, Any]] = {}  # transactional tables by type
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            self.load()

    # --- semantic / claims ----------------------------------------------------

    def remember_claim(self, claim: Claim) -> Claim:
        self.claims[claim.claim_id] = claim
        return claim

    def learn_alias(self, alias: str, canonical: str, source: str = "user") -> None:
        self.ontology.add_alias(alias, canonical, source=source)
        self.remember_claim(
            Claim(
                subject=canonical,
                relationship="alias",
                value=alias,
                source=source,
                evidence_status=EvidenceStatus.ASSERTED,
            )
        )

    def current_claims(
        self,
        subject: Optional[str] = None,
        relationship: Optional[str] = None,
        at: Optional[str] = None,
    ) -> list[Claim]:
        """Return claims valid at time `at` (default now). Later-arriving reports
        about past periods do not overwrite current inventory blindly."""
        when = at or utc_now()
        out: list[Claim] = []
        for c in self.claims.values():
            if subject and c.subject != subject:
                continue
            if relationship and c.relationship != relationship:
                continue
            if c.validity_start and c.validity_start > when:
                continue
            if c.validity_end and c.validity_end < when:
                continue
            out.append(c)
        return out

    # --- transactional records ------------------------------------------------

    def upsert_record(self, kind: str, record_id: str, data: dict[str, Any]) -> None:
        table = self.records.setdefault(kind, {})
        existing = table.get(record_id, {})
        merged = {**existing, **data, "id": record_id, "updated_at": utc_now()}
        table[record_id] = merged

    def get_records(self, kind: str, **filters: Any) -> list[dict[str, Any]]:
        rows = list(self.records.get(kind, {}).values())
        for key, val in filters.items():
            rows = [r for r in rows if r.get(key) == val]
        return rows

    def search(self, query: str, kinds: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """Lexical / BM25-lite retrieval over records and claim text."""
        tokens = [t for t in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(t) > 1]
        scored: list[tuple[float, dict]] = []
        kinds = kinds or list(self.records.keys())
        for kind in kinds:
            for row in self.records.get(kind, {}).values():
                blob = json.dumps(row, default=str).lower()
                score = sum(blob.count(tok) for tok in tokens)
                # synonym boost via ontology aliases
                for alias, canon in self.ontology.aliases.items():
                    if alias in query.lower() and canon in blob:
                        score += 1.5
                if score > 0:
                    scored.append((score, {"kind": kind, **row, "_score": score}))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:50]]

    # --- episodic / procedural / preference -----------------------------------

    def add_episode(self, episode: Episode) -> Episode:
        self.episodes.append(episode)
        return episode

    def add_procedure(self, proc: Procedure) -> None:
        self.procedures[proc.name] = proc

    def set_preference(self, key: str, value: Any) -> None:
        self.preferences[key] = value

    # --- persistence ----------------------------------------------------------

    def save(self, path: Optional[Path | str] = None) -> Path:
        target = Path(path) if path else self._path
        if target is None:
            raise ValueError("No path for MemorySystem.save()")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ontology": self.ontology.to_dict(),
            "claims": [c.to_dict() for c in self.claims.values()],
            "records": self.records,
            "episodes": [e.__dict__ for e in self.episodes],
            "procedures": {k: v.__dict__ for k, v in self.procedures.items()},
            "preferences": self.preferences,
        }
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self._path = target
        return target

    def load(self, path: Optional[Path | str] = None) -> None:
        source = Path(path) if path else self._path
        if source is None or not source.exists():
            return
        data = json.loads(source.read_text(encoding="utf-8"))
        self.records = data.get("records", {})
        self.preferences = data.get("preferences", {})
        self.claims = {}
        for cd in data.get("claims", []):
            status = EvidenceStatus(cd.pop("evidence_status", "asserted"))
            claim = Claim(**{**cd, "evidence_status": status})
            self.claims[claim.claim_id] = claim
        self.episodes = [Episode(**e) for e in data.get("episodes", [])]
        self.procedures = {
            k: Procedure(**v) for k, v in data.get("procedures", {}).items()
        }
        # aliases
        for alias, canon in data.get("ontology", {}).get("aliases", {}).items():
            self.ontology.add_alias(alias, canon)
        self._path = source

    def summary(self) -> dict[str, Any]:
        return {
            "domain": self.ontology.domain,
            "claims": len(self.claims),
            "record_kinds": {k: len(v) for k, v in self.records.items()},
            "episodes": len(self.episodes),
            "procedures": len(self.procedures),
            "preferences": len(self.preferences),
        }
