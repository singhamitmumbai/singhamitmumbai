"""Five memory kinds + provenance facts (blueprint §7)."""

from __future__ import annotations

from typing import Any, Optional

from .contracts import FactRecord


class ProvenanceMemory:
    """Working / episodic / semantic / procedural / source memory with supersession."""

    KINDS = ("working", "episodic", "semantic", "procedural", "source")

    def __init__(self) -> None:
        self.facts: dict[str, FactRecord] = {}
        self.working: dict[str, Any] = {
            "active_project": None,
            "selected_document": None,
            "current_task": None,
            "unresolved_question": None,
        }
        self.episodes: list[dict[str, Any]] = []
        self.procedures: dict[str, dict[str, Any]] = {}

    def set_working(self, **kwargs: Any) -> dict[str, Any]:
        for k, v in kwargs.items():
            if k in self.working:
                self.working[k] = v
        return dict(self.working)

    def record_fact(
        self,
        subject: str,
        relation: str,
        value: Any,
        *,
        kind: str = "semantic",
        verification_status: str = "asserted",
        source_id: str = "",
        source_version: str = "1",
        page_or_row: Optional[str] = None,
        units: Optional[str] = None,
        supersedes_fact_id: Optional[str] = None,
        access_scope: str = "local",
    ) -> FactRecord:
        # Soft-supersede prior active facts with same subject+relation
        if supersedes_fact_id is None:
            for f in self.facts.values():
                if (
                    f.subject == subject
                    and f.relation == relation
                    and f.valid_until is None
                    and f.kind == kind
                ):
                    supersedes_fact_id = f.fact_id
                    break
        if supersedes_fact_id and supersedes_fact_id in self.facts:
            prior = self.facts[supersedes_fact_id]
            prior.valid_until = FactRecord(
                subject=subject, relation=relation, value=value
            ).recorded_at

        fact = FactRecord(
            subject=subject,
            relation=relation,
            value=value,
            units=units,
            source_id=source_id,
            source_version=source_version,
            page_or_row=page_or_row,
            verification_status=verification_status,
            access_scope=access_scope,
            supersedes_fact_id=supersedes_fact_id,
            kind=kind,
        )
        self.facts[fact.fact_id] = fact
        return fact

    def correct_fact(
        self,
        subject: str,
        relation: str,
        new_value: Any,
        *,
        source_id: str = "user_correction",
        verification_status: str = "user_said",
    ) -> FactRecord:
        return self.record_fact(
            subject,
            relation,
            new_value,
            verification_status=verification_status,
            source_id=source_id,
            kind="semantic",
        )

    def query(
        self,
        *,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        kind: Optional[str] = None,
        current_only: bool = True,
    ) -> list[FactRecord]:
        out: list[FactRecord] = []
        for f in self.facts.values():
            if subject and f.subject != subject:
                continue
            if relation and f.relation != relation:
                continue
            if kind and f.kind != kind:
                continue
            if current_only and f.valid_until is not None:
                continue
            out.append(f)
        return out

    def record_episode(self, summary: str, evidence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        ep = {"summary": summary, "evidence": evidence or {}}
        self.episodes.append(ep)
        return ep

    def register_procedure(self, name: str, steps: list[str], template: Optional[dict[str, Any]] = None) -> None:
        self.procedures[name] = {"steps": steps, "template": template or {}}

    def snapshot(self) -> dict[str, Any]:
        return {
            "working": dict(self.working),
            "fact_count": len(self.facts),
            "current_facts": [f.to_dict() for f in self.query()],
            "episodes": list(self.episodes[-20:]),
            "procedures": list(self.procedures.keys()),
        }
