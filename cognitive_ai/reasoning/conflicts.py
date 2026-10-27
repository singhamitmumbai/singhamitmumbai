"""Detect and expose conflicting claims / evidence (blueprint §4, §7, §15)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.records import Claim, EvidenceStatus
from ..core.working_state import WorkingState, UncertaintyNote
from ..core.records import UncertaintyKind
from ..memory.store import MemorySystem


@dataclass
class ConflictReport:
    subject: str
    relationship: str
    values: list[Any] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relationship": self.relationship,
            "values": list(self.values),
            "claim_ids": list(self.claim_ids),
            "sources": list(self.sources),
            "message": self.message,
        }


class ConflictDetector:
    """Compare claims that share subject+relationship but disagree on value."""

    def scan_state(self, state: WorkingState) -> list[ConflictReport]:
        groups: dict[tuple[str, str], list[Claim]] = {}
        for claim in state.claims.values():
            key = (claim.subject, claim.relationship)
            groups.setdefault(key, []).append(claim)
        return self._from_groups(groups)

    def scan_memory_claims(self, memory: MemorySystem) -> list[ConflictReport]:
        groups: dict[tuple[str, str], list[Claim]] = {}
        claims_obj = getattr(memory, "claims", {}) or {}
        claim_iter = claims_obj.values() if isinstance(claims_obj, dict) else claims_obj
        for claim in claim_iter:
            if not isinstance(claim, Claim):
                continue
            key = (claim.subject, claim.relationship)
            groups.setdefault(key, []).append(claim)
        # Also scan invoice payment_status vs payment records as soft conflicts
        reports = self._from_groups(groups)
        reports.extend(self._invoice_payment_conflicts(memory))
        return reports

    def apply_to_state(self, state: WorkingState, reports: list[ConflictReport]) -> list[UncertaintyNote]:
        notes: list[UncertaintyNote] = []
        for r in reports:
            note = UncertaintyNote(
                kind=UncertaintyKind.CONFLICTING_EVIDENCE,
                message=r.message or f"Conflict on {r.subject}.{r.relationship}: {r.values}",
                related_ids=list(r.claim_ids),
            )
            state.uncertainties.append(note)
            notes.append(note)
            # Mark claims conflicting
            for cid in r.claim_ids:
                if cid in state.claims:
                    state.claims[cid].evidence_status = EvidenceStatus.CONFLICTING
                    state.claims[cid].conflicts_with = [x for x in r.claim_ids if x != cid]
        return notes

    def _from_groups(self, groups: dict[tuple[str, str], list[Claim]]) -> list[ConflictReport]:
        out: list[ConflictReport] = []
        for (subj, rel), claims in groups.items():
            values = []
            seen = set()
            for c in claims:
                key = repr(c.value)
                if key not in seen:
                    seen.add(key)
                    values.append(c.value)
            if len(values) <= 1:
                continue
            out.append(
                ConflictReport(
                    subject=subj,
                    relationship=rel,
                    values=values,
                    claim_ids=[c.claim_id for c in claims],
                    sources=[c.source for c in claims],
                    message=(
                        f"Conflicting evidence for {subj} {rel}: "
                        + ", ".join(f"{c.source}→{c.value!r}" for c in claims)
                    ),
                )
            )
        return out

    def _invoice_payment_conflicts(self, memory: MemorySystem) -> list[ConflictReport]:
        """Invoice marked unpaid while a posted payment covers its amount."""
        out: list[ConflictReport] = []
        payments = memory.get_records("payment")
        for inv in memory.get_records("invoice"):
            if inv.get("payment_status") != "unpaid":
                continue
            inv_id = inv.get("id")
            paid = sum(
                float(p.get("amount_inr") or 0)
                for p in payments
                if p.get("invoice_id") == inv_id and p.get("status") == "posted"
            )
            amount = float(inv.get("amount_inr") or 0)
            if amount > 0 and paid + 1e-6 >= amount:
                out.append(
                    ConflictReport(
                        subject=str(inv_id),
                        relationship="payment_status",
                        values=["unpaid", "paid_by_payments"],
                        claim_ids=[],
                        sources=["invoice_register", "payment_register"],
                        message=(
                            f"Invoice {inv_id} is marked unpaid but posted payments "
                            f"(₹{paid}) cover amount ₹{amount}."
                        ),
                    )
                )
        return out
