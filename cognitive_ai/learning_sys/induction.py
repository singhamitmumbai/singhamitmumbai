"""Simple program / procedure induction from reviewed examples (blueprint §13, §19).

Learns reusable operator sequences from positive examples; rejects overlap with
counterexamples. No LLM, no self-labeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import LearningExample, utc_now
from ..memory.store import MemorySystem, Procedure


@dataclass
class InducedProgram:
    name: str
    steps: list[str]
    applicability: list[str] = field(default_factory=list)
    support: int = 0
    status: str = "candidate"  # candidate|rejected|released
    program_id: str = field(default_factory=lambda: new_id("prog"))
    created_at: str = field(default_factory=utc_now)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "name": self.name,
            "steps": list(self.steps),
            "applicability": list(self.applicability),
            "support": self.support,
            "status": self.status,
            "created_at": self.created_at,
            "notes": self.notes,
        }


class ProgramInducer:
    """Induce a small procedure from reviewed LearningExample traces."""

    def induce(
        self,
        name: str,
        positives: list[LearningExample],
        counterexamples: Optional[list[LearningExample]] = None,
        *,
        applicability: Optional[list[str]] = None,
        memory: Optional[MemorySystem] = None,
        release: bool = False,
    ) -> InducedProgram:
        counterexamples = counterexamples or []
        if not positives:
            return InducedProgram(
                name=name,
                steps=[],
                status="rejected",
                notes="no_positive_examples",
            )

        # Extract operator sequences from example metadata / verified targets
        sequences: list[list[str]] = []
        for ex in positives:
            seq = _extract_steps(ex)
            if seq:
                sequences.append(seq)
        if not sequences:
            # Fall back to name-based template
            sequences = [[name]]

        # Longest common subsequence-ish: majority vote per position of shortest
        steps = _consensus_steps(sequences)

        # Reject if any counterexample shares identical step sequence
        pos_keys = {"|".join(s) for s in sequences}
        neg_keys = {"|".join(_extract_steps(e) or []) for e in counterexamples}
        if pos_keys & neg_keys:
            return InducedProgram(
                name=name,
                steps=steps,
                support=len(positives),
                status="rejected",
                notes="overlap_with_counterexamples",
                applicability=list(applicability or []),
            )

        prog = InducedProgram(
            name=name,
            steps=steps,
            support=len(positives),
            status="candidate",
            applicability=list(applicability or ["reviewed"]),
            notes=f"induced_from_{len(positives)}_examples",
        )
        if release and memory is not None:
            memory.add_procedure(
                Procedure(
                    name=name,
                    steps=steps,
                    applicability=prog.applicability,
                    version="induced_v1",
                )
            )
            prog.status = "released"
        return prog


def _extract_steps(ex: LearningExample) -> list[str]:
    meta = {}
    if isinstance(ex.input_data, dict):
        meta = ex.input_data
    target = ex.verified_target if isinstance(ex.verified_target, dict) else {}
    for key in ("steps", "operators", "plan_steps"):
        val = target.get(key) or meta.get(key)
        if isinstance(val, list) and val:
            return [str(v) for v in val]
    op = target.get("operation") or meta.get("operation")
    if op:
        # Default Finderbay query workflow
        if str(op).startswith("query_"):
            return ["retrieve_records", "reason_over_records", "build_table", "respond"]
        return [str(op), "respond"]
    return []


def _consensus_steps(sequences: list[list[str]]) -> list[str]:
    if not sequences:
        return []
    # Prefer the most common full sequence
    from collections import Counter

    keys = Counter(tuple(s) for s in sequences)
    best = keys.most_common(1)[0][0]
    return list(best)
