"""Pillar scorecard for AGI-gap evaluation (honest: not an AGI certificate)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PILLARS = (
    "general_learning",
    "open_world",
    "deep_transfer",
    "long_horizon",
    "multimodality",
    "lifelong_memory",
    "metacognition",
    "alignment_eval",
)


@dataclass
class PillarScore:
    name: str
    score: float  # 0..1
    passed: int = 0
    total: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AGIGapScorecard:
    pillars: list[PillarScore] = field(default_factory=list)
    overall: float = 0.0
    human_level_agi: bool = False
    certifies_agi: bool = False
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pillars": [p.to_dict() for p in self.pillars],
            "overall": self.overall,
            "human_level_agi": False,
            "certifies_agi": False,
            "version": self.version,
            "note": "Product competence scorecard across gap pillars — not AGI certification.",
        }


def build_scorecard(case_results: list[dict[str, Any]], *, version: str = "") -> AGIGapScorecard:
    """Map named eval cases onto pillars via prefix/keyword heuristics."""
    mapping = {
        "general_learning": ("learn_", "curriculum", "learning_"),
        "open_world": ("open_world", "world_"),
        "deep_transfer": ("transfer", "compose_transfer", "schema_"),
        "long_horizon": ("long_horizon", "horizon", "stack_horizon"),
        "multimodality": ("multimodal", "fuse"),
        "lifelong_memory": ("lifelong", "dream", "consolidat"),
        "metacognition": ("meta_", "critique", "debate"),
        "alignment_eval": ("align_", "hard_novel", "constitution", "redteam", "self_eval"),
    }
    buckets: dict[str, list[bool]] = {p: [] for p in PILLARS}
    notes: dict[str, list[str]] = {p: [] for p in PILLARS}
    for case in case_results:
        name = str(case.get("name") or "")
        ok = bool(case.get("ok"))
        placed = False
        for pillar, keys in mapping.items():
            if any(k in name for k in keys):
                buckets[pillar].append(ok)
                if not ok:
                    notes[pillar].append(name)
                placed = True
                break
        if not placed:
            buckets["alignment_eval"].append(ok)

    pillars: list[PillarScore] = []
    scores: list[float] = []
    for p in PILLARS:
        vals = buckets[p]
        total = len(vals)
        passed = sum(1 for v in vals if v)
        score = (passed / total) if total else 0.0
        scores.append(score)
        pillars.append(
            PillarScore(name=p, score=round(score, 3), passed=passed, total=total, notes=notes[p][:5])
        )
    overall = sum(scores) / max(1, len(scores))
    return AGIGapScorecard(
        pillars=pillars,
        overall=round(overall, 3),
        version=version,
    )
