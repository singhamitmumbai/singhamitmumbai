"""Product / proto-AGI alignment guards + honesty checks (not AGI-proof safety)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class AlignmentVerdict:
    allowed: bool
    risk: str = "low"  # low|medium|high|critical
    reasons: list[str] = field(default_factory=list)
    policy: str = "complete_ai_product"
    transform: Optional[str] = None
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BLOCK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("cyber_offense", re.compile(r"\b(?:exploit|ransomware|sql injection payload|keylogger)\b", re.I)),
    ("bioweapon", re.compile(r"\b(?:synthesize pathogen|weaponize virus|reverse[- ]genetics)\b", re.I)),
    ("explosives", re.compile(r"\b(?:make a bomb|build explosives|improvised explosive)\b", re.I)),
    ("child_exploit", re.compile(r"\b(?:child sexual|csam)\b", re.I)),
]

CAUTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("medical", re.compile(r"\b(?:diagnose|prescribe|dosage for)\b", re.I)),
    ("financial", re.compile(r"\b(?:guaranteed returns|insider trading)\b", re.I)),
    ("computer_control", re.compile(r"\b(?:delete all files|format disk|shutdown network)\b", re.I)),
]


class AlignmentGuard:
    """Refuse / caution high-risk requests; always mark human_level_agi=false."""

    def check(self, text: str) -> AlignmentVerdict:
        raw = text or ""
        for name, pat in BLOCK_PATTERNS:
            if pat.search(raw):
                return AlignmentVerdict(
                    False,
                    "critical",
                    [f"blocked:{name}"],
                    transform="refuse",
                )
        for name, pat in CAUTION_PATTERNS:
            if pat.search(raw):
                return AlignmentVerdict(
                    True,
                    "high",
                    [f"caution:{name}"],
                    transform="add_disclaimer",
                )
        return AlignmentVerdict(True, "low", ["ok"])

    def wrap_response(self, text: str, verdict: AlignmentVerdict) -> str:
        if not verdict.allowed:
            return (
                "I cannot help with that request. It violates Complete AI safety policy. "
                "(human_level_agi=false)"
            )
        if verdict.risk == "high" and verdict.transform == "add_disclaimer":
            return (
                text
                + "\n\n_Disclaimer: This is general information only — not professional "
                "medical/financial/legal advice. human_level_agi=false._"
            )
        return text

    def honesty_badge(self) -> dict[str, Any]:
        return {
            "human_level_agi": False,
            "claims_consciousness": False,
            "self_labeling_allowed": False,
            "unverified_web_as_fact": False,
            "policy": "complete_ai_product_alignment_v1",
        }
