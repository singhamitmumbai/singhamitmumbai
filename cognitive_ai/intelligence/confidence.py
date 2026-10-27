"""Response / decision confidence scoring (honest uncertainty)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ConfidenceScore:
    value: float  # 0..1
    band: str  # high|medium|low|abstain
    reasons: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_response(
    *,
    verified: Optional[bool] = None,
    supported: bool = True,
    intent_confidence: Optional[float] = None,
    missing_fields: Optional[list[str]] = None,
    engine_stub: bool = False,
    alert: bool = False,
    unknown_objects: int = 0,
    outcome_unknown: bool = False,
) -> ConfidenceScore:
    reasons: list[str] = []
    unknowns: list[str] = []
    score = 0.55

    if not supported:
        return ConfidenceScore(0.15, "abstain", ["unsupported_request"], ["capability_gap"])

    if verified is True:
        score += 0.25
        reasons.append("verified_outcome")
    elif verified is False:
        score -= 0.2
        reasons.append("verification_failed")
        unknowns.append("unverified_result")

    if intent_confidence is not None:
        score = 0.5 * score + 0.5 * max(0.0, min(1.0, intent_confidence))
        reasons.append(f"intent_confidence={intent_confidence:.2f}")

    if missing_fields:
        score -= 0.08 * min(3, len(missing_fields))
        unknowns.extend(f"missing:{m}" for m in missing_fields)
        reasons.append("missing_fields")

    if engine_stub:
        score -= 0.08
        reasons.append("specialist_engine_stub")
        unknowns.append("model_weights_optional")

    if unknown_objects > 0:
        score -= 0.05 * min(4, unknown_objects)
        unknowns.append(f"unknown_objects:{unknown_objects}")

    if outcome_unknown:
        score = min(score, 0.4)
        reasons.append("outcome_unknown")
        unknowns.append("reconcile_before_retry")

    if alert:
        reasons.append("safety_alert_present")

    score = max(0.0, min(1.0, score))
    if score >= 0.75:
        band = "high"
    elif score >= 0.5:
        band = "medium"
    elif score >= 0.3:
        band = "low"
    else:
        band = "abstain"
    return ConfidenceScore(round(score, 3), band, reasons, unknowns)
