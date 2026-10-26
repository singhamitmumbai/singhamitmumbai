"""Calibration utilities — classifier scores ≠ calibrated probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class CalibrationReport:
    n: int
    brier: float
    bins: list[dict]
    message: str

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "brier": self.brier,
            "bins": self.bins,
            "message": self.message,
        }


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    if not probs:
        return 0.0
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes length mismatch")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def reliability_bins(
    probs: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> list[dict]:
    """Build a reliability diagram table."""
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "count": 0, "sum_prob": 0.0, "sum_out": 0.0} for i in range(n_bins)]
    for p, o in zip(probs, outcomes):
        idx = min(n_bins - 1, max(0, int(p * n_bins)))
        bins[idx]["count"] += 1
        bins[idx]["sum_prob"] += p
        bins[idx]["sum_out"] += o
    out = []
    for b in bins:
        if b["count"] == 0:
            out.append({**b, "avg_prob": None, "avg_out": None, "gap": None})
        else:
            avg_p = b["sum_prob"] / b["count"]
            avg_o = b["sum_out"] / b["count"]
            out.append({**b, "avg_prob": avg_p, "avg_out": avg_o, "gap": abs(avg_p - avg_o)})
    return out


def calibrate_report(probs: Sequence[float], outcomes: Sequence[int]) -> CalibrationReport:
    return CalibrationReport(
        n=len(probs),
        brier=brier_score(probs, outcomes),
        bins=reliability_bins(probs, outcomes),
        message="Scores are empirical; do not multiply unrelated confidences into a joint probability.",
    )
