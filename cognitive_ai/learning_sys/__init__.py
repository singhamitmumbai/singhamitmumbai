"""Controlled learning: facts, rules, datasets, rollback + 24/7 continuous loop.

ContinuousLearner is lazy-imported to avoid circular imports with system/agi_gap.
"""

from __future__ import annotations

from typing import Any

from .journal import LearningJournal
from .learner import LearningSystem

__all__ = [
    "LearningSystem",
    "LearningJournal",
    "ContinuousLearner",
    "get_continuous_learner",
]


def __getattr__(name: str) -> Any:
    if name == "ContinuousLearner":
        from .continuous import ContinuousLearner

        return ContinuousLearner
    if name == "get_continuous_learner":
        from .continuous import get_continuous_learner

        return get_continuous_learner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
