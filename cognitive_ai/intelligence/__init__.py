"""Cross-cutting intelligence helpers — routing, confidence, self-checks (non-LLM)."""

from .router import SmartRouter, RouteDecision
from .confidence import ConfidenceScore, score_response
from .hardening import harden_text, SelfCheck

__all__ = [
    "SmartRouter",
    "RouteDecision",
    "ConfidenceScore",
    "score_response",
    "harden_text",
    "SelfCheck",
]
