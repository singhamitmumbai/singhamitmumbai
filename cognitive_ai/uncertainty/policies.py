"""Map uncertainty kinds to appropriate system responses."""

from __future__ import annotations

from ..core.records import UncertaintyKind


POLICY = {
    UncertaintyKind.RECOGNITION: "Reprocess or ask about the unclear input.",
    UncertaintyKind.MISSING_INFORMATION: "Retrieve or request the missing field.",
    UncertaintyKind.CONFLICTING_EVIDENCE: "Compare sources and expose the conflict.",
    UncertaintyKind.STALE_INFORMATION: "Refresh the relevant source.",
    UncertaintyKind.UNSUPPORTED_TASK: "Explain the missing capability.",
    UncertaintyKind.FORECAST: "Report an appropriate range or qualification.",
}


def respond_to_uncertainty(kind: UncertaintyKind | str) -> str:
    if isinstance(kind, str):
        kind = UncertaintyKind(kind)
    return POLICY.get(kind, "Qualify the answer and request clarification if needed.")
