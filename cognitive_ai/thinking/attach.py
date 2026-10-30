"""Shared helper — attach think→plan deliberation to any handler response."""

from __future__ import annotations

from typing import Any, Optional

from .deliberator import Deliberation, ThinkingSystem


def get_thinker(owner: Any) -> ThinkingSystem:
    """Lazy-bind a ThinkingSystem on any object as ``_thinker``."""
    thinker = getattr(owner, "_thinker", None)
    if thinker is None:
        thinker = ThinkingSystem()
        try:
            owner._thinker = thinker
        except Exception:  # noqa: BLE001
            pass
    return thinker


def deliberate(
    owner: Any,
    text: str,
    *,
    context: Optional[dict[str, Any]] = None,
) -> Deliberation:
    """Run universal thinking for any situation / question."""
    return get_thinker(owner).think(text, context=context)


def merge_thinking(
    result: dict[str, Any],
    deliberation: Optional[Deliberation],
) -> dict[str, Any]:
    """Copy deliberation fields onto a response dict (idempotent)."""
    if deliberation is None:
        return result
    out = dict(result)
    out.setdefault("thinking", deliberation.to_dict())
    out.setdefault("thinking_brief", deliberation.as_brief())
    out.setdefault("thinking_markdown", deliberation.as_markdown())
    out.setdefault("human_level_agi", False)
    return out


def think_or_abstain(
    owner: Any,
    text: str,
    *,
    context: Optional[dict[str, Any]] = None,
) -> tuple[Deliberation, Optional[dict[str, Any]]]:
    """Think first; if policy says abstain, return a ready refusal payload."""
    d = deliberate(owner, text, context=context)
    if d.should_abstain and d.strategy == "abstain":
        refuse = {
            "text": (
                "I need to stop and refuse that request after thinking it through.\n\n"
                + d.as_markdown()
            ),
            "thinking": d.to_dict(),
            "thinking_brief": d.as_brief(),
            "thinking_markdown": d.as_markdown(),
            "faithful": True,
            "human_level_agi": False,
            "route": "think_abstain",
            "ok": False,
        }
        return d, refuse
    return d, None
