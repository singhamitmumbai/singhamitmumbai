"""Strong(er) metacognition — know limits, pick strategies, abstain, reflect."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class MetaDecision:
    strategy: str
    confidence: float
    should_abstain: bool = False
    reasons: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    next_questions: list[str] = field(default_factory=list)
    risk: str = "low"  # low|medium|high
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetaCognition:
    """Strategy selection + honesty about capability gaps."""

    STRATEGIES = (
        "local_kb",
        "web_search",
        "tool_finderbay",
        "tool_vision",
        "tool_coding",
        "long_horizon",
        "transfer",
        "ask_clarify",
        "abstain",
        "creative_template",
        "math",
    )

    CAPABILITY_LIMITS = [
        "No human-level AGI / consciousness",
        "Open-world coverage is curated + combinatorial, not complete",
        "Long-horizon loops are bounded and heuristic",
        "Vision/speech/OCR depend on optional engines (often stub)",
        "Continual learning rejects unverified self-labels",
        "Alignment checks are product-grade, not AGI-proof",
    ]

    def decide(
        self,
        text: str,
        *,
        open_world: Optional[dict[str, Any]] = None,
        has_tools_hint: bool = False,
    ) -> MetaDecision:
        lower = (text or "").lower().strip()
        reasons: list[str] = []
        unknowns = list((open_world or {}).get("unknowns") or [])
        conf = 0.55

        if not lower:
            return MetaDecision(
                "abstain", 0.1, True, ["empty_input"], next_questions=["What would you like to do?"]
            )

        if re.search(r"\b(?:calculate|compute|\d+\s*[+\-*/])\b", lower):
            return MetaDecision("math", 0.9, False, ["math_cue"])

        if re.search(
            r"\b(?:invoices?|warehouse|shortages?|purchase orders?|vendors?|materials?|delivery|deliveries)\b",
            lower,
        ):
            return MetaDecision("tool_finderbay", 0.88, False, ["finderbay_domain"], risk="low")

        if re.search(r"\b(?:image|photo|thermal|detect|yolo|ocr)\b", lower):
            return MetaDecision("tool_vision", 0.8, False, ["vision_cue"], risk="medium")

        if re.search(r"\b(?:generate function|refactor|unit test|python code)\b", lower):
            return MetaDecision("tool_coding", 0.82, False, ["coding_cue"])

        if re.search(r"\b(?:search the web|look up|wikipedia|latest|current news)\b", lower):
            return MetaDecision("web_search", 0.8, False, ["search_cue"], risk="medium")

        if re.search(r"\b(?:plan over|multi-?step|then verify|long(?:-|\s)?term|roadmap)\b", lower):
            return MetaDecision("long_horizon", 0.75, False, ["horizon_cue"], risk="medium")

        if re.search(r"\b(?:transfer|reuse procedure|analog(?:y|ous)|same for project)\b", lower):
            return MetaDecision("transfer", 0.7, False, ["transfer_cue"])

        if re.search(r"\b(?:poem|brainstorm|outline|draft email|pros and cons)\b", lower):
            return MetaDecision("creative_template", 0.7, False, ["creative_cue"])

        if len(lower.split()) < 3 and not lower.endswith("?"):
            return MetaDecision(
                "ask_clarify",
                0.4,
                False,
                ["too_short"],
                next_questions=["Can you give a bit more detail?"],
            )

        # Unknown-heavy open world → search or clarify
        if len(unknowns) >= 3 and not has_tools_hint:
            reasons.append("many_unknown_entities")
            return MetaDecision(
                "web_search",
                0.55,
                False,
                reasons,
                unknowns=unknowns,
                next_questions=[f"What is {u}?" for u in unknowns[:2]],
                risk="medium",
            )

        if re.search(r"\b(?:what is|who is|explain|why|how)\b", lower):
            conf = 0.72
            return MetaDecision("local_kb", conf, False, ["question_form"], unknowns=unknowns)

        # High-stakes abstain patterns
        if re.search(r"\b(?:medical diagnosis|buy this stock|delete all|weapon|exploit)\b", lower):
            return MetaDecision(
                "abstain",
                0.95,
                True,
                ["alignment_high_risk"],
                next_questions=["I can discuss general info, not unsafe/actionable harm."],
                risk="high",
            )

        return MetaDecision("local_kb", conf, False, ["default"], unknowns=unknowns)

    def reflect(self, decision: MetaDecision, outcome: dict[str, Any]) -> dict[str, Any]:
        ok = bool(outcome.get("ok", outcome.get("supported", True)))
        note = "strategy_ok" if ok else "strategy_miss"
        return {
            "decision": decision.to_dict(),
            "outcome_ok": ok,
            "reflection": note,
            "adjust_next": (
                None
                if ok
                else ("web_search" if decision.strategy == "local_kb" else "ask_clarify")
            ),
            "limits": self.CAPABILITY_LIMITS,
            "human_level_agi": False,
        }

    def critique(self, text: str, draft_answer: str, *, decision: Optional[MetaDecision] = None) -> dict[str, Any]:
        """Post-hoc critique: unsupported claims, missing uncertainty, policy drift."""
        issues: list[str] = []
        draft = draft_answer or ""
        if re.search(r"\b(?:definitely|guaranteed|100%|as an agi|i am conscious)\b", draft, re.I):
            issues.append("overclaim")
        if "human_level_agi" not in draft.lower() and decision and decision.risk in ("high",):
            issues.append("missing_honesty_marker")
        if len(draft.strip()) < 8:
            issues.append("too_short")
        if decision and decision.unknowns and not any(u in draft.lower() for u in decision.unknowns[:2]):
            issues.append("ignored_unknowns")
        score = max(0.0, 1.0 - 0.2 * len(issues))
        return {
            "ok": not issues,
            "issues": issues,
            "quality": round(score, 3),
            "suggestion": (
                "Add uncertainty and avoid overclaim." if issues else "Draft looks acceptable."
            ),
            "human_level_agi": False,
        }

    def debate_strategies(self, text: str, *, open_world: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Propose top-2 strategies and pick winner by simple scoring."""
        primary = self.decide(text, open_world=open_world)
        # Force an alternate by masking primary cues lightly
        alt_text = text
        if primary.strategy == "local_kb":
            alt = MetaDecision("web_search", 0.5, False, ["alt_search"])
        elif primary.strategy == "web_search":
            alt = MetaDecision("long_horizon", 0.5, False, ["alt_horizon"])
        elif primary.strategy.startswith("tool_"):
            alt = MetaDecision("ask_clarify", 0.45, False, ["alt_clarify"])
        else:
            alt = MetaDecision("local_kb", 0.45, False, ["alt_local"])
        winner = primary if primary.confidence >= alt.confidence else alt
        return {
            "primary": primary.to_dict(),
            "alternate": alt.to_dict(),
            "winner": winner.strategy,
            "human_level_agi": False,
        }

    def self_model(self) -> dict[str, Any]:
        return {
            "strategies": list(self.STRATEGIES),
            "limits": list(self.CAPABILITY_LIMITS),
            "human_level_agi": False,
            "note": "Metacognition is rule/heuristic based — not full self-awareness.",
        }
