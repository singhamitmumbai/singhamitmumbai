"""Proposer / Critic / Verifier rule libraries for Mother AI.

Heuristic role separation inside one process — not independently trained models.
human_level_agi=false.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .gateway import InputInterpretation


INTENT_ACTION_MAP: dict[str, list[dict[str, Any]]] = {
    "conversation": [
        {
            "action": "dialogue",
            "steps": ["greet_or_meta", "use_symbolic_dialogue"],
            "expected": "friendly local reply",
            "confidence": 0.85,
            "cost": 1,
        }
    ],
    "meta": [
        {
            "action": "dialogue",
            "steps": ["capabilities_overview"],
            "expected": "capability summary",
            "confidence": 0.88,
            "cost": 1,
        }
    ],
    "inspect": [
        {
            "action": "inspect_repo",
            "steps": ["status", "diagnostics", "summarize_gaps"],
            "expected": "diagnostic report",
            "confidence": 0.8,
            "cost": 3,
        }
    ],
    "coding": [
        {
            "action": "coding",
            "steps": ["route_coding_system", "verify_if_possible"],
            "expected": "code result",
            "confidence": 0.75,
            "cost": 4,
        }
    ],
    "question": [
        {
            "action": "complete_ai",
            "steps": ["think_plan", "route_complete_ai", "observe"],
            "expected": "task answer via CompleteAI",
            "confidence": 0.78,
            "cost": 3,
        }
    ],
    "task": [
        {
            "action": "complete_ai",
            "steps": ["think_plan", "route_complete_ai", "observe"],
            "expected": "task answer via CompleteAI",
            "confidence": 0.78,
            "cost": 3,
        }
    ],
    "planning": [
        {
            "action": "complete_ai",
            "steps": ["think_plan", "route_complete_ai", "observe"],
            "expected": "plan via CompleteAI",
            "confidence": 0.76,
            "cost": 3,
        }
    ],
    "tool": [
        {
            "action": "complete_ai",
            "steps": ["think_plan", "route_complete_ai", "observe"],
            "expected": "tool result via CompleteAI",
            "confidence": 0.74,
            "cost": 3,
        }
    ],
    "vision": [
        {
            "action": "complete_ai",
            "steps": ["think_plan", "route_complete_ai", "observe"],
            "expected": "vision/perception answer",
            "confidence": 0.7,
            "cost": 4,
        }
    ],
    "learning": [
        {
            "action": "complete_ai",
            "steps": ["think_plan", "route_complete_ai", "observe"],
            "expected": "learning status/answer",
            "confidence": 0.72,
            "cost": 2,
        }
    ],
    "memory": [
        {
            "action": "complete_ai",
            "steps": ["think_plan", "route_complete_ai", "observe"],
            "expected": "memory recall/update",
            "confidence": 0.72,
            "cost": 2,
        }
    ],
}

DIALOGUE_FALLBACK = {
    "action": "dialogue",
    "steps": ["fallback_conversation"],
    "expected": "safe conversational reply",
    "confidence": 0.5,
    "cost": 1,
}

# Hard verifier gates — action rejected unless conditions hold
VERIFIER_RULES: list[dict[str, Any]] = [
    {
        "id": "no_empty_inspect",
        "action": "inspect_repo",
        "reject_if": "empty_normalized",
        "fallback": {"action": "dialogue", "steps": ["clarify"], "expected": "ask clarify", "confidence": 0.4, "cost": 1},
        "reason": "inspect requires non-empty input",
    },
    {
        "id": "no_coding_without_code_cues",
        "action": "coding",
        "reject_if": "no_coding_cues",
        "fallback": {
            "action": "complete_ai",
            "steps": ["think_plan", "route_complete_ai"],
            "expected": "clarify coding request",
            "confidence": 0.55,
            "cost": 2,
        },
        "reason": "coding action needs code-like cues or explicit coding intent",
    },
    {
        "id": "budget_block_expensive",
        "action": "*",
        "reject_if": "attention_exhausted",
        "fallback": DIALOGUE_FALLBACK,
        "reason": "attention budget exhausted — prefer cheap dialogue",
    },
]

CRITIC_RULES: list[dict[str, Any]] = [
    {"id": "low_conf", "when": "interp_confidence_lt_0.4", "risk": "low_interpretation_confidence", "prefer_low_cost": True},
    {"id": "partial_diag", "when": "intent_eq_inspect", "risk": "diagnostics_may_be_partial"},
    {"id": "no_proposals", "when": "empty_proposals", "risk": "no_proposals"},
    {"id": "hinglish_partial", "when": "lang_hi_or_hinglish", "risk": "hinglish_coverage_partial"},
    {"id": "high_cost_uncertain", "when": "interp_confidence_lt_0.5_and_cost_ge_3", "risk": "expensive_under_uncertainty", "prefer_low_cost": True},
    {"id": "world_model_error", "when": "avg_prediction_error_gt_0.6", "risk": "elevated_prediction_error", "prefer_low_cost": True},
]


@dataclass
class CritiqueReport:
    risks: list[str] = field(default_factory=list)
    prefer_low_cost_if_uncertain: bool = False
    rule_hits: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "risks": list(self.risks),
            "prefer_low_cost_if_uncertain": self.prefer_low_cost_if_uncertain,
            "rule_hits": list(self.rule_hits),
            "notes": list(self.notes),
            "human_level_agi": False,
        }


class RoleLibrary:
    """Proposer / Critic / Verifier with explicit rule IDs."""

    version = "mother_roles_v1"

    def propose(
        self,
        interp: InputInterpretation,
        *,
        goal: Any = None,
        world_summary: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        intent = interp.intent
        proposals: list[dict[str, Any]] = []
        for p in INTENT_ACTION_MAP.get(intent, []):
            proposals.append(dict(p))
        # Goal-bearing non-conversation intents get complete_ai if missing
        if goal is not None and intent not in ("conversation", "meta", "system_control") and not any(
            x.get("action") == "complete_ai" for x in proposals
        ):
            proposals.append(
                {
                    "action": "complete_ai",
                    "steps": ["think_plan", "route_complete_ai", "observe"],
                    "expected": "goal-directed CompleteAI response",
                    "confidence": 0.7,
                    "cost": 3,
                }
            )
        proposals.append(dict(DIALOGUE_FALLBACK))
        # Annotate with world-model resource hint
        attn = float((world_summary or {}).get("resources", {}).get("attention", 1.0))
        for p in proposals:
            p["attn_ok"] = attn >= 0.15 or p.get("cost", 9) <= 1
            p["human_level_agi"] = False
        return proposals

    def critique(
        self,
        proposals: list[dict[str, Any]],
        interp: InputInterpretation,
        *,
        world_summary: Optional[dict[str, Any]] = None,
    ) -> CritiqueReport:
        report = CritiqueReport()
        avg_err = float((world_summary or {}).get("avg_prediction_error") or 0.0)
        for rule in CRITIC_RULES:
            hit = False
            when = rule["when"]
            if when == "interp_confidence_lt_0.4" and interp.confidence < 0.4:
                hit = True
            elif when == "intent_eq_inspect" and interp.intent == "inspect":
                hit = True
            elif when == "empty_proposals" and not proposals:
                hit = True
            elif when == "lang_hi_or_hinglish" and interp.language in ("hi", "hinglish"):
                hit = True
            elif when == "interp_confidence_lt_0.5_and_cost_ge_3":
                if interp.confidence < 0.5 and any(float(p.get("cost") or 0) >= 3 for p in proposals):
                    hit = True
            elif when == "avg_prediction_error_gt_0.6" and avg_err > 0.6:
                hit = True
            if hit:
                report.rule_hits.append(rule["id"])
                report.risks.append(rule["risk"])
                if rule.get("prefer_low_cost"):
                    report.prefer_low_cost_if_uncertain = True
        return report

    def verify_select(
        self,
        proposals: list[dict[str, Any]],
        critique: CritiqueReport | dict[str, Any],
        interp: InputInterpretation,
        *,
        world_summary: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if isinstance(critique, CritiqueReport):
            crit = critique
        else:
            crit = CritiqueReport(
                risks=list(critique.get("risks") or []),
                prefer_low_cost_if_uncertain=bool(critique.get("prefer_low_cost_if_uncertain")),
            )
        if not proposals:
            return dict(DIALOGUE_FALLBACK)

        ranked = sorted(
            proposals,
            key=lambda p: (-float(p.get("confidence") or 0), float(p.get("cost") or 9)),
        )
        chosen = dict(ranked[0])
        if crit.prefer_low_cost_if_uncertain:
            cheap = sorted(proposals, key=lambda p: float(p.get("cost") or 9))[0]
            if float(cheap.get("confidence") or 0) >= 0.45:
                chosen = dict(cheap)

        attn = float((world_summary or {}).get("resources", {}).get("attention", 1.0))
        has_code_cues = bool(
            interp.intent == "coding"
            or any(
                k in (interp.normalized or "").lower()
                for k in ("def ", "function", "class ", "generate", "refactor", "code", "python", "implement")
            )
        )
        for rule in VERIFIER_RULES:
            action_match = rule["action"] in ("*", chosen.get("action"))
            if not action_match:
                continue
            reject = False
            cond = rule["reject_if"]
            if cond == "empty_normalized" and not interp.normalized:
                reject = True
            elif cond == "no_coding_cues" and chosen.get("action") == "coding" and not has_code_cues:
                reject = True
            elif cond == "attention_exhausted" and attn < 0.1 and float(chosen.get("cost") or 0) > 1:
                reject = True
            if reject:
                chosen = dict(rule["fallback"])
                chosen["verifier_reject"] = rule["id"]
                chosen["verifier_reason"] = rule["reason"]
                break

        chosen["critique_risks"] = list(crit.risks)
        chosen["human_level_agi"] = False
        return chosen
