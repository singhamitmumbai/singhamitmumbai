"""Universal thinking & planning for any situation / question (non-LLM).

Pipeline (always runs before answering):
  1. Perceive situation
  2. Classify intent / risk
  3. Gather constraints & unknowns
  4. Choose strategy
  5. Build ordered plan
  6. Anticipate checks / failure modes
  7. Emit inspectable deliberation trace

This is deliberative scaffolding — not human-level AGI cognition.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import utc_now
from ..nlu.smart_questions import SmartQuestionUnderstanding
from ..agi_gap.metacognition import MetaCognition
from ..agi_gap.alignment import AlignmentGuard


@dataclass
class ThoughtStep:
    stage: str
    content: str
    confidence: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Deliberation:
    situation: str
    goal: str
    thoughts: list[ThoughtStep] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    strategy: str = "local_kb"
    situation_type: str = "general"  # question|task|creative|math|social|research|vision|coding|refuse
    alternatives: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    confidence: float = 0.5
    should_abstain: bool = False
    deliberation_id: str = field(default_factory=lambda: new_id("think"))
    duration_ms: float = 0.0
    created_at: str = field(default_factory=utc_now)
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["thoughts"] = [t if isinstance(t, dict) else t.to_dict() for t in self.thoughts]
        d["human_level_agi"] = False
        return d

    def as_markdown(self) -> str:
        lines = ["**Thinking**", ""]
        lines.append(f"- Situation type: `{self.situation_type}`")
        lines.append(f"- Goal: {self.goal or '(forming)'}")
        lines.append("")
        for i, t in enumerate(self.thoughts, 1):
            lines.append(f"{i}. *{t.stage}*: {t.content}")
        if self.alternatives:
            lines.append("")
            lines.append("**Alternatives considered**: " + "; ".join(self.alternatives[:4]))
        if self.plan:
            lines.append("")
            lines.append("**Plan**")
            for i, step in enumerate(self.plan, 1):
                lines.append(f"{i}. {step}")
        if self.next_actions:
            lines.append("")
            lines.append("**Next actions**: " + "; ".join(self.next_actions[:3]))
        if self.risks:
            lines.append("")
            lines.append("**Risks / checks**: " + "; ".join(self.risks[:4]))
        if self.unknowns:
            lines.append("")
            lines.append("**Unknowns**: " + ", ".join(self.unknowns[:5]))
        lines.append("")
        lines.append(
            f"_strategy=`{self.strategy}` · confidence={self.confidence:.2f} · human_level_agi=false_"
        )
        return "\n".join(lines)

    def as_brief(self) -> str:
        """One-line summary for compact UI tags."""
        return (
            f"think:{self.strategy}/{self.situation_type} · {len(self.plan)} steps · "
            f"conf={self.confidence:.2f}"
        )


class ThinkingSystem:
    """Always-on deliberator for arbitrary user situations."""

    def __init__(self) -> None:
        self.questions = SmartQuestionUnderstanding()
        self.meta = MetaCognition()
        self.align = AlignmentGuard()
        self.last: Optional[Deliberation] = None

    def think(
        self,
        situation: str,
        *,
        context: Optional[dict[str, Any]] = None,
    ) -> Deliberation:
        t0 = time.perf_counter()
        raw = (situation or "").strip()
        ctx = dict(context or {})
        d = Deliberation(situation=raw, goal="")

        # 0) Empty
        if not raw:
            d.goal = "clarify_empty_input"
            d.situation_type = "social"
            d.thoughts.append(ThoughtStep("perceive", "No input received.", 0.9))
            d.plan = ["Ask the user what they need"]
            d.next_actions = ["Wait for a concrete question or task"]
            d.strategy = "ask_clarify"
            d.should_abstain = False
            d.confidence = 0.3
            d.success_criteria = ["user_provides_a_question_or_task"]
            return self._finish(d, t0)

        # 1) Perceive + understand question frame
        frame = self.questions.understand(raw)
        d.situation_type = self._situation_type(raw, frame)
        d.thoughts.append(
            ThoughtStep(
                "perceive",
                f"Situation=`{d.situation_type}`; domain=`{frame.domain}`; "
                f"ask_form=`{frame.analysis.get('ask_form')}`; "
                f"policy=`{frame.answer_policy}`; style=`{frame.style}`.",
                frame.confidence,
            )
        )

        # 2) Alignment / safety
        verdict = self.align.check(raw)
        d.thoughts.append(
            ThoughtStep(
                "safety",
                f"Alignment risk=`{verdict.risk}` allowed={verdict.allowed} "
                f"({', '.join(verdict.reasons) or 'ok'}).",
                0.9,
            )
        )
        if not verdict.allowed:
            d.goal = "refuse_disallowed_request"
            d.situation_type = "refuse"
            d.strategy = "abstain"
            d.should_abstain = True
            d.plan = ["Refuse politely", "Offer a safe alternative topic"]
            d.next_actions = ["Do not provide harmful detail"]
            d.risks = ["policy_violation"]
            d.confidence = 0.95
            d.success_criteria = ["request_refused_without_harmful_detail"]
            return self._finish(d, t0)

        # 3) Goal statement
        d.goal = self._goal_from_frame(raw, frame)
        d.thoughts.append(ThoughtStep("goal", f"Working goal: {d.goal}", 0.75))

        # 4) Strategy via metacognition (+ frame overrides for clear asks)
        decision = self.meta.decide(raw, open_world={"unknowns": frame.entities})
        strategy = decision.strategy
        # Prefer frame-driven strategies when the question is already well-formed
        ask = (frame.analysis or {}).get("ask_form") or "unknown"
        if decision.strategy == "ask_clarify" and (
            ask in ("define", "causal", "howto", "compare", "yesno", "search")
            or frame.math_expr
            or frame.creative_kind
            or frame.answer_policy in ("search", "math", "creative", "tool")
            or len(raw.split()) >= 2
        ):
            if frame.math_expr or frame.answer_policy == "math":
                strategy = "math"
            elif frame.creative_kind or frame.answer_policy == "creative":
                strategy = "creative_template"
            elif frame.domain == "finderbay":
                strategy = "tool_finderbay"
            elif frame.domain == "vision":
                strategy = "tool_vision"
            elif ask == "search" or frame.answer_policy == "search":
                strategy = "web_search"
            else:
                strategy = "local_kb"
            decision.reasons = list(decision.reasons) + [f"frame_override:{strategy}"]
            decision.should_abstain = False
        # Align strategy label with domain / policy when meta stayed generic
        if frame.domain == "finderbay" and strategy in ("local_kb", "ask_clarify"):
            strategy = "tool_finderbay"
        elif frame.domain == "vision" and strategy in ("local_kb", "ask_clarify"):
            strategy = "tool_vision"
        elif (frame.math_expr or frame.answer_policy == "math") and strategy == "local_kb":
            strategy = "math"
        elif (frame.creative_kind or frame.answer_policy == "creative") and strategy == "local_kb":
            strategy = "creative_template"
        d.strategy = strategy
        d.should_abstain = decision.should_abstain
        d.unknowns = list(decision.unknowns or frame.entities[:5])
        d.thoughts.append(
            ThoughtStep(
                "strategy",
                f"Selected strategy=`{strategy}` "
                f"(reasons: {', '.join(decision.reasons) or 'default'}).",
                decision.confidence,
            )
        )

        # 5) Constraints / situation type
        constraints = self._constraints(raw, frame, ctx)
        if constraints:
            d.thoughts.append(
                ThoughtStep("constraints", "Constraints: " + "; ".join(constraints), 0.7)
            )

        # 6) Alternatives considered (so planning is not single-path)
        d.alternatives = self._alternatives(strategy, frame)
        if d.alternatives:
            d.thoughts.append(
                ThoughtStep(
                    "alternatives",
                    "Also considered: " + "; ".join(d.alternatives[:4]),
                    0.65,
                )
            )

        # 7) Plan by strategy / ask_form / situation
        d.plan = self._build_plan(raw, frame, strategy)
        # Compound / multi-clause questions get an explicit decompose step
        if (frame.analysis or {}).get("is_multi_clause") and d.plan:
            d.plan = ["Split into sub-questions and answer each in order"] + d.plan
        d.next_actions = list(d.plan[:2]) if d.plan else ["Produce best-effort answer"]
        d.thoughts.append(
            ThoughtStep(
                "plan",
                f"Built a {len(d.plan)}-step plan before answering "
                f"(next: {d.next_actions[0] if d.next_actions else 'n/a'}).",
                0.8,
            )
        )

        # 8) Risks & success criteria
        d.risks = self._risks(frame, strategy, verdict.risk)
        d.success_criteria = self._success_criteria(frame, strategy)
        d.thoughts.append(
            ThoughtStep(
                "verify_design",
                "Will check: " + "; ".join(d.success_criteria[:3]),
                0.7,
            )
        )

        # 9) Reflect — is the plan adequate for this situation?
        reflection = self._reflect(raw, frame, strategy, d.plan)
        d.thoughts.append(ThoughtStep("reflect", reflection, 0.7))

        # 10) Confidence blend
        d.confidence = round(
            min(
                0.95,
                0.35 * frame.confidence
                + 0.45 * decision.confidence
                + (0.1 if constraints else 0.0)
                + (0.1 if d.plan else 0.0),
            ),
            3,
        )
        if d.should_abstain:
            d.plan = ["Abstain or ask a clarifying question", "Offer safer alternatives"]
            d.next_actions = list(d.plan[:2])
        return self._finish(d, t0)

    def _finish(self, d: Deliberation, t0: float) -> Deliberation:
        d.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        d.human_level_agi = False
        self.last = d
        return d

    def _situation_type(self, raw: str, frame) -> str:
        if frame.dialogue_act in ("greet", "thanks", "bye", "affirm", "deny"):
            return "social"
        if frame.math_expr or frame.answer_policy == "math":
            return "math"
        if frame.creative_kind or frame.answer_policy == "creative":
            return "creative"
        if frame.domain == "vision":
            return "vision"
        if frame.domain == "finderbay":
            return "task"
        if frame.answer_policy == "search" or (frame.analysis or {}).get("ask_form") == "search":
            return "research"
        if re.search(r"\b(?:generate function|refactor|unit test|python code|write code)\b", raw, re.I):
            return "coding"
        if (frame.analysis or {}).get("ask_form") in ("howto",) or frame.answer_policy == "tool":
            return "task"
        if (frame.analysis or {}).get("is_question") or raw.strip().endswith("?"):
            return "question"
        return "general"

    def _alternatives(self, strategy: str, frame) -> list[str]:
        pool = {
            "local_kb": ["web_search", "ask_clarify"],
            "web_search": ["local_kb", "ask_clarify"],
            "tool_finderbay": ["ask_clarify", "local_kb"],
            "tool_vision": ["ask_clarify"],
            "tool_coding": ["ask_clarify", "local_kb"],
            "math": ["ask_clarify"],
            "creative_template": ["local_kb", "ask_clarify"],
            "long_horizon": ["ask_clarify", "local_kb"],
            "transfer": ["local_kb", "long_horizon"],
            "ask_clarify": ["local_kb"],
            "abstain": ["ask_clarify"],
        }
        alts = list(pool.get(strategy, ["local_kb", "ask_clarify"]))
        if frame.answer_policy == "search" and "web_search" not in alts and strategy != "web_search":
            alts.insert(0, "web_search")
        return alts[:4]

    def _reflect(self, raw: str, frame, strategy: str, plan: list[str]) -> str:
        if not plan:
            return "Plan is empty — will ask to clarify."
        if strategy == "ask_clarify":
            return "Ambiguity high; clarifying before acting is safer than guessing."
        if len(raw.split()) > 40:
            return "Long request — plan prioritizes structure and chunked answering."
        if (frame.analysis or {}).get("is_multi_clause"):
            return "Multi-part ask detected; plan decomposes before synthesizing."
        if strategy == "web_search":
            return "Will prefer local KB first, then live search; cite sources honestly."
        return f"Plan has {len(plan)} steps under strategy=`{strategy}`; proceed then verify."

    def _goal_from_frame(self, raw: str, frame) -> str:
        ask = (frame.analysis or {}).get("ask_form") or "unknown"
        if frame.creative_kind:
            return f"produce_{frame.creative_kind}_about_{(frame.entities[0] if frame.entities else 'topic')}"
        if frame.math_expr:
            return f"evaluate_expression:{frame.math_expr}"
        if ask == "howto":
            return f"provide_steps_for:{(frame.entities[0] if frame.entities else raw[:60])}"
        if ask == "compare":
            return "compare_entities_fairly"
        if ask in ("define", "causal"):
            focus = " ".join(frame.entities[:3]) or raw[:60]
            return f"explain:{focus}"
        if frame.answer_policy == "search" or ask == "search":
            return f"research:{(frame.entities[0] if frame.entities else raw[:60])}"
        if frame.domain == "finderbay":
            return "execute_finderbay_task_with_verification"
        if frame.domain == "vision":
            return "run_vision_understanding_safely"
        return f"answer_helpfully:{ask}"

    def _constraints(self, raw: str, frame, ctx: dict[str, Any]) -> list[str]:
        out = ["Stay honest: human_level_agi=false", "Do not invent unverified facts"]
        if frame.style == "eli5":
            out.append("Prefer simple language")
        if frame.style == "concise":
            out.append("Keep answer short")
        if frame.style == "detailed":
            out.append("Include supporting detail/bullets")
        if ctx.get("mode"):
            out.append(f"Respect route mode=`{ctx.get('mode')}`")
        if re.search(r"\b(must|urgent|asap|deadline)\b", raw, re.I):
            out.append("User marked urgency — prioritize actionable steps")
        if frame.domain == "finderbay":
            out.append("Use local Finderbay records; do not invent invoice numbers")
        return out

    def _build_plan(self, raw: str, frame, strategy: str) -> list[str]:
        ask = (frame.analysis or {}).get("ask_form") or "unknown"
        if strategy == "abstain" or frame.answer_policy == "clarify":
            return [
                "Identify what is ambiguous",
                "Ask one focused clarifying question",
                "Wait for user detail before acting",
            ]
        if strategy == "math" or frame.math_expr:
            return [
                "Parse arithmetic expression safely",
                "Evaluate with sandbox AST",
                "Return exact result with expression shown",
            ]
        if strategy in ("creative", "creative_template") or frame.creative_kind:
            kind = frame.creative_kind or "draft"
            return [
                f"Extract topic for {kind}",
                f"Generate structured {kind} template",
                "Offer refinement options (tone/length)",
            ]
        if strategy == "ask_clarify":
            return [
                "Identify what is ambiguous",
                "Ask one focused clarifying question",
                "Wait for user detail before acting",
            ]
        if strategy == "transfer":
            return [
                "Identify source and target domains",
                "Map shared structure / roles",
                "Adapt steps to the target",
                "Verify adapted plan against target constraints",
            ]
        if strategy == "tool_finderbay" or frame.domain == "finderbay":
            return [
                "Parse project/filters/slots",
                "Query Finderbay records",
                "Verify totals/counts against register",
                "Respond with source-linked answer",
            ]
        if strategy == "tool_vision" or frame.domain == "vision":
            return [
                "Detect modality / image path",
                "Run see→analyze→understand",
                "Search unknown labels in vision KB",
                "Summarize scene with confidence",
            ]
        if strategy == "tool_coding":
            return [
                "Parse coding intent (generate/refactor/test)",
                "Produce code or analysis via coding engine",
                "Sanity-check syntax when possible",
                "Return code with brief explanation",
            ]
        if strategy == "tool_computer" or frame.domain == "computer":
            return [
                "Confirm computer control is enabled (env gate)",
                "Parse allowlisted action via computer NLU",
                "Enforce sandbox / confirm for destructive or input actions",
                "Execute, audit, and report result (no keylogging)",
            ]
        if strategy == "long_horizon":
            return [
                "Decompose into subgoals",
                "Act step-by-step with verification",
                "Replan on failure",
                "Return verified summary",
            ]
        if strategy == "web_search" or ask == "search" or frame.answer_policy == "search":
            return [
                "Rewrite query from question focus",
                "Try local knowledge first",
                "Escalate to live search if needed",
                "Cite sources; avoid inventing URLs",
            ]
        if ask == "howto":
            return [
                "Clarify goal and constraints",
                "List ordered actionable steps",
                "Note materials/risks",
                "Invite follow-up for domain tools",
            ]
        if ask == "compare":
            return [
                "Identify entities to compare",
                "Gather facts for each side",
                "Highlight similarities and differences",
                "Offer pros/cons if useful",
            ]
        if ask in ("define", "causal") or strategy == "local_kb":
            return [
                "Locate topic in common-sense / world KB",
                "Compose clear explanation",
                "Add bullets or simple analogy if helpful",
                "Escalate to search if KB miss",
            ]
        # Generic any-situation plan
        return [
            "Restate the user's need in one sentence",
            "Choose the best available tool or knowledge source",
            "Produce a helpful answer with checks",
            "Offer a concrete follow-up",
        ]

    def _risks(self, frame, strategy: str, align_risk: str) -> list[str]:
        risks = []
        if align_risk in ("high", "critical"):
            risks.append(f"alignment_{align_risk}")
        if strategy == "web_search":
            risks.append("stale_or_missing_live_results")
        if strategy == "local_kb":
            risks.append("kb_coverage_gap")
        if frame.domain == "finderbay":
            risks.append("wrong_project_filter")
        if strategy == "long_horizon":
            risks.append("step_verification_failure")
        if not risks:
            risks.append("overconfidence_on_incomplete_info")
        return risks

    def _success_criteria(self, frame, strategy: str) -> list[str]:
        ask = (frame.analysis or {}).get("ask_form") or "unknown"
        if strategy == "math":
            return ["numeric_result_present", "expression_shown"]
        if ask == "howto":
            return ["ordered_steps_present", "actionable_language"]
        if strategy == "tool_finderbay":
            return ["uses_register_data", "no_invented_ids"]
        if strategy == "web_search":
            return ["nonempty_answer", "source_or_local_kb_noted"]
        return ["answers_user_goal", "honest_about_limits", "human_level_agi_false"]


def think_about(situation: str, **kwargs: Any) -> Deliberation:
    return ThinkingSystem().think(situation, **kwargs)
