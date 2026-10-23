"""Self-model: introspect modules, thoughts, capabilities, and limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Thought:
    """A recorded cognitive event for reflection / explanation."""

    kind: str  # perceive | reason | plan | act | reflect | learn | speak
    content: str
    detail: dict[str, Any] = field(default_factory=dict)
    thought_id: str = field(default_factory=lambda: uuid4().hex[:10])
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "thought_id": self.thought_id,
            "kind": self.kind,
            "content": self.content,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


@dataclass
class ModuleInfo:
    name: str
    role: str
    capabilities: list[str]
    limits: list[str]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "capabilities": list(self.capabilities),
            "limits": list(self.limits),
        }


DEFAULT_MODULES = [
    ModuleInfo(
        name="knowledge",
        role="Long-term world memory (facts + rules)",
        capabilities=[
            "store and query triples",
            "persist to JSON/SQLite",
            "hold production rules",
        ],
        limits=["no sensory grounding", "no natural-language semantics beyond tokens"],
    ),
    ModuleInfo(
        name="beliefs",
        role="Working memory and situational awareness",
        capabilities=[
            "track confidence-weighted beliefs",
            "hold open questions",
            "describe current situation",
        ],
        limits=["beliefs decay; not a full probabilistic model"],
    ),
    ModuleInfo(
        name="inference",
        role="Symbolic reasoner",
        capabilities=[
            "forward chaining",
            "backward chaining",
            "justify conclusions (why)",
        ],
        limits=["no statistical generalization", "finite rule depth"],
    ),
    ModuleInfo(
        name="planner",
        role="Means-ends / hierarchical planner",
        capabilities=[
            "decompose goals into subgoals",
            "select actions by effects",
            "detect already-satisfied goals",
        ],
        limits=["small action library", "no continuous control"],
    ),
    ModuleInfo(
        name="learner",
        role="Acquire facts/rules from interaction and failure",
        capabilities=[
            "write new facts from user input",
            "induce simple rules from repeated patterns",
            "record failed-plan lessons",
        ],
        limits=["no neural learning", "induction is heuristic"],
    ),
    ModuleInfo(
        name="self_model",
        role="Introspection and self-explanation",
        capabilities=[
            "answer what/why/trying-to-do questions",
            "list modules and limits",
            "replay recent thoughts",
        ],
        limits=["cannot report internal GPU/LLM state — none exists"],
    ),
    ModuleInfo(
        name="think_loop",
        role="Cognitive cycle controller",
        capabilities=[
            "perceive → believe → reason → plan → act → reflect",
            "run N autonomous steps",
            "stop when goals satisfied",
        ],
        limits=["discrete symbolic steps only"],
    ),
]


class SelfModel:
    """Explicit model of the agent's own mind for introspection."""

    def __init__(self, name: str = "SymbolicBrain") -> None:
        self.name = name
        self.modules = {m.name: m for m in DEFAULT_MODULES}
        self.thoughts: list[Thought] = []
        self.max_thoughts = 200
        self.identity = {
            "name": name,
            "kind": "symbolic cognitive system",
            "llm": False,
            "paradigm": "production rules + means-ends planning + working memory",
        }
        self._brain_ref: Any = None  # set by Brain

    def attach(self, brain: Any) -> None:
        self._brain_ref = brain

    def record(
        self,
        kind: str,
        content: str,
        detail: Optional[dict[str, Any]] = None,
    ) -> Thought:
        thought = Thought(kind=kind, content=content, detail=detail or {})
        self.thoughts.append(thought)
        if len(self.thoughts) > self.max_thoughts:
            self.thoughts = self.thoughts[-self.max_thoughts :]
        return thought

    def recent_thoughts(self, n: int = 10, kind: Optional[str] = None) -> list[Thought]:
        items = self.thoughts
        if kind:
            items = [t for t in items if t.kind == kind]
        return items[-n:]

    def describe_modules(self) -> list[dict]:
        return [m.to_dict() for m in self.modules.values()]

    def capabilities(self) -> list[str]:
        caps: list[str] = []
        for m in self.modules.values():
            caps.extend(f"{m.name}: {c}" for c in m.capabilities)
        return caps

    def limits(self) -> list[str]:
        lims: list[str] = [
            "I do not use large language models or neural text generation.",
            "I operate on symbolic facts, rules, goals, and actions.",
        ]
        for m in self.modules.values():
            lims.extend(f"{m.name}: {x}" for x in m.limits)
        return lims

    def answer(self, question: str) -> str:
        """Route common introspective questions to structured answers."""
        q = question.strip().lower()
        brain = self._brain_ref

        # Goal / activity queries before identity ("what are you" is a substring)
        if any(
            p in q
            for p in (
                "what are you trying",
                "what are you doing",
                "current goal",
                "your goal",
                "trying to do",
            )
        ):
            return self._trying_to_do()

        if any(p in q for p in ("who are you", "your name")) or q.rstrip("?") in (
            "what are you",
            "what are you?",
        ):
            return (
                f"I am {self.name}, a {self.identity['kind']}. "
                f"Paradigm: {self.identity['paradigm']}. "
                "I have no LLM backend."
            )

        if "what do you know" in q or "what know" in q or q in ("know?", "knowledge?"):
            return self._what_do_you_know()

        if q.startswith("why") or "why did you conclude" in q or "why conclude" in q:
            return self._why(question)

        if "how do you think" in q or "how think" in q:
            return (
                "Each think cycle: perceive inputs → update beliefs → "
                "forward/backward inference → plan toward open goals → "
                "execute symbolic actions → reflect and optionally learn. "
                "No neural sampling is involved."
            )

        if "capabilities" in q or "what can you" in q:
            return "Capabilities:\n- " + "\n- ".join(self.capabilities()[:20])

        if "limit" in q or "cannot" in q:
            return "Limits:\n- " + "\n- ".join(self.limits()[:15])

        if "modules" in q or "architecture" in q:
            lines = [
                f"- {m.name}: {m.role}" for m in self.modules.values()
            ]
            return "Modules:\n" + "\n".join(lines)

        if "recent thought" in q or "last thought" in q or "thinking" in q:
            recent = self.recent_thoughts(5)
            if not recent:
                return "I have no recorded thoughts yet."
            return "Recent thoughts:\n" + "\n".join(
                f"- [{t.kind}] {t.content}" for t in recent
            )

        if "belief" in q:
            if brain is None:
                return "No brain attached."
            beliefs = brain.working.active_beliefs()
            if not beliefs:
                return "Working memory has no active beliefs."
            return "Active beliefs:\n" + "\n".join(
                f"- {b.content} (conf={b.confidence:.2f}, src={b.source})"
                for b in beliefs[:20]
            )

        if brain is not None and hasattr(brain, "ask_about_world"):
            world = brain.ask_about_world(question)
            if world:
                return world

        return (
            "I did not match that to an introspective query. "
            "Try: 'what do you know?', 'why did you conclude X?', "
            "'what are you trying to do?', 'capabilities', 'modules'."
        )

    def _what_do_you_know(self) -> str:
        brain = self._brain_ref
        if brain is None:
            return "No knowledge attached."
        kb = brain.kb
        facts = kb.facts()
        rules = kb.rules()
        sample = facts[:15]
        lines = [
            f"Domain: {kb.domain}",
            f"Facts: {len(facts)}, Rules: {len(rules)}",
            "Sample facts:",
        ]
        lines.extend(f"  {t}" for t in sample)
        if len(facts) > 15:
            lines.append(f"  ... and {len(facts) - 15} more")
        if rules:
            lines.append("Rules:")
            for r in rules[:10]:
                lines.append(f"  {r.name}: {len(r.antecedents)}→{len(r.consequents)}")
        return "\n".join(lines)

    def _why(self, question: str) -> str:
        brain = self._brain_ref
        if brain is None:
            return "No brain attached to explain conclusions."
        # Parse "why did you conclude (a b c)" or "why (a b c)"
        import re

        m = re.search(
            r"\((\S+)\s+(\S+)\s+(\S+)\)",
            question,
        )
        if m:
            return brain.engine.why(m.group(1), m.group(2), m.group(3))
        # Try last inference step
        if brain.engine.history:
            last = brain.engine.history[-1]
            if last.steps:
                s = last.steps[-1]
                return (
                    f"Most recent conclusion: {s.conclusion} "
                    f"via rule '{s.rule_name}' from "
                    + ", ".join(str(p) for p in s.premises)
                )
            return last.explanation
        return (
            "Provide a triple, e.g. why did you conclude (tweety can fly)?"
        )

    def _trying_to_do(self) -> str:
        brain = self._brain_ref
        if brain is None:
            return "No planner attached."
        goals = brain.planner.open_goals()
        if not goals:
            sat = [g for g in brain.planner.goals.values() if g.status == "satisfied"]
            if sat:
                return (
                    "No open goals. Recently satisfied: "
                    + "; ".join(g.description for g in sat[-3:])
                )
            return "I am idle — no open goals."
        lines = ["I am trying to:"]
        for g in goals:
            lines.append(f"- [{g.status}] {g.description} → {g.target}")
        if brain.planner.plan_history:
            last = brain.planner.plan_history[-1]
            if last.steps:
                lines.append(
                    "Current plan steps: "
                    + ", ".join(s.grounded_action_name() for s in last.steps)
                )
        return "\n".join(lines)

    def dump(self) -> dict:
        return {
            "identity": dict(self.identity),
            "modules": self.describe_modules(),
            "recent_thoughts": [t.to_dict() for t in self.recent_thoughts(20)],
            "capabilities": self.capabilities(),
            "limits": self.limits(),
        }
