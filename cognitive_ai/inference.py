"""Inference engine: forward chaining and backward chaining over rules + facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .knowledge import KnowledgeBase, Rule, Triple


@dataclass
class InferenceStep:
    """One justified derivation."""

    conclusion: Triple
    rule_name: str
    bindings: dict[str, str]
    premises: list[Triple]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "conclusion": str(self.conclusion),
            "rule": self.rule_name,
            "bindings": dict(self.bindings),
            "premises": [str(p) for p in self.premises],
            "confidence": self.confidence,
        }


@dataclass
class InferenceTrace:
    """Full trace of an inference run for self-explanation."""

    mode: str
    steps: list[InferenceStep] = field(default_factory=list)
    derived: list[Triple] = field(default_factory=list)
    goal: Optional[Triple] = None
    success: bool = False
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "goal": str(self.goal) if self.goal else None,
            "success": self.success,
            "explanation": self.explanation,
            "derived": [str(t) for t in self.derived],
            "steps": [s.to_dict() for s in self.steps],
        }


def _resolve(var: str, env: dict[str, str]) -> str:
    """Follow binding chains to a non-var or unbound var."""
    seen: set[str] = set()
    while var.startswith("?") and var in env and var not in seen:
        seen.add(var)
        var = env[var]
    return var


def substitute(triple: Triple, bindings: dict[str, str]) -> Triple:
    def sub(x: str) -> str:
        if x.startswith("?"):
            return _resolve(x, bindings)
        return x

    return Triple(sub(triple.subject), sub(triple.predicate), sub(triple.object))


def unify(a: Triple, b: Triple, bindings: Optional[dict[str, str]] = None) -> Optional[dict[str, str]]:
    """Unify two triples; variables start with '?'."""
    env = dict(bindings or {})
    for left, right in zip(a.as_tuple(), b.as_tuple()):
        lv = _resolve(left, env) if left.startswith("?") else left
        rv = _resolve(right, env) if right.startswith("?") else right
        if lv.startswith("?") and rv.startswith("?"):
            if lv != rv:
                env[lv] = rv
        elif lv.startswith("?"):
            env[lv] = rv
        elif rv.startswith("?"):
            env[rv] = lv
        elif lv == "*" or rv == "*":
            continue
        elif lv != rv:
            return None
    return env


_fresh_counter = 0


def freshen_rule(rule: Rule) -> Rule:
    """Rename rule variables so nested applications do not clash."""
    global _fresh_counter
    mapping: dict[str, str] = {}

    def rename_slot(x: str) -> str:
        global _fresh_counter
        if not x.startswith("?"):
            return x
        if x not in mapping:
            _fresh_counter += 1
            mapping[x] = f"{x}_{_fresh_counter}"
        return mapping[x]

    def rename_triple(t: Triple) -> Triple:
        return Triple(rename_slot(t.subject), rename_slot(t.predicate), rename_slot(t.object))

    return Rule(
        name=rule.name,
        antecedents=[rename_triple(t) for t in rule.antecedents],
        consequents=[rename_triple(t) for t in rule.consequents],
        confidence=rule.confidence,
        learned=rule.learned,
    )


class InferenceEngine:
    """Forward and backward chaining over a KnowledgeBase (+ optional extras)."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.history: list[InferenceTrace] = []

    def _all_facts(self, extra: Optional[list[Triple]] = None) -> list[Triple]:
        facts = list(self.kb.facts())
        if extra:
            seen = {t.as_tuple() for t in facts}
            for t in extra:
                if t.as_tuple() not in seen:
                    facts.append(t)
                    seen.add(t.as_tuple())
        return facts

    def forward_chain(
        self,
        extra_facts: Optional[list[Triple]] = None,
        max_iterations: int = 50,
        write_back: bool = True,
    ) -> InferenceTrace:
        """Derive all reachable conclusions via production rules."""
        trace = InferenceTrace(mode="forward")
        working = {t.as_tuple(): t for t in self._all_facts(extra_facts)}
        changed = True
        iterations = 0

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for rule in self.kb.rules():
                for bindings in self._match_antecedents(rule.antecedents, list(working.values())):
                    premises = [substitute(a, bindings) for a in rule.antecedents]
                    for cons in rule.consequents:
                        conclusion = substitute(cons, bindings)
                        # Skip if still has unbound variables
                        if any(x.startswith("?") for x in conclusion.as_tuple()):
                            continue
                        key = conclusion.as_tuple()
                        if key in working:
                            continue
                        working[key] = conclusion
                        step = InferenceStep(
                            conclusion=conclusion,
                            rule_name=rule.name,
                            bindings=bindings,
                            premises=premises,
                            confidence=rule.confidence,
                        )
                        trace.steps.append(step)
                        trace.derived.append(conclusion)
                        if write_back:
                            self.kb.assert_fact(*key)
                        changed = True

        trace.success = bool(trace.derived)
        if trace.derived:
            trace.explanation = (
                f"Forward chaining derived {len(trace.derived)} fact(s) "
                f"in {len(trace.steps)} step(s)."
            )
        else:
            trace.explanation = "Forward chaining produced no new facts."
        self.history.append(trace)
        return trace

    def backward_chain(
        self,
        goal: Triple,
        extra_facts: Optional[list[Triple]] = None,
        max_depth: int = 12,
    ) -> InferenceTrace:
        """Try to prove a goal by reducing it via rules."""
        trace = InferenceTrace(mode="backward", goal=goal)
        facts = self._all_facts(extra_facts)
        proven, steps = self._prove(goal, facts, {}, max_depth, set())
        trace.steps = steps
        trace.success = proven
        if proven:
            grounded = substitute(goal, steps[-1].bindings) if steps else goal
            # Prefer fully grounded from successful proof tree leaves
            for step in steps:
                if step.conclusion.as_tuple() == substitute(goal, step.bindings).as_tuple() or (
                    not any(x.startswith("?") for x in step.conclusion.as_tuple())
                    and unify(goal, step.conclusion) is not None
                ):
                    grounded = step.conclusion
            # Collect all grounded conclusions that unify with goal
            for step in steps:
                if unify(goal, step.conclusion) is not None and not any(
                    x.startswith("?") for x in step.conclusion.as_tuple()
                ):
                    if step.conclusion not in trace.derived:
                        trace.derived.append(step.conclusion)
            if not trace.derived and not any(x.startswith("?") for x in grounded.as_tuple()):
                trace.derived.append(grounded)
            trace.explanation = (
                f"Proved {goal} via backward chaining "
                f"({len(trace.steps)} inference step(s))."
            )
        else:
            # Also check if goal is already a known fact (with bindings)
            for fact in facts:
                env = unify(goal, fact)
                if env is not None:
                    trace.success = True
                    trace.derived.append(fact)
                    trace.explanation = f"{goal} is already known as {fact}."
                    self.history.append(trace)
                    return trace
            trace.explanation = f"Could not prove {goal}."
        self.history.append(trace)
        return trace

    def why(self, subject: str, predicate: str, obj: str) -> str:
        """Explain why a conclusion was reached, using history."""
        target = Triple(subject, predicate, obj)
        # Direct fact?
        if self.kb.has_fact(subject, predicate, obj):
            for trace in reversed(self.history):
                for step in trace.steps:
                    if step.conclusion.as_tuple() == target.as_tuple():
                        premises = ", ".join(str(p) for p in step.premises)
                        return (
                            f"I concluded {target} by rule '{step.rule_name}' "
                            f"from premises: {premises} "
                            f"(confidence={step.confidence:.2f})."
                        )
            return f"{target} is a stored fact in knowledge (domain={self.kb.domain})."
        return f"I have no record of concluding {target}."

    def _match_antecedents(
        self,
        antecedents: list[Triple],
        facts: list[Triple],
        bindings: Optional[dict[str, str]] = None,
    ) -> list[dict[str, str]]:
        if not antecedents:
            return [dict(bindings or {})]

        first, *rest = antecedents
        results: list[dict[str, str]] = []
        for fact in facts:
            env = unify(first, fact, bindings)
            if env is None:
                continue
            for deeper in self._match_antecedents(rest, facts, env):
                results.append(deeper)
        return results

    def _prove(
        self,
        goal: Triple,
        facts: list[Triple],
        bindings: dict[str, str],
        depth: int,
        visiting: set[tuple],
    ) -> tuple[bool, list[InferenceStep]]:
        if depth <= 0:
            return False, []

        grounded = substitute(goal, bindings)
        gkey = grounded.as_tuple()
        if gkey in visiting:
            return False, []

        # Fact base match
        for fact in facts:
            env = unify(grounded, fact, bindings)
            if env is not None:
                return True, []

        next_visiting = set(visiting)
        next_visiting.add(gkey)

        # Try rules whose consequent unifies with goal (fresh vars each time)
        for rule in self.kb.rules():
            fresh = freshen_rule(rule)
            for cons in fresh.consequents:
                env = unify(cons, grounded, bindings)
                if env is None:
                    continue
                rule_key = ("rule", rule.name) + substitute(cons, env).as_tuple()
                if rule_key in next_visiting:
                    continue
                branch_visiting = set(next_visiting)
                branch_visiting.add(rule_key)
                ok, steps = self._prove_all(
                    fresh.antecedents, facts, env, depth - 1, branch_visiting
                )
                if ok:
                    conclusion = substitute(cons, env)
                    # Resolve through binding chains for readable premises
                    resolved = {k: _resolve(k, env) for k in env}
                    premises = [substitute(a, resolved) for a in fresh.antecedents]
                    step = InferenceStep(
                        conclusion=substitute(cons, resolved),
                        rule_name=rule.name,
                        bindings=resolved,
                        premises=premises,
                        confidence=rule.confidence,
                    )
                    return True, steps + [step]
        return False, []

    def _prove_all(
        self,
        goals: list[Triple],
        facts: list[Triple],
        bindings: dict[str, str],
        depth: int,
        visiting: set[tuple],
    ) -> tuple[bool, list[InferenceStep]]:
        if not goals:
            return True, []
        if depth <= 0:
            return False, []

        first, *rest = goals
        grounded = substitute(first, bindings)
        # Fully resolve variables in grounded pattern for visit key stability
        grounded = Triple(
            *(_resolve(x, bindings) if x.startswith("?") else x for x in grounded.as_tuple())
        )
        gkey = grounded.as_tuple()
        if gkey in visiting:
            return False, []

        candidates: list[tuple[dict[str, str], list[InferenceStep]]] = []

        for fact in facts:
            env = unify(grounded, fact, bindings)
            if env is not None:
                candidates.append((env, []))

        next_visiting = set(visiting)
        next_visiting.add(gkey)

        for rule in self.kb.rules():
            fresh = freshen_rule(rule)
            for cons in fresh.consequents:
                env = unify(cons, grounded, bindings)
                if env is None:
                    continue
                rule_key = ("rule", rule.name) + substitute(cons, env).as_tuple()
                if rule_key in next_visiting:
                    continue
                branch_visiting = set(next_visiting)
                branch_visiting.add(rule_key)
                ok, steps = self._prove_all(
                    fresh.antecedents, facts, env, depth - 1, branch_visiting
                )
                if ok:
                    resolved = {k: _resolve(k, env) for k in env}
                    conclusion = substitute(cons, resolved)
                    premises = [substitute(a, resolved) for a in fresh.antecedents]
                    step = InferenceStep(
                        conclusion=conclusion,
                        rule_name=rule.name,
                        bindings=resolved,
                        premises=premises,
                        confidence=rule.confidence,
                    )
                    candidates.append((env, steps + [step]))

        for env, prefix_steps in candidates:
            ok, more = self._prove_all(rest, facts, env, depth, next_visiting)
            if ok:
                return True, prefix_steps + more
        return False, []
