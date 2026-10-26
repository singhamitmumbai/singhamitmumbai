"""Tests for knowledge base persistence and queries."""

from pathlib import Path

from cognitive_ai.knowledge import KnowledgeBase, Rule, Triple
from cognitive_ai.learning import Learner
from cognitive_ai.planner import Plan, Goal


def test_sqlite_roundtrip(tmp_path: Path):
    path = tmp_path / "mind.sqlite"
    kb = KnowledgeBase()
    kb.set_domain("test")
    kb.assert_fact("a", "rel", "b")
    kb.add_rule(
        Rule(
            name="r",
            antecedents=[Triple("?x", "rel", "?y")],
            consequents=[Triple("?y", "rel", "?x")],
        )
    )
    kb.save(path)

    kb2 = KnowledgeBase(path)
    assert kb2.domain == "test"
    assert kb2.has_fact("a", "rel", "b")
    assert kb2.get_rule("r") is not None


def test_json_roundtrip(tmp_path: Path):
    path = tmp_path / "mind.json"
    kb = KnowledgeBase()
    kb.assert_fact("x", "y", "z")
    kb.save(path)
    kb2 = KnowledgeBase()
    kb2.load(path)
    assert kb2.has_fact("x", "y", "z")


def test_query_wildcards():
    kb = KnowledgeBase()
    kb.assert_fact("tweety", "is_a", "bird")
    kb.assert_fact("opus", "is_a", "penguin")
    hits = kb.query("*", "is_a", "*")
    assert len(hits) == 2


def test_learner_fact_and_induction():
    kb = KnowledgeBase()
    learner = Learner(kb)
    learner.learn_fact("sparky", "is_a", "dog")
    assert kb.has_fact("sparky", "is_a", "dog")

    a = Triple("sparky", "is_a", "dog")
    b = Triple("sparky", "can", "bark")
    learner.observe_cooccurrence(a, b)
    learner.observe_cooccurrence(a, b)
    kb.assert_fact("sparky", "can", "bark")
    induced = learner.induce_rules(min_support=2)
    assert induced
    assert any(r.learned for r in kb.rules())


def test_learn_from_failed_plan():
    kb = KnowledgeBase()
    learner = Learner(kb)
    goal = Goal("g", Triple("a", "b", "c"))
    plan = Plan(goal=goal, success=False, subgoals=[Goal("sg", Triple("x", "y", "z"))])
    lesson = learner.learn_from_failed_plan(plan)
    assert lesson.kind == "failure"
    assert kb.query(goal.goal_id, "plan_failed_for", "*")
