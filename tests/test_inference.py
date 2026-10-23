"""Tests for forward/backward chaining and explanations."""

from cognitive_ai.brain import Brain
from cognitive_ai.knowledge import Rule, Triple


def test_forward_chain_inherits_can_fly():
    brain = Brain()
    brain.load_domain("animals")
    trace = brain.engine.forward_chain()
    assert any(
        t.as_tuple() == ("tweety", "can", "fly") for t in trace.derived
    ) or brain.kb.has_fact("tweety", "can", "fly")
    # After forward chain, fact should be in KB
    assert brain.kb.has_fact("tweety", "can", "fly")
    assert brain.kb.has_fact("tweety", "is_a", "bird")


def test_backward_chain_prove_tweety_fly():
    brain = Brain()
    brain.load_domain("animals")
    # Don't rely on forward write-back — pure backward proof
    trace = brain.engine.backward_chain(Triple("tweety", "can", "fly"))
    assert trace.success
    assert trace.mode == "backward"


def test_why_explains_derived_fact():
    brain = Brain()
    brain.load_domain("animals")
    brain.engine.forward_chain()
    explanation = brain.why("tweety", "can", "fly")
    assert "tweety" in explanation
    assert "fly" in explanation
    assert "rule" in explanation.lower() or "fact" in explanation.lower()


def test_custom_rule_fires():
    brain = Brain()
    brain.assert_fact("water", "is_a", "liquid")
    brain.add_rule(
        Rule(
            name="liquid_flows",
            antecedents=[Triple("?x", "is_a", "liquid")],
            consequents=[Triple("?x", "can", "flow")],
        )
    )
    trace = brain.engine.forward_chain()
    assert brain.kb.has_fact("water", "can", "flow")
    assert any(s.rule_name == "liquid_flows" for s in trace.steps)
