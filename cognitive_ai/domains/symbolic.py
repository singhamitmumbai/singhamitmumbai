"""Small built-in knowledge domains for demos and tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cognitive_ai.knowledge import Rule, Triple
from cognitive_ai.planner import Action

if TYPE_CHECKING:
    from cognitive_ai.brain import Brain


def load_domain(brain: "Brain", name: str) -> None:
    loaders = {
        "animals": load_animals,
        "blocks": load_blocks,
        "kitchen": load_kitchen,
    }
    if name not in loaders:
        raise ValueError(f"Unknown domain '{name}'. Choose from: {sorted(loaders)}")
    loaders[name](brain)


def load_animals(brain: "Brain") -> None:
    """Classic birds/mammals inheritance-style domain."""
    brain.kb.clear()
    brain.kb.set_domain("animals")
    brain.working.clear()
    brain.planner.goals.clear()
    brain.planner.actions.clear()
    brain.engine.history.clear()

    facts = [
        ("tweety", "is_a", "canary"),
        ("canary", "is_a", "bird"),
        ("bird", "can", "fly"),
        ("bird", "has", "wings"),
        ("opus", "is_a", "penguin"),
        ("penguin", "is_a", "bird"),
        ("penguin", "can", "swim"),
        ("penguin", "cannot", "fly"),
        ("mammal", "is_a", "animal"),
        ("bird", "is_a", "animal"),
        ("felix", "is_a", "cat"),
        ("cat", "is_a", "mammal"),
        ("cat", "can", "meow"),
        ("animal", "needs", "food"),
    ]
    for s, p, o in facts:
        brain.kb.assert_fact(s, p, o)

    brain.kb.add_rule(
        Rule(
            name="transitive_is_a",
            antecedents=[
                Triple("?x", "is_a", "?y"),
                Triple("?y", "is_a", "?z"),
            ],
            consequents=[Triple("?x", "is_a", "?z")],
        )
    )
    brain.kb.add_rule(
        Rule(
            name="inherit_can",
            antecedents=[
                Triple("?x", "is_a", "?y"),
                Triple("?y", "can", "?z"),
            ],
            consequents=[Triple("?x", "can", "?z")],
            confidence=0.85,
        )
    )
    brain.kb.add_rule(
        Rule(
            name="inherit_has",
            antecedents=[
                Triple("?x", "is_a", "?y"),
                Triple("?y", "has", "?z"),
            ],
            consequents=[Triple("?x", "has", "?z")],
        )
    )
    brain.kb.add_rule(
        Rule(
            name="inherit_needs",
            antecedents=[
                Triple("?x", "is_a", "?y"),
                Triple("?y", "needs", "?z"),
            ],
            consequents=[Triple("?x", "needs", "?z")],
        )
    )
    # Exception-ish: if cannot fly, we still may derive can fly via inheritance;
    # a more advanced system would use priorities. We keep both facts visible.

    brain.register_action(
        Action(
            name="observe_animal",
            description="Directly observe an animal property",
            preconditions=[Triple("?x", "is_a", "animal")],
            effects=[Triple("?x", "observed", "true")],
            cost=1.0,
        )
    )
    brain.register_action(
        Action(
            name="feed",
            description="Feed an animal that needs food",
            preconditions=[Triple("?x", "needs", "food")],
            effects=[Triple("?x", "fed", "true")],
            cost=1.0,
        )
    )
    brain.register_action(
        Action(
            name="teach_fact",
            description="Record that an agent knows a derived capability",
            preconditions=[Triple("?x", "can", "?y")],
            effects=[Triple("agent", "knows_can", "?x")],
            cost=0.5,
        )
    )


def load_blocks(brain: "Brain") -> None:
    """Tiny blocks-world for planning demos."""
    brain.kb.clear()
    brain.kb.set_domain("blocks")
    brain.working.clear()
    brain.planner.goals.clear()
    brain.planner.actions.clear()
    brain.engine.history.clear()

    for s, p, o in [
        ("block_a", "is_a", "block"),
        ("block_b", "is_a", "block"),
        ("block_c", "is_a", "block"),
        ("block_a", "on", "table"),
        ("block_b", "on", "table"),
        ("block_c", "on", "block_a"),
        ("block_b", "clear", "true"),
        ("block_c", "clear", "true"),
        ("hand", "empty", "true"),
    ]:
        brain.kb.assert_fact(s, p, o)

    brain.register_action(
        Action(
            name="pickup",
            description="Pick up a clear block when hand is empty",
            preconditions=[
                Triple("?x", "clear", "true"),
                Triple("?x", "on", "?y"),
                Triple("hand", "empty", "true"),
            ],
            effects=[
                Triple("hand", "holding", "?x"),
                Triple("?x", "held", "true"),
            ],
            deletes=[
                Triple("hand", "empty", "true"),
                Triple("?x", "on", "?y"),
                Triple("?x", "clear", "true"),
            ],
            cost=1.0,
        )
    )
    brain.register_action(
        Action(
            name="put_on_table",
            description="Put held block on table",
            preconditions=[Triple("hand", "holding", "?x")],
            effects=[
                Triple("?x", "on", "table"),
                Triple("?x", "clear", "true"),
                Triple("hand", "empty", "true"),
            ],
            deletes=[
                Triple("hand", "holding", "?x"),
                Triple("?x", "held", "true"),
            ],
            cost=1.0,
        )
    )
    brain.register_action(
        Action(
            name="stack",
            description="Stack held block onto a clear block",
            preconditions=[
                Triple("hand", "holding", "?x"),
                Triple("?y", "clear", "true"),
                Triple("?y", "is_a", "block"),
            ],
            effects=[
                Triple("?x", "on", "?y"),
                Triple("?x", "clear", "true"),
                Triple("hand", "empty", "true"),
            ],
            deletes=[
                Triple("hand", "holding", "?x"),
                Triple("?y", "clear", "true"),
                Triple("?x", "held", "true"),
            ],
            cost=1.5,
        )
    )


def load_kitchen(brain: "Brain") -> None:
    """Simple kitchen/cooking domain mixing rules and goals."""
    brain.kb.clear()
    brain.kb.set_domain("kitchen")
    brain.working.clear()
    brain.planner.goals.clear()
    brain.planner.actions.clear()
    brain.engine.history.clear()

    for s, p, o in [
        ("egg", "is_a", "ingredient"),
        ("butter", "is_a", "ingredient"),
        ("pan", "is_a", "tool"),
        ("egg", "location", "fridge"),
        ("butter", "location", "fridge"),
        ("pan", "location", "cabinet"),
        ("fridge", "contains", "egg"),
        ("fridge", "contains", "butter"),
        ("chef", "is_a", "agent"),
        ("omelette", "requires", "egg"),
        ("omelette", "requires", "butter"),
        ("omelette", "requires", "pan"),
    ]:
        brain.kb.assert_fact(s, p, o)

    brain.kb.add_rule(
        Rule(
            name="have_if_location_counter",
            antecedents=[Triple("?x", "location", "counter")],
            consequents=[Triple("chef", "has", "?x")],
        )
    )

    brain.register_action(
        Action(
            name="get_from_fridge",
            description="Take an item from the fridge to the counter",
            preconditions=[Triple("?x", "location", "fridge")],
            effects=[Triple("?x", "location", "counter")],
            deletes=[Triple("?x", "location", "fridge")],
            cost=1.0,
        )
    )
    brain.register_action(
        Action(
            name="get_pan",
            description="Get pan from cabinet",
            preconditions=[Triple("pan", "location", "cabinet")],
            effects=[Triple("pan", "location", "counter")],
            deletes=[Triple("pan", "location", "cabinet")],
            cost=1.0,
        )
    )
    brain.register_action(
        Action(
            name="cook_omelette",
            description="Cook an omelette when ingredients are on the counter",
            preconditions=[
                Triple("egg", "location", "counter"),
                Triple("butter", "location", "counter"),
                Triple("pan", "location", "counter"),
            ],
            effects=[Triple("omelette", "status", "ready")],
            cost=2.0,
        )
    )
