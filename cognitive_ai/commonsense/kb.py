"""Curated everyday common-sense triples — explicit, inspectable, non-LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..knowledge import KnowledgeBase, Rule, Triple


# Seed facts covering everyday physics, biology, social, geography, time, tools.
SEED_FACTS: list[tuple[str, str, str]] = [
    # biology / animals
    ("bird", "can", "fly"),
    ("bird", "has", "wings"),
    ("bird", "is_a", "animal"),
    ("fish", "can", "swim"),
    ("fish", "lives_in", "water"),
    ("fish", "is_a", "animal"),
    ("mammal", "is_a", "animal"),
    ("dog", "is_a", "mammal"),
    ("dog", "can", "bark"),
    ("cat", "is_a", "mammal"),
    ("cat", "can", "meow"),
    ("human", "is_a", "mammal"),
    ("human", "can", "think"),
    ("penguin", "is_a", "bird"),
    ("penguin", "cannot", "fly"),
    ("penguin", "can", "swim"),
    # physics / materials
    ("ice", "is", "frozen_water"),
    ("ice", "melts_to", "water"),
    ("water", "boils_at_c", "100"),
    ("water", "freezes_at_c", "0"),
    ("fire", "needs", "oxygen"),
    ("metal", "can", "conduct_electricity"),
    ("wood", "can", "burn"),
    ("glass", "is", "fragile"),
    ("sun", "provides", "light"),
    ("sun", "provides", "heat"),
    ("moon", "orbits", "earth"),
    ("earth", "orbits", "sun"),
    # everyday objects / tools
    ("knife", "used_for", "cutting"),
    ("key", "used_for", "opening_lock"),
    ("umbrella", "used_for", "rain_protection"),
    ("phone", "used_for", "communication"),
    ("clock", "used_for", "telling_time"),
    ("book", "used_for", "reading"),
    ("car", "needs", "fuel_or_electricity"),
    ("car", "used_for", "transport"),
    # social / norms
    ("promise", "should_be", "kept"),
    ("lying", "is_generally", "wrong"),
    ("thank_you", "is", "polite"),
    ("queue", "requires", "waiting_turn"),
    ("doctor", "helps", "patients"),
    ("teacher", "helps", "students"),
    # time / calendar
    ("week", "has_days", "7"),
    ("year", "has_months", "12"),
    ("day", "has_hours", "24"),
    ("yesterday", "is_before", "today"),
    ("tomorrow", "is_after", "today"),
    # geography (minimal)
    ("india", "is_a", "country"),
    ("mumbai", "is_in", "india"),
    ("delhi", "is_in", "india"),
    ("pacific", "is_a", "ocean"),
    # causation / affordances
    ("rain", "can_cause", "wet_ground"),
    ("sleep", "reduces", "tiredness"),
    ("food", "reduces", "hunger"),
    ("exercise", "can_improve", "health"),
    ("study", "can_improve", "knowledge"),
]

SEED_RULES: list[dict[str, Any]] = [
    {
        "name": "inherit_is_a",
        "antecedents": [("?x", "is_a", "?y"), ("?y", "is_a", "?z")],
        "consequents": [("?x", "is_a", "?z")],
    },
    {
        "name": "inherit_can_unless_cannot",
        "antecedents": [("?x", "is_a", "?y"), ("?y", "can", "?a")],
        "consequents": [("?x", "can", "?a")],
    },
    {
        "name": "tool_affordance",
        "antecedents": [("?t", "used_for", "?p"), ("?x", "has", "?t")],
        "consequents": [("?x", "can_perform", "?p")],
    },
]


class CommonSenseKB:
    """Load curated common-sense into a symbolic knowledge base."""

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.kb = KnowledgeBase(path)
        self.kb.set_domain("commonsense")
        if len(self.kb) == 0:
            self.seed()

    def seed(self) -> None:
        for s, p, o in SEED_FACTS:
            self.kb.assert_fact(s, p, o)
        for rd in SEED_RULES:
            rule = Rule(
                name=rd["name"],
                antecedents=[Triple(*t) for t in rd["antecedents"]],
                consequents=[Triple(*t) for t in rd["consequents"]],
            )
            self.kb.add_rule(rule)

    def ask_fact(self, subject: str, predicate: str = "*", obj: str = "*") -> list[Triple]:
        return self.kb.query(subject, predicate, obj)

    def related(self, entity: str) -> list[Triple]:
        return self.kb.related(entity.lower())

    def export_json(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "facts": [list(t.as_tuple()) for t in self.kb.facts()],
            "rules": [r.to_dict() for r in self.kb.rules()],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def summary(self) -> dict[str, Any]:
        return {"facts": len(self.kb), "rules": len(self.kb.rules()), "domain": self.kb.domain}
