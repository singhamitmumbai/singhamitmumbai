"""Much larger learned/curated world + common-sense knowledge (non-LLM).

Expands the symbolic KB with combinatorial world facts and provides an
open-world understanding layer (entity typing, schema hints, unknown handling).
Still curated/generated — not scraped web AGI knowledge.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Optional

from ..commonsense.generate_kb import materialize
from ..commonsense.kb import CommonSenseKB
from ..knowledge import Rule, Triple


# Extra taxonomies for open-world scale-up
SCIENCE_CONCEPTS = {
    "force": ["gravity", "friction", "tension", "magnetism", "electricity"],
    "wave": ["light", "sound", "radio_wave", "ocean_wave"],
    "energy": ["kinetic_energy", "potential_energy", "thermal_energy", "chemical_energy"],
    "matter_state": ["solid", "liquid", "gas", "plasma"],
}

GEO_EXTRA = {
    "continent": ["africa", "antarctica", "asia", "europe", "north_america", "south_america", "australia"],
    "ocean": ["pacific", "atlantic", "indian", "arctic", "southern"],
}

TECH_STACK = {
    "language": ["python", "javascript", "typescript", "java", "go", "rust", "c", "sql"],
    "database": ["sqlite", "postgres", "mysql", "mongodb", "redis"],
    "protocol": ["http", "https", "tcp", "udp", "smtp", "dns"],
}

EVERYDAY_EVENTS = [
    ("rain", "can_cause", "wet_ground"),
    ("rain", "can_cause", "flooding"),
    ("sun", "provides", "light"),
    ("sun", "provides", "heat"),
    ("night", "is_darker_than", "day"),
    ("sleep", "restores", "energy"),
    ("exercise", "improves", "fitness"),
    ("study", "improves", "knowledge"),
    ("practice", "improves", "skill"),
    ("apology", "can_repair", "relationship"),
    ("promise", "creates", "expectation"),
    ("deadline", "creates", "urgency"),
    ("budget", "limits", "spending"),
    ("map", "helps", "navigation"),
    ("recipe", "helps", "cooking"),
    ("manual", "helps", "repair"),
    ("experiment", "tests", "hypothesis"),
    ("hypothesis", "is_a", "proposal"),
    ("evidence", "supports_or_refutes", "hypothesis"),
    ("measurement", "reduces", "uncertainty"),
]

OPEN_WORLD_SCHEMAS: dict[str, list[str]] = {
    "animal": ["can", "cannot", "lives_in", "has", "is_a", "eats"],
    "tool": ["used_for", "is_a", "made_of", "requires"],
    "place": ["is_in", "is_a", "contains"],
    "person_role": ["is_a", "helps", "works_at"],
    "concept": ["is_a", "related_to", "defined_as"],
    "event": ["can_cause", "happens_in", "requires"],
    "substance": ["is_a", "state", "used_for"],
}


def iter_expanded_world_facts() -> Iterator[tuple[str, str, str]]:
    """Yield thousands of additional explicit world facts (deterministic)."""
    # Science hierarchy
    for kind, members in SCIENCE_CONCEPTS.items():
        yield (kind, "is_a", "science_concept")
        for m in members:
            yield (m, "is_a", kind)
            yield (m, "is_a", "science_concept")
            yield (m, "studied_in", "science")

    for kind, members in GEO_EXTRA.items():
        yield (kind, "is_a", "geo_category")
        for m in members:
            yield (m, "is_a", kind)

    for kind, members in TECH_STACK.items():
        yield (kind, "is_a", "tech_category")
        for m in members:
            yield (m, "is_a", kind)
            yield (m, "used_in", "computing")

    for t in EVERYDAY_EVENTS:
        yield t

    # Combinatorial: each tech language can_build software artifact
    artifacts = ["script", "api", "service", "library", "test", "cli"]
    for lang in TECH_STACK["language"]:
        for art in artifacts:
            yield (lang, "can_build", art)

    # Combinatorial: continents contain placeholder countries pattern
    for cont in GEO_EXTRA["continent"]:
        yield (cont, "has_property", "landmass")
        yield (cont, "part_of", "earth")

    # Days / time commons
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for d in days:
        yield (d, "is_a", "weekday" if d not in ("saturday", "sunday") else "weekend_day")
        yield (d, "is_a", "day_of_week")

    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    for i, m in enumerate(months, 1):
        yield (m, "is_a", "month")
        yield (m, "month_index", str(i))

    # Numbers / quantities meta
    for n in range(0, 21):
        yield (str(n), "is_a", "integer")
        if n % 2 == 0:
            yield (str(n), "is", "even")
        else:
            yield (str(n), "is", "odd")

    # Emotions basic
    emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
    for e in emotions:
        yield (e, "is_a", "emotion")
        yield (e, "experienced_by", "human")

    # Professions × workplace grid
    roles = ["doctor", "teacher", "engineer", "farmer", "pilot", "chef", "nurse", "lawyer"]
    places = ["hospital", "school", "office", "farm", "airport", "kitchen", "clinic", "court"]
    for role, place in zip(roles, places):
        yield (role, "often_works_at", place)
        yield (role, "is_a", "profession")
        yield (place, "is_a", "workplace")

    # Materials × states
    materials = ["water", "iron", "wood", "glass", "plastic", "gold", "silver", "copper"]
    for mat in materials:
        yield (mat, "is_a", "material")
        for st in ("solid", "liquid", "gas"):
            yield (mat, "may_exist_as", st)

    # Causal chains (shallow)
    causal = [
        ("spark", "can_cause", "fire"),
        ("fire", "can_cause", "smoke"),
        ("smoke", "can_cause", "alarm"),
        ("drought", "can_cause", "crop_failure"),
        ("crop_failure", "can_cause", "food_shortage"),
        ("infection", "can_cause", "fever"),
        ("fever", "can_indicate", "illness"),
        ("overheating", "can_cause", "shutdown"),
        ("backup", "can_prevent", "data_loss"),
        ("training", "can_improve", "performance"),
    ]
    for t in causal:
        yield t

    # Spatial relations vocabulary
    for rel in ("above", "below", "inside", "outside", "near", "far", "left_of", "right_of"):
        yield (rel, "is_a", "spatial_relation")


def expand_rules() -> list[Rule]:
    specs = [
        ("science_inherit", [("?x", "is_a", "?y"), ("?y", "is_a", "science_concept")], [("?x", "in_domain", "science")]),
        ("tech_inherit", [("?x", "is_a", "?y"), ("?y", "is_a", "tech_category")], [("?x", "in_domain", "technology")]),
        ("cause_chain", [("?x", "can_cause", "?y"), ("?y", "can_cause", "?z")], [("?x", "may_lead_to", "?z")]),
        ("profession_place", [("?r", "often_works_at", "?p"), ("?p", "is_a", "workplace")], [("?r", "has_workplace", "?p")]),
    ]
    out = []
    for name, ants, cons in specs:
        out.append(
            Rule(
                name=name,
                antecedents=[Triple(*a) for a in ants],
                consequents=[Triple(*c) for c in cons],
            )
        )
    return out


@dataclass
class OpenWorldFrame:
    text: str
    entities: list[str] = field(default_factory=list)
    types: dict[str, str] = field(default_factory=dict)
    known_facts: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    schema_hints: list[str] = field(default_factory=list)
    confidence: float = 0.0
    open_world: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorldKnowledgeHub:
    """Larger world KB + open-world framing."""

    def __init__(self, kb: Optional[CommonSenseKB] = None, *, expand: bool = True) -> None:
        self.kb = kb or CommonSenseKB()
        self.expanded = False
        if expand:
            self.ensure_expanded()

    def ensure_expanded(self) -> dict[str, Any]:
        if self.expanded:
            return self.stats()
        base_facts, _ = materialize()
        extra = list(iter_expanded_world_facts())
        before = len(self.kb.kb)
        for s, p, o in extra:
            self.kb.kb.assert_fact(s, p, o)
        for rule in expand_rules():
            # avoid dup rule names
            existing = {r.name for r in self.kb.kb.rules()}
            if rule.name not in existing:
                self.kb.kb.add_rule(rule)
        self.expanded = True
        return {
            "base_fact_templates": len(base_facts),
            "extra_facts_asserted": len(extra),
            "kb_facts_after": len(self.kb.kb),
            "kb_before": before,
            "human_level_agi": False,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "facts": len(self.kb.kb),
            "rules": len(self.kb.kb.rules()),
            "expanded": self.expanded,
            "human_level_agi": False,
            "note": "Curated/combinatorial expansion — not AGI-scale world model.",
        }

    def understand(self, text: str) -> OpenWorldFrame:
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", (text or "").lower())
        stop = {
            "the", "and", "for", "what", "who", "where", "when", "why", "how",
            "can", "does", "about", "with", "from", "this", "that", "have", "tell",
        }
        ents = [t for t in tokens if t not in stop]
        types: dict[str, str] = {}
        known: list[str] = []
        unknown: list[str] = []
        schemas: list[str] = []
        for e in ents[:12]:
            rel = self.kb.related(e)
            if rel:
                known.extend(str(t) for t in rel[:4])
                # type guess from is_a
                isa = [t for t in rel if getattr(t, "predicate", "") == "is_a"]
                if isa:
                    types[e] = str(isa[0].object)
                    for schema, preds in OPEN_WORLD_SCHEMAS.items():
                        if types[e] == schema or types[e] in (
                            "animal", "bird", "fish", "mammal", "tool", "profession", "material"
                        ):
                            schemas.append(schema)
                else:
                    types[e] = "known_entity"
            else:
                unknown.append(e)
                types[e] = "unknown"
        # Suggest schema probes for unknowns
        for u in unknown[:5]:
            schemas.append(f"probe:{u}:is_a|used_for|can|located_in")
        conf = 0.3 + 0.05 * len(known) - 0.03 * len(unknown)
        return OpenWorldFrame(
            text=text,
            entities=ents[:12],
            types=types,
            known_facts=known[:20],
            unknowns=unknown[:10],
            schema_hints=list(dict.fromkeys(schemas))[:10],
            confidence=max(0.1, min(0.95, conf)),
        )
