"""Domain ontology: concepts, relationships, units and constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Concept:
    name: str
    parent: Optional[str] = None
    attributes: dict[str, str] = field(default_factory=dict)  # attr → type/unit
    constraints: list[str] = field(default_factory=list)
    relationships: list[tuple[str, str]] = field(default_factory=list)  # (rel, target)


class Ontology:
    """Typed concept graph for a domain package."""

    def __init__(self, domain: str = "generic") -> None:
        self.domain = domain
        self.concepts: dict[str, Concept] = {}
        self.aliases: dict[str, str] = {}  # alias → canonical concept/entity id

    def add_concept(self, concept: Concept) -> None:
        self.concepts[concept.name] = concept

    def add_alias(self, alias: str, canonical: str, *, source: str = "seed") -> None:
        del source
        self.aliases[alias.lower()] = canonical

    def resolve(self, name: str) -> Optional[str]:
        key = name.lower().strip()
        if key in self.concepts:
            return key
        return self.aliases.get(key)

    def related(self, concept: str) -> list[tuple[str, str]]:
        c = self.concepts.get(concept)
        return list(c.relationships) if c else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "concepts": {
                k: {
                    "parent": v.parent,
                    "attributes": v.attributes,
                    "constraints": v.constraints,
                    "relationships": v.relationships,
                }
                for k, v in self.concepts.items()
            },
            "aliases": dict(self.aliases),
        }


def finderbay_ontology() -> Ontology:
    """Business ontology for Finderbay: projects, invoices, deliveries, tasks."""
    onto = Ontology("finderbay")
    specs = [
        Concept(
            "project",
            attributes={"name": "string", "status": "string"},
            relationships=[("contains", "task"), ("owns", "invoice"), ("expects", "delivery")],
        ),
        Concept(
            "task",
            parent="project",
            attributes={"title": "string", "status": "string", "deadline": "datetime"},
            relationships=[("requires", "material")],
        ),
        Concept(
            "supplier",
            attributes={"name": "string"},
            relationships=[("provides", "item")],
        ),
        Concept(
            "item",
            attributes={"sku": "string", "unit": "string"},
        ),
        Concept(
            "delivery",
            attributes={
                "expected_qty": "number",
                "received_qty": "number",
                "status": "string",
                "week": "string",
            },
            relationships=[("satisfies", "order"), ("for_project", "project")],
            constraints=["received_qty >= 0", "status in pending|partial|complete|unknown"],
        ),
        Concept(
            "invoice",
            attributes={
                "amount_inr": "currency_inr",
                "payment_status": "string",
                "dispute_status": "string",
                "due_date": "date",
            },
            relationships=[("refers_to", "contract"), ("for_project", "project")],
            constraints=["amount_inr >= 0"],
        ),
        Concept(
            "payment",
            attributes={"amount_inr": "currency_inr", "paid_at": "datetime"},
            relationships=[("settles", "invoice")],
        ),
        Concept(
            "material",
            attributes={"available": "number", "reserved": "number", "unit": "string"},
        ),
        Concept(
            "warehouse",
            attributes={"name": "string"},
            relationships=[("stores", "material")],
        ),
    ]
    for c in specs:
        onto.add_concept(c)
    onto.add_alias("central store", "warehouse")
    onto.add_alias("bill", "invoice")
    onto.add_alias("shipment", "delivery")
    return onto
