"""Curated common-sense knowledge base — large, explicit, non-LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..knowledge import KnowledgeBase, Rule, Triple
from .generate_kb import materialize


class CommonSenseKB:
    """Load curated common-sense into a symbolic knowledge base."""

    def __init__(self, path: Optional[Path | str] = None, *, large: bool = True) -> None:
        self.kb = KnowledgeBase(path)
        self.kb.set_domain("commonsense")
        self.large = large
        if len(self.kb) == 0:
            self.seed()

    def seed(self) -> None:
        facts, rules = materialize()
        for s, p, o in facts:
            self.kb.assert_fact(s, p, o)
        for rule in rules:
            self.kb.add_rule(rule)
        # Keep a few high-value meta facts
        self.kb.assert_fact("commonsense_kb", "fact_count", str(len(facts)))
        self.kb.assert_fact("commonsense_kb", "rule_count", str(len(rules)))

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
        return {
            "facts": len(self.kb),
            "rules": len(self.kb.rules()),
            "domain": self.kb.domain,
            "large": self.large,
        }
