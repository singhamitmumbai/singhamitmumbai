"""World / knowledge memory: structured facts as triples + persistence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass(frozen=True)
class Triple:
    """A factual relation: subject --predicate--> object."""

    subject: str
    predicate: str
    object: str

    def matches(self, pattern: "Triple") -> bool:
        """Wildcard match where '*' matches any slot."""

        def slot(a: str, b: str) -> bool:
            return a == "*" or b == "*" or a == b

        return (
            slot(self.subject, pattern.subject)
            and slot(self.predicate, pattern.predicate)
            and slot(self.object, pattern.object)
        )

    def bind(self, pattern: "Triple") -> dict[str, str]:
        """Bind pattern variables (?x) against this grounded triple."""
        binding: dict[str, str] = {}
        for val, pat in (
            (self.subject, pattern.subject),
            (self.predicate, pattern.predicate),
            (self.object, pattern.object),
        ):
            if pat.startswith("?"):
                if pat in binding and binding[pat] != val:
                    return {}
                binding[pat] = val
            elif pat != "*" and pat != val:
                return {}
        return binding

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def __str__(self) -> str:
        return f"({self.subject} {self.predicate} {self.object})"


@dataclass
class Rule:
    """Production rule: if all antecedents match, assert consequents."""

    name: str
    antecedents: list[Triple]
    consequents: list[Triple]
    confidence: float = 1.0
    learned: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "antecedents": [asdict(t) for t in self.antecedents],
            "consequents": [asdict(t) for t in self.consequents],
            "confidence": self.confidence,
            "learned": self.learned,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Rule":
        return cls(
            name=data["name"],
            antecedents=[Triple(**t) for t in data["antecedents"]],
            consequents=[Triple(**t) for t in data["consequents"]],
            confidence=float(data.get("confidence", 1.0)),
            learned=bool(data.get("learned", False)),
        )


class KnowledgeBase:
    """Triple store with optional SQLite / JSON persistence."""

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self._facts: set[tuple[str, str, str]] = set()
        self._rules: dict[str, Rule] = {}
        self._path = Path(path) if path else None
        self._meta: dict[str, str] = {"domain": "empty"}
        if self._path and self._path.exists():
            self.load()

    # --- facts ----------------------------------------------------------------

    def assert_fact(self, subject: str, predicate: str, obj: str) -> bool:
        """Add a fact. Returns True if newly added."""
        key = (subject, predicate, obj)
        if key in self._facts:
            return False
        self._facts.add(key)
        return True

    def retract_fact(self, subject: str, predicate: str, obj: str) -> bool:
        key = (subject, predicate, obj)
        if key not in self._facts:
            return False
        self._facts.remove(key)
        return True

    def has_fact(self, subject: str, predicate: str, obj: str) -> bool:
        return (subject, predicate, obj) in self._facts

    def facts(self) -> list[Triple]:
        return [Triple(*t) for t in sorted(self._facts)]

    def query(
        self,
        subject: str = "*",
        predicate: str = "*",
        obj: str = "*",
    ) -> list[Triple]:
        pattern = Triple(subject, predicate, obj)
        return [t for t in self.facts() if t.matches(pattern)]

    def query_bindings(self, pattern: Triple) -> list[dict[str, str]]:
        """Return variable bindings for pattern against all facts."""
        results: list[dict[str, str]] = []
        for fact in self.facts():
            binding = fact.bind(pattern)
            if binding or self._fully_grounded_match(fact, pattern):
                if binding or (
                    pattern.subject == fact.subject
                    and pattern.predicate == fact.predicate
                    and pattern.object == fact.object
                ):
                    results.append(binding)
                elif not any(p.startswith("?") for p in pattern.as_tuple()):
                    if fact.matches(pattern):
                        results.append({})
        # Deduplicate while preserving order
        seen: set[tuple[tuple[str, str], ...]] = set()
        unique: list[dict[str, str]] = []
        for b in results:
            key = tuple(sorted(b.items()))
            if key not in seen:
                seen.add(key)
                unique.append(b)
        return unique

    @staticmethod
    def _fully_grounded_match(fact: Triple, pattern: Triple) -> bool:
        for f, p in zip(fact.as_tuple(), pattern.as_tuple()):
            if p.startswith("?") or p == "*":
                continue
            if f != p:
                return False
        return True

    def related(self, entity: str) -> list[Triple]:
        out: list[Triple] = []
        for t in self.facts():
            if t.subject == entity or t.object == entity:
                out.append(t)
        return out

    # --- rules ----------------------------------------------------------------

    def add_rule(self, rule: Rule) -> None:
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> bool:
        if name not in self._rules:
            return False
        del self._rules[name]
        return True

    def rules(self) -> list[Rule]:
        return list(self._rules.values())

    def get_rule(self, name: str) -> Optional[Rule]:
        return self._rules.get(name)

    # --- meta / persistence ---------------------------------------------------

    def set_domain(self, name: str) -> None:
        self._meta["domain"] = name

    @property
    def domain(self) -> str:
        return self._meta.get("domain", "unknown")

    def clear(self) -> None:
        self._facts.clear()
        self._rules.clear()

    def save(self, path: Optional[Path | str] = None) -> Path:
        target = Path(path) if path else self._path
        if target is None:
            raise ValueError("No path configured for KnowledgeBase.save()")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._path = target

        if target.suffix.lower() == ".json":
            return self._save_json(target)
        return self._save_sqlite(target)

    def load(self, path: Optional[Path | str] = None) -> None:
        source = Path(path) if path else self._path
        if source is None or not source.exists():
            return
        self._path = source
        if source.suffix.lower() == ".json":
            self._load_json(source)
        else:
            self._load_sqlite(source)

    def _save_json(self, path: Path) -> Path:
        payload = {
            "meta": self._meta,
            "facts": [list(t) for t in sorted(self._facts)],
            "rules": [r.to_dict() for r in self.rules()],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _load_json(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.clear()
        self._meta = dict(data.get("meta", {}))
        for s, p, o in data.get("facts", []):
            self._facts.add((s, p, o))
        for rd in data.get("rules", []):
            rule = Rule.from_dict(rd)
            self._rules[rule.name] = rule

    def _save_sqlite(self, path: Path) -> Path:
        if path.exists():
            path.unlink()
        conn = sqlite3.connect(path)
        try:
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            cur.execute(
                "CREATE TABLE facts ("
                "subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,"
                "PRIMARY KEY (subject, predicate, object))"
            )
            cur.execute(
                "CREATE TABLE rules (name TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            for k, v in self._meta.items():
                cur.execute("INSERT INTO meta VALUES (?, ?)", (k, v))
            cur.executemany(
                "INSERT INTO facts VALUES (?, ?, ?)", sorted(self._facts)
            )
            for rule in self.rules():
                cur.execute(
                    "INSERT INTO rules VALUES (?, ?)",
                    (rule.name, json.dumps(rule.to_dict())),
                )
            conn.commit()
        finally:
            conn.close()
        return path

    def _load_sqlite(self, path: Path) -> None:
        conn = sqlite3.connect(path)
        try:
            cur = conn.cursor()
            self.clear()
            for key, value in cur.execute("SELECT key, value FROM meta"):
                self._meta[key] = value
            for row in cur.execute("SELECT subject, predicate, object FROM facts"):
                self._facts.add(tuple(row))  # type: ignore[arg-type]
            for name, payload in cur.execute("SELECT name, payload FROM rules"):
                rule = Rule.from_dict(json.loads(payload))
                self._rules[name] = rule
        finally:
            conn.close()

    def summary(self) -> dict:
        return {
            "domain": self.domain,
            "fact_count": len(self._facts),
            "rule_count": len(self._rules),
            "path": str(self._path) if self._path else None,
        }

    def __iter__(self) -> Iterator[Triple]:
        return iter(self.facts())

    def __len__(self) -> int:
        return len(self._facts)

    def extend_facts(self, triples: Iterable[Triple | tuple[str, str, str]]) -> int:
        added = 0
        for t in triples:
            if isinstance(t, Triple):
                s, p, o = t.as_tuple()
            else:
                s, p, o = t
            if self.assert_fact(s, p, o):
                added += 1
        return added
