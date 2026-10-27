"""Local object knowledge for vision understanding (non-LLM).

Known objects get structured facts. Unknown labels trigger search.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional


# Curated starter knowledge — COCO-ish + site/materials useful for Finderbay
OBJECT_KB: dict[str, dict[str, Any]] = {
    "person": {
        "category": "human",
        "aliases": ["people", "human", "worker", "man", "woman"],
        "attributes": ["pose", "clothing", "activity"],
        "uses": ["labor", "supervision", "safety_monitoring"],
        "safety": "PPE may be required on construction sites",
        "related": ["hardhat", "vest", "helmet"],
    },
    "car": {
        "category": "vehicle",
        "aliases": ["automobile", "sedan"],
        "attributes": ["color", "make", "wheels"],
        "uses": ["transport", "site_access"],
        "related": ["truck", "bus", "motorcycle"],
    },
    "truck": {
        "category": "vehicle",
        "aliases": ["lorry", "tipper"],
        "attributes": ["load", "wheels"],
        "uses": ["material_delivery", "haulage"],
        "related": ["car", "bus"],
        "site_note": "Often delivers cement, aggregate, steel",
    },
    "bus": {"category": "vehicle", "aliases": ["coach"], "uses": ["passenger_transport"], "related": ["car"]},
    "motorcycle": {"category": "vehicle", "aliases": ["bike", "motorbike"], "uses": ["transport"], "related": ["bicycle"]},
    "bicycle": {"category": "vehicle", "aliases": ["cycle"], "uses": ["transport"], "related": ["motorcycle"]},
    "bottle": {
        "category": "container",
        "aliases": ["flask"],
        "attributes": ["liquid", "volume"],
        "uses": ["storage", "drinking"],
        "related": ["cup"],
    },
    "cup": {"category": "container", "aliases": ["mug"], "uses": ["drinking"], "related": ["bottle"]},
    "chair": {"category": "furniture", "aliases": ["seat"], "uses": ["sitting"], "related": ["couch", "dining table"]},
    "couch": {"category": "furniture", "aliases": ["sofa"], "uses": ["sitting"], "related": ["chair"]},
    "bed": {"category": "furniture", "uses": ["sleeping"], "related": ["chair"]},
    "dining table": {"category": "furniture", "aliases": ["table"], "uses": ["work_surface"], "related": ["chair"]},
    "tv": {"category": "electronics", "aliases": ["television", "monitor"], "uses": ["display"], "related": ["laptop"]},
    "laptop": {
        "category": "electronics",
        "aliases": ["notebook", "computer"],
        "attributes": ["screen", "keyboard"],
        "uses": ["computing", "documentation"],
        "related": ["keyboard", "mouse", "cell phone"],
    },
    "cell phone": {
        "category": "electronics",
        "aliases": ["phone", "mobile", "smartphone"],
        "uses": ["communication", "camera"],
        "related": ["laptop"],
    },
    "keyboard": {"category": "electronics", "uses": ["input"], "related": ["laptop", "mouse"]},
    "mouse": {"category": "electronics", "uses": ["input"], "related": ["laptop", "keyboard"]},
    "remote": {"category": "electronics", "uses": ["control"], "related": ["tv"]},
    "book": {"category": "document", "aliases": ["manual", "ledger"], "uses": ["reference", "records"], "related": ["suitcase"]},
    "backpack": {"category": "container", "uses": ["carry"], "related": ["handbag", "suitcase"]},
    "handbag": {"category": "container", "uses": ["carry"], "related": ["backpack"]},
    "suitcase": {"category": "container", "uses": ["carry", "travel"], "related": ["backpack"]},
    "potted plant": {"category": "nature", "aliases": ["plant"], "uses": ["decoration"], "related": []},
    "cat": {"category": "animal", "uses": [], "related": ["dog"]},
    "dog": {"category": "animal", "uses": ["guarding"], "related": ["cat"]},
    "fire": {
        "category": "hazard",
        "aliases": ["flame", "blaze"],
        "attributes": ["smoke", "heat"],
        "uses": [],
        "safety": "ALERT — evacuate and extinguish; not a normal site object",
        "related": ["smoke"],
    },
    "smoke": {
        "category": "hazard",
        "aliases": ["fumes"],
        "safety": "ALERT — investigate fire/ventilation",
        "related": ["fire"],
    },
    "cement": {
        "category": "material",
        "aliases": ["cement_bags", "opc"],
        "attributes": ["bags", "weight"],
        "uses": ["construction", "concrete"],
        "site_note": "Track in inventory.balance / shortage reports",
        "related": ["steel", "sand", "aggregate"],
    },
    "steel": {
        "category": "material",
        "aliases": ["rebar", "steel_bars", "tmt"],
        "uses": ["reinforcement", "construction"],
        "site_note": "Common shortage item on civil sites",
        "related": ["cement"],
    },
    "hardhat": {
        "category": "ppe",
        "aliases": ["helmet", "safety helmet"],
        "uses": ["head_protection"],
        "safety": "Required PPE on many sites",
        "related": ["vest", "person"],
    },
    "vest": {
        "category": "ppe",
        "aliases": ["safety vest", "hi-vis"],
        "uses": ["visibility"],
        "related": ["hardhat", "person"],
    },
    "object": {
        "category": "unknown",
        "aliases": [],
        "uses": [],
        "note": "Generic detection — needs analysis and search",
    },
}


class ObjectKnowledge:
    """Lookup + learn object facts. Search fills gaps when understanding fails."""

    def __init__(self, seed: Optional[dict[str, dict[str, Any]]] = None) -> None:
        self.kb: dict[str, dict[str, Any]] = deepcopy(seed or OBJECT_KB)
        self.learned: dict[str, dict[str, Any]] = {}
        # alias → canonical
        self._aliases: dict[str, str] = {}
        self._rebuild_aliases()

    def _rebuild_aliases(self) -> None:
        self._aliases = {}
        for name, fact in {**self.kb, **self.learned}.items():
            self._aliases[name.lower()] = name
            for a in fact.get("aliases") or []:
                self._aliases[str(a).lower()] = name

    def canonicalize(self, label: str) -> str:
        key = (label or "").strip().lower().replace("_", " ")
        return self._aliases.get(key, key)

    def known(self, label: str) -> bool:
        canon = self.canonicalize(label)
        return canon in self.kb or canon in self.learned

    def lookup(self, label: str) -> Optional[dict[str, Any]]:
        canon = self.canonicalize(label)
        fact = self.learned.get(canon) or self.kb.get(canon)
        if not fact:
            return None
        out = deepcopy(fact)
        out["label"] = canon
        out["source"] = "learned" if canon in self.learned else "local_kb"
        out["understood"] = True
        return out

    def remember(self, label: str, fact: dict[str, Any], *, source: str = "search") -> dict[str, Any]:
        canon = self.canonicalize(label) if self.known(label) else (label or "").strip().lower()
        entry = deepcopy(fact)
        entry.setdefault("category", fact.get("category") or "learned")
        entry["learned_from"] = source
        self.learned[canon] = entry
        # also register aliases
        for a in entry.get("aliases") or []:
            self._aliases[str(a).lower()] = canon
        self._aliases[canon] = canon
        out = deepcopy(entry)
        out["label"] = canon
        out["source"] = source
        out["understood"] = True
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "seed_objects": len(self.kb),
            "learned_objects": len(self.learned),
            "total": len(self.kb) + len(self.learned),
        }
