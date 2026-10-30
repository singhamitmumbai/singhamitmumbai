"""Search for unknown objects when vision does not yet understand them.

Order (blueprint-style): local KB variants → document/memory FTS → curated
encyclopedia snippets → optional web fetch (offline-safe stub by default).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .knowledge import ObjectKnowledge


# Compact encyclopedia for labels YOLO may emit that are thin in seed KB
ENCYCLOPEDIA: dict[str, dict[str, Any]] = {
    "traffic light": {
        "category": "signal",
        "summary": "Road signalling device with red/amber/green lamps controlling traffic flow.",
        "uses": ["traffic_control"],
        "attributes": ["color_state"],
    },
    "stop sign": {
        "category": "sign",
        "summary": "Regulatory road sign requiring vehicles to stop.",
        "uses": ["traffic_control"],
    },
    "bench": {
        "category": "furniture",
        "summary": "Long seat for several people, often outdoor.",
        "uses": ["sitting"],
    },
    "umbrella": {
        "category": "accessory",
        "summary": "Portable canopy for rain or sun protection.",
        "uses": ["weather_protection"],
    },
    "handbag": {
        "category": "container",
        "summary": "Bag carried by hand for personal items.",
        "uses": ["carry"],
    },
    "tie": {"category": "clothing", "summary": "Neckwear accessory.", "uses": ["clothing"]},
    "suitcase": {
        "category": "container",
        "summary": "Travel case for clothes and belongings.",
        "uses": ["travel", "carry"],
    },
    "frisbee": {"category": "sport", "summary": "Flying disc used in recreation.", "uses": ["recreation"]},
    "skis": {"category": "sport", "summary": "Long runners for snow travel.", "uses": ["recreation"]},
    "sports ball": {"category": "sport", "summary": "Ball used in sports.", "uses": ["recreation"]},
    "kite": {"category": "sport", "summary": "Tethered flying toy.", "uses": ["recreation"]},
    "baseball bat": {"category": "sport", "summary": "Bat for baseball.", "uses": ["recreation"]},
    "baseball glove": {"category": "sport", "summary": "Glove for baseball fielding.", "uses": ["recreation"]},
    "skateboard": {"category": "sport", "summary": "Board with wheels for riding.", "uses": ["recreation"]},
    "surfboard": {"category": "sport", "summary": "Board for surfing waves.", "uses": ["recreation"]},
    "tennis racket": {"category": "sport", "summary": "Racket for tennis.", "uses": ["recreation"]},
    "wine glass": {"category": "container", "summary": "Stemmed glass for wine.", "uses": ["drinking"]},
    "fork": {"category": "utensil", "summary": "Eating utensil with tines.", "uses": ["eating"]},
    "knife": {"category": "utensil", "summary": "Cutting tool or utensil.", "uses": ["cutting", "eating"], "safety": "Sharp — handle carefully"},
    "spoon": {"category": "utensil", "summary": "Eating/serving utensil.", "uses": ["eating"]},
    "bowl": {"category": "container", "summary": "Round deep dish.", "uses": ["food"]},
    "banana": {"category": "food", "summary": "Tropical fruit.", "uses": ["food"]},
    "apple": {"category": "food", "summary": "Common fruit.", "uses": ["food"]},
    "sandwich": {"category": "food", "summary": "Food between bread slices.", "uses": ["food"]},
    "orange": {"category": "food", "summary": "Citrus fruit.", "uses": ["food"]},
    "broccoli": {"category": "food", "summary": "Green vegetable.", "uses": ["food"]},
    "carrot": {"category": "food", "summary": "Root vegetable.", "uses": ["food"]},
    "hot dog": {"category": "food", "summary": "Sausage in a bun.", "uses": ["food"]},
    "pizza": {"category": "food", "summary": "Baked flatbread with toppings.", "uses": ["food"]},
    "donut": {"category": "food", "summary": "Fried sweet dough.", "uses": ["food"]},
    "cake": {"category": "food", "summary": "Baked sweet dessert.", "uses": ["food"]},
    "clock": {"category": "device", "summary": "Timekeeping device.", "uses": ["time"]},
    "vase": {"category": "container", "summary": "Vessel for flowers.", "uses": ["decoration"]},
    "scissors": {"category": "tool", "summary": "Cutting tool with two blades.", "uses": ["cutting"], "safety": "Sharp"},
    "teddy bear": {"category": "toy", "summary": "Stuffed toy bear.", "uses": ["recreation"]},
    "hair drier": {"category": "appliance", "summary": "Device that blows hot air to dry hair.", "uses": ["grooming"]},
    "toothbrush": {"category": "hygiene", "summary": "Brush for cleaning teeth.", "uses": ["hygiene"]},
    "refrigerator": {"category": "appliance", "summary": "Cooling cabinet for food storage.", "uses": ["storage"]},
    "oven": {"category": "appliance", "summary": "Heated chamber for cooking.", "uses": ["cooking"]},
    "toaster": {"category": "appliance", "summary": "Device for toasting bread.", "uses": ["cooking"]},
    "sink": {"category": "fixture", "summary": "Basin with water supply for washing.", "uses": ["washing"]},
    "toilet": {"category": "fixture", "summary": "Sanitation fixture.", "uses": ["sanitation"]},
    "microwave": {"category": "appliance", "summary": "Oven using microwave radiation.", "uses": ["cooking"]},
    "rebar": {
        "category": "material",
        "summary": "Reinforcing steel bar used in concrete structures.",
        "aliases": ["steel_bars", "tmt"],
        "uses": ["reinforcement", "construction"],
        "site_note": "Track lengths/diameters in material reports",
    },
    "helmet": {
        "category": "ppe",
        "summary": "Protective hard head covering.",
        "aliases": ["hardhat"],
        "uses": ["head_protection"],
        "safety": "Required PPE on many sites",
    },
    "thermal hotspot": {
        "category": "thermal_anomaly",
        "summary": "Region of elevated heat on a thermal/IR image; may indicate overload or friction.",
        "aliases": ["hotspot", "hot region"],
        "uses": ["inspection", "predictive_maintenance"],
        "safety": "Inspect equipment before continuing operation",
    },
    "flir": {
        "category": "sensor",
        "summary": "Thermal imaging camera brand/type commonly used for heat inspection.",
        "uses": ["thermography"],
    },
    "night vision": {
        "category": "sensor",
        "summary": "Low-light imaging modality amplifying residual light.",
        "uses": ["security", "navigation"],
    },

    "crane": {
        "category": "equipment",
        "summary": "Lifting machine for heavy loads on construction sites.",
        "uses": ["lifting", "construction"],
        "safety": "Exclusion zones and certified operators required",
    },
    "excavat": {  # stem match via startswith helpers
        "category": "equipment",
        "summary": "Earth-moving construction machine.",
        "uses": ["excavation"],
        "safety": "Keep clear of swing radius",
    },
}


class ObjectSearch:
    def __init__(self, knowledge: Optional[ObjectKnowledge] = None) -> None:
        self.knowledge = knowledge or ObjectKnowledge()
        self.history: list[dict[str, Any]] = []

    def search(self, label: str, *, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Find explanations for an unrecognized or poorly understood object."""
        query = (label or "").strip().lower().replace("_", " ")
        steps: list[dict[str, Any]] = []

        # 1) Direct / alias KB
        hit = self.knowledge.lookup(query)
        if hit:
            steps.append({"source": "local_kb", "ok": True})
            result = {
                "ok": True,
                "query": query,
                "found": True,
                "fact": hit,
                "steps": steps,
                "method": "local_kb",
            }
            self.history.append(result)
            return result

        # 2) Encyclopedia exact / fuzzy
        enc = self._encyclopedia_lookup(query)
        if enc:
            steps.append({"source": "encyclopedia", "ok": True})
            remembered = self.knowledge.remember(query, enc, source="encyclopedia")
            result = {
                "ok": True,
                "query": query,
                "found": True,
                "fact": remembered,
                "steps": steps,
                "method": "encyclopedia",
                "summary": enc.get("summary"),
            }
            self.history.append(result)
            return result

        # 3) Morphological / partial match in encyclopedia + KB
        partial = self._partial_match(query)
        if partial:
            steps.append({"source": "partial_match", "ok": True, "matched": partial["label"]})
            remembered = self.knowledge.remember(query, partial, source="partial_match")
            result = {
                "ok": True,
                "query": query,
                "found": True,
                "fact": remembered,
                "steps": steps,
                "method": "partial_match",
            }
            self.history.append(result)
            return result

        # 4) Context hints (filename, OCR text, site materials)
        ctx_fact = self._from_context(query, context or {})
        if ctx_fact:
            steps.append({"source": "context", "ok": True})
            remembered = self.knowledge.remember(query, ctx_fact, source="context")
            result = {
                "ok": True,
                "query": query,
                "found": True,
                "fact": remembered,
                "steps": steps,
                "method": "context",
            }
            self.history.append(result)
            return result

        # 5) Synthetic structured hypothesis (still honest about uncertainty)
        hypothesis = {
            "category": "unknown",
            "summary": f"Unrecognized object labeled '{query}'. Needs human confirmation or richer model.",
            "uses": [],
            "attributes": list((context or {}).get("observed_attributes") or []),
            "confidence": 0.2,
            "needs_review": True,
        }
        steps.append({"source": "hypothesis", "ok": False})
        # Do not auto-activate low-confidence hypotheses into KB as understood
        result = {
            "ok": True,
            "query": query,
            "found": False,
            "fact": {**hypothesis, "label": query, "understood": False, "source": "hypothesis"},
            "steps": steps,
            "method": "hypothesis",
            "message": "Object not in knowledge base; recorded as unresolved for review",
        }
        self.history.append(result)
        return result

    def _encyclopedia_lookup(self, query: str) -> Optional[dict[str, Any]]:
        if query in ENCYCLOPEDIA:
            return dict(ENCYCLOPEDIA[query])
        # alias scan
        for key, val in ENCYCLOPEDIA.items():
            aliases = [key] + list(val.get("aliases") or [])
            if query in {a.lower() for a in aliases}:
                return dict(val) | {"matched_key": key}
        return None

    def _partial_match(self, query: str) -> Optional[dict[str, Any]]:
        tokens = set(re.findall(r"[a-z0-9]+", query))
        best = None
        best_score = 0
        pool = {**{k: v for k, v in ENCYCLOPEDIA.items()}, **self.knowledge.kb}
        for key, val in pool.items():
            key_tokens = set(re.findall(r"[a-z0-9]+", key))
            for a in val.get("aliases") or []:
                key_tokens |= set(re.findall(r"[a-z0-9]+", str(a).lower()))
            score = len(tokens & key_tokens)
            if query.startswith(key[:4]) or key.startswith(query[:4]):
                score += 1
            if score > best_score:
                best_score = score
                best = {"label": key, **dict(val)}
        if best and best_score > 0:
            best["summary"] = best.get("summary") or f"Matched related concept '{best['label']}'."
            return best
        return None

    def _from_context(self, query: str, context: dict[str, Any]) -> Optional[dict[str, Any]]:
        blob = " ".join(
            str(x)
            for x in (
                context.get("image_path"),
                context.get("ocr_text"),
                context.get("filename"),
                " ".join(context.get("hints") or []),
            )
            if x
        ).lower()
        if not blob:
            return None
        # material / site hints
        for mat in ("cement", "steel", "rebar", "helmet", "hardhat", "crane"):
            if mat in blob or mat in query:
                enc = self._encyclopedia_lookup(mat) or self.knowledge.lookup(mat)
                if enc:
                    return dict(enc) if "summary" in enc or "category" in enc else dict(enc)
                if mat in ENCYCLOPEDIA:
                    return dict(ENCYCLOPEDIA[mat])
                hit = self.knowledge.lookup(mat)
                if hit:
                    return {
                        "category": hit.get("category"),
                        "summary": f"Identified via context as {mat}.",
                        "uses": hit.get("uses") or [],
                        "site_note": hit.get("site_note"),
                        "aliases": hit.get("aliases") or [],
                    }
        return None
