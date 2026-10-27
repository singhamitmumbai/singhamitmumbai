"""Smart multi-signal router — pick the strongest specialist path (non-LLM)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RouteDecision:
    route: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartRouter:
    """Score candidate routes from lexical + structural cues."""

    ROUTES = (
        "vision_thermal",
        "vision_understand",
        "vision",
        "jarvis",
        "coding",
        "finderbay",
        "memory",
        "computer",
        "learning",
        "planning",
        "general",
    )

    def decide(self, text: str) -> RouteDecision:
        t = (text or "").strip()
        lower = t.lower()
        scores: dict[str, float] = {r: 0.0 for r in self.ROUTES}
        reasons: list[str] = []
        hints: dict[str, Any] = {}

        # --- vision / thermal ---
        if re.search(r"\b(thermal|thermography|flir|hotspot|infrared|\bir\b)\b", lower):
            scores["vision_thermal"] += 4.5
            reasons.append("thermal_cue")
        if re.search(
            r"\b(understand (the )?scene|understand (this |the )?image|what do you see|"
            r"describe (this |the )?(image|scene)|perceive image|identify objects)\b",
            lower,
        ):
            scores["vision_understand"] += 4.2
            reasons.append("understand_cue")
        if re.search(
            r"\b(detect|count|yolo|vision|image|photo|jpg|png|camera|night vision|x-?ray|depth)\b",
            lower,
        ):
            scores["vision"] += 2.8
            reasons.append("vision_cue")
        m_img = re.search(
            r"((?:/|\./)?[^\s\"']+\.(?:jpg|jpeg|png|bmp|webp|tif|tiff|dcm))",
            t,
            flags=re.I,
        )
        if m_img:
            hints["image_path"] = m_img.group(1)
            scores["vision"] += 1.5
            scores["vision_understand"] += 1.2
            if re.search(r"thermal|flir|infrared|night|xray|x-ray", m_img.group(1), re.I):
                scores["vision_thermal"] += 2.0
                scores["vision_understand"] += 1.5

        # --- jarvis workflows ---
        if re.search(
            r"\b(shortage|material balance|yaad dilana|remind me|stop this task|"
            r"cancel this job|jarvis)\b",
            lower,
        ):
            scores["jarvis"] += 4.0
            reasons.append("jarvis_workflow")
        if re.search(r"\b(schedule|tomorrow morning|kal subah)\b", lower):
            scores["jarvis"] += 1.5

        # --- finderbay domain ---
        if re.search(
            r"\b(invoice|invoices|payment|delivery|deliveries|purchase order|\bpo\b|"
            r"contract|vendor|warehouse|project [a-z0-9]|unpaid|overdue|reserve\b|"
            r"material|stock|alert)\b",
            lower,
        ):
            scores["finderbay"] += 3.5
            reasons.append("finderbay_domain")

        # --- coding ---
        if re.search(
            r"\b(generate function|write (a )?class|refactor|pytest|python code|"
            r"implement|bugfix|lint|ast |unit test)\b",
            lower,
        ):
            scores["coding"] += 4.0
            reasons.append("coding_cue")

        # --- memory / learning / computer / plan ---
        if re.search(r"\b(remember|recall|what do you remember)\b", lower):
            scores["memory"] += 4.0
            reasons.append("memory_cue")
        if re.search(r"\b(correction:|learn alias|is also called)\b", lower):
            scores["learning"] += 4.0
            reasons.append("learning_cue")
        if lower.startswith("computer ") or lower.startswith("control "):
            scores["computer"] += 5.0
            reasons.append("computer_cue")
        if lower.startswith("plan ") or lower.startswith("how would you "):
            scores["planning"] += 4.5
            reasons.append("planning_cue")

        # Mild prior for general chat questions
        if re.search(r"\b(who are you|what can you do|explain|why|how does)\b", lower):
            scores["general"] += 1.2

        # Pick best
        best_route = max(scores.items(), key=lambda kv: kv[1])
        route, conf_raw = best_route
        if conf_raw < 1.0:
            route = "general"
            conf = 0.35
            reasons.append("low_signal_default_general")
        else:
            # normalize roughly
            conf = min(0.95, 0.4 + conf_raw / 8.0)

        # Prefer more specific vision routes when close
        if route == "vision" and scores["vision_thermal"] >= scores["vision"] - 0.5 and scores["vision_thermal"] >= 3:
            route = "vision_thermal"
            reasons.append("prefer_thermal")
        if route == "vision" and scores["vision_understand"] >= scores["vision"] - 0.2 and scores["vision_understand"] >= 3:
            route = "vision_understand"
            reasons.append("prefer_understand")

        return RouteDecision(
            route=route,
            confidence=round(conf, 3),
            scores={k: round(v, 2) for k, v in scores.items() if v > 0},
            reasons=reasons,
            hints=hints,
        )
