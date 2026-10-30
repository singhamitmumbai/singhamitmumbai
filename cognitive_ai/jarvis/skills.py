"""Skills / workflows — procedures that combine tools (blueprint §1, §5D, §13)."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Optional

from .registry import JarvisToolRegistry


@dataclass
class SkillResult:
    ok: bool
    skill: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    verified: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "skill": self.skill,
            "steps": self.steps,
            "output": self.output,
            "missing": self.missing,
            "verified": self.verified,
            "message": self.message,
        }


class MaterialShortageSkill:
    """Worked example: prepare tomorrow's material shortage report (blueprint §13)."""

    name = "materials.shortage_report"
    description = "Resolve site/date, read demand+stock, calculate shortages, generate+verify report"

    def __init__(self, tools: JarvisToolRegistry) -> None:
        self.tools = tools

    def run(
        self,
        *,
        site: str,
        date: str,
        demand: list[dict[str, Any]],
        stock: list[dict[str, Any]],
        committed: Optional[list[dict[str, Any]]] = None,
        unusable: Optional[list[dict[str, Any]]] = None,
        expected_deliveries: Optional[list[dict[str, Any]]] = None,
    ) -> SkillResult:
        steps: list[dict[str, Any]] = []
        missing: list[str] = []
        if not site:
            missing.append("site")
        if not date:
            missing.append("date")
        if not demand:
            missing.append("demand")
        if missing:
            return SkillResult(ok=False, skill=self.name, missing=missing, message="needs_input")

        # Step 1: resolve project/site
        resolved = self.tools.invoke("projects.resolve", {"query": site})
        steps.append({"step": "resolve", "result": resolved})

        # Step 2–3: normalize + available stock
        # Available = on-hand - committed - unusable (blueprint formula)
        stock_map = {s["item"]: Decimal(str(s["qty"])) for s in stock}
        for c in committed or []:
            stock_map[c["item"]] = stock_map.get(c["item"], Decimal("0")) - Decimal(str(c["qty"]))
        for u in unusable or []:
            stock_map[u["item"]] = stock_map.get(u["item"], Decimal("0")) - Decimal(str(u["qty"]))
        available_list = [{"item": k, "qty": float(max(Decimal("0"), v))} for k, v in stock_map.items()]

        short = self.tools.invoke(
            "inventory.shortages",
            {"site": site, "date": date, "demand": demand, "stock": available_list},
        )
        steps.append({"step": "calculate", "result": short})

        # Step 4: generate report
        sections = {
            "site": site,
            "date": date,
            "project_id": resolved.get("project_id"),
            "shortages": short.get("shortages", []),
            "expected_deliveries": expected_deliveries or [],
            "source_note": "deterministic inventory calculation; deliveries shown separately",
        }
        report = self.tools.invoke(
            "documents.create_report",
            {"title": f"Material Shortage {site} {date}", "sections": sections},
        )
        steps.append({"step": "generate", "result": report})

        # Step 5: verify
        verify = self.tools.invoke(
            "documents.verify",
            {"artifact_id": report.get("artifact_id") or "unknown"},
        )
        steps.append({"step": "verify", "result": verify})

        # Recheck arithmetic with Decimal
        recalc_ok = True
        for row in short.get("shortages", []):
            expected = max(0.0, float(row["required"]) - float(row["available"]))
            if abs(expected - float(row["shortage"])) > 1e-9:
                recalc_ok = False
                break

        ok = bool(short.get("ok") and report.get("ok") and verify.get("verified") and recalc_ok)
        return SkillResult(
            ok=ok,
            skill=self.name,
            steps=steps,
            output={
                "report": report,
                "shortages": short.get("shortages", []),
                "total_shortage_units": short.get("total_shortage_units"),
                "expected_deliveries": expected_deliveries or [],
                "recalc_ok": recalc_ok,
            },
            verified=ok,
            message="shortage_report_verified" if ok else "shortage_report_incomplete",
        )


class SkillRegistry:
    def __init__(self, tools: JarvisToolRegistry) -> None:
        self.tools = tools
        self.skills: dict[str, Any] = {
            MaterialShortageSkill.name: MaterialShortageSkill(tools),
        }

    def list_skills(self) -> list[dict[str, str]]:
        return [{"name": k, "description": getattr(v, "description", "")} for k, v in self.skills.items()]

    def run(self, name: str, **kwargs: Any) -> SkillResult:
        skill = self.skills.get(name)
        if not skill:
            return SkillResult(ok=False, skill=name, message="unknown_skill")
        return skill.run(**kwargs)
