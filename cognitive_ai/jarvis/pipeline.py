"""JARVIS request pipeline: Capture→Interpret→Resolve→Plan→Execute→Verify→Respond→Record."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .contracts import JobState
from .jobs import JobStore
from .nlu import IntentClassifier
from .provenance import ProvenanceMemory
from .registry import JarvisToolRegistry
from .skills import SkillRegistry, SkillResult


@dataclass
class PipelineResult:
    ok: bool
    stage: str
    intent: str = ""
    slots: dict[str, Any] = field(default_factory=dict)
    job_id: Optional[str] = None
    plan: list[str] = field(default_factory=list)
    skill_result: Optional[dict[str, Any]] = None
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    response: str = ""
    missing: list[str] = field(default_factory=list)
    verified: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "intent": self.intent,
            "slots": self.slots,
            "job_id": self.job_id,
            "plan": self.plan,
            "skill_result": self.skill_result,
            "tool_results": self.tool_results,
            "response": self.response,
            "missing": self.missing,
            "verified": self.verified,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


# Intent patterns — defined command set (blueprint §5A)
_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("inventory.shortages", re.compile(r"(material|shortage|stock).*(report|shortage|balance)|shortage report|material balance", re.I)),
    ("files.search", re.compile(r"(find|search|locate).*(file|report|document)|latest.*report", re.I)),
    ("documents.extract", re.compile(r"(read|extract|parse).*(invoice|document|pdf)", re.I)),
    ("schedule.create", re.compile(r"(remind|schedule|yaad dilana|kal\s)", re.I)),
    ("jobs.cancel", re.compile(r"\b(stop|cancel|abort)\b.*(task|job|this)?", re.I)),
    ("memory.query", re.compile(r"(what.*(remember|know)|recall|memory)", re.I)),
    ("math.calculate", re.compile(r"(calculate|compute|how much is|\d+\s*[\+\-\*/])", re.I)),
    ("system.health", re.compile(r"(health|status|are you (ok|ready))", re.I)),
]


class JarvisPipeline:
    """Non-LLM controller implementing the blueprint control loop."""

    def __init__(
        self,
        tools: Optional[JarvisToolRegistry] = None,
        jobs: Optional[JobStore] = None,
        memory: Optional[ProvenanceMemory] = None,
        skills: Optional[SkillRegistry] = None,
    ) -> None:
        self.tools = tools or JarvisToolRegistry()
        self.jobs = jobs or JobStore()
        self.memory = memory or ProvenanceMemory()
        self.skills = skills or SkillRegistry(self.tools)
        self.classifier = IntentClassifier()
        # Seed procedural memory
        self.memory.register_procedure(
            "materials.shortage_report",
            [
                "resolve site/date",
                "read demand and stock",
                "normalize units",
                "calculate shortages",
                "generate report",
                "verify output",
            ],
        )

    def capture(self, text: str, *, source: str = "text", meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return {
            "raw_text": text.strip(),
            "source": source,
            "meta": meta or {},
        }

    def interpret(self, captured: dict[str, Any]) -> tuple[str, dict[str, Any], list[str]]:
        """Hybrid interpret: regex cues + TF-IDF classifier; keep higher-confidence intent."""
        text = captured["raw_text"]
        regex_intent = "unsupported"
        for name, pat in _INTENT_PATTERNS:
            if pat.search(text):
                regex_intent = name
                break
        clf = self.classifier.predict(text, threshold=0.20)
        # Prefer classifier when confident; otherwise regex; merge slots
        if regex_intent == "jobs.cancel":
            intent = "jobs.cancel"  # safety-critical
        elif regex_intent != "unsupported":
            # Regex cue present: allow classifier override only if clearly stronger same-family
            if clf.supported and clf.confidence >= 0.28 and clf.intent == regex_intent:
                intent = clf.intent
            else:
                intent = regex_intent
        elif clf.supported and clf.confidence >= 0.30:
            # No regex hit — require stronger classifier confidence to avoid false accepts
            intent = clf.intent
        else:
            intent = "unsupported"

        slots = self._extract_slots(text, intent)
        # Merge classifier slots when richer
        if intent != "unsupported":
            for k, v in (clf.slots or {}).items():
                if k not in slots or slots.get(k) in (None, ""):
                    slots[k] = v
        missing = list(slots.pop("_missing", []))
        if intent != "unsupported":
            for m in clf.missing or []:
                if m not in missing:
                    missing.append(m)
        if intent == "unsupported":
            slots["_intent_confidence"] = float(clf.confidence or 0.0)
        elif clf.supported and clf.intent == intent:
            slots["_intent_confidence"] = float(clf.confidence)
        else:
            slots["_intent_confidence"] = 0.7 if regex_intent != "unsupported" else float(clf.confidence or 0.5)
        slots["_interpret_method"] = "hybrid"
        return intent, slots, missing

    def resolve(self, intent: str, slots: dict[str, Any]) -> dict[str, Any]:
        """Resolve project/file/date against working memory and tools."""
        resolved = dict(slots)
        if not resolved.get("site") and self.memory.working.get("active_project"):
            resolved["site"] = self.memory.working["active_project"]
        if resolved.get("site"):
            proj = self.tools.invoke("projects.resolve", {"query": str(resolved["site"])})
            resolved["project_id"] = proj.get("project_id")
            self.memory.set_working(active_project=resolved["site"])
        if resolved.get("relative_date") == "tomorrow" and not resolved.get("date"):
            from datetime import datetime, timedelta, timezone

            resolved["date"] = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
        return resolved

    def plan(self, intent: str, slots: dict[str, Any]) -> list[str]:
        if intent == "inventory.shortages":
            return [
                "projects.resolve",
                "inventory.shortages",
                "documents.create_report",
                "documents.verify",
            ]
        if intent in self.tools.contracts:
            return [intent]
        return []

    def execute(
        self,
        intent: str,
        slots: dict[str, Any],
        plan: list[str],
        *,
        job_id: str,
        demand: Optional[list] = None,
        stock: Optional[list] = None,
    ) -> tuple[list[dict[str, Any]], Optional[SkillResult]]:
        self.jobs.transition(job_id, JobState.RUNNING)
        results: list[dict[str, Any]] = []
        skill_result: Optional[SkillResult] = None

        if intent == "inventory.shortages":
            skill_result = self.skills.run(
                "materials.shortage_report",
                site=slots.get("site") or slots.get("project") or "default_site",
                date=slots.get("date") or "tomorrow",
                demand=demand
                or slots.get("demand")
                or [
                    {"item": "cement_bags", "qty": 100},
                    {"item": "steel_bars", "qty": 50},
                ],
                stock=stock
                or slots.get("stock")
                or [
                    {"item": "cement_bags", "qty": 70},
                    {"item": "steel_bars", "qty": 50},
                ],
                committed=slots.get("committed"),
                unusable=slots.get("unusable"),
                expected_deliveries=slots.get("expected_deliveries"),
            )
            for s in skill_result.steps:
                self.jobs.add_step(job_id, s)
            results.append(skill_result.to_dict())
            return results, skill_result

        for tool_name in plan:
            args = self._args_for_tool(tool_name, slots)
            res = self.tools.invoke(tool_name, args)
            results.append(res)
            self.jobs.add_result(job_id, res)
            if not res.get("ok") and res.get("needs_confirm"):
                self.jobs.transition(job_id, JobState.NEEDS_INPUT)
                break
            if not res.get("ok"):
                self.jobs.transition(job_id, JobState.FAILED, error=str(res.get("error")))
                break
        return results, skill_result

    def verify(self, results: list[dict[str, Any]], skill_result: Optional[SkillResult]) -> tuple[bool, dict[str, Any]]:
        if skill_result is not None:
            return skill_result.verified, {"skill": skill_result.skill, "output": skill_result.output}
        if not results:
            return False, {"reason": "no_results"}
        last = results[-1]
        return bool(last.get("ok") and last.get("verified", last.get("ok"))), {"last": last}

    def respond(self, intent: str, verified: bool, evidence: dict[str, Any], missing: list[str]) -> str:
        if missing:
            return f"I need more information: {', '.join(missing)}."
        if intent == "inventory.shortages":
            out = evidence.get("output") or {}
            shortages = out.get("shortages") or []
            n = sum(1 for s in shortages if s.get("shortage", 0) > 0)
            if verified:
                return (
                    f"Material shortage report verified. "
                    f"{n} item(s) short; total shortage units={out.get('total_shortage_units')}. "
                    f"Draft wording: report generated and reopened for checks."
                )
            return "Shortage workflow ran but verification did not pass."
        if intent == "unsupported":
            return (
                "That request is outside the supported command set. "
                "Try material shortage report, file search, schedule, or system health."
            )
        if verified:
            return f"Completed and verified: {intent}."
        return f"Finished {intent}, but outcome not fully verified."

    def record(self, intent: str, slots: dict[str, Any], evidence: dict[str, Any], verified: bool) -> None:
        self.memory.record_episode(
            f"intent={intent} verified={verified}",
            evidence={"slots": slots, "evidence_keys": list(evidence.keys())},
        )
        if slots.get("site"):
            self.memory.record_fact(
                slots["site"],
                "last_intent",
                intent,
                verification_status="observed",
                source_id="pipeline",
                kind="episodic",
            )

    def run(
        self,
        text: str,
        *,
        source: str = "text",
        attempt_id: Optional[str] = None,
        demand: Optional[list] = None,
        stock: Optional[list] = None,
        cancel: bool = False,
    ) -> PipelineResult:
        captured = self.capture(text, source=source)
        intent, slots, missing = self.interpret(captured)
        job = self.jobs.create(captured["raw_text"], intent=intent, attempt_id=attempt_id)

        if cancel or intent == "jobs.cancel":
            self.jobs.request_cancel(job.job_id)
            # stop scheduling further steps; mark cancelled if not completed
            if job.state == JobState.CANCEL_REQUESTED.value:
                self.jobs.transition(job.job_id, JobState.CANCELLED)
            return PipelineResult(
                ok=True,
                stage="cancelled",
                intent=intent,
                job_id=job.job_id,
                response="Stop acknowledged: no further steps scheduled; speech queue would be cleared.",
                verified=True,
            )

        if missing and intent != "unsupported":
            self.jobs.transition(job.job_id, JobState.NEEDS_INPUT)
            job.missing_fields = missing
            return PipelineResult(
                ok=False,
                stage="resolve",
                intent=intent,
                slots=slots,
                job_id=job.job_id,
                missing=missing,
                response=self.respond(intent, False, {}, missing),
            )

        slots = self.resolve(intent, slots)
        plan = self.plan(intent, slots)
        if intent == "unsupported" or not plan:
            self.jobs.transition(job.job_id, JobState.FAILED, error="unsupported")
            return PipelineResult(
                ok=False,
                stage="plan",
                intent=intent,
                slots=slots,
                job_id=job.job_id,
                response=self.respond("unsupported", False, {}, []),
            )

        results, skill_result = self.execute(
            intent, slots, plan, job_id=job.job_id, demand=demand, stock=stock
        )
        self.jobs.transition(job.job_id, JobState.VERIFYING)
        verified, evidence = self.verify(results, skill_result)
        if verified:
            self.jobs.transition(job.job_id, JobState.COMPLETED)
        elif any(r.get("ok") for r in results):
            self.jobs.transition(job.job_id, JobState.PARTIAL)
        else:
            self.jobs.transition(job.job_id, JobState.FAILED, error="verify_failed")

        response = self.respond(intent, verified, evidence, [])
        self.record(intent, slots, evidence, verified)
        try:
            from ..intelligence import score_response

            conf = score_response(
                verified=verified,
                supported=intent != "unsupported",
                intent_confidence=float(slots.get("_intent_confidence") or 0.5),
                missing_fields=[],
            ).to_dict()
        except Exception:  # noqa: BLE001
            conf = None
        return PipelineResult(
            ok=verified,
            stage="respond",
            intent=intent,
            slots=slots,
            job_id=job.job_id,
            plan=plan,
            skill_result=skill_result.to_dict() if skill_result else None,
            tool_results=results,
            response=response,
            verified=verified,
            evidence=evidence,
            confidence=conf,
        )

    def _extract_slots(self, text: str, intent: str) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        missing: list[str] = []
        # relative date
        if re.search(r"\btomorrow\b|kal\b", text, re.I):
            slots["relative_date"] = "tomorrow"
        # time like 10 baje / 10:00
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(baje|am|pm)?\b", text, re.I)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2) or 0)
            ampm = (m.group(3) or "").lower()
            if ampm == "pm" and hour < 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
            if ampm == "baje" and hour <= 12 and "subah" in text.lower():
                pass  # morning
            if ampm in ("", "baje") and hour <= 12 and "subah" not in text.lower() and "am" not in text.lower() and "pm" not in text.lower():
                slots["_ampm_ambiguous"] = True
            slots["local_time"] = f"{hour:02d}:{minute:02d}"
            slots["timezone"] = "Asia/Kolkata"
        # site / project
        m = re.search(r"(?:for|site|project)\s+([A-Za-z0-9_\- ]+?)(?:\s*$|\s+report|\s+tomorrow)", text, re.I)
        if m:
            slots["site"] = m.group(1).strip()
        elif self.memory.working.get("active_project"):
            slots["site"] = self.memory.working["active_project"]
        elif intent == "inventory.shortages":
            # allow selected site phrasing
            if re.search(r"selected site|this site", text, re.I):
                slots["site"] = self.memory.working.get("active_project") or "selected_site"
            else:
                missing.append("site")
        # schedule task phrase
        if intent == "schedule.create":
            m = re.search(r"(?:remind me(?: to)?|yaad dilana)\s+(.+?)(?:\s+kal|\s+at|\s+\d|$)", text, re.I)
            if m:
                slots["task"] = m.group(1).strip(" .,")
            else:
                # Hinglish: "tender review yaad dilana"
                m2 = re.search(r"(.+?)\s+yaad dilana", text, re.I)
                slots["task"] = m2.group(1).strip() if m2 else "reminder"
            when_parts = []
            if slots.get("relative_date"):
                when_parts.append(slots["relative_date"])
            if slots.get("local_time"):
                when_parts.append(slots["local_time"])
            slots["when"] = " ".join(when_parts) or "unspecified"
        # math expression
        if intent == "math.calculate":
            m = re.search(r"([\d\.\s\+\-\*/\(\)]+)", text)
            if m:
                slots["expression"] = m.group(1).strip()
            else:
                missing.append("expression")
        if intent == "files.search":
            slots["query"] = text
        if intent == "documents.extract":
            slots["path"] = "document.pdf"
        if intent == "jobs.cancel":
            slots["job_id"] = "current"
        if intent == "memory.query":
            slots["query"] = text
        slots["_missing"] = missing
        return slots

    def _args_for_tool(self, tool_name: str, slots: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "schedule.create":
            return {"task": slots.get("task", "task"), "when": slots.get("when", "unspecified")}
        if tool_name == "files.search":
            return {"query": slots.get("query", "")}
        if tool_name == "documents.extract":
            return {"path": slots.get("path", "")}
        if tool_name == "math.calculate":
            return {"expression": slots.get("expression", "0")}
        if tool_name == "memory.query":
            return {"query": slots.get("query", "")}
        if tool_name == "jobs.cancel":
            return {"job_id": slots.get("job_id", "current")}
        if tool_name == "system.health":
            return {}
        if tool_name == "projects.resolve":
            return {"query": slots.get("site") or "default"}
        return dict(slots)
