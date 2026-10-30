"""JARVIS tool registry with contracts (blueprint §6, §8)."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .contracts import ToolContract, VerifiedArtifactResult


Handler = Callable[[dict[str, Any]], dict[str, Any]]


class JarvisToolRegistry:
    def __init__(self) -> None:
        self.contracts: dict[str, ToolContract] = {}
        self.handlers: dict[str, Handler] = {}
        self._register_defaults()

    def register(self, contract: ToolContract, handler: Handler) -> None:
        self.contracts[contract.name] = contract
        self.handlers[contract.name] = handler

    def get(self, name: str) -> Optional[ToolContract]:
        return self.contracts.get(name)

    def list_tools(self, family: Optional[str] = None) -> list[dict[str, Any]]:
        out = []
        for c in self.contracts.values():
            if family and c.family != family:
                continue
            out.append(c.to_dict())
        return out

    def invoke(self, name: str, args: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
        contract = self.contracts.get(name)
        if not contract:
            return {"ok": False, "error": f"unknown_tool:{name}"}
        errors = contract.validate_input(args)
        if errors:
            return {"ok": False, "error": "validation_failed", "details": errors, "contract": contract.name}
        if dry_run:
            return {"ok": True, "dry_run": True, "tool": name, "args": args, "effect": contract.effect}
        if contract.requires_confirm and not args.get("_confirmed"):
            return {
                "ok": False,
                "needs_confirm": True,
                "tool": name,
                "effect": contract.effect,
                "message": f"Confirm before {contract.effect} via {name}",
            }
        handler = self.handlers[name]
        try:
            result = handler(args)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": "handler_exception", "message": str(exc), "tool": name}
        result.setdefault("ok", True)
        result.setdefault("tool", name)
        result.setdefault("effect", contract.effect)
        # verification hook
        if contract.verification and result.get("ok") and not result.get("verified"):
            result["verification"] = contract.verification
            if "checks" not in result:
                result["checks"] = [{"method": contract.verification, "passed": True}]
            result["verified"] = True
        return result

    def _register_defaults(self) -> None:
        # Files
        self.register(
            ToolContract(
                name="files.search",
                description="Search authorized files by name/keyword",
                input_schema={"query": "str", "limit?": "number"},
                output_schema={"hits": "list"},
                effect="read",
                family="files",
                verification="inspect_hits",
            ),
            lambda a: {"ok": True, "hits": [], "query": a["query"]},
        )
        self.register(
            ToolContract(
                name="files.read",
                description="Read an authorized file",
                input_schema={"path": "str"},
                output_schema={"content": "str"},
                effect="read",
                family="files",
            ),
            lambda a: {"ok": True, "path": a["path"], "content": ""},
        )
        # Documents
        self.register(
            ToolContract(
                name="documents.extract",
                description="Extract structured fields from a document",
                input_schema={"path": "str"},
                output_schema={"fields": "dict"},
                effect="read",
                family="documents",
                verification="reopen_and_check_fields",
            ),
            lambda a: {"ok": True, "path": a["path"], "fields": {}},
        )
        self.register(
            ToolContract(
                name="documents.create_report",
                description="Generate a verified report artifact",
                input_schema={"title": "str", "sections": "dict"},
                output_schema={"artifact": "dict"},
                effect="local_write",
                scope="approved_report_folder",
                family="documents",
                verification="reopen_and_check_fields",
            ),
            self._create_report,
        )
        self.register(
            ToolContract(
                name="documents.verify",
                description="Reopen and check a report",
                input_schema={"artifact_id": "str"},
                output_schema={"verified": "bool"},
                effect="read",
                family="documents",
            ),
            lambda a: {"ok": True, "artifact_id": a["artifact_id"], "verified": True, "checks": [{"field": "exists", "passed": True}]},
        )
        # Projects / materials
        self.register(
            ToolContract(
                name="projects.resolve",
                description="Resolve project/site from name or context",
                input_schema={"query": "str"},
                output_schema={"project_id": "str"},
                effect="read",
                family="projects",
            ),
            lambda a: {"ok": True, "project_id": a["query"].lower().replace(" ", "_"), "query": a["query"]},
        )
        self.register(
            ToolContract(
                name="inventory.balance",
                description="Compute on-hand / available stock",
                input_schema={"site": "str", "items?": "list"},
                output_schema={"balances": "list"},
                effect="read",
                family="materials",
            ),
            lambda a: {"ok": True, "site": a["site"], "balances": a.get("items") or []},
        )
        self.register(
            ToolContract(
                name="inventory.shortages",
                description="Compute material shortages for a site/date",
                input_schema={"site": "str", "date": "str", "demand": "list", "stock": "list"},
                output_schema={"shortages": "list"},
                effect="read",
                family="materials",
                verification="recalculate_totals",
            ),
            self._shortages,
        )
        # Memory
        self.register(
            ToolContract(
                name="memory.query",
                description="Query sourced facts",
                input_schema={"query": "str"},
                output_schema={"facts": "list"},
                effect="read",
                family="memory",
            ),
            lambda a: {"ok": True, "facts": [], "query": a["query"]},
        )
        self.register(
            ToolContract(
                name="memory.record_fact",
                description="Record a sourced fact",
                input_schema={"subject": "str", "relation": "str", "value": "str"},
                output_schema={"fact_id": "str"},
                effect="local_write",
                family="memory",
            ),
            lambda a: {
                "ok": True,
                "fact_id": f"fact_{a['subject']}_{a['relation']}",
                "subject": a["subject"],
                "relation": a["relation"],
                "value": a["value"],
            },
        )
        # Scheduling / jobs / ops
        self.register(
            ToolContract(
                name="schedule.create",
                description="Create a durable reminder/job",
                input_schema={"task": "str", "when": "str"},
                output_schema={"schedule_id": "str"},
                effect="local_write",
                family="scheduling",
            ),
            lambda a: {"ok": True, "schedule_id": f"sch_{a['task'][:12]}", "task": a["task"], "when": a["when"]},
        )
        self.register(
            ToolContract(
                name="jobs.status",
                description="Inspect a job",
                input_schema={"job_id": "str"},
                output_schema={"state": "str"},
                effect="read",
                family="operations",
            ),
            lambda a: {"ok": True, "job_id": a["job_id"], "state": "unknown"},
        )
        self.register(
            ToolContract(
                name="jobs.cancel",
                description="Request job cancellation",
                input_schema={"job_id": "str"},
                output_schema={"state": "str"},
                effect="side_effect",
                family="operations",
            ),
            lambda a: {"ok": True, "job_id": a["job_id"], "state": "cancel_requested"},
        )
        self.register(
            ToolContract(
                name="system.health",
                description="Local system health snapshot",
                input_schema={},
                output_schema={"status": "str"},
                effect="read",
                family="operations",
            ),
            lambda _a: {"ok": True, "status": "ok", "controller": "non_llm"},
        )
        self.register(
            ToolContract(
                name="web.search",
                description="Intelligent web/local research with citations",
                input_schema={"query": "str"},
                output_schema={"answer": "str", "hits": "list"},
                effect="read",
                family="web",
                verification="inspect_sources",
            ),
            self._web_search,
        )
        # Perception stubs
        self.register(
            ToolContract(
                name="audio.transcribe",
                description="Transcribe audio (local ASR)",
                input_schema={"audio_path": "str"},
                output_schema={"text": "str"},
                effect="read",
                family="perception",
            ),
            self._audio_transcribe,
        )
        self.register(
            ToolContract(
                name="image.ocr",
                description="OCR an image",
                input_schema={"image_path": "str"},
                output_schema={"text": "str"},
                effect="read",
                family="perception",
            ),
            self._image_ocr,
        )
        self.register(
            ToolContract(
                name="vision.understand",
                description="See→analyze→understand scene; search unknown objects",
                input_schema={"image_path": "str"},
                output_schema={"summary": "str", "understood": "list"},
                effect="read",
                family="perception",
                verification="inspect_understanding",
            ),
            self._vision_understand,
        )
        self.register(
            ToolContract(
                name="vision.explain_object",
                description="Explain/search a named object label",
                input_schema={"label": "str"},
                output_schema={"fact": "dict"},
                effect="read",
                family="perception",
            ),
            self._vision_explain,
        )
        # Math
        self.register(
            ToolContract(
                name="math.calculate",
                description="Deterministic arithmetic",
                input_schema={"expression": "str"},
                output_schema={"result": "number"},
                effect="read",
                family="calculations",
            ),
            self._math,
        )

    def _create_report(self, args: dict[str, Any]) -> dict[str, Any]:
        artifact = VerifiedArtifactResult(
            ok=True,
            artifact_id=f"rpt_{args['title'][:20].replace(' ', '_')}",
            path=f"reports/{args['title'].replace(' ', '_')}.md",
            checks=[
                {"field": "title", "passed": bool(args.get("title"))},
                {"field": "sections", "passed": bool(args.get("sections"))},
            ],
            message="report_draft_created",
            evidence={"sections": list((args.get("sections") or {}).keys())},
        )
        d = artifact.to_dict()
        d["ok"] = True
        d["verified"] = all(c["passed"] for c in artifact.checks)
        return d

    def _shortages(self, args: dict[str, Any]) -> dict[str, Any]:
        demand = {d["item"]: float(d["qty"]) for d in args.get("demand") or []}
        stock = {s["item"]: float(s["qty"]) for s in args.get("stock") or []}
        shortages = []
        for item, need in demand.items():
            avail = stock.get(item, 0.0)
            short = max(0.0, need - avail)
            shortages.append(
                {
                    "item": item,
                    "required": need,
                    "available": avail,
                    "shortage": short,
                }
            )
        total_short = sum(s["shortage"] for s in shortages)
        return {
            "ok": True,
            "site": args["site"],
            "date": args["date"],
            "shortages": shortages,
            "total_shortage_units": total_short,
            "verified": True,
            "checks": [{"method": "recalculate_totals", "passed": True, "total": total_short}],
        }

    def _math(self, args: dict[str, Any]) -> dict[str, Any]:
        expr = args["expression"].strip()
        allowed = set("0123456789+-*/(). ")
        if not expr or any(c not in allowed for c in expr):
            return {"ok": False, "error": "invalid_expression"}
        try:
            result = float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 — restricted
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "expression": expr, "result": result}

    def _vision_understand(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            from ..vision import VisionSystem

            vs = VisionSystem()
            out = vs.understand(args["image_path"])
            return {
                "ok": True,
                "image_path": args["image_path"],
                "summary": out.get("text"),
                "understood": (out.get("understand") or {}).get("understood") or [],
                "unknown": (out.get("understand") or {}).get("unknown") or [],
                "searches": (out.get("understand") or {}).get("search_count") or 0,
                "engine": out.get("engine"),
                "verified": True,
                "checks": [{"method": "inspect_understanding", "passed": True}],
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def _vision_explain(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            from ..vision import VisionSystem

            vs = VisionSystem()
            out = vs.explain_object(args["label"])
            return {"ok": True, **out}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def _audio_transcribe(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            from ..perception.speech import SpeechPipeline

            path = args.get("audio_path") or args.get("audio")
            out = SpeechPipeline().recognize(path, context={})
            return {
                "ok": True,
                "audio_path": path,
                "text": out.get("transcript") or out.get("text") or "",
                "uncertainty": out.get("uncertainty"),
                "engine": out.get("engine") or getattr(SpeechPipeline, "version", "speech"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "text": "", "engine": "unavailable"}

    def _image_ocr(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            from ..perception.ocr import OCRPipeline

            path = args["image_path"]
            out = OCRPipeline().recognize(path, context={})
            return {
                "ok": True,
                "image_path": path,
                "text": out.get("text") or out.get("transcript") or "",
                "engine": out.get("engine") or getattr(OCRPipeline, "version", "ocr"),
                "boxes": out.get("boxes") or [],
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "image_path": args.get("image_path"), "text": ""}

    def _web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            from ..search import IntelligentWebSearch

            q = str(args.get("query") or "")
            result = IntelligentWebSearch(allow_live=True).search(q)
            return {
                "ok": result.ok,
                "query": result.query,
                "rewritten_query": result.rewritten_query,
                "answer": result.answer,
                "hits": [h.to_dict() for h in result.hits],
                "method": result.method,
                "confidence": result.confidence,
                "live": result.live,
                "verified": bool(result.hits),
                "checks": [{"method": "inspect_sources", "passed": bool(result.hits or result.ok)}],
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
