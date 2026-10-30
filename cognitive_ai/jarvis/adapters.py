"""Adapters bridging JARVIS tools to Finderbay / perception / filesystem (blueprint §9–§10)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .documents import DocumentToolkit
from .policy import PolicyGuard


class FinderbayAdapters:
    """Prefer official/local data; Playwright/accessibility are later phases."""

    def __init__(
        self,
        *,
        finderbay: Any = None,
        docs: Optional[DocumentToolkit] = None,
        policy: Optional[PolicyGuard] = None,
    ) -> None:
        self.finderbay = finderbay
        self.docs = docs or DocumentToolkit()
        self.policy = policy or PolicyGuard()

    def files_search(self, query: str, *, limit: int = 10, roots: Optional[list[str]] = None) -> dict[str, Any]:
        roots = roots or ["data", "reports", "."]
        hits = []
        q = query.lower()
        for root in roots:
            base = Path(root)
            if not base.exists():
                continue
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                if q in p.name.lower() or q in str(p).lower():
                    hits.append({"path": str(p), "name": p.name, "bytes": p.stat().st_size})
                    if len(hits) >= limit:
                        return {"ok": True, "hits": hits, "query": query}
        return {"ok": True, "hits": hits, "query": query}

    def files_read(self, path: str) -> dict[str, Any]:
        auth = self.policy.authorize_tool("files.read", "read", {"path": path})
        if not auth.get("ok"):
            return auth
        p = Path(path)
        if not p.exists():
            return {"ok": False, "error": "file_not_found", "path": path}
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "path": path}
        return {"ok": True, "path": path, "content": text[:50_000], "bytes": p.stat().st_size}

    def inventory_from_finderbay(self, site: str) -> dict[str, Any]:
        """Pull material rows from Finderbay world model when available."""
        if self.finderbay is None:
            return {"ok": False, "error": "finderbay_unavailable", "site": site}
        try:
            wm = getattr(self.finderbay, "world", None) or getattr(self.finderbay, "world_model", None)
            materials = []
            if wm is not None:
                store = getattr(wm, "materials", None) or getattr(wm, "entities", None)
                if isinstance(store, dict):
                    for mid, row in store.items():
                        if isinstance(row, dict):
                            materials.append(
                                {
                                    "item": row.get("name") or row.get("sku") or mid,
                                    "qty": float(row.get("available") or row.get("qty") or row.get("on_hand") or 0),
                                    "site": row.get("site") or site,
                                }
                            )
                elif hasattr(wm, "query"):
                    rows = wm.query("material") or []
                    for row in rows:
                        if isinstance(row, dict):
                            materials.append(
                                {
                                    "item": row.get("name") or row.get("id"),
                                    "qty": float(row.get("available") or row.get("qty") or 0),
                                    "site": row.get("site") or site,
                                }
                            )
            return {"ok": True, "site": site, "balances": materials, "source": "finderbay"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "site": site}

    def audio_transcribe(self, audio: Any) -> dict[str, Any]:
        try:
            from ..perception.speech import SpeechPipeline

            sp = SpeechPipeline()
            result = sp.recognize(audio, context={})
            return {
                "ok": True,
                "text": result.get("transcript") or "",
                "uncertainty": result.get("uncertainty"),
                "engine": result.get("engine") or getattr(sp, "version", "speech"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "text": "", "engine": "unavailable"}

    def image_ocr(self, image_path: str) -> dict[str, Any]:
        try:
            from ..perception.ocr import OCRPipeline

            ocr = OCRPipeline()
            result = ocr.run(image_path)
            return {
                "ok": True,
                "image_path": image_path,
                "text": result.get("text") or "",
                "engine": result.get("engine") or "ocr",
                "boxes": result.get("boxes") or [],
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "image_path": image_path, "text": ""}

    def bind_registry(self, registry) -> None:
        """Replace default handlers with adapter-backed implementations."""
        docs = self.docs

        registry.handlers["files.search"] = lambda a: self.files_search(
            a["query"], limit=int(a.get("limit") or 10)
        )
        registry.handlers["files.read"] = lambda a: self.files_read(a["path"])
        registry.handlers["documents.extract"] = lambda a: docs.extract(a["path"])
        registry.handlers["documents.create_report"] = lambda a: docs.create_report(
            a["title"], a["sections"]
        )
        registry.handlers["documents.verify"] = lambda a: docs.verify(a["artifact_id"])
        registry.handlers["audio.transcribe"] = lambda a: self.audio_transcribe(
            a.get("audio") or a.get("audio_path")
        )
        registry.handlers["image.ocr"] = lambda a: self.image_ocr(a["image_path"])
        registry.handlers["inventory.balance"] = self._balance_handler

    def _balance_handler(self, args: dict[str, Any]) -> dict[str, Any]:
        site = args["site"]
        fb = self.inventory_from_finderbay(site)
        if fb.get("ok") and fb.get("balances"):
            return fb
        items = args.get("items") or []
        return {"ok": True, "site": site, "balances": items, "source": "args"}
