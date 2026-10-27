"""Document extract / report create / verify (blueprint §10)."""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import utc_now
from .contracts import VerifiedArtifactResult


class DocumentToolkit:
    def __init__(self, report_dir: str | Path = "data/jarvis/reports") -> None:
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts: dict[str, dict[str, Any]] = {}

    def extract(self, path: str) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {
                "ok": False,
                "error": "file_not_found",
                "path": path,
                "fields": {},
                "page_refs": [],
            }
        text = ""
        suffix = p.suffix.lower()
        try:
            if suffix in (".txt", ".md", ".csv", ".json"):
                text = p.read_text(encoding="utf-8", errors="replace")
            elif suffix == ".pdf":
                try:
                    from pypdf import PdfReader  # type: ignore

                    reader = PdfReader(str(p))
                    pages = []
                    for i, page in enumerate(reader.pages):
                        pages.append({"page": i + 1, "text": page.extract_text() or ""})
                    text = "\n".join(x["text"] for x in pages)
                    return {
                        "ok": True,
                        "path": path,
                        "fields": {"page_count": len(pages), "char_count": len(text)},
                        "page_refs": pages,
                        "engine": "pypdf",
                    }
                except Exception as exc:  # noqa: BLE001
                    return {"ok": False, "error": str(exc), "path": path, "fields": {}}
            else:
                text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "path": path, "fields": {}}
        return {
            "ok": True,
            "path": path,
            "fields": {"char_count": len(text), "preview": text[:400]},
            "page_refs": [{"page": 1, "text": text[:2000]}],
            "engine": "text",
        }

    def create_report(
        self,
        title: str,
        sections: dict[str, Any],
        *,
        fmt: str = "md",
    ) -> dict[str, Any]:
        artifact_id = new_id("rpt")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:60]
        path = self.report_dir / f"{safe}_{artifact_id}.{fmt}"
        if fmt == "json":
            body = json.dumps({"title": title, "sections": sections, "created_at": utc_now()}, indent=2)
        else:
            lines = [f"# {title}", "", f"_generated: {utc_now()}_", ""]
            for key, val in sections.items():
                lines.append(f"## {key}")
                if isinstance(val, list):
                    for row in val:
                        lines.append(f"- {row}")
                elif isinstance(val, dict):
                    for k, v in val.items():
                        lines.append(f"- **{k}**: {v}")
                else:
                    lines.append(str(val))
                lines.append("")
            body = "\n".join(lines)
        path.write_text(body, encoding="utf-8")
        checks = [
            {"field": "title", "passed": bool(title)},
            {"field": "sections", "passed": bool(sections)},
            {"field": "file_exists", "passed": path.exists()},
            {"field": "non_empty", "passed": path.stat().st_size > 0},
        ]
        result = VerifiedArtifactResult(
            ok=True,
            artifact_id=artifact_id,
            path=str(path),
            checks=checks,
            message="report_created",
            evidence={"section_keys": list(sections.keys()), "bytes": path.stat().st_size},
        )
        d = result.to_dict()
        d["verified"] = all(c["passed"] for c in checks)
        self.artifacts[artifact_id] = d
        return d

    def verify(self, artifact_id: str, *, expected_fields: Optional[list[str]] = None) -> dict[str, Any]:
        art = self.artifacts.get(artifact_id)
        if not art:
            # try reopen by scanning
            return {"ok": False, "artifact_id": artifact_id, "verified": False, "error": "unknown_artifact"}
        path = Path(art.get("path") or "")
        checks = [
            {"field": "exists", "passed": path.exists()},
            {"field": "artifact_id", "passed": art.get("artifact_id") == artifact_id},
        ]
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            checks.append({"field": "readable", "passed": len(text) > 0})
            for f in expected_fields or []:
                checks.append({"field": f"contains:{f}", "passed": f in text})
        verified = all(c["passed"] for c in checks)
        return {
            "ok": verified,
            "artifact_id": artifact_id,
            "path": str(path),
            "verified": verified,
            "checks": checks,
            "message": "reopened_and_checked" if verified else "verification_failed",
        }

    @staticmethod
    def money(value: Any) -> str:
        """Exact decimal money string (blueprint §10)."""
        d = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return format(d, "f")
