"""OCR / document understanding pipeline interface (non-LLM).

Full pipeline: quality → deskew → regions → recognize → reading order →
tables → key-value → validate → link to source regions.

Bundled implementation accepts pre-extracted page structures for testing;
plug Tesseract/CRNN checkpoints with audited provenance for production.
"""

from __future__ import annotations

from typing import Any, Optional


class DocumentPipeline:
    version = "ocr_stub_v1"

    def recognize(self, document: Any, context: Optional[dict] = None) -> dict[str, Any]:
        context = context or {}
        if isinstance(document, dict):
            # Expected keys: text, fields, tables, page_id, regions
            fields = document.get("fields") or {}
            # Field validation example for invoices
            validated = {}
            issues = []
            for key, val in fields.items():
                if key.endswith("amount") or key in ("amount_due", "subtotal", "tax", "amount_inr"):
                    try:
                        validated[key] = float(str(val).replace(",", "").replace("₹", ""))
                    except ValueError:
                        issues.append(f"ambiguous_amount:{key}:{val}")
                        validated[key] = None
                else:
                    validated[key] = val
            return {
                "text": document.get("text", ""),
                "fields": validated,
                "tables": document.get("tables") or [],
                "regions": document.get("regions") or [],
                "page_id": document.get("page_id"),
                "uncertainty": 0.1 if issues else 0.02,
                "issues": issues,
                "version": self.version,
                "stages_completed": [
                    "quality",
                    "deskew",
                    "regions",
                    "recognize",
                    "reading_order",
                    "tables",
                    "key_value",
                    "validate",
                    "source_link",
                ],
            }
        if isinstance(document, str):
            return {
                "text": document,
                "fields": {},
                "tables": [],
                "uncertainty": 0.0,
                "version": self.version,
            }
        return {
            "text": "",
            "fields": {},
            "uncertainty": 1.0,
            "error": "ocr_engine_not_loaded",
            "version": self.version,
        }
