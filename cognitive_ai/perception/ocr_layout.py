"""Multi-page invoice OCR with layout/table heuristics (Tesseract-based)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


@dataclass
class PageResult:
    page_index: int
    text: str
    fields: dict[str, Any]
    tables: list[list[list[str]]] = field(default_factory=list)
    regions: list[dict] = field(default_factory=list)
    uncertainty: float = 0.0


@dataclass
class DocumentResult:
    pages: list[PageResult]
    fields: dict[str, Any]
    tables: list[list[list[str]]]
    text: str
    page_count: int
    version: str = "tesseract_layout_v1"
    engine: str = "tesseract"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": [
                {
                    "page_index": p.page_index,
                    "text": p.text,
                    "fields": p.fields,
                    "tables": p.tables,
                    "uncertainty": p.uncertainty,
                    "regions": p.regions[:50],
                }
                for p in self.pages
            ],
            "fields": self.fields,
            "tables": self.tables,
            "text": self.text,
            "page_count": self.page_count,
            "version": self.version,
            "engine": self.engine,
            "uncertainty": max((p.uncertainty for p in self.pages), default=0.0),
        }


class LayoutOCR:
    """Recognize one or many pages; reconstruct simple tables from TSV."""

    def __init__(self, lang: str = "eng+hin") -> None:
        self.lang = lang

    def recognize_document(self, source: Any) -> DocumentResult:
        pages_in = self._normalize_pages(source)
        pages: list[PageResult] = []
        for i, page in enumerate(pages_in):
            pages.append(self._recognize_page(i, page))
        merged_fields: dict[str, Any] = {}
        tables: list[list[list[str]]] = []
        texts = []
        for p in pages:
            texts.append(p.text)
            for k, v in p.fields.items():
                merged_fields.setdefault(k, v)
            tables.extend(p.tables)
        return DocumentResult(
            pages=pages,
            fields=merged_fields,
            tables=tables,
            text="\n\n".join(texts),
            page_count=len(pages),
        )

    def _normalize_pages(self, source: Any) -> list[Any]:
        if isinstance(source, dict):
            if "pages" in source:
                return list(source["pages"])
            return [source]
        if isinstance(source, list):
            return source
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                files = sorted(
                    [
                        p
                        for p in path.iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
                    ]
                )
                return files
            return [path]
        return [source]

    def _recognize_page(self, index: int, page: Any) -> PageResult:
        from .ocr_production import ProductionOCR

        prod = ProductionOCR(lang=self.lang)
        if isinstance(page, dict) and not any(
            isinstance(page.get(k), (str, Path)) and Path(str(page.get(k))).exists()
            for k in ("path", "image")
        ):
            # structured page
            raw = prod.recognize(page)
            tables = page.get("tables") or self._tables_from_text(raw.get("text") or "")
            return PageResult(
                page_index=index,
                text=raw.get("text") or "",
                fields=raw.get("fields") or {},
                tables=tables,
                regions=raw.get("regions") or [],
                uncertainty=float(raw.get("uncertainty") or 0.0),
            )

        image_src = page
        if isinstance(page, dict):
            image_src = page.get("path") or page.get("image") or page
        raw = prod.recognize(image_src)
        text = raw.get("text") or ""
        tables = self._tables_from_tesseract_image(image_src) or self._tables_from_text(text)
        return PageResult(
            page_index=index,
            text=text,
            fields=raw.get("fields") or {},
            tables=tables,
            regions=raw.get("regions") or [],
            uncertainty=float(raw.get("uncertainty") or 0.0),
        )

    def _tables_from_tesseract_image(self, image_src: Any) -> list[list[list[str]]]:
        try:
            import pytesseract
            from PIL import Image
        except Exception:
            return []
        try:
            if isinstance(image_src, (str, Path)):
                img = Image.open(image_src)
            else:
                img = image_src
            tsv = pytesseract.image_to_data(img, lang=self.lang, output_type=pytesseract.Output.DICT)
        except Exception:
            return []
        # Group words by block/paragraph/line
        lines: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
        n = len(tsv.get("text") or [])
        for i in range(n):
            word = (tsv["text"][i] or "").strip()
            if not word or float(tsv["conf"][i]) < 0:
                continue
            key = (int(tsv["block_num"][i]), int(tsv["par_num"][i]), int(tsv["line_num"][i]))
            lines.setdefault(key, []).append((int(tsv["left"][i]), word))
        if len(lines) < 2:
            return []
        # Heuristic: consecutive lines with 2+ tokens → table rows
        rows = []
        for key in sorted(lines):
            words = [w for _, w in sorted(lines[key])]
            if len(words) >= 2:
                rows.append(words)
        if len(rows) >= 2:
            return [rows]
        return []

    def _tables_from_text(self, text: str) -> list[list[list[str]]]:
        rows = []
        for line in text.splitlines():
            if "\t" in line:
                rows.append([c.strip() for c in line.split("\t") if c.strip()])
            elif re.search(r"\s{2,}", line):
                parts = [p.strip() for p in re.split(r"\s{2,}", line.strip()) if p.strip()]
                if len(parts) >= 2:
                    rows.append(parts)
        return [rows] if len(rows) >= 2 else []

    def analyze_layout(self, doc: DocumentResult) -> dict[str, Any]:
        """Summarize multi-page layout: columns, tables, key fields."""
        col_hints: list[str] = []
        for table in doc.tables[:5]:
            if table and isinstance(table[0], list):
                col_hints.extend(str(c) for c in table[0][:8])
        dense_pages = sum(1 for p in doc.pages if len((p.text or "").split()) > 20)
        return {
            "page_count": doc.page_count,
            "table_count": len(doc.tables),
            "field_keys": sorted(doc.fields.keys()),
            "column_hints": list(dict.fromkeys(col_hints))[:20],
            "dense_pages": dense_pages,
            "engine": doc.engine,
            "version": doc.version,
            "has_multi_page": doc.page_count > 1,
        }

    def export_invoice_fields(self, doc: DocumentResult) -> dict[str, Any]:
        """Map layout OCR fields + tables into a Finderbay-ready invoice dict.

        Production path: multi-page OCR → structured invoice fields for import.
        Does not claim perfect extraction — callers should verify.
        """
        fields = dict(doc.fields or {})
        # Normalize common aliases
        invoice = {
            "invoice_number": fields.get("invoice_number") or fields.get("invoice_no") or fields.get("inv_no"),
            "vendor": fields.get("vendor") or fields.get("supplier") or fields.get("from"),
            "amount_due": _as_float(fields.get("amount_due") or fields.get("total") or fields.get("grand_total")),
            "subtotal": _as_float(fields.get("subtotal") or fields.get("sub_total")),
            "tax": _as_float(fields.get("tax") or fields.get("gst") or fields.get("vat")),
            "currency": fields.get("currency") or "INR",
            "due_date": fields.get("due_date") or fields.get("due"),
            "invoice_date": fields.get("invoice_date") or fields.get("date"),
            "project_hint": fields.get("project") or fields.get("project_id"),
            "line_items": [],
            "raw_fields": fields,
            "page_count": doc.page_count,
            "table_count": len(doc.tables),
            "source_engine": doc.engine,
            "version": "invoice_export_v1",
            "human_level_agi": False,
        }
        # First table → line items when header-like
        if doc.tables:
            table = doc.tables[0]
            if table and len(table) >= 2:
                headers = [str(h).lower().strip() for h in table[0]]
                for row in table[1:]:
                    item: dict[str, Any] = {}
                    for i, cell in enumerate(row):
                        key = headers[i] if i < len(headers) else f"col_{i}"
                        item[key] = cell
                    if item:
                        invoice["line_items"].append(item)
        invoice["ok"] = bool(
            invoice.get("invoice_number")
            or invoice.get("amount_due") is not None
            or invoice.get("line_items")
        )
        invoice["tables_csv"] = [_table_to_csv(t) for t in doc.tables[:3]]
        return invoice


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").replace("₹", "").replace("rs", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", text, flags=re.I)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _table_to_csv(table: list[list[str]]) -> str:
    lines = []
    for row in table:
        lines.append(",".join('"' + str(c).replace('"', '""') + '"' for c in row))
    return "\n".join(lines)
