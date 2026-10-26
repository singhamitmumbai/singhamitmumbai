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
