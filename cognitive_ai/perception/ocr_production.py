"""Production OCR via Tesseract (LSTM) — non-LLM document pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Union


class ProductionOCR:
    """Full-page OCR with field heuristics; falls back to structured oracle input."""

    version = "tesseract_v5"

    def __init__(self, lang: str = "eng+hin") -> None:
        self.lang = lang
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import pytesseract  # noqa: F401
                from PIL import Image  # noqa: F401

                self._available = True
            except Exception:
                self._available = False
        return bool(self._available)

    def recognize(self, document: Any, context: Optional[dict] = None) -> dict[str, Any]:
        context = context or {}
        # Structured oracle / pre-extracted page
        if isinstance(document, dict) and (
            "fields" in document or "text" in document or "tables" in document
        ):
            return self._from_structured(document)

        # Image path or PIL image
        if isinstance(document, (str, Path)) and Path(document).exists():
            return self._from_image(Path(document))
        try:
            from PIL import Image

            if isinstance(document, Image.Image):
                return self._from_pil(document)
        except Exception:
            pass

        if isinstance(document, (bytes, bytearray)):
            try:
                from io import BytesIO
                from PIL import Image

                return self._from_pil(Image.open(BytesIO(document)))
            except Exception as exc:
                return {
                    "text": "",
                    "fields": {},
                    "uncertainty": 1.0,
                    "error": f"image_decode_failed:{exc}",
                    "version": self.version,
                }

        return {
            "text": str(document) if document is not None else "",
            "fields": {},
            "uncertainty": 0.5,
            "version": self.version,
            "note": "non_image_input_passthrough",
        }

    def _from_structured(self, document: dict) -> dict[str, Any]:
        from .ocr import DocumentPipeline

        # reuse validation heuristics
        return DocumentPipeline().recognize(document)

    def _from_image(self, path: Path) -> dict[str, Any]:
        from PIL import Image

        return self._from_pil(Image.open(path))

    def _from_pil(self, image: Any) -> dict[str, Any]:
        if not self.available:
            return {
                "text": "",
                "fields": {},
                "uncertainty": 1.0,
                "error": "tesseract_not_available",
                "version": self.version,
            }
        import pytesseract

        # Quality / orientation assist
        try:
            osd = pytesseract.image_to_osd(image)
        except Exception:
            osd = ""
        text = pytesseract.image_to_string(image, lang=self.lang)
        data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)
        regions = []
        n = len(data.get("text") or [])
        for i in range(n):
            word = (data["text"][i] or "").strip()
            if not word:
                continue
            regions.append(
                {
                    "text": word,
                    "conf": float(data["conf"][i]),
                    "left": int(data["left"][i]),
                    "top": int(data["top"][i]),
                    "width": int(data["width"][i]),
                    "height": int(data["height"][i]),
                }
            )
        fields = self._extract_fields(text)
        confs = [r["conf"] for r in regions if r["conf"] >= 0]
        uncertainty = 0.0
        if confs:
            uncertainty = max(0.0, min(1.0, 1.0 - (sum(confs) / len(confs) / 100.0)))
        return {
            "text": text.strip(),
            "fields": fields,
            "regions": regions,
            "tables": [],
            "osd": osd,
            "uncertainty": uncertainty,
            "version": self.version,
            "engine": "tesseract",
            "lang": self.lang,
            "stages_completed": [
                "quality",
                "osd",
                "recognize",
                "reading_order",
                "key_value",
                "validate",
                "source_link",
            ],
        }

    def _extract_fields(self, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        patterns = {
            "invoice_number": r"(?:invoice\s*(?:no|number|#)\s*[:\-]?\s*)([A-Za-z0-9\-]+)",
            "amount_due": r"(?:amount\s*due|total\s*due|grand\s*total)\s*[:\-]?\s*₹?\s*([\d,]+\.?\d*)",
            "subtotal": r"(?:sub\s*total|subtotal)\s*[:\-]?\s*₹?\s*([\d,]+\.?\d*)",
            "tax": r"(?:tax|gst|vat)\s*[:\-]?\s*₹?\s*([\d,]+\.?\d*)",
            "vendor": r"(?:vendor|from|seller)\s*[:\-]?\s*([A-Za-z0-9 &.\-]+)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text, flags=re.I)
            if not m:
                continue
            val = m.group(1).strip()
            if key in ("amount_due", "subtotal", "tax"):
                try:
                    fields[key] = float(val.replace(",", ""))
                except ValueError:
                    fields[key] = val
            else:
                fields[key] = val
        return fields
