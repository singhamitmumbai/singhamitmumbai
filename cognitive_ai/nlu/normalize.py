"""Cautious text normalization for English, Hindi and Hinglish."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
    "lakh": 100_000,
    "lac": 100_000,
    "crore": 10_000_000,
}


def detect_language(text: str) -> str:
    """Return en | hi | hinglish | mixed based on script and tokens."""
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    lower = text.lower()
    hinglish_markers = (
        "kal",
        "aaj",
        "kitna",
        "dikhao",
        "batao",
        "pending",
        "invoice",
        "project",
        "unpaid",
        "bhejo",
        "mat",
    )
    has_latin = bool(re.search(r"[A-Za-z]", text))
    hits = sum(1 for m in hinglish_markers if m in lower)
    if has_devanagari and has_latin:
        return "mixed"
    if has_devanagari:
        return "hi"
    if hits >= 2 and has_latin:
        return "hinglish"
    return "en"


def normalize_text(text: str) -> str:
    """Light normalization — do not aggressively rewrite meaning."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_money(text: str) -> list[dict]:
    """Extract INR amounts including lakh/crore expressions.

    Indian-unit patterns are applied first so ``₹1 lakh`` becomes 100000,
    not a bare ``₹1``.
    """
    results: list[dict] = []
    lower = text.lower()
    covered: list[tuple[int, int]] = []

    def overlaps(span: tuple[int, int]) -> bool:
        a, b = span
        return any(not (b <= x or a >= y) for x, y in covered)

    unit_pat = re.compile(
        r"(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lac|crore)s?\b",
        re.I,
    )
    for m in unit_pat.finditer(lower):
        raw = m.group(1).replace(",", "")
        try:
            num = float(raw)
        except ValueError:
            continue
        unit = m.group(2).lower()
        if unit in ("lakh", "lac"):
            num *= 100_000
        elif unit == "crore":
            num *= 10_000_000
        covered.append(m.span())
        results.append(
            {"raw": m.group(0), "amount_inr": num, "span": m.span(), "kind": "indian_unit"}
        )

    for pat, kind in (
        (r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)", "inr_symbol"),
        (r"([\d,]+(?:\.\d+)?)\s*(?:rupees|inr)\b", "inr_word"),
    ):
        for m in re.finditer(pat, lower, flags=re.I):
            if overlaps(m.span()):
                continue
            raw = m.group(1).replace(",", "")
            try:
                num = float(raw)
            except ValueError:
                continue
            covered.append(m.span())
            results.append(
                {"raw": m.group(0), "amount_inr": num, "span": m.span(), "kind": kind}
            )
    return results


def extract_dates_relative(text: str, reference_weekday: Optional[str] = None) -> list[dict]:
    """Extract relative time expressions. 'kal' is flagged as ambiguous."""
    del reference_weekday  # reserved for tense resolution
    lower = text.lower()
    out: list[dict] = []
    mapping = {
        "this week": {"relative": "this_week", "ambiguous": False},
        "last week": {"relative": "last_week", "ambiguous": False},
        "last month": {"relative": "last_month", "ambiguous": False},
        "aaj": {"relative": "today", "ambiguous": False},
        "today": {"relative": "today", "ambiguous": False},
        "yesterday": {"relative": "yesterday", "ambiguous": False},
        "tomorrow": {"relative": "tomorrow", "ambiguous": False},
        "kal": {"relative": "kal", "ambiguous": True},  # yesterday OR tomorrow
    }
    for phrase, meta in mapping.items():
        if phrase in lower:
            out.append({"phrase": phrase, **meta})
    return out
