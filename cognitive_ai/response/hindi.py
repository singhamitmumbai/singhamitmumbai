"""Limited Hindi / Hinglish surface realization from structured answer plans."""

from __future__ import annotations

from typing import Any, Optional


def format_inr_hi(amount: float) -> str:
    if amount >= 10_000_000:
        return f"₹{amount / 10_000_000:.2f} करोड़".replace(".00", "")
    if amount >= 100_000:
        val = amount / 100_000
        if abs(val - round(val)) < 1e-9:
            return f"₹{int(round(val))} लाख"
        return f"₹{val:.1f} लाख"
    return f"₹{amount:,.0f}"


NUM_HI = {
    0: "शून्य",
    1: "एक",
    2: "दो",
    3: "तीन",
    4: "चार",
    5: "पाँच",
    6: "छह",
    7: "सात",
    8: "आठ",
    9: "नौ",
    10: "दस",
}


def realize_invoice_hi(structured: dict[str, Any]) -> str:
    count = int(structured.get("count") or 0)
    total = float(structured.get("total_inr") or 0)
    unresolved = int(structured.get("unresolved_count") or 0)
    as_of = structured.get("as_of")
    num = NUM_HI.get(count, str(count))
    if count == 0:
        text = "दी गई शर्तों से कोई इनवॉइस मेल नहीं खाता।"
    else:
        text = f"{num} इनवॉइस मिले, कुल {format_inr_hi(total)}"
        if as_of:
            text += f", रजिस्टर समय {as_of} के अनुसार।"
        else:
            text += "।"
    if unresolved:
        u = NUM_HI.get(unresolved, str(unresolved))
        text += f" {u} अन्य इनवॉइस की भुगतान/विवाद स्थिति अनसुलझी है।"
    return text


def realize_pending_hi(structured: dict[str, Any], draft: Optional[dict] = None) -> str:
    pending = int(structured.get("pending_count") or 0)
    uncertain = int(structured.get("uncertain_count") or 0)
    text = f"{pending} पुष्ट लंबित डिलीवरी आइटम मिले।"
    if uncertain:
        text += (
            f" {uncertain} अतिरिक्त आइटम की रसीद स्थिति अनिश्चित है "
            f"और उन्हें पुष्ट लंबित से अलग रखा गया।"
        )
    if draft:
        text += " फॉलो-अप ड्राफ्ट तैयार किया गया और भेजा नहीं गया।"
        text += "\n\n--- Draft ---\n" + draft.get("body", "")
    return text


def prefer_hindi(language: str) -> bool:
    return language in ("hi", "hinglish", "mixed")
