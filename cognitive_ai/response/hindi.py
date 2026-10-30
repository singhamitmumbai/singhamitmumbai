"""Hindi / Hinglish surface realization from structured answer plans."""

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
    11: "ग्यारह",
    12: "बारह",
    13: "तेरह",
    14: "चौदह",
    15: "पंद्रह",
    16: "सोलह",
    17: "सत्रह",
    18: "अठारह",
    19: "उन्नीस",
    20: "बीस",
}


def num_hi(n: int) -> str:
    return NUM_HI.get(int(n), str(n))


def realize_invoice_hi(structured: dict[str, Any]) -> str:
    count = int(structured.get("count") or 0)
    total = float(structured.get("total_inr") or 0)
    unresolved = int(structured.get("unresolved_count") or 0)
    as_of = structured.get("as_of")
    if count == 0:
        text = "दी गई शर्तों से कोई इनवॉइस मेल नहीं खाता।"
    else:
        text = f"{num_hi(count)} इनवॉइस मिले, कुल {format_inr_hi(total)}"
        if as_of:
            text += f", रजिस्टर समय {as_of} के अनुसार।"
        else:
            text += "।"
    if unresolved:
        text += f" {num_hi(unresolved)} अन्य इनवॉइस की भुगतान/विवाद स्थिति अनसुलझी है।"
    return text


def realize_pending_hi(structured: dict[str, Any], draft: Optional[dict] = None) -> str:
    pending = int(structured.get("pending_count") or 0)
    uncertain = int(structured.get("uncertain_count") or 0)
    text = f"{num_hi(pending)} पुष्ट लंबित डिलीवरी आइटम मिले।"
    if uncertain:
        text += (
            f" {num_hi(uncertain)} अतिरिक्त आइटम की रसीद स्थिति अनिश्चित है "
            f"और उन्हें पुष्ट लंबित से अलग रखा गया।"
        )
    if draft:
        text += " फॉलो-अप ड्राफ्ट तैयार किया गया और भेजा नहीं गया।"
        text += "\n\n--- Draft ---\n" + draft.get("body", "")
    return text


def realize_materials_hi(structured: dict[str, Any]) -> str:
    count = int(structured.get("count") or 0)
    if count == 0:
        return "कोई सामग्री रिकॉर्ड नहीं मिला।"
    return f"{num_hi(count)} सामग्री रिकॉर्ड मिले।"


def realize_reserve_hi(structured: dict[str, Any]) -> str:
    if not structured.get("reserved"):
        return "स्टॉक रिज़र्व नहीं हो सका।"
    qty = structured.get("qty_reserved")
    avail = structured.get("available")
    item = structured.get("item") or structured.get("material") or "सामग्री"
    return f"{item} की {qty} मात्रा रिज़र्व हुई। उपलब्ध शेष: {avail}।"


def realize_task_hi(structured: dict[str, Any]) -> str:
    if structured.get("created"):
        title = structured.get("title") or (structured.get("task") or {}).get("title") or "कार्य"
        return f"कार्य बनाया गया: {title}।"
    count = int(structured.get("count") or 0)
    return f"{num_hi(count)} कार्य सूची में हैं।"


def realize_alerts_hi(structured: dict[str, Any]) -> str:
    n = int(structured.get("count") or 0)
    by = structured.get("by_severity") or {}
    return (
        f"{num_hi(n)} अलर्ट: "
        f"{num_hi(int(by.get('critical', 0)))} गंभीर, "
        f"{num_hi(int(by.get('warn', 0)))} चेतावनी, "
        f"{num_hi(int(by.get('info', 0)))} सूचना।"
    )


def realize_payment_hi(structured: dict[str, Any]) -> str:
    pay = structured.get("payment") or {}
    return (
        f"भुगतान {pay.get('id')} दर्ज: ₹{pay.get('amount_inr')} "
        f"इनवॉइस {pay.get('invoice_id')} के लिए।"
    )


def realize_vision_hi(structured: dict[str, Any], op: str) -> str:
    engine = structured.get("engine", "stub")
    if op == "count_people":
        n = int(structured.get("people_count") or structured.get("count") or 0)
        return f"{num_hi(n)} व्यक्ति गिने गए (इंजन={engine})।"
    if op == "detect_vehicles":
        n = int(structured.get("vehicle_count") or structured.get("count") or 0)
        return f"{num_hi(n)} वाहन पहचाने गए (इंजन={engine})।"
    if op == "detect_fire":
        alert = "हाँ" if structured.get("alert") else "नहीं"
        return f"आग/धुआँ अलर्ट: {alert} (इंजन={engine})।"
    if op == "vision_status":
        return "कंप्यूटर विज़न तैयार है (YOLO_Projects)।"
    if op == "understand_scene":
        und = int((structured.get("understand") or {}).get("understood_count") or 0)
        unk = int((structured.get("understand") or {}).get("unknown_count") or 0)
        return (
            f"दृश्य समझ: {num_hi(und)} वस्तुएँ समझीं, "
            f"{num_hi(unk)} अज्ञात (इंजन={engine})।"
        )
    n = int(structured.get("count") or 0)
    return f"{num_hi(n)} वस्तुएँ पहचानी गईं (इंजन={engine})।"


def realize_status_hi() -> str:
    return (
        "मैं Complete AI हूँ — Finderbay कार्य, विज़न, कोडिंग और सामान्य चैट। "
        "human_level_agi=false।"
    )


def prefer_hindi(language: str) -> bool:
    return language in ("hi", "hinglish", "mixed")
