"""Session message search + message variation (non-LLM paraphrases).

Search finds turns in a chat transcript. Variation produces alternate
phrasings of a message (formal, simple, short, polite, Hinglish, …).
Also expands a search query into useful alternate query strings.

human_level_agi=false — lexical/template transforms, not generative AGI.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional


# Lightweight synonym expansion for search recall
_SYNONYMS: dict[str, list[str]] = {
    "invoice": ["bill", "invoices", "billing"],
    "bill": ["invoice", "invoices"],
    "show": ["list", "display", "find", "get", "dikhao"],
    "list": ["show", "display", "find"],
    "find": ["search", "show", "locate", "lookup"],
    "search": ["find", "lookup", "query"],
    "payment": ["payments", "settlement", "paid"],
    "unpaid": ["outstanding", "due", "pending payment", "baaki"],
    "explain": ["describe", "what is", "tell me about"],
    "help": ["assist", "support"],
    "project": ["proj", "site"],
    "message": ["msg", "turn", "chat", "reply"],
    "gravity": ["gravitation"],
    "code": ["function", "program", "script"],
}


@dataclass
class MessageHit:
    index: int
    role: str
    text: str
    route: str = ""
    score: float = 0.0
    matched: list[str] = field(default_factory=list)
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Variation:
    style: str
    text: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MessageSearch:
    """Lexical + synonym search over session turns."""

    version = "message_search_v1"

    def search(
        self,
        messages: Iterable[dict[str, Any] | Any],
        query: str,
        *,
        role: Optional[str] = None,
        route: Optional[str] = None,
        limit: int = 20,
        include_query_variations: bool = True,
    ) -> dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {
                "ok": False,
                "error": "empty_query",
                "hits": [],
                "human_level_agi": False,
            }
        rows = list(messages)
        variants = self.query_variations(q) if include_query_variations else [q]
        tokens = _tokens(q)
        expanded = set(tokens)
        for t in list(tokens):
            expanded.update(_SYNONYMS.get(t, []))

        hits: list[MessageHit] = []
        for i, raw in enumerate(rows):
            row = _as_row(raw)
            if role and row["role"] != role:
                continue
            if route and route.lower() not in (row["route"] or "").lower():
                continue
            text = row["text"]
            lower = text.lower()
            score = 0.0
            matched: list[str] = []
            # Exact / substring
            if q.lower() in lower:
                score += 5.0
                matched.append("substring")
            for v in variants[1:6]:
                if v.lower() in lower and v.lower() != q.lower():
                    score += 2.0
                    matched.append(f"variant:{v[:40]}")
                    break
            # Token overlap + synonyms
            msg_toks = set(_tokens(text))
            overlap = expanded & msg_toks
            if overlap:
                score += 1.5 * len(overlap)
                matched.extend(sorted(overlap)[:8])
            # Role boost for assistant answers when query looks like a topic
            if row["role"] == "assistant" and score > 0:
                score += 0.2
            if score <= 0:
                continue
            hits.append(
                MessageHit(
                    index=i,
                    role=row["role"],
                    text=text,
                    route=row["route"],
                    score=round(score, 3),
                    matched=list(dict.fromkeys(matched))[:12],
                    ts=float(row.get("ts") or 0),
                )
            )
        hits.sort(key=lambda h: (-h.score, -h.index))
        limited = hits[: max(1, limit)]
        return {
            "ok": True,
            "query": q,
            "query_variations": variants,
            "hits": [h.to_dict() for h in limited],
            "total_hits": len(hits),
            "scanned": len(rows),
            "version": self.version,
            "human_level_agi": False,
        }

    def query_variations(self, query: str, *, n: int = 8) -> list[str]:
        """Alternate search strings for better recall."""
        q = (query or "").strip()
        out = [q]
        lower = q.lower()
        # Synonym swaps on first matching token
        for tok in _tokens(q):
            for syn in _SYNONYMS.get(tok, [])[:3]:
                alt = re.sub(rf"\b{re.escape(tok)}\b", syn, lower, count=1, flags=re.I)
                if alt not in out:
                    out.append(alt)
        # Soft wrappers
        for wrap in (
            f"show {q}",
            f"find {q}",
            f"about {q}",
            f"{q}?",
        ):
            if wrap not in out and wrap.strip() != q:
                out.append(wrap)
        # Drop filler
        stripped = re.sub(
            r"\b(?:please|kindly|can you|could you|i want to|search for|find messages? (?:for|about))\b",
            "",
            lower,
            flags=re.I,
        )
        stripped = re.sub(r"\s+", " ", stripped).strip(" ?")
        if stripped and stripped not in out:
            out.insert(1, stripped)
        return out[:n]


class MessageVariation:
    """Produce alternate phrasings of a message (template / lexical)."""

    version = "message_variation_v1"

    STYLES = (
        "formal",
        "simple",
        "short",
        "polite",
        "question",
        "imperative",
        "hinglish",
        "email",
        "bullet",
        "concise",
    )

    def vary(
        self,
        text: str,
        *,
        styles: Optional[list[str]] = None,
        n: Optional[int] = None,
    ) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {"ok": False, "error": "empty_text", "variations": [], "human_level_agi": False}
        wanted = [s.lower() for s in (styles or list(self.STYLES))]
        wanted = [s for s in wanted if s in self.STYLES]
        if not wanted:
            wanted = list(self.STYLES)
        if n is not None:
            wanted = wanted[: max(1, int(n))]

        variations: list[Variation] = []
        for style in wanted:
            fn = getattr(self, f"_style_{style}", None)
            if not fn:
                continue
            alt = fn(raw)
            if alt and alt.strip() and alt.strip() != raw:
                variations.append(Variation(style=style, text=alt.strip(), notes="template"))
            elif alt and alt.strip() == raw:
                # still show with light mark so UI has something
                variations.append(
                    Variation(style=style, text=alt.strip(), notes="unchanged_core")
                )
        # Deduplicate by text keeping first style
        seen: set[str] = set()
        unique: list[Variation] = []
        for v in variations:
            key = v.text.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(v)
        return {
            "ok": True,
            "original": raw,
            "variations": [v.to_dict() for v in unique],
            "count": len(unique),
            "version": self.version,
            "human_level_agi": False,
        }

    # --- style transforms -----------------------------------------------------

    def _style_formal(self, text: str) -> str:
        t = text
        reps = [
            (r"\bcan't\b", "cannot"),
            (r"\bwon't\b", "will not"),
            (r"\bdon't\b", "do not"),
            (r"\bI'm\b", "I am"),
            (r"\blet's\b", "let us"),
            (r"\bshow\b", "please provide"),
            (r"\bfind\b", "kindly locate"),
            (r"\btell me\b", "please explain"),
            (r"\bgimme\b", "please give me"),
        ]
        for pat, repl in reps:
            t = re.sub(pat, repl, t, flags=re.I)
        if not re.search(r"\bplease\b|\bkindly\b", t, flags=re.I):
            t = "Please " + t[0].lower() + t[1:] if t else t
        if not t.endswith((".", "?", "!")):
            t += "."
        return t[0].upper() + t[1:] if t else t

    def _style_simple(self, text: str) -> str:
        t = text
        reps = [
            (r"\bprovide\b", "give"),
            (r"\blocate\b", "find"),
            (r"\bassist\b", "help"),
            (r"\binquire\b", "ask"),
            (r"\bregarding\b", "about"),
            (r"\butilize\b", "use"),
            (r"\bapproximately\b", "about"),
            (r"\bgravitation\b", "gravity"),
        ]
        for pat, repl in reps:
            t = re.sub(pat, repl, t, flags=re.I)
        return t

    def _style_short(self, text: str) -> str:
        t = re.sub(
            r"\b(?:please|kindly|could you|can you|would you|i (?:want|need) to|i would like to)\b",
            "",
            text,
            flags=re.I,
        )
        t = re.sub(r"\s+", " ", t).strip(" ,")
        # Keep first clause
        t = re.split(r"[.;]\s+", t)[0].strip()
        return t[:120]

    def _style_concise(self, text: str) -> str:
        return self._style_short(text)

    def _style_polite(self, text: str) -> str:
        core = self._style_short(text)
        if not core:
            core = text
        if not re.search(r"\bplease\b", core, flags=re.I):
            core = "Please " + core[0].lower() + core[1:] if core else core
        if not core.endswith((".", "?", "!")):
            core += "."
        if not core.lower().endswith("thank you.") and "thank" not in core.lower():
            core = core.rstrip(".") + ". Thank you."
        return core[0].upper() + core[1:] if core else core

    def _style_question(self, text: str) -> str:
        t = text.rstrip(".! ")
        if t.endswith("?"):
            return t
        # Lead with question word if not present
        if not re.match(r"^(?:what|why|how|when|where|who|which|can|could|is|are|do|does)\b", t, flags=re.I):
            t = f"Could you {t[0].lower() + t[1:] if t else t}"
        return t + "?"

    def _style_imperative(self, text: str) -> str:
        t = re.sub(
            r"^(?:please|kindly|could you|can you|would you|i (?:want|need) to)\s+",
            "",
            text,
            flags=re.I,
        )
        t = t.strip().rstrip(".?")
        # Capitalize first verb-ish token
        if t:
            t = t[0].upper() + t[1:]
        return t

    def _style_hinglish(self, text: str) -> str:
        t = text
        reps = [
            (r"\bplease\b", "kripya"),
            (r"\bshow\b", "dikhao"),
            (r"\blist\b", "dikhao"),
            (r"\bfind\b", "dhundo"),
            (r"\btell me\b", "batao"),
            (r"\bexplain\b", "samjhao"),
            (r"\bhelp\b", "madad"),
            (r"\bthank you\b", "dhanyavaad"),
            (r"\binvoices?\b", "invoice"),
            (r"\bunpaid\b", "baaki"),
            (r"\btoday\b", "aaj"),
            (r"\btomorrow\b", "kal"),
        ]
        for pat, repl in reps:
            t = re.sub(pat, repl, t, flags=re.I)
        return t

    def _style_email(self, text: str) -> str:
        subject = self._style_short(text)[:80] or "Request"
        body = self._style_polite(text)
        return f"Subject: {subject}\n\nHello,\n\n{body}\n\nRegards"

    def _style_bullet(self, text: str) -> str:
        parts = re.split(r"[.;]\s+|\band then\b|\bthen\b|, and\s+", text, flags=re.I)
        parts = [p.strip(" .") for p in parts if p and p.strip(" .")]
        if len(parts) <= 1:
            # Split on commas for lists
            parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            parts = [text]
        return "\n".join(f"- {p[0].upper() + p[1:] if p else p}" for p in parts[:8])


def parse_search_command(text: str) -> Optional[str]:
    """Extract query from NL search-message commands."""
    lower = (text or "").strip()
    patterns = [
        r"^(?:search|find)\s+messages?\s+(?:for|about|with)?\s*(.+)$",
        r"^(?:message\s+search|search\s+chat|search\s+history)\s*:?\s*(.+)$",
        r"^(?:/search)\s+(.+)$",
        r"^find\s+(?:in\s+)?(?:chat|history|messages?)\s*:?\s*(.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, lower, flags=re.I)
        if m:
            return m.group(1).strip().strip("\"'")
    return None


def parse_vary_command(text: str) -> Optional[str]:
    """Extract text from NL vary/paraphrase commands."""
    lower = (text or "").strip()
    patterns = [
        r"^(?:vary|paraphrase|rephrase|rewrite)\s+(?:this\s+)?(?:message\s*)?:?\s*(.+)$",
        r"^(?:message\s+variations?|variations?\s+of)\s*:?\s*(.+)$",
        r"^(?:/vary)\s+(.+)$",
        r"^(?:give|show)\s+(?:me\s+)?(?:message\s+)?variations?\s+(?:of|for)\s+(.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, lower, flags=re.I | re.S)
        if m:
            return m.group(1).strip().strip("\"'")
    return None


def format_search_reply(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"Message search failed: {result.get('error')}"
    hits = result.get("hits") or []
    if not hits:
        return (
            f"No messages matched **{result.get('query')}**.\n"
            f"Tried variants: {', '.join((result.get('query_variations') or [])[:4])}"
        )
    lines = [
        f"Found **{result.get('total_hits', len(hits))}** message(s) for **{result.get('query')}** "
        f"(showing {len(hits)}):",
        "",
    ]
    for h in hits:
        snippet = (h.get("text") or "")[:160].replace("\n", " ")
        lines.append(
            f"- `#{h.get('index')}` **{h.get('role')}**"
            + (f" · {h.get('route')}" if h.get("route") else "")
            + f" · score={h.get('score')}: {snippet}"
        )
    vars_ = result.get("query_variations") or []
    if len(vars_) > 1:
        lines.append("")
        lines.append("Query variations: " + "; ".join(f"`{v}`" for v in vars_[:5]))
    lines.append("")
    lines.append("_human_level_agi=false_")
    return "\n".join(lines)


def format_vary_reply(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"Message variation failed: {result.get('error')}"
    lines = [
        f"Variations of: **{(result.get('original') or '')[:120]}**",
        "",
    ]
    for v in result.get("variations") or []:
        lines.append(f"**{v.get('style')}**")
        lines.append(v.get("text") or "")
        lines.append("")
    lines.append("_human_level_agi=false_")
    return "\n".join(lines).rstrip() + "\n"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]{2,}", (text or "").lower())


def _as_row(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return {
            "role": str(raw.get("role") or "user"),
            "text": str(raw.get("text") or ""),
            "route": str(raw.get("route") or ""),
            "ts": raw.get("ts") or 0,
        }
    return {
        "role": str(getattr(raw, "role", "user") or "user"),
        "text": str(getattr(raw, "text", "") or ""),
        "route": str(getattr(raw, "route", "") or ""),
        "ts": getattr(raw, "ts", 0) or 0,
    }
