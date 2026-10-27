"""ChatGPT-style answer composition for the non-LLM brain.

Turns structured KB triples, search hits, and intent frames into natural,
helpful prose — introductions, explanations, bullets, caveats, and follow-ups —
without calling an LLM.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ChatGPTStyleAnswer:
    text: str
    style: str = "helpful"  # helpful|eli5|detailed|concise|steps|compare
    sections: list[str] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "chatgpt_style"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Predicate → natural English gloss
PRED_GLOSS = {
    "is_a": "is a kind of",
    "can": "can",
    "cannot": "cannot",
    "has": "has",
    "needs": "needs",
    "used_for": "is used for",
    "part_of": "is part of",
    "lives_in": "lives in",
    "is_in": "is found in",
    "is_in_continent": "is found in",
    "made_of": "is made of",
    "causes": "can cause",
    "capable_of": "is capable of",
    "at_location": "is typically found at",
    "has_property": "is typically",
    "desires": "typically wants",
    "requires": "requires",
}


def _parse_triple(t: Any) -> Optional[tuple[str, str, str]]:
    s = str(t).strip()
    m = re.match(r"^\(?\s*([^,\s()]+)\s+([^\s()]+)\s+(.+?)\s*\)?$", s)
    if m:
        return m.group(1), m.group(2), m.group(3).rstrip(")")
    # Triple-like object
    if hasattr(t, "subject") and hasattr(t, "predicate"):
        return str(t.subject), str(t.predicate), str(getattr(t, "object", "*"))
    return None


def _gloss(pred: str) -> str:
    return PRED_GLOSS.get(pred.lower(), pred.replace("_", " "))


def triples_to_sentences(triples: list[Any], *, limit: int = 8) -> list[str]:
    """Convert symbolic triples into readable English sentences."""
    out: list[str] = []
    seen: set[str] = set()
    for t in triples[: limit * 2]:
        parsed = _parse_triple(t)
        if not parsed:
            continue
        subj, pred, obj = parsed
        if obj in ("*", "?"):
            continue
        key = f"{subj}|{pred}|{obj}".lower()
        if key in seen:
            continue
        seen.add(key)
        gloss = _gloss(pred)
        if pred.lower() == "cannot":
            sent = f"{subj.capitalize()} cannot {obj}."
        elif pred.lower() == "can":
            sent = f"{subj.capitalize()} can {obj}."
        elif pred.lower() == "is_a":
            article = "an" if obj[:1].lower() in "aeiou" else "a"
            sent = f"{subj.capitalize()} is {article} {obj}."
        elif pred.lower() == "used_for":
            sent = f"{subj.capitalize()} is used for {obj}."
        else:
            sent = f"{subj.capitalize()} {_gloss(pred)} {obj}."
        out.append(sent)
        if len(out) >= limit:
            break
    return out


class ChatGPTStyleComposer:
    """Compose ChatGPT-like answers from non-LLM structured evidence."""

    def compose(
        self,
        *,
        topic: str = "",
        ask_form: str = "define",
        triples: Optional[list[Any]] = None,
        facts: Optional[list[str]] = None,
        summary: Optional[str] = None,
        style: str = "helpful",
        sources: Optional[list[str]] = None,
        confidence: float = 0.7,
        yesno: Optional[bool] = None,
        extra_bullets: Optional[list[str]] = None,
    ) -> ChatGPTStyleAnswer:
        triples = triples or []
        facts = facts or []
        topic_clean = (topic or "this").strip().strip("?.")
        sentences = triples_to_sentences(triples)
        bullets = list(extra_bullets or [])
        bullets.extend(facts)
        if not bullets and sentences:
            bullets = [s.rstrip(".") for s in sentences]

        follow = self._follow_ups(topic_clean, ask_form)

        if ask_form == "yesno" and yesno is not None:
            return self._yesno(topic_clean, yesno, bullets, sentences, style, confidence, follow)

        if ask_form in ("howto",):
            return self._howto(topic_clean, bullets, style, confidence, follow, summary)

        if ask_form in ("compare",):
            return self._compare(topic_clean, bullets, style, confidence, follow, summary)

        if ask_form in ("causal",):
            return self._causal(topic_clean, bullets, sentences, style, confidence, follow, summary)

        if ask_form in ("list",):
            return self._list(topic_clean, bullets, style, confidence, follow, summary)

        if style == "eli5":
            return self._eli5(topic_clean, bullets, sentences, summary, confidence, follow, sources)

        if style == "concise":
            return self._concise(topic_clean, bullets, sentences, summary, confidence, follow, sources)

        return self._define(
            topic_clean, bullets, sentences, summary, style, confidence, follow, sources
        )

    def compose_search(
        self,
        *,
        query: str,
        answer_core: str,
        hits: Optional[list[dict[str, Any]]] = None,
        rewritten: str = "",
        confidence: float = 0.7,
    ) -> ChatGPTStyleAnswer:
        hits = hits or []
        topic = rewritten or query
        parts: list[str] = []
        parts.append(f"**{topic.strip().title() if len(topic.split()) <= 4 else topic.strip()}**")
        core = (answer_core or "").strip()
        # Drop trailing local-kb marker for cleaner prose, re-add as note
        core = re.sub(r"\s*\(Local knowledge base\.\)\s*$", "", core).strip()
        if core:
            parts.append("")
            parts.append(core)
        if hits:
            cites = []
            for h in hits[:3]:
                title = h.get("title") or "Source"
                url = h.get("url") or ""
                src = h.get("source") or ""
                if url.startswith("local://"):
                    cites.append(f"- {title} (local knowledge)")
                elif url:
                    cites.append(f"- {title}: {url}" + (f" [{src}]" if src else ""))
                else:
                    cites.append(f"- {title}")
            if cites:
                parts.append("")
                parts.append("**Sources**")
                parts.extend(cites)
        parts.append("")
        parts.append(
            "Want a simpler explanation, more detail, or a related comparison? Just ask."
        )
        text = "\n".join(parts)
        return ChatGPTStyleAnswer(
            text=text,
            style="helpful",
            sections=["summary", "sources"],
            follow_ups=[
                f"Explain {topic} simply",
                f"What else is related to {topic}?",
            ],
            confidence=confidence,
            method="chatgpt_style_search",
        )

    def compose_writing(
        self,
        *,
        kind: str,
        topic: str,
        confidence: float = 0.65,
    ) -> ChatGPTStyleAnswer:
        """Template writing assists (poem/outline/brainstorm/email) without an LLM."""
        t = topic.strip() or "your topic"
        if kind == "poem":
            text = (
                f"Here is a short poem about {t}:\n\n"
                f"  Quiet thoughts of {t},\n"
                f"  gathered carefully in light —\n"
                f"  a small clear answer.\n\n"
                "Want a longer poem, a different mood, or a rhyme scheme? Say the style."
            )
        elif kind == "email":
            text = (
                f"Draft email about {t}:\n\n"
                f"Subject: Regarding {t}\n\n"
                f"Hi,\n\nI wanted to follow up on {t}. "
                "Please let me know a good time to discuss next steps.\n\nThanks,\n[Your name]\n\n"
                "Tell me tone (formal/casual) or key points to include and I will refine."
            )
        elif kind == "outline":
            text = (
                f"Outline for '{t}':\n"
                f"1. Context — why {t} matters\n"
                f"2. Core ideas — main points about {t}\n"
                "3. Examples or evidence\n"
                "4. Practical next steps\n"
                "5. Open questions / risks\n\n"
                "Ask me to expand any section into paragraphs."
            )
        elif kind == "brainstorm":
            text = (
                f"Brainstorm ideas for '{t}':\n"
                f"- Clarify the goal of {t} in one sentence\n"
                f"- List constraints (time, money, tools)\n"
                f"- Generate 5 options, then pick the safest small start\n"
                f"- Measure a simple success signal\n"
                f"- Iterate with feedback\n\n"
                "Say a domain (business, coding, study) for more specific ideas."
            )
        elif kind == "pros_cons":
            text = (
                f"Pros and cons of {t}:\n\n"
                f"**Pros**\n- Potential upside depending on your goals around {t}\n"
                "- Can be started with a small experiment\n"
                "- Easy to measure early feedback\n\n"
                f"**Cons**\n- Needs clear constraints to avoid wasted effort\n"
                "- Trade-offs depend on context you have not specified yet\n"
                "- Risk of over-general advice without your details\n\n"
                "Share your constraints and I will tailor this."
            )
        else:
            text = (
                f"Here is a structured take on '{t}'. "
                "Ask for an outline, pros/cons, steps, or a short draft."
            )
        return ChatGPTStyleAnswer(
            text=text,
            style="helpful",
            sections=["draft"],
            follow_ups=[f"Make the {kind} longer", f"Make it more formal"],
            confidence=confidence,
            method=f"chatgpt_style_{kind}",
        )

    def compose_math(self, expression: str, value: Any, *, ok: bool) -> ChatGPTStyleAnswer:
        if not ok:
            return ChatGPTStyleAnswer(
                text=(
                    f"I could not safely evaluate `{expression}`. "
                    "Try a simple arithmetic expression like `12 * (3 + 4)`."
                ),
                style="concise",
                confidence=0.3,
                method="chatgpt_style_math",
            )
        return ChatGPTStyleAnswer(
            text=f"**{expression}** = **{value}**",
            style="concise",
            sections=["result"],
            follow_ups=["Calculate another expression"],
            confidence=0.95,
            method="chatgpt_style_math",
        )

    # --- internal templates -------------------------------------------------

    def _define(
        self,
        topic: str,
        bullets: list[str],
        sentences: list[str],
        summary: Optional[str],
        style: str,
        confidence: float,
        follow: list[str],
        sources: Optional[list[str]],
    ) -> ChatGPTStyleAnswer:
        parts: list[str] = []
        if summary:
            parts.append(summary.strip())
        else:
            parts.append(
                f"**{topic.capitalize()}** — here is a clear, practical overview:"
            )
        if bullets:
            parts.append("")
            for b in bullets[:8]:
                line = b if b.startswith("-") or b.startswith("•") else f"- {b}"
                parts.append(line)
        elif sentences:
            parts.append("")
            parts.append(" ".join(sentences[:5]))
        else:
            parts.append(
                f"I have limited local detail on '{topic}'. "
                "You can ask me to search the web for a richer answer."
            )
        if style == "detailed" and sentences:
            parts.append("")
            parts.append("**In more detail**")
            parts.extend(f"- {s}" for s in sentences[:8])
        if sources:
            parts.append("")
            parts.append("**Sources**: " + "; ".join(sources[:3]))
        parts.append("")
        parts.append(self._closing(topic))
        return ChatGPTStyleAnswer(
            "\n".join(parts),
            style=style,
            sections=["overview", "points"],
            follow_ups=follow,
            confidence=confidence,
        )

    def _eli5(
        self,
        topic: str,
        bullets: list[str],
        sentences: list[str],
        summary: Optional[str],
        confidence: float,
        follow: list[str],
        sources: Optional[list[str]],
    ) -> ChatGPTStyleAnswer:
        body = summary or (" ".join(sentences[:3]) if sentences else "")
        if not body and bullets:
            body = bullets[0]
        if not body:
            body = f"{topic} is a thing people ask about — I need a bit more context."
        # Simplify a bit
        body = re.sub(r"\([^)]*\)", "", body)
        text = (
            f"Simple version: **{topic}**.\n\n"
            f"{body}\n\n"
            "Think of it like a plain idea first; details can come after.\n"
            + (f"\nSources: {'; '.join(sources[:2])}" if sources else "")
        )
        return ChatGPTStyleAnswer(
            text.strip(),
            style="eli5",
            sections=["eli5"],
            follow_ups=follow,
            confidence=confidence,
            method="chatgpt_style_eli5",
        )

    def _concise(
        self,
        topic: str,
        bullets: list[str],
        sentences: list[str],
        summary: Optional[str],
        confidence: float,
        follow: list[str],
        sources: Optional[list[str]],
    ) -> ChatGPTStyleAnswer:
        if summary:
            text = summary.strip()
        elif sentences:
            text = sentences[0]
        elif bullets:
            text = bullets[0]
        else:
            text = f"No short answer found for {topic}."
        return ChatGPTStyleAnswer(
            text,
            style="concise",
            follow_ups=follow,
            confidence=confidence,
            method="chatgpt_style_concise",
        )

    def _howto(
        self,
        topic: str,
        bullets: list[str],
        style: str,
        confidence: float,
        follow: list[str],
        summary: Optional[str],
    ) -> ChatGPTStyleAnswer:
        steps = bullets[:6] if bullets else [
            "Clarify the exact goal and constraints",
            "Gather the required materials or information",
            "Do the smallest safe first action",
            "Check the result against your goal",
            "Iterate and fix issues",
        ]
        parts = [f"Here is a practical way to **{topic}**:", ""]
        if summary:
            parts.append(summary)
            parts.append("")
        for i, s in enumerate(steps, 1):
            s_clean = re.sub(r"^[-•\d.)\s]+", "", s).strip()
            parts.append(f"{i}. {s_clean}")
        parts.append("")
        parts.append("If this maps to Finderbay, vision, or coding tools, ask me that task next.")
        return ChatGPTStyleAnswer(
            "\n".join(parts),
            style="steps",
            sections=["steps"],
            follow_ups=follow,
            confidence=confidence,
            method="chatgpt_style_howto",
        )

    def _compare(
        self,
        topic: str,
        bullets: list[str],
        style: str,
        confidence: float,
        follow: list[str],
        summary: Optional[str],
    ) -> ChatGPTStyleAnswer:
        parts = [f"**Comparison — {topic}**", ""]
        if summary:
            parts.append(summary)
            parts.append("")
        if bullets:
            parts.extend(f"- {b}" for b in bullets[:10])
        else:
            parts.append("Share two items (e.g. 'compare dog and cat') for a sharper contrast.")
        parts.append("")
        parts.append("Want a table-style pros/cons breakdown? Ask for pros and cons.")
        return ChatGPTStyleAnswer(
            "\n".join(parts),
            style="compare",
            sections=["compare"],
            follow_ups=follow,
            confidence=confidence,
            method="chatgpt_style_compare",
        )

    def _causal(
        self,
        topic: str,
        bullets: list[str],
        sentences: list[str],
        style: str,
        confidence: float,
        follow: list[str],
        summary: Optional[str],
    ) -> ChatGPTStyleAnswer:
        parts = [f"**Why — {topic}**", ""]
        if summary:
            parts.append(summary)
        elif sentences:
            parts.append(" ".join(sentences[:4]))
        elif bullets:
            parts.extend(f"- {b}" for b in bullets[:6])
        else:
            parts.append(
                "I do not have a causal explanation in local knowledge yet. "
                "Try 'search the web for why …' for live sources."
            )
        parts.append("")
        parts.append("I can also explain this more simply or in more detail.")
        return ChatGPTStyleAnswer(
            "\n".join(parts),
            style=style,
            sections=["why"],
            follow_ups=follow,
            confidence=confidence,
            method="chatgpt_style_causal",
        )

    def _list(
        self,
        topic: str,
        bullets: list[str],
        style: str,
        confidence: float,
        follow: list[str],
        summary: Optional[str],
    ) -> ChatGPTStyleAnswer:
        parts = [f"**{topic.capitalize()}**", ""]
        if summary:
            parts.append(summary)
            parts.append("")
        items = bullets or ["(no items found locally)"]
        parts.extend(f"- {b}" for b in items[:12])
        return ChatGPTStyleAnswer(
            "\n".join(parts),
            style=style,
            sections=["list"],
            follow_ups=follow,
            confidence=confidence,
            method="chatgpt_style_list",
        )

    def _yesno(
        self,
        topic: str,
        yesno: bool,
        bullets: list[str],
        sentences: list[str],
        style: str,
        confidence: float,
        follow: list[str],
    ) -> ChatGPTStyleAnswer:
        lead = "Yes." if yesno else "No."
        detail = ""
        if sentences:
            detail = " " + " ".join(sentences[:3])
        elif bullets:
            detail = " " + bullets[0]
        text = f"**{lead}**{detail}\n\n" + self._closing(topic)
        return ChatGPTStyleAnswer(
            text,
            style=style,
            sections=["yesno"],
            follow_ups=follow,
            confidence=confidence,
            method="chatgpt_style_yesno",
        )

    def _follow_ups(self, topic: str, ask_form: str) -> list[str]:
        t = topic or "this"
        base = [
            f"Explain {t} simply",
            f"Give more detail on {t}",
        ]
        if ask_form == "howto":
            base.append(f"What materials do I need for {t}?")
        if ask_form == "compare":
            base.append("Show pros and cons")
        if ask_form == "define":
            base.append(f"Search the web for {t}")
        return base[:4]

    def _closing(self, topic: str) -> str:
        return (
            f"Ask a follow-up anytime — e.g. explain {topic} simply, "
            "compare it with something else, or search the web for more."
        )


def compose_from_chat_payload(
    *,
    topic: str,
    ask_form: str,
    triples: Optional[list[Any]] = None,
    summary: Optional[str] = None,
    style: str = "helpful",
    confidence: float = 0.7,
    yesno: Optional[bool] = None,
) -> ChatGPTStyleAnswer:
    return ChatGPTStyleComposer().compose(
        topic=topic,
        ask_form=ask_form,
        triples=triples,
        summary=summary,
        style=style,
        confidence=confidence,
        yesno=yesno,
    )
