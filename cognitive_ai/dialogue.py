"""Symbolic natural-language dialogue — pattern routing, no LLM."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from .knowledge import Triple

if TYPE_CHECKING:
    from .brain import Brain


HELP_TEXT = """I am a symbolic Cognitive OS brain (no LLM). I understand:

Conversation
  • greetings, help, who/what I am, what I know, capabilities

Knowledge questions
  • "can tweety fly?"  ·  "is opus a bird?"  ·  "does bird have wings?"
  • "tell me about tweety"  ·  "what do you know about cats?"
  • "why can tweety fly?" / "why tweety can fly"

Actions
  • "feed tweety" / "feed the bird" → set goal and think
  • "make an omelette" (kitchen domain) → plan & act
  • think / reason / prove s p o / assert s p o
  • domain animals|blocks|kitchen

Commands
  assert <s> <p> <o>   goal <desc> | <s> <p> <o>
  query [s [p [o]]]    why <s> <p> <o>   status
"""


def respond(brain: "Brain", message: str) -> str:
    """Produce a helpful symbolic reply for free-form chat text."""
    msg = message.strip()
    if not msg:
        return "Say something — try “help”, “who are you?”, or “can tweety fly?”."

    lower = msg.lower().strip()
    lower = re.sub(r"[!?]+$", "", lower).strip()

    # --- greetings / meta -------------------------------------------------
    if _is_greeting(lower):
        st = brain.status()
        return (
            f"Hello. I am {brain.name}, a symbolic cognitive system "
            f"(domain={st['domain']}, {st['facts']} facts, {st['rules']} rules). "
            "Ask about entities I know, or say “help”."
        )

    if lower in ("help", "?", "commands", "how to use", "what should i say"):
        return HELP_TEXT

    if lower in ("thanks", "thank you", "thx", "ty"):
        return "You're welcome."

    if lower in ("bye", "goodbye", "see you", "exit"):
        return "Goodbye. Knowledge stays in memory until domain reload."

    # --- feed / cook style intents ----------------------------------------
    feed = _match_feed(lower, brain)
    if feed:
        return feed

    cook = _match_cook(lower, brain)
    if cook:
        return cook

    # --- why / can / is / does / about ------------------------------------
    why = _match_why(lower, brain)
    if why:
        return why

    qa = _match_capability_question(lower, brain)
    if qa:
        return qa

    about = _match_about(lower, brain)
    if about:
        return about

    # entity mention dump
    entity = _find_known_entity(lower, brain)
    if entity and len(lower.split()) <= 4:
        return _describe_entity(brain, entity)

    # Fall through to self-model introspective router
    ans = brain.self_model.answer(msg)
    if not ans.startswith("I did not match"):
        return ans

    # Last resort: search any token that matches a known entity
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", lower)
    known = []
    for t in tokens:
        if brain.kb.related(t) or brain.kb.query(t, "*", "*") or brain.kb.query("*", "*", t):
            known.append(t)
    if known:
        # Prefer longer / more specific
        entity = max(set(known), key=lambda x: (len(brain.kb.related(x)), len(x)))
        return (
            f"I don't parse that full sentence, but I know about **{entity}**:\n"
            + _describe_entity(brain, entity)
            + "\n\nTry: “can {0} fly?”, “is {0} a bird?”, or “tell me about {0}”.".format(
                entity
            )
        )

    st = brain.status()
    sample = [str(t) for t in brain.kb.facts()[:5]]
    return (
        "I couldn't interpret that as a symbolic query.\n"
        f"Current domain: {st['domain']} ({st['facts']} facts).\n"
        "Examples I understand:\n"
        "  • can tweety fly?\n"
        "  • tell me about opus\n"
        "  • feed tweety\n"
        "  • help\n"
        + ("Known sample facts:\n  " + "\n  ".join(sample) if sample else "")
    )


def _is_greeting(lower: str) -> bool:
    greetings = (
        "hi",
        "hello",
        "hey",
        "howdy",
        "good morning",
        "good evening",
        "good afternoon",
        "hi there",
        "hello there",
    )
    if lower in greetings:
        return True
    if lower.startswith(("hi ", "hello ", "hey ")):
        return True
    if "how are you" in lower:
        return True
    return False


def _match_feed(lower: str, brain: "Brain") -> Optional[str]:
    # feed tweety / feed the bird / feed opus
    m = re.search(r"\bfeed\s+(?:the\s+)?([a-z0-9_]+)\b", lower)
    if not m:
        return None
    target = m.group(1)
    if target in ("bird", "animal"):
        # pick a known bird instance
        for cand in ("tweety", "opus"):
            if brain.kb.query(cand, "is_a", "*") or brain.engine.backward_chain(
                Triple(cand, "is_a", "bird")
            ).success:
                target = cand
                break
    # Ensure needs food if possible
    brain.reason()
    if not brain.kb.has_fact(target, "needs", "food"):
        # try prove
        brain.engine.backward_chain(Triple(target, "needs", "food"))
        brain.reason()
    g = brain.set_goal(f"Feed {target}", target, "fed", "true")
    report = brain.think(steps=4)
    lines = [
        f"Goal set: Feed {target} ({g.goal_id}).",
        f"Think → {report.stopped_reason}",
    ]
    for c in report.cycles:
        for a in c.actions:
            lines.append(f"  {a}")
        for u in c.utterances:
            lines.append(f"  {u}")
    if brain.kb.has_fact(target, "fed", "true"):
        lines.append(f"Result: ({target} fed true) is now known.")
    return "\n".join(lines)


def _match_cook(lower: str, brain: "Brain") -> Optional[str]:
    if not any(w in lower for w in ("omelette", "omelet", "cook breakfast", "make breakfast")):
        return None
    if brain.kb.domain != "kitchen":
        brain.load_domain("kitchen")
        brain.reason()
        switched = "Switched to kitchen domain. "
    else:
        switched = ""
    g = brain.set_goal("Make omelette", "omelette", "status", "ready")
    report = brain.think(steps=8)
    lines = [
        f"{switched}Goal: Make omelette ({g.goal_id}).",
        f"Think → {report.stopped_reason}",
    ]
    for c in report.cycles:
        for a in c.actions:
            lines.append(f"  {a}")
        for u in c.utterances:
            lines.append(f"  {u}")
    if brain.kb.has_fact("omelette", "status", "ready"):
        lines.append("Result: omelette is ready.")
    return "\n".join(lines)


def _match_why(lower: str, brain: "Brain") -> Optional[str]:
    # why can tweety fly / why did you conclude (tweety can fly) / why tweety can fly
    m = re.search(
        r"why(?:\s+did\s+you\s+conclude)?\s+(?:can|does|is)?\s*"
        r"(?:\()?([a-z0-9_]+)\s+(can|cannot|has|is_a|is|needs|location)?\s*([a-z0-9_]+)?",
        lower,
    )
    # simpler patterns
    m2 = re.search(r"why\s+(?:can|does)\s+([a-z0-9_]+)\s+([a-z0-9_]+)", lower)
    if m2:
        s, o = m2.group(1), m2.group(2)
        pred = "can" if "can" in lower else "has"
        if "does" in lower and o in ("wings", "food"):
            pred = "has" if o == "wings" else "needs"
        brain.reason()
        # ensure derived
        brain.prove(s, pred, o)
        return brain.why(s, pred, o)

    m3 = re.search(r"why\s+([a-z0-9_]+)\s+(can|has|is_a|needs)\s+([a-z0-9_]+)", lower)
    if m3:
        s, p, o = m3.group(1), m3.group(2), m3.group(3)
        brain.prove(s, p, o)
        return brain.why(s, p, o)

    m4 = re.search(r"\(([a-z0-9_]+)\s+([a-z0-9_]+)\s+([a-z0-9_]+)\)", lower)
    if "why" in lower and m4:
        s, p, o = m4.group(1), m4.group(2), m4.group(3)
        brain.prove(s, p, o)
        return brain.why(s, p, o)
    return None


def _match_capability_question(lower: str, brain: "Brain") -> Optional[str]:
    # can X Y? / does X have Y? / is X a Y? / is X Y?
    m = re.match(r"can\s+([a-z0-9_]+)\s+([a-z0-9_]+)\s*$", lower)
    if m:
        return _prove_or_deny(brain, m.group(1), "can", m.group(2))

    m = re.match(r"does\s+([a-z0-9_]+)\s+have\s+([a-z0-9_]+)\s*$", lower)
    if m:
        return _prove_or_deny(brain, m.group(1), "has", m.group(2))

    m = re.match(r"does\s+([a-z0-9_]+)\s+need\s+([a-z0-9_]+)\s*$", lower)
    if m:
        return _prove_or_deny(brain, m.group(1), "needs", m.group(2))

    m = re.match(r"is\s+([a-z0-9_]+)\s+an?\s+([a-z0-9_]+)\s*$", lower)
    if m:
        return _prove_or_deny(brain, m.group(1), "is_a", m.group(2))

    m = re.match(r"is\s+([a-z0-9_]+)\s+([a-z0-9_]+)\s*$", lower)
    if m and m.group(2) not in ("ready", "true", "false"):
        # ambiguous: try is_a then other
        s, o = m.group(1), m.group(2)
        ans = _prove_or_deny(brain, s, "is_a", o, quiet_fail=True)
        if ans:
            return ans
        # try as object of any predicate
        hits = brain.kb.query(s, "*", o)
        if hits:
            return f"Yes — known as:\n" + "\n".join(f"  {t}" for t in hits[:10])
        return f"No — I cannot establish ({s} is_a {o})."

    # where is X
    m = re.match(r"where\s+is\s+(?:the\s+)?([a-z0-9_]+)\s*$", lower)
    if m:
        s = m.group(1)
        hits = brain.kb.query(s, "location", "*")
        if hits:
            return f"{s} location:\n" + "\n".join(f"  {t}" for t in hits)
        return f"I don't know where {s} is."

    return None


def _prove_or_deny(
    brain: "Brain",
    s: str,
    p: str,
    o: str,
    quiet_fail: bool = False,
) -> Optional[str]:
    if brain.kb.has_fact(s, p, o):
        return f"Yes — ({s} {p} {o}) is a known fact."
    proof = brain.prove(s, p, o)
    if proof["success"]:
        why = brain.why(s, p, o)
        derived = proof.get("derived") or []
        extra = f"\nDerived: {derived[0]}" if derived else ""
        return f"Yes — I can prove ({s} {p} {o}).{extra}\n{why}"
    # cannot
    if brain.kb.has_fact(s, "cannot", o) and p == "can":
        return f"No — I know ({s} cannot {o})."
    if quiet_fail:
        return None
    related = brain.kb.related(s)[:8]
    hint = ""
    if related:
        hint = "\nRelated facts:\n" + "\n".join(f"  {t}" for t in related)
    elif brain.kb.domain != "animals" and s in ("tweety", "opus", "felix"):
        hint = "\nTip: switch domain with “domain animals”."
    return f"No — I cannot establish ({s} {p} {o}).{hint}"


def _match_about(lower: str, brain: "Brain") -> Optional[str]:
    m = re.search(
        r"(?:tell me about|what do you know about|describe|who is|what is)\s+(?:the\s+)?([a-z0-9_]+)",
        lower,
    )
    if not m:
        return None
    return _describe_entity(brain, m.group(1))


def _find_known_entity(lower: str, brain: "Brain") -> Optional[str]:
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", lower)
    stop = {
        "a", "an", "the", "is", "are", "do", "does", "can", "what", "who", "how",
        "about", "tell", "me", "know", "please", "just", "show",
    }
    best = None
    best_n = 0
    for t in tokens:
        if t in stop:
            continue
        n = len(brain.kb.related(t))
        if n > best_n:
            best, best_n = t, n
    return best if best_n else None


def _describe_entity(brain: "Brain", entity: str) -> str:
    brain.reason()
    related = brain.kb.related(entity)
    # also query as object
    as_obj = brain.kb.query("*", "*", entity)
    seen = {t.as_tuple() for t in related}
    for t in as_obj:
        if t.as_tuple() not in seen:
            related.append(t)
    if not related:
        # try case variants
        return (
            f"I don't have facts about '{entity}' in domain '{brain.kb.domain}'.\n"
            "Try another name, or “domain animals”, or assert new facts."
        )
    lines = [f"What I know about {entity} ({len(related)} relations):"]
    for t in related[:25]:
        lines.append(f"  {t}")
    # highlight capabilities
    cans = [t for t in related if t.predicate == "can" and t.subject == entity]
    if cans:
        lines.append("Capabilities: " + ", ".join(t.object for t in cans))
    types = [t for t in related if t.predicate == "is_a" and t.subject == entity]
    if types:
        lines.append("Types: " + ", ".join(t.object for t in types))
    return "\n".join(lines)
