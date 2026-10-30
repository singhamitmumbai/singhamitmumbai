"""Intelligent internet search for the non-LLM brain (ChatGPT-level synthesis).

Pipeline:
  1. Rewrite / expand the query from question understanding
  2. Answer from local curated knowledge when possible
  3. Optional live fetch (DuckDuckGo Instant Answer + Wikipedia summary)
  4. Rank, fuse snippets, compose ChatGPT-style cited answers
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..response.chatgpt_style import ChatGPTStyleComposer


USER_AGENT = "CompleteAI-Finderbay/1.14 (non-LLM research; +https://localhost)"


# Curated offline answers for common open questions (honest, sourced as local_kb)
LOCAL_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "gravity": {
        "summary": (
            "Gravity is the attraction between masses. On Earth it gives weight and "
            "makes objects fall toward the ground. In Newton's view it is a force; "
            "in Einstein's general relativity it is curved spacetime."
        ),
        "topics": ["physics", "force"],
        "keywords": ["gravity", "gravitation", "why things fall", "newton gravity"],
        "bullets": [
            "Stronger for larger masses and closer distances",
            "Earth's surface gravity ≈ 9.8 m/s²",
            "Keeps planets in orbit around the Sun",
        ],
    },
    "photosynthesis": {
        "summary": (
            "Photosynthesis is how plants (and some bacteria/algae) convert light energy, "
            "water, and carbon dioxide into sugars and oxygen — the foundation of most food chains."
        ),
        "topics": ["biology"],
        "keywords": ["photosynthesis", "plants make food", "chlorophyll"],
        "bullets": [
            "Needs light, CO₂, and water",
            "Produces glucose (food) and oxygen",
            "Happens mainly in chloroplasts",
        ],
    },
    "water boiling point": {
        "summary": "At standard atmospheric pressure (1 atm), water boils at 100°C (212°F). At higher altitude the boiling point drops.",
        "topics": ["chemistry", "physics"],
        "keywords": ["boiling point of water", "water boil", "100 celsius"],
        "bullets": ["100°C at sea level", "Lower temperature at high altitude", "Raised by dissolved salt (slightly)"],
    },
    "speed of light": {
        "summary": "The speed of light in vacuum is exactly 299,792,458 m/s (about 299,792 km/s or ≈3×10^8 m/s). It is a universal constant denoted c.",
        "topics": ["physics"],
        "keywords": ["speed of light", "c light", "299792"],
        "bullets": ["≈ 3×10^8 m/s in vacuum", "Slower in glass/water", "Sets the cosmic speed limit for information"],
    },
    "python programming": {
        "summary": "Python is a high-level, readable programming language used for scripting, data science, web backends, automation, and AI tooling.",
        "topics": ["computing"],
        "keywords": ["what is python", "python language", "python programming"],
        "bullets": ["Dynamic typing and large standard library", "Popular for ML/data and scripting", "Indentation defines blocks"],
    },
    "artificial intelligence": {
        "summary": (
            "AI is software that perceives, reasons, plans, or acts. Complete AI combines "
            "specialist engines (tasks, vision, coding, search) and optional ChatGPT-class "
            "API chat — without claiming human-level AGI."
        ),
        "topics": ["computing"],
        "keywords": ["what is ai", "artificial intelligence", "machine learning"],
        "bullets": ["Narrow AI solves specific tasks", "LLMs are one kind of AI system", "Complete AI keeps human_level_agi=false"],
    },
    "internet": {
        "summary": "The Internet is a global network of networks that exchanges data using standard protocols (notably IP and TCP).",
        "topics": ["computing"],
        "keywords": ["what is the internet", "internet", "world wide web"],
        "bullets": ["Built from linked networks", "Uses IP addressing and routing", "The Web is one application on top of it"],
    },
    "sqlite": {
        "summary": "SQLite is an embedded SQL database engine stored in a local file; useful for on-device memory and FTS5 full-text search.",
        "topics": ["computing"],
        "keywords": ["sqlite", "what is sqlite"],
        "bullets": ["Zero-config file database", "Great for apps and agents", "Supports SQL and FTS5"],
    },
    "ocr": {
        "summary": "OCR (optical character recognition) converts images of text into machine-readable characters.",
        "topics": ["computing", "vision"],
        "keywords": ["what is ocr", "optical character recognition"],
        "bullets": ["Reads printed or handwritten text from images", "Used in document pipelines", "Accuracy depends on image quality"],
    },
    "thermal imaging": {
        "summary": "Thermal imaging cameras sense infrared radiation to visualize relative heat; absolute °C needs calibration.",
        "topics": ["vision", "sensors"],
        "keywords": ["thermal imaging", "thermography", "flir", "infrared camera"],
        "bullets": ["Sees heat differences, not visible color", "Useful for electrical/mechanical inspection", "Needs calibration for absolute temperatures"],
    },
    "machine learning": {
        "summary": "Machine learning lets systems improve at a task from data/examples instead of only hand-written rules.",
        "topics": ["computing"],
        "keywords": ["machine learning", "what is ml", "deep learning"],
        "bullets": ["Supervised, unsupervised, and reinforcement styles", "Needs data and evaluation", "Deep learning uses neural nets"],
    },
    "dna": {
        "summary": "DNA (deoxyribonucleic acid) stores genetic instructions in living organisms as a double helix of base pairs.",
        "topics": ["biology"],
        "keywords": ["what is dna", "deoxyribonucleic", "genes"],
        "bullets": ["Four bases: A, T, C, G", "Codes for proteins via genes", "Copied during cell division"],
    },
    "climate change": {
        "summary": "Climate change refers to long-term shifts in temperatures and weather patterns, largely driven today by human greenhouse-gas emissions.",
        "topics": ["science", "earth"],
        "keywords": ["climate change", "global warming", "greenhouse"],
        "bullets": ["CO₂ and methane trap heat", "Impacts weather, oceans, and ecosystems", "Mitigation + adaptation are both needed"],
    },
    "bitcoin": {
        "summary": "Bitcoin is a decentralized digital currency that records transactions on a public blockchain secured by proof-of-work mining.",
        "topics": ["computing", "finance"],
        "keywords": ["what is bitcoin", "btc", "cryptocurrency bitcoin"],
        "bullets": ["Peer-to-peer value transfer", "Limited supply issuance schedule", "Volatile price; not legal tender everywhere"],
    },
    "http": {
        "summary": "HTTP is the application protocol that browsers and servers use to request and transfer web pages and APIs.",
        "topics": ["computing"],
        "keywords": ["what is http", "https", "hypertext transfer"],
        "bullets": ["Request/response model", "HTTPS adds TLS encryption", "Methods include GET, POST, PUT, DELETE"],
    },
    "algorithm": {
        "summary": "An algorithm is a finite, well-defined sequence of steps to solve a problem or compute a result.",
        "topics": ["computing"],
        "keywords": ["what is an algorithm", "algorithms"],
        "bullets": ["Input → steps → output", "Judged by correctness and efficiency", "Foundation of computer science"],
    },
    "electricity": {
        "summary": "Electricity is the flow or presence of electric charge; useful power is often the flow of electrons through a conductor (current).",
        "topics": ["physics"],
        "keywords": ["what is electricity", "electric current", "voltage"],
        "bullets": ["Voltage is electrical pressure", "Current is charge flow", "Power ≈ voltage × current"],
    },
    "solar system": {
        "summary": "The Solar System is the Sun and everything bound to it by gravity — planets, moons, asteroids, comets, and dust.",
        "topics": ["astronomy"],
        "keywords": ["solar system", "planets", "sun planets"],
        "bullets": ["Eight planets orbit the Sun", "Earth is the third planet", "Kuiper belt and Oort cloud beyond the planets"],
    },
}


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    source: str  # local_kb|ddg|wikipedia|cache
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    ok: bool
    query: str
    rewritten_query: str
    answer: str
    hits: list[SearchHit] = field(default_factory=list)
    method: str = "none"
    confidence: float = 0.0
    live: bool = False
    notes: list[str] = field(default_factory=list)
    question: Optional[dict[str, Any]] = None
    related_queries: list[str] = field(default_factory=list)
    style: str = "helpful"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hits"] = [h if isinstance(h, dict) else h.to_dict() for h in self.hits]
        return d


class IntelligentWebSearch:
    """Smarter search: rewrite → expand → local → live → ChatGPT-style synthesize."""

    def __init__(self, *, allow_live: bool = True, timeout: float = 4.0) -> None:
        self.allow_live = allow_live
        self.timeout = timeout
        self.cache: dict[str, SearchResult] = {}
        self.composer = ChatGPTStyleComposer()

    def search(
        self,
        query: str,
        *,
        question: Optional[dict[str, Any]] = None,
        allow_live: Optional[bool] = None,
        max_hits: int = 5,
        style: str = "helpful",
    ) -> SearchResult:
        raw = (query or "").strip()
        if not raw:
            return SearchResult(False, "", "", "Empty query.", method="empty", confidence=0.0)

        rewritten = self.rewrite_query(raw, question=question)
        expansions = self.expand_queries(rewritten, question=question)
        cache_key = f"{rewritten.lower()}|{style}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            cached.notes = list(cached.notes) + ["cache_hit"]
            return cached

        # 1) Local knowledge
        local = self._local_lookup(rewritten, raw, style=style)
        if local and local.confidence >= 0.7:
            local.related_queries = self._related(rewritten)
            local.question = question
            local.style = style
            self.cache[cache_key] = local
            return local

        live_ok = self.allow_live if allow_live is None else allow_live
        hits: list[SearchHit] = list(local.hits) if local else []
        notes: list[str] = list(local.notes) if local else []
        answer_parts: list[str] = []
        local_bullets: list[str] = []
        if local and local.answer:
            answer_parts.append(re.sub(r"\s*\(Local knowledge base\.\)\s*$", "", local.answer).strip())
            notes.append("local_partial")

        # 2) Live providers (try primary + one expansion)
        if live_ok:
            for qtry in [rewritten] + expansions[:1]:
                ddg = self._duckduckgo(qtry)
                if ddg:
                    hits.extend(ddg)
                    notes.append(f"ddg_live:{qtry[:40]}")
                wiki = self._wikipedia(qtry)
                if wiki:
                    hits.append(wiki)
                    notes.append(f"wikipedia_live:{qtry[:40]}")
                if hits:
                    break

        hits = self._rank_hits(hits, rewritten)[:max_hits]
        if not hits and not answer_parts:
            result = SearchResult(
                ok=False,
                query=raw,
                rewritten_query=rewritten,
                answer=(
                    "I could not find a reliable answer offline, and live search "
                    "returned nothing (or network is unavailable).\n\n"
                    "Try a clearer entity question, check connectivity, "
                    "or rephrase with a concrete noun."
                ),
                method="miss",
                confidence=0.15,
                live=bool(live_ok),
                notes=notes + ["no_hits"],
                question=question,
                related_queries=self._related(rewritten),
                style=style,
            )
            return result

        # Synthesize ChatGPT-style answer
        core = ""
        if answer_parts:
            core = answer_parts[0]
        if hits:
            top = hits[0]
            if not core or len(core) < 40:
                core = top.snippet
            elif top.source != "local_kb" and top.snippet and top.snippet not in core:
                # Fuse: local + top live snippet briefly
                core = core.rstrip(".") + ". " + top.snippet[:280]

        composed = self.composer.compose_search(
            query=raw,
            answer_core=core,
            hits=[h.to_dict() for h in hits],
            rewritten=rewritten,
            confidence=0.55 + 0.1 * min(3, len(hits)),
        )
        conf = composed.confidence
        if any(h.source == "local_kb" for h in hits):
            conf = max(conf, 0.78)
        if any(h.source == "wikipedia" for h in hits):
            conf = max(conf, 0.72)

        # Style variants
        answer_text = composed.text
        if style == "eli5" and core:
            answer_text = self.composer.compose(
                topic=rewritten,
                ask_form="define",
                summary=core[:400],
                style="eli5",
                confidence=conf,
                sources=[h.url for h in hits[:2] if h.url],
            ).text
        elif style == "concise" and core:
            answer_text = core.split(".")[0].strip() + "."

        result = SearchResult(
            ok=True,
            query=raw,
            rewritten_query=rewritten,
            answer=answer_text,
            hits=hits,
            method="hybrid" if local and hits else ("local_kb" if local else "live"),
            confidence=min(0.93, conf),
            live=any(h.source in ("ddg", "wikipedia") for h in hits),
            notes=notes,
            question=question,
            related_queries=self._related(rewritten),
            style=style,
        )
        self.cache[cache_key] = result
        return result

    def rewrite_query(self, text: str, *, question: Optional[dict[str, Any]] = None) -> str:
        """Turn a natural question into a sharper search query."""
        q = text.strip()
        q = re.sub(
            r"^(?:please\s+)?(?:search(?:\s+the\s+web)?(?:\s+for)?|look\s+up|google|find\s+online|"
            r"what\s+is|what's|whats|who\s+is|tell\s+me\s+about|explain|summarize|eli5)\s+",
            "",
            q,
            flags=re.I,
        )
        q = re.sub(r"\b(?:in simple terms|simply|in detail|briefly|please)\b", " ", q, flags=re.I)
        q = q.strip(" ?.")
        if question:
            focus = question.get("focus") or []
            if isinstance(focus, list) and focus and len(" ".join(str(f) for f in focus)) >= 3:
                if len(q.split()) <= 2 or question.get("ask_form") in ("define", "causal", "howto", "search"):
                    q = " ".join(str(f) for f in focus[:2])
            ask = question.get("ask_form")
            wh = question.get("wh_type")
            if ask == "define" or wh == "what":
                if "definition" not in q.lower():
                    q = f"{q}"
            elif ask == "howto":
                q = f"how to {re.sub(r'^how to\s+', '', q, flags=re.I)}"
            elif ask == "causal" or wh == "why":
                if not q.lower().startswith("why"):
                    q = f"why {q}"
        q = re.sub(r"\b(?:please|kindly|exactly|basically)\b", " ", q, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip(" ?.")
        return q or text.strip()

    def expand_queries(
        self, rewritten: str, *, question: Optional[dict[str, Any]] = None
    ) -> list[str]:
        """Multi-query expansion for smarter recall."""
        base = rewritten.strip()
        if not base:
            return []
        out = []
        ask = (question or {}).get("ask_form")
        if ask == "define" or True:
            out.append(f"{base} definition")
            out.append(f"{base} overview")
        if ask == "howto":
            out.append(f"how to {base} steps")
        if ask == "causal":
            out.append(f"why {base} explanation")
        # Dedup
        seen = {base.lower()}
        uniq = []
        for q in out:
            k = q.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(q)
        return uniq[:3]

    def _related(self, query: str) -> list[str]:
        q = query.strip()
        return [
            f"Explain {q} simply",
            f"Compare {q} with a related concept",
            f"Search the web for {q} applications",
        ]

    def _local_lookup(self, rewritten: str, raw: str, *, style: str = "helpful") -> Optional[SearchResult]:
        blob = f"{rewritten} {raw}".lower()
        best_key = None
        best_score = 0.0
        for key, fact in LOCAL_KNOWLEDGE.items():
            score = 0.0
            if key in blob:
                score += 2.0
            for kw in fact.get("keywords") or []:
                if kw in blob:
                    score += 1.5
            keys = set(re.findall(r"[a-z0-9]+", key))
            toks = set(re.findall(r"[a-z0-9]+", blob))
            score += 0.4 * len(keys & toks)
            if score > best_score:
                best_score = score
                best_key = key
        if not best_key or best_score < 1.2:
            return None
        fact = LOCAL_KNOWLEDGE[best_key]
        hit = SearchHit(
            title=best_key.title(),
            url="local://knowledge/" + best_key.replace(" ", "_"),
            snippet=fact["summary"],
            source="local_kb",
            score=best_score,
        )
        composed = self.composer.compose(
            topic=best_key,
            ask_form="define",
            summary=fact["summary"],
            extra_bullets=list(fact.get("bullets") or []),
            style=style if style in ("eli5", "detailed", "concise", "helpful") else "helpful",
            confidence=min(0.92, 0.65 + 0.08 * best_score),
            sources=["local knowledge base"],
        )
        return SearchResult(
            ok=True,
            query=raw,
            rewritten_query=rewritten,
            answer=composed.text,
            hits=[hit],
            method="local_kb",
            confidence=composed.confidence,
            live=False,
            notes=["local_kb", "chatgpt_style"],
            related_queries=self._related(best_key),
            style=style,
        )

    def _duckduckgo(self, query: str) -> list[SearchHit]:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "no_redirect": 1, "no_html": 1, "skip_disambig": 1}
        )
        data = self._http_json(url)
        if not data:
            return []
        hits: list[SearchHit] = []
        abstract = (data.get("AbstractText") or "").strip()
        abs_url = (data.get("AbstractURL") or "").strip()
        heading = (data.get("Heading") or query).strip()
        if abstract:
            hits.append(
                SearchHit(
                    title=heading or "DuckDuckGo Abstract",
                    url=abs_url or "https://duckduckgo.com/?" + urllib.parse.urlencode({"q": query}),
                    snippet=abstract[:500],
                    source="ddg",
                    score=2.5,
                )
            )
        for topic in (data.get("RelatedTopics") or [])[:5]:
            if not isinstance(topic, dict):
                continue
            text = (topic.get("Text") or "").strip()
            first = topic.get("FirstURL") or ""
            if text and first:
                hits.append(
                    SearchHit(
                        title=text.split(" - ")[0][:80],
                        url=first,
                        snippet=text[:400],
                        source="ddg",
                        score=1.5,
                    )
                )
        return hits

    def _wikipedia(self, query: str) -> Optional[SearchHit]:
        title = query.strip().replace(" ", "_")
        title = re.sub(r"_+(?:definition|overview)$", "", title, flags=re.I)
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
        data = self._http_json(url)
        if not data or data.get("type") == "disambiguation":
            search_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
                {"action": "opensearch", "search": query, "limit": 1, "namespace": 0, "format": "json"}
            )
            sdata = self._http_json(search_url)
            if not sdata or len(sdata) < 4 or not sdata[1]:
                return None
            title = sdata[1][0]
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(
                title.replace(" ", "_")
            )
            data = self._http_json(url)
            if not data:
                return None
        extract = (data.get("extract") or "").strip()
        page_url = (data.get("content_urls") or {}).get("desktop", {}).get("page") or data.get(
            "canonicalurl"
        ) or ""
        if not extract:
            return None
        return SearchHit(
            title=data.get("title") or query,
            url=page_url or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(str(title))}",
            snippet=extract[:500],
            source="wikipedia",
            score=2.2,
        )

    def _rank_hits(self, hits: list[SearchHit], query: str) -> list[SearchHit]:
        qtoks = set(re.findall(r"[a-z0-9]+", query.lower()))
        ranked: list[SearchHit] = []
        seen_urls: set[str] = set()
        for h in hits:
            if h.url in seen_urls:
                continue
            seen_urls.add(h.url)
            boost = h.score
            ht = set(re.findall(r"[a-z0-9]+", (h.title + " " + h.snippet).lower()))
            boost += 0.3 * len(qtoks & ht)
            if h.source == "local_kb":
                boost += 0.5
            if h.source == "wikipedia":
                boost += 0.3
            # Prefer longer informative snippets
            boost += min(0.4, len(h.snippet) / 800.0)
            ranked.append(SearchHit(h.title, h.url, h.snippet, h.source, round(boost, 3)))
        ranked.sort(key=lambda x: -x.score)
        return ranked

    def _http_json(self, url: str) -> Optional[Any]:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
        except Exception:  # noqa: BLE001 — network optional
            return None
