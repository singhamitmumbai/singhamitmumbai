"""Intelligent internet search for the non-LLM brain.

Pipeline:
  1. Rewrite / normalize the query from question understanding
  2. Answer from local curated knowledge when possible
  3. Optional live fetch (DuckDuckGo Instant Answer + Wikipedia summary)
  4. Rank, cite sources, attach confidence — never invent URLs
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


USER_AGENT = "CompleteAI-Finderbay/1.12 (non-LLM research; +https://localhost)"


# Curated offline answers for common open questions (honest, sourced as local_kb)
LOCAL_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "gravity": {
        "summary": "Gravity is the attraction between masses; on Earth it gives weight and makes objects fall toward the ground.",
        "topics": ["physics", "force"],
        "keywords": ["gravity", "gravitation", "why things fall"],
    },
    "photosynthesis": {
        "summary": "Photosynthesis is how plants convert light, water, and carbon dioxide into sugars and oxygen.",
        "topics": ["biology"],
        "keywords": ["photosynthesis", "plants make food"],
    },
    "water boiling point": {
        "summary": "At standard atmospheric pressure, water boils at 100°C (212°F).",
        "topics": ["chemistry", "physics"],
        "keywords": ["boiling point of water", "water boil"],
    },
    "speed of light": {
        "summary": "The speed of light in vacuum is about 299,792 km/s (≈3×10^8 m/s).",
        "topics": ["physics"],
        "keywords": ["speed of light", "c light"],
    },
    "python programming": {
        "summary": "Python is a high-level programming language used for scripting, data, web, and AI tooling.",
        "topics": ["computing"],
        "keywords": ["what is python", "python language"],
    },
    "artificial intelligence": {
        "summary": "AI is software that perceives, reasons, plans, or acts; Complete AI combines specialist engines without claiming human-level AGI.",
        "topics": ["computing"],
        "keywords": ["what is ai", "artificial intelligence"],
    },
    "internet": {
        "summary": "The Internet is a global network of networks that exchanges data using standard protocols (IP/TCP).",
        "topics": ["computing"],
        "keywords": ["what is the internet", "internet"],
    },
    "sqlite": {
        "summary": "SQLite is an embedded SQL database engine stored in a local file; useful for on-device memory and FTS5 search.",
        "topics": ["computing"],
        "keywords": ["sqlite", "what is sqlite"],
    },
    "ocr": {
        "summary": "OCR (optical character recognition) converts images of text into machine-readable characters.",
        "topics": ["computing", "vision"],
        "keywords": ["what is ocr", "optical character recognition"],
    },
    "thermal imaging": {
        "summary": "Thermal imaging cameras sense infrared radiation to visualize relative heat; absolute °C needs calibration.",
        "topics": ["vision", "sensors"],
        "keywords": ["thermal imaging", "thermography", "flir"],
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

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hits"] = [h if isinstance(h, dict) else h.to_dict() for h in self.hits]
        return d


class IntelligentWebSearch:
    """Smarter search: rewrite → local → live → synthesize with citations."""

    def __init__(self, *, allow_live: bool = True, timeout: float = 4.0) -> None:
        self.allow_live = allow_live
        self.timeout = timeout
        self.cache: dict[str, SearchResult] = {}

    def search(
        self,
        query: str,
        *,
        question: Optional[dict[str, Any]] = None,
        allow_live: Optional[bool] = None,
        max_hits: int = 5,
    ) -> SearchResult:
        raw = (query or "").strip()
        if not raw:
            return SearchResult(False, "", "", "Empty query.", method="empty", confidence=0.0)

        rewritten = self.rewrite_query(raw, question=question)
        cache_key = rewritten.lower()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            cached.notes = list(cached.notes) + ["cache_hit"]
            return cached

        # 1) Local knowledge
        local = self._local_lookup(rewritten, raw)
        if local and local.confidence >= 0.7:
            self.cache[cache_key] = local
            local.question = question
            return local

        live_ok = self.allow_live if allow_live is None else allow_live
        hits: list[SearchHit] = list(local.hits) if local else []
        notes: list[str] = list(local.notes) if local else []
        answer_parts: list[str] = []
        if local and local.answer:
            answer_parts.append(local.answer)
            notes.append("local_partial")

        # 2) Live providers
        if live_ok:
            ddg = self._duckduckgo(rewritten)
            if ddg:
                hits.extend(ddg)
                notes.append("ddg_live")
            wiki = self._wikipedia(rewritten)
            if wiki:
                hits.append(wiki)
                notes.append("wikipedia_live")

        # Rank / dedupe
        hits = self._rank_hits(hits, rewritten)[:max_hits]
        if not hits and not answer_parts:
            result = SearchResult(
                ok=False,
                query=raw,
                rewritten_query=rewritten,
                answer=(
                    "I could not find a reliable answer offline, and live search "
                    "returned nothing (or network is unavailable). "
                    "Try a clearer entity question, or check connectivity."
                ),
                method="miss",
                confidence=0.15,
                live=bool(live_ok),
                notes=notes + ["no_hits"],
                question=question,
            )
            return result

        # Synthesize answer from top snippets (extractive, not generative essay)
        if hits:
            top = hits[0]
            answer_parts.append(top.snippet)
            if top.url:
                answer_parts.append(f"Source: {top.title} — {top.url}")
            if len(hits) > 1:
                answer_parts.append(
                    "Also: " + "; ".join(f"{h.title}" for h in hits[1:3])
                )

        answer = " ".join(answer_parts)
        conf = 0.55 + 0.1 * min(3, len(hits))
        if any(h.source == "local_kb" for h in hits):
            conf = max(conf, 0.75)
        if any(h.source == "wikipedia" for h in hits):
            conf = max(conf, 0.7)
        result = SearchResult(
            ok=True,
            query=raw,
            rewritten_query=rewritten,
            answer=answer,
            hits=hits,
            method="hybrid" if local and hits else ("local_kb" if local else "live"),
            confidence=min(0.92, conf),
            live=any(h.source in ("ddg", "wikipedia") for h in hits),
            notes=notes,
            question=question,
        )
        self.cache[cache_key] = result
        return result

    def rewrite_query(self, text: str, *, question: Optional[dict[str, Any]] = None) -> str:
        """Turn a natural question into a sharper search query."""
        q = text.strip()
        # Strip conversational wrappers
        q = re.sub(
            r"^(?:please\s+)?(?:search(?:\s+the\s+web)?(?:\s+for)?|look\s+up|google|find\s+online|what\s+is|what's|whats|who\s+is|tell\s+me\s+about|explain)\s+",
            "",
            q,
            flags=re.I,
        )
        q = q.strip(" ?.")
        if question:
            focus = question.get("focus") or []
            if focus and len(" ".join(focus)) >= 3:
                # Prefer analyzer focus for short queries
                if len(q.split()) <= 2 or question.get("ask_form") in ("define", "causal", "howto"):
                    q = " ".join(focus[:2])
            wh = question.get("wh_type")
            ask = question.get("ask_form")
            if ask == "define" or wh == "what":
                q = f"{q} definition"
            elif ask == "howto":
                q = f"how to {re.sub(r'^how to\s+', '', q, flags=re.I)}"
            elif ask == "causal" or wh == "why":
                q = f"why {q}"
        # Drop filler
        q = re.sub(r"\b(?:please|kindly|exactly|basically)\b", " ", q, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip(" ?.")
        return q or text.strip()

    def _local_lookup(self, rewritten: str, raw: str) -> Optional[SearchResult]:
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
            # token overlap
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
        return SearchResult(
            ok=True,
            query=raw,
            rewritten_query=rewritten,
            answer=fact["summary"] + " (Local knowledge base.)",
            hits=[hit],
            method="local_kb",
            confidence=min(0.9, 0.6 + 0.1 * best_score),
            live=False,
            notes=["local_kb"],
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
        # Use summary REST API
        title = query.strip().replace(" ", "_")
        # Strip trailing 'definition'
        title = re.sub(r"_+definition$", "", title, flags=re.I)
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
        data = self._http_json(url)
        if not data or data.get("type") == "disambiguation":
            # try search API
            search_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
                {"action": "opensearch", "search": query, "limit": 1, "namespace": 0, "format": "json"}
            )
            sdata = self._http_json(search_url)
            if not sdata or len(sdata) < 4 or not sdata[1]:
                return None
            title = sdata[1][0]
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_"))
            data = self._http_json(url)
            if not data:
                return None
        extract = (data.get("extract") or "").strip()
        page_url = (data.get("content_urls") or {}).get("desktop", {}).get("page") or data.get("canonicalurl") or ""
        if not extract:
            return None
        return SearchHit(
            title=data.get("title") or query,
            url=page_url or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}",
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
            ranked.append(
                SearchHit(h.title, h.url, h.snippet, h.source, round(boost, 3))
            )
        ranked.sort(key=lambda x: -x.score)
        return ranked

    def _http_json(self, url: str) -> Optional[Any]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
        except Exception:  # noqa: BLE001 — network optional
            return None
