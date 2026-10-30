"""Unified memory hub for the Finderbay Assistant."""

from __future__ import annotations

from typing import Any, Optional


class MemoryHub:
    """Preference + episodic + semantic facades over existing stores."""

    def __init__(self, finderbay_ai, *, user_memory: Optional[dict[str, str]] = None) -> None:
        self.ai = finderbay_ai
        self.user_memory = user_memory if user_memory is not None else {}
        self.episodes: list[dict[str, Any]] = []

    def remember_preference(self, key: str, value: str) -> dict[str, Any]:
        self.user_memory[str(key)] = str(value)
        # Also store as asserted claim when memory system present
        try:
            self.ai.learner.learn_fact(
                subject="user",
                relationship=f"prefers_{key}",
                value=value,
                source="assistant",
                reviewer="user",
            )
        except Exception:
            pass
        return {"ok": True, "key": key, "value": value}

    def recall(self, key: str) -> Optional[str]:
        if key in self.user_memory:
            return self.user_memory[key]
        # Search claims
        try:
            claims = self.ai.memory.claims
            iterable = claims.values() if isinstance(claims, dict) else claims
            for c in iterable:
                if getattr(c, "subject", None) == "user" and key in str(getattr(c, "relationship", "")):
                    return str(c.value)
        except Exception:
            pass
        return None

    def note_episode(self, summary: str, *, tags: Optional[list[str]] = None, data: Optional[dict] = None) -> dict[str, Any]:
        ep = {"summary": summary, "tags": tags or [], "data": data or {}}
        self.episodes.append(ep)
        try:
            from ..memory.store import Episode

            self.ai.memory.add_episode(
                Episode(summary=summary[:200], observations=[], actions=[], outcome="noted")
            )
        except Exception:
            pass
        return {"ok": True, "episode": ep}

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        q = (query or "").lower()
        hits: list[dict[str, Any]] = []
        for k, v in self.user_memory.items():
            if q in k.lower() or q in str(v).lower():
                hits.append({"kind": "preference", "key": k, "value": v})
        for ep in reversed(self.episodes):
            if q in ep.get("summary", "").lower():
                hits.append({"kind": "episode", **ep})
            if len(hits) >= limit:
                break
        try:
            from ..memory.retrieval import BM25Index

            docs = []
            for kind, rows in (getattr(self.ai.memory, "records", {}) or {}).items():
                if isinstance(rows, dict):
                    for rid, row in rows.items():
                        docs.append({"kind": kind, "id": rid, "text": str(row)})
            if docs:
                idx = BM25Index()
                idx.build(docs, text_fn=lambda d: d["text"])
                for score, doc in idx.search(query, top_k=limit):
                    hits.append({"kind": "retrieval", "score": score, "item": doc})
        except Exception:
            pass
        return hits[:limit]

    def snapshot(self) -> dict[str, Any]:
        return {
            "preferences": dict(self.user_memory),
            "episodes": len(self.episodes),
            "memory_records": len(getattr(self.ai.memory, "records", {}) or {}),
        }
