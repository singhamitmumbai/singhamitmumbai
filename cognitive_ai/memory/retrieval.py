"""Lexical retrieval: BM25-lite + TF character features + curated synonyms."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Optional


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(t) > 1]


class BM25Index:
    """In-memory BM25 over document dicts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[dict[str, Any]] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.df: Counter[str] = Counter()
        self.avgdl: float = 0.0
        self.synonyms: dict[str, list[str]] = {}

    def set_synonyms(self, mapping: dict[str, list[str]]) -> None:
        self.synonyms = {k.lower(): [x.lower() for x in v] for k, v in mapping.items()}

    def build(self, documents: Iterable[dict[str, Any]], text_fn=None) -> None:
        text_fn = text_fn or (lambda d: str(d))
        self.docs = list(documents)
        self.doc_tokens = []
        self.doc_len = []
        self.df = Counter()
        for d in self.docs:
            toks = tokenize(text_fn(d))
            self.doc_tokens.append(toks)
            self.doc_len.append(len(toks))
            for t in set(toks):
                self.df[t] += 1
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0

    def _expand_query(self, tokens: list[str]) -> list[str]:
        out = list(tokens)
        for t in tokens:
            out.extend(self.synonyms.get(t, []))
        return out

    def search(self, query: str, top_k: int = 20) -> list[tuple[float, dict[str, Any]]]:
        q = self._expand_query(tokenize(query))
        if not q or not self.docs:
            return []
        N = len(self.docs)
        scored: list[tuple[float, dict[str, Any]]] = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            tf = Counter(self.doc_tokens[i])
            dl = self.doc_len[i] or 1
            for term in q:
                if term not in tf:
                    continue
                n_q = self.df.get(term, 0)
                idf = math.log(1 + (N - n_q + 0.5) / (n_q + 0.5))
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((score, {**doc, "_score": score}))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]
