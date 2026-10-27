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
        q_chars = _char_ngrams(query.lower())
        scored: list[tuple[float, dict[str, Any]]] = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            tf = Counter(self.doc_tokens[i])
            dl = self.doc_len[i] or 1
            for term in q:
                if term not in tf:
                    # fuzzy: prefix / edit-light match against doc terms
                    fuzzy = _fuzzy_term_hit(term, tf)
                    if not fuzzy:
                        continue
                    freq = fuzzy
                    n_q = self.df.get(term, 1)
                else:
                    freq = tf[term]
                    n_q = self.df.get(term, 0)
                idf = math.log(1 + (N - n_q + 0.5) / (n_q + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * (freq * (self.k1 + 1)) / denom
            # Character n-gram boost helps Hinglish / typos
            if q_chars:
                doc_chars = _char_ngrams(" ".join(self.doc_tokens[i]))
                inter = len(q_chars & doc_chars)
                if inter:
                    score += 0.15 * inter / max(1, len(q_chars))
            if score > 0:
                scored.append((score, {**doc, "_score": score}))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    t = re.sub(r"\s+", " ", text.lower())
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def _fuzzy_term_hit(term: str, tf: Counter) -> float:
    """Return a fractional tf if a close token exists (prefix or 1-char edit-ish)."""
    if not term:
        return 0.0
    for tok, freq in tf.items():
        if tok.startswith(term) or term.startswith(tok):
            if min(len(tok), len(term)) >= 3:
                return float(freq) * 0.6
        if abs(len(tok) - len(term)) <= 1 and len(term) >= 4:
            # cheap Hamming-ish
            mismatches = sum(1 for a, b in zip(tok, term) if a != b) + abs(len(tok) - len(term))
            if mismatches <= 1:
                return float(freq) * 0.5
    return 0.0
