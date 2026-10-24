"""Okapi BM25 over pre-tokenised documents (no external dependency)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.4, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n_docs = len(corpus)
        self.doc_len = np.array([len(d) for d in corpus], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.tf: list[Counter[str]] = [Counter(d) for d in corpus]
        df: Counter[str] = Counter()
        for d in self.tf:
            df.update(d.keys())
        self.df = df
        # BM25+ style floor keeps IDF positive for very common terms.
        self.idf = {t: math.log(1 + (self.n_docs - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.max_idf = math.log(1 + (self.n_docs + 0.5) / 0.5) if self.n_docs else 1.0

    def idf_of(self, term: str) -> float:
        """IDF for ``term``; unseen terms get the maximum (they are maximally informative)."""
        return self.idf.get(term, self.max_idf)

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n_docs)
        if not self.n_docs:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.tf], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: list[str], k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]
