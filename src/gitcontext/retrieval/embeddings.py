"""Pluggable embedders. The default is local, deterministic and dependency-light."""

from __future__ import annotations

import hashlib
from itertools import pairwise
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .text import tokenize


@runtime_checkable
class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> Embedder: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class TfidfEmbedder:
    """Word (code-aware) + character n-gram TF-IDF, optionally projected with LSA.

    Character n-grams give fuzzy matching (``refresh``/``refreshing``/``refreshed``, typos) that is
    complementary to BM25's exact-term matching; LSA adds a little co-occurrence "semantics".
    """

    name = "tfidf"

    def __init__(self, lsa_dim: int | None = 192) -> None:
        self.lsa_dim = lsa_dim
        self._vec = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        tokenizer=tokenize,
                        lowercase=False,
                        token_pattern=None,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=1,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2
                    ),
                ),
            ]
        )
        self._svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> TfidfEmbedder:
        x = self._vec.fit_transform(texts)
        if self.lsa_dim and min(x.shape) > self.lsa_dim + 1:
            self._svd = TruncatedSVD(n_components=self.lsa_dim, random_state=0).fit(x)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        x = self._vec.transform(texts)
        dense = self._svd.transform(x) if self._svd is not None else x.toarray()
        return _l2(np.asarray(dense, dtype=np.float32))


class HashingEmbedder:
    """Stateless hashing-trick embedder (numpy only). Useful for streaming / incremental indexes."""

    name = "hashing"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def fit(self, texts: list[str]) -> HashingEmbedder:
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokenize(text)
            feats = toks + [f"{a}_{b}" for a, b in pairwise(toks)]
            for tok in feats:
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        out = np.sign(out) * np.log1p(np.abs(out))
        return _l2(out)


class SentenceTransformerEmbedder:
    """Neural embeddings via sentence-transformers (``pip install gitcontext[st]``)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "pip install gitcontext[st] to use SentenceTransformerEmbedder"
            ) from exc
        self.name = f"st:{model_name}"
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> SentenceTransformerEmbedder:
        return self

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs model download
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def make_embedder(name: str = "tfidf") -> Embedder:
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "tfidf-raw":
        return TfidfEmbedder(lsa_dim=None)
    if name == "hashing":
        return HashingEmbedder()
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name[3:])
    raise ValueError(f"unknown embedder '{name}'")
