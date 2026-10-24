"""Code-aware tokenisation shared by BM25, the reranker and evidence scoring.

``refreshAccessToken`` and ``refresh_access_token`` both yield ``refresh``, ``access``, ``token`` so
identifiers match natural-language questions.
"""

from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its me my of on or our so than that the their them then there these they this
    those to up us was we were what when where which who whom why will with would you your
    about after before should self none true false return import def class work works working
    happen happens happened explain describe tell does doing done get gets make makes need
    needs""".split()  # noqa: SIM905
)


def stem(word: str) -> str:
    """Very light suffix stripping; deliberately conservative to keep identifiers recognisable."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _IDENT.findall(text):
        parts = [p.lower() for p in _CAMEL.findall(raw)] if not raw.isdigit() else [raw]
        if len(parts) > 1:
            whole = raw.lower()
            if whole not in STOPWORDS:
                tokens.append(whole)
        for p in parts:
            if p in STOPWORDS or (len(p) < 2 and not p.isdigit()):
                continue
            tokens.append(stem(p))
    return tokens


def query_terms(text: str) -> list[str]:
    """Unique tokens of a query, order preserved."""
    seen: dict[str, None] = {}
    for t in tokenize(text):
        seen.setdefault(t, None)
    return list(seen)
