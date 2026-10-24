"""Ranking metrics over document-level relevance (binary gold labels)."""

from __future__ import annotations

import math

from ..models import Chunk, SourceType


def matches_gold(chunk: Chunk, gold: str) -> bool:
    """Gold ids are exact doc ids, or ``commit~<subject substring>`` for commits."""
    if gold.startswith("commit~"):
        return chunk.source_type == SourceType.COMMIT and gold[7:].lower() in chunk.title.lower()
    return chunk.doc_id == gold


def ranked_gold_hits(chunks: list[Chunk], gold: list[str]) -> list[str | None]:
    """Collapse a chunk ranking to unique documents; label each with the gold id it matches."""
    seen: set[str] = set()
    labels: list[str | None] = []
    for c in chunks:
        if c.doc_id in seen:
            continue
        seen.add(c.doc_id)
        labels.append(next((g for g in gold if matches_gold(c, g)), None))
    return labels


def recall_at_k(labels: list[str | None], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len({lbl for lbl in labels[:k] if lbl}) / len(gold)


def mrr(labels: list[str | None], k: int = 10) -> float:
    for rank, lbl in enumerate(labels[:k], start=1):
        if lbl:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels: list[str | None], gold: list[str], k: int) -> float:
    seen: set[str] = set()
    dcg = 0.0
    for rank, lbl in enumerate(labels[:k], start=1):
        if lbl and lbl not in seen:
            seen.add(lbl)
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0
