"""Evaluation harness: retrieval ablation (BM25 / vector / hybrid / hybrid+rerank) and answer
quality (citation precision, gold coverage, refusal accuracy)."""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from ..config import RetrievalMode, Settings
from ..engine import GitContext
from ..llm import LLMClient
from ..retrieval.rerank import HeuristicReranker
from ..samples import build_sample_repo
from .metrics import matches_gold, mrr, ndcg_at_k, ranked_gold_hits, recall_at_k

MODES: tuple[RetrievalMode, ...] = ("bm25", "vector", "hybrid", "hybrid_rerank")
DEFAULT_QUESTIONS = next(
    (
        p
        for p in (
            Path(__file__).resolve().parents[3] / "evals" / "questions.yaml",
            Path.cwd() / "evals" / "questions.yaml",
        )
        if p.exists()
    ),
    Path("evals/questions.yaml"),
)


class EvalQuestion(BaseModel):
    id: str
    question: str
    gold: list[str]


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[EvalQuestion]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [EvalQuestion.model_validate(q) for q in data["questions"]]


def evaluate_retrieval(
    engine: GitContext, questions: list[EvalQuestion], k: int = 5
) -> dict[str, dict[str, float]]:
    answerable = [q for q in questions if q.gold]
    results: dict[str, dict[str, float]] = {}
    for mode in MODES:
        recalls, mrrs, ndcgs, hit1, latencies = [], [], [], [], []
        for q in answerable:
            t0 = time.perf_counter()
            hits = engine.search(q.question, k=30, mode=mode)
            latencies.append((time.perf_counter() - t0) * 1000)
            labels = ranked_gold_hits([h.chunk for h in hits], q.gold)
            recalls.append(recall_at_k(labels, q.gold, k))
            mrrs.append(mrr(labels, 10))
            ndcgs.append(ndcg_at_k(labels, q.gold, k))
            hit1.append(1.0 if labels and labels[0] else 0.0)
        results[mode] = {
            f"recall@{k}": statistics.mean(recalls),
            "mrr@10": statistics.mean(mrrs),
            f"ndcg@{k}": statistics.mean(ndcgs),
            "hit@1": statistics.mean(hit1),
            "p50_latency_ms": statistics.median(latencies),
        }
    return results


def evaluate_reranker_ablation(
    engine: GitContext, questions: list[EvalQuestion]
) -> dict[str, dict[str, float]]:
    """Leave-one-feature-out ablation of the heuristic reranker (feature weight set to 0)."""
    answerable = [q for q in questions if q.gold]
    original = engine.retriever.reranker
    out: dict[str, dict[str, float]] = {}
    try:
        for feature in ("none", *HeuristicReranker(idf=lambda t: 1.0).w):
            weights = {} if feature == "none" else {feature: 0.0}
            engine.retriever.reranker = HeuristicReranker(
                idf=engine.retriever.bm25.idf_of, weights=weights
            )
            hit1, mrrs = [], []
            for q in answerable:
                hits = engine.search(q.question, k=30, mode="hybrid_rerank")
                labels = ranked_gold_hits([h.chunk for h in hits], q.gold)
                hit1.append(1.0 if labels and labels[0] else 0.0)
                mrrs.append(mrr(labels, 10))
            out[feature] = {"hit@1": statistics.mean(hit1), "mrr@10": statistics.mean(mrrs)}
    finally:
        engine.retriever.reranker = original
    return out


def evaluate_answers(engine: GitContext, questions: list[EvalQuestion]) -> dict[str, Any]:
    precisions, coverages, per_question = [], [], []
    false_refusals = correct_refusals = 0
    answerable = [q for q in questions if q.gold]
    unanswerable = [q for q in questions if not q.gold]
    for q in questions:
        ans = engine.ask(q.question)
        cited = [engine.retriever.by_id[c.chunk_id] for c in ans.citations]
        good = [c for c in cited if any(matches_gold(c, g) for g in q.gold)]
        row: dict[str, Any] = {
            "id": q.id,
            "refused": ans.refused,
            "n_citations": len(cited),
            "evidence": round(ans.evidence_score, 3),
        }
        if q.gold:
            if ans.refused:
                false_refusals += 1
            else:
                precisions.append(len(good) / len(cited))
                covered = {g for g in q.gold for c in cited if matches_gold(c, g)}
                coverages.append(len(covered) / len(q.gold))
                row |= {"citation_precision": precisions[-1], "gold_coverage": coverages[-1]}
        elif ans.refused:
            correct_refusals += 1
        per_question.append(row)
    return {
        "citation_precision": statistics.mean(precisions) if precisions else 0.0,
        "gold_source_coverage": statistics.mean(coverages) if coverages else 0.0,
        "answered": len(answerable) - false_refusals,
        "answerable": len(answerable),
        "false_refusal_rate": false_refusals / len(answerable) if answerable else 0.0,
        "refusal_accuracy_unanswerable": correct_refusals / len(unanswerable)
        if unanswerable
        else 0.0,
        "unanswerable": len(unanswerable),
        "per_question": per_question,
    }


def run_eval(
    questions_path: Path = DEFAULT_QUESTIONS,
    llm: LLMClient | None = None,
    k: int = 5,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    with tempfile.TemporaryDirectory() as tmp:
        repo = build_sample_repo(Path(tmp) / "shopfront-api")
        settings = Settings(index_dir=Path(tmp) / "index")
        engine = GitContext.build(repo, settings=settings, llm=llm)
        retrieval = evaluate_retrieval(engine, questions, k=k)
        ablation = evaluate_reranker_ablation(engine, questions)
        answers = evaluate_answers(engine, questions)
    return {
        "n_questions": len(questions),
        "n_chunks": len(engine.chunks),
        "model": engine.llm.model,
        "retrieval": retrieval,
        "reranker_ablation": ablation,
        "answers": answers,
    }


def format_report(report: dict[str, Any], k: int = 5) -> str:
    lines = [
        f"Questions: {report['n_questions']}  |  Chunks indexed: {report['n_chunks']}"
        f"  |  Generator: {report['model']}",
        "",
        f"| Retrieval mode | Recall@{k} | MRR@10 | nDCG@{k} | Hit@1 | p50 latency (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for mode, m in report["retrieval"].items():
        lines.append(
            f"| {mode} | {m[f'recall@{k}']:.3f} | {m['mrr@10']:.3f} | {m[f'ndcg@{k}']:.3f} | "
            f"{m['hit@1']:.3f} | {m['p50_latency_ms']:.1f} |"
        )
    lines += ["", "| Reranker feature removed | Hit@1 | MRR@10 |", "|---|---|---|"]
    for feat, m in report["reranker_ablation"].items():
        lines.append(f"| {feat} | {m['hit@1']:.3f} | {m['mrr@10']:.3f} |")
    a = report["answers"]
    lines += [
        "",
        "| Answer metric (hybrid_rerank) | Value |",
        "|---|---|",
        f"| Citation precision | {a['citation_precision']:.3f} |",
        f"| Gold-source coverage of citations | {a['gold_source_coverage']:.3f} |",
        f"| Answered (answerable questions) | {a['answered']}/{a['answerable']} |",
        f"| False refusal rate | {a['false_refusal_rate']:.3f} |",
        f"| Refusal accuracy (unanswerable) | "
        f"{a['refusal_accuracy_unanswerable']:.3f} ({a['unanswerable']} questions) |",
    ]
    return "\n".join(lines)


def to_yaml(report: dict[str, Any]) -> str:
    return yaml.safe_dump(report, sort_keys=False)
