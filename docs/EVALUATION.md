# Evaluation

All numbers on this page were produced by `make eval` (`python -m gitcontext.cli eval --out
evals/results.json`) in this repository's development environment, with the **offline extractive
generator**. They are reproducible: the sample repository, its git history and the question set are
deterministic.

## Corpus

`gitcontext.samples.build_sample_repo` replays `examples/sample_history.yaml` into a real git
repository ("shopfront-api"), so ingestion exercises actual `git log` output:

| Source type | Documents | Chunks |
|---|---|---|
| Code (Python + TypeScript) | 11 | 43 |
| Docs (README, auth, billing, deployment, runbook) | 5 | 15 |
| Commits | 15 | 15 |
| Pull requests (#118, #134, #141, #152) | 4 | 8 |
| Discussions (#27, #31) | 2 | 4 |
| Incident postmortems (INC-2026-014, -022, -029) | 3 | 10 |
| **Total** | **40** | **95** |

The content is fictional but written to have the cross-references real repos have: the token
refresh code was changed by PR #134 because of postmortem INC-2026-014, the webhook idempotency
check came from INC-2026-022 via PR #152, and so on.

## Question set

`evals/questions.yaml` holds 30 questions:

- **25 answerable**, each labelled with the minimal set of documents a good answer should draw on
  (1-4 gold documents, 2.0 on average). Commits are matched by subject substring
  (`commit~validate rate-limit settings`) because shas depend on the git version.
- **5 unanswerable** (Kubernetes autoscaling, GraphQL schema, push notifications, recommendation
  model, Postgres key rotation): the index has no evidence for them and the system should refuse.

Question types: "how does X work" (code + docs), "why" (PRs, postmortems), "who/when" (commits),
operational (runbooks), and decision questions (discussions).

## Metrics

Retrieval metrics are computed at **document level**: the ranked chunk list is collapsed to unique
documents in first-seen order, then compared with the gold documents (binary relevance).

- **Recall@5**: share of gold documents among the top 5 documents.
- **MRR@10**: reciprocal rank of the first gold document.
- **nDCG@5**: binary-gain DCG over the top 5, normalised by the ideal ordering.
- **Hit@1**: the top document is gold.

Answer metrics run the full `ask()` pipeline (hybrid + rerank, diversification, evidence gate,
generator):

- **Citation precision**: share of cited documents that are gold, averaged over answered questions.
  This is a *lower bound*: gold lists are minimal, so citing `docs/auth.md` for a question whose
  gold is PR #134 counts as wrong even when the doc is relevant.
- **Gold-source coverage**: share of gold documents that the answer cites.
- **False refusal rate** on answerable questions and **refusal accuracy** on unanswerable ones.

## Results

### Retrieval ablation

| Retrieval mode | Recall@5 | MRR@10 | nDCG@5 | Hit@1 | p50 latency (ms) |
|---|---|---|---|---|---|
| BM25 only | 0.973 | 0.820 | 0.847 | 0.680 | 0.2 |
| Vector only (TF-IDF + LSA) | 0.950 | 0.827 | 0.836 | 0.680 | 1.7 |
| Hybrid (RRF, k=60) | 0.960 | 0.847 | 0.858 | 0.720 | 2.0 |
| **Hybrid + heuristic rerank** | **0.973** | **0.913** | **0.890** | **0.840** | 7.4 |

### Reranker ablation (leave one feature out)

| Feature removed | Hit@1 | MRR@10 |
|---|---|---|
| none (full reranker) | 0.840 | 0.913 |
| IDF-weighted coverage | 0.720 | 0.853 |
| title / path / symbol match | 0.920 | 0.953 |
| query bigrams | 0.840 | 0.913 |
| fused first-stage score | 0.840 | 0.913 |
| query-intent source prior | 0.800 | 0.880 |
| explicit `#PR` / path reference | 0.840 | 0.913 |

### Answers (hybrid + rerank, offline extractive generator)

| Metric | Value |
|---|---|
| Citation precision | 0.637 |
| Gold-source coverage of citations | 0.767 |
| Answered / answerable | 25 / 25 (false refusal rate 0.000) |
| Refusal accuracy on unanswerable | 0.800 (4 / 5) |

Evidence scores (the refusal gate input) ranged from 0.408 to 1.0 on answerable questions and from
0.099 to 0.501 on unanswerable ones, with the threshold at 0.35.

## Analysis

**Recall is saturated; ordering is not.** With 40 documents, every method finds most gold documents
in the top 5. The meaningful differences are MRR, nDCG and Hit@1. RRF gives a small, consistent gain
over either retriever; the reranker gives the large one (Hit@1 0.72 to 0.84, MRR 0.85 to 0.91).

**What the reranker gains come from.** The ablation shows IDF-weighted coverage carries most of the
gain (removing it drops Hit@1 back to 0.72) and the query-intent prior adds the rest ("why ...
customers" boosts postmortems, "who changed ... when" boosts commits). Bigrams, the fused score and
explicit references are neutral on this set (explicit references matter for questions like "what
did #152 change", which is covered by a unit test rather than the eval). The title/path/symbol
feature *hurts* here: removing it raises Hit@1 to 0.92, because breadcrumb titles such as
"Authentication > Token refresh" match many questions equally. The default weights were left
unchanged because a two-question swing on a 25-question development set is not strong evidence;
the feature is the first candidate for removal or down-weighting once a larger eval set exists.

**BM25 is a strong baseline for code.** Identifiers like `invalid_grant` or `REFRESH_SKEW_SECONDS`
are exact-match problems. The code-aware tokenizer (camelCase and snake_case splitting) is a large
part of why BM25 alone reaches 0.973 Recall@5 here.

**The generator is the bottleneck in offline mode.** The extractive stand-in cites every source
that has a sentence overlapping the question above a threshold, which over-cites. Examples:

- q02 "Why do we refresh access tokens 60 seconds before they expire?" cites PR #134 and the
  postmortem (gold) but also `docs/auth.md` and the module docstring (not in the minimal gold).
- q19 "How are prorated amounts rounded?" cites only `docs/billing.md`, which answers it, but
  misses the code and PR #118, so gold coverage is 0.33.

**The missed refusal is instructive.** u05 "How do we rotate the Postgres encryption keys?" scores
0.50 evidence because "rotate" (refresh-token rotation), "Postgres" (cache stampede discussion) and
"keys" (API keys) all appear in the corpus. A purely lexical gate cannot tell that the *combination*
is unsupported. With a real LLM, the prompt-level `INSUFFICIENT_EVIDENCE` path is the second line of
defence; a calibrated, learned gate is on the roadmap.

## Caveats

- Small, synthetic corpus written by the author of the system; treat it as a regression suite and a
  demonstration of the methodology, not as a benchmark.
- The reranker weights were set before the first evaluation run and not tuned on it, but
  diversification and the offline answerer were adjusted while inspecting failures, so this is a
  development set rather than a held-out test set.
- Answer metrics measure *citation* behaviour, not answer faithfulness or fluency. Measuring those
  needs a real generator and an LLM or human judge.

## Reproduce

```bash
make install
make eval                       # prints the tables and writes evals/results.json
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... make eval   # same harness, Claude as generator
```

`evals/results.json` contains per-question rows (refused, evidence, citation precision, gold
coverage) for failure analysis.
