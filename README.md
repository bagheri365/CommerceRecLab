# CommerceRecLab

**Research-oriented e-commerce recommender system for session intent, candidate generation, funnel-aware ranking, cold start, catalog exposure, and serving tradeoffs.**

> **Research question:** How should an e-commerce recommender balance user intent, product relevance, conversion opportunity, availability, catalog coverage, freshness, and serving cost?

CommerceRecLab is a system-design and experimentation project built around the Retailrocket e-commerce dataset. The project emphasizes reproducible offline evaluation, temporal correctness, candidate-generation quality, ranking tradeoffs, catalog behavior, and serving architecture rather than a single benchmark model.

## Scientific principle

```text
baseline → measurable failure → targeted intervention → evaluate → retain / reject
```

Complexity must earn its place through evidence. Negative results and failed interventions remain part of the project record.

## Primary empirical dataset

CommerceRecLab uses Retailrocket's public e-commerce behavior data:

```text
events.csv
item_properties_part1.csv
item_properties_part2.csv
category_tree.csv
```

The behavioral funnel contains timestamped:

```text
view → addtocart → transaction
```

The item-property tables provide time-varying product state, and the category tree provides hierarchical catalog structure.

Raw data are kept under `data/raw/` and are not committed to Git.

## Observation semantics

An observed event means that Retailrocket recorded a visitor-item event at a timestamp. It does **not** mean that every item without an event was shown and rejected.

```text
observed (visitor, item, event, time)
    = recorded behavior

missing visitor-item pair
    = no recorded event
    ≠ observed negative
```

Recommendation experiments therefore define candidate sets, labels, temporal cutoffs, and negative/comparison construction explicitly.

Time-varying item properties must be joined **as of recommendation time**. Future product state must never leak into historical predictions.

See [`docs/data_semantics.md`](docs/data_semantics.md) and the full [`docs/roadmap.md`](docs/roadmap.md).

## Evaluation discipline

Every empirical experiment must state:

1. the prediction or ranking estimand;
2. the temporal train / validation / test split;
3. the candidate universe;
4. label construction;
5. negative/comparison construction;
6. point-in-time feature rules;
7. prohibited leakage;
8. which claims the result supports.

The project distinguishes tasks such as next-item retrieval, event-type prediction, funnel-aware ranking, and transaction-oriented reranking rather than treating them as interchangeable.

## Planned system components

```text
data audit
→ observation & temporal semantics
→ behavioral baselines
→ session / user intent
→ candidate generation
→ feature hydration
→ ranking
→ funnel-aware reranking
→ catalog / availability policy
→ serving & graceful degradation
```

Planned experiments include:

- popularity and recency baselines;
- co-visitation and session-based retrieval;
- collaborative and latent retrieval;
- multi-source candidate generation;
- click / cart / transaction-aware ranking;
- new-item and sparse-history behavior;
- time-varying item-property features;
- category-aware retrieval and reranking;
- catalog exposure concentration and coverage;
- exact vs. approximate retrieval at larger scale;
- online serving, latency budgets, caching, and fallbacks.

## Repository structure

```text
CommerceRecLab/
├── README.md
├── LICENSE
├── pyproject.toml
├── CITATION.cff
├── configs/
├── experiments/
├── artifacts/
├── docs/
│   ├── roadmap.md
│   └── data_semantics.md
├── src/
│   └── commercereclab/
│       ├── data/
│       ├── audit/
│       ├── eligibility/
│       ├── retrieval/
│       ├── features/
│       ├── models/
│       ├── ranking/
│       ├── reciprocity/
│       ├── policy/
│       ├── simulation/
│       ├── serving/
│       └── evaluation/
└── tests/
```


## Development

Create and activate a virtual environment, then install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

Run lint checks:

```bash
ruff check .
```

## Roadmap

The detailed, scientifically revised build plan lives in [`docs/roadmap.md`](docs/roadmap.md).

The first Retailrocket milestones are methodological: establish the dataset contract, temporal observation semantics, and leakage-safe evaluation protocol before adding complex models.

## Scope and claim discipline

CommerceRecLab is not presented as a reproduction of any production retailer's recommender system. Results from Retailrocket support claims about the defined offline tasks and dataset. Systems simulations and scaling experiments are labeled separately and are not presented as observed production effects.

## v0.1 — Observation semantics

After completing the dataset audit, measure which funnel paths are actually present in the event log:

```bash
python -m commercereclab.evaluation.observation \
  data/raw/retailrocket/events.csv \
  --output-dir artifacts/v0_1_observation_semantics
```

This milestone is descriptive. It does not infer recommendation impressions or convert missing visitor-item pairs into negatives. See [`docs/v0_1_observation_semantics.md`](docs/v0_1_observation_semantics.md).

## v0.2 — Temporal evaluation protocol

Freeze time-respecting train/validation/test cutoffs, sessionization, candidate-set semantics, and point-in-time leakage rules before fitting ranking models:

```bash
python -m commercereclab.evaluation.temporal \
  data/raw/retailrocket/events.csv \
  --output-dir artifacts/v0_2_temporal_protocol
```

See [`docs/v0_2_temporal_evaluation_protocol.md`](docs/v0_2_temporal_evaluation_protocol.md).

## v0.3 — Behavioral baselines

Measure how far simple train-only heuristics can go before introducing learned personalization:

```bash
python -m commercereclab.evaluation.baselines \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --output-dir artifacts/v0_3_behavioral_baselines
```

The baseline table keeps future view, cart, and transaction objectives separate and uses the frozen v0.2 session and temporal contracts. See [`docs/v0_3_behavioral_baselines.md`](docs/v0_3_behavioral_baselines.md).

## v0.4 — Hybrid behavioral scoring + ablation

Test whether category context, co-visitation, recency, and visitor history provide complementary signal before introducing a learned ranker:

```bash
python -m commercereclab.evaluation.hybrid \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --output-dir artifacts/v0_4_hybrid_ablation
```

The fusion uses fixed equal-weight reciprocal-rank fusion and validation-only retain/reject logic. See [`docs/v0_4_hybrid_behavioral_ablation.md`](docs/v0_4_hybrid_behavioral_ablation.md).

### v0.5 — Learned task-specific rankers

v0.5 tests whether separate lightweight logistic rankers can learn task-specific weights over category, co-visitation, recency, and visitor-history retrieval signals without using validation outcomes for fitting. See `docs/v0_5_learned_task_specific_rankers.md`.

### v0.6 — Candidate generation / retrieval

v0.6 moves upstream from reranking and tests whether deeper behavioral and category-hierarchy retrieval can increase candidate recall without future metadata leakage. It reports candidate recall together with candidate-set size and fixed-RRF ranking diagnostics. See `docs/v0_6_candidate_retrieval.md`.

```bash
python -m commercereclab.evaluation.retrieval \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --output-dir artifacts/v0_6_candidate_retrieval
```

### v0.7 — Retrieval efficiency / Pareto frontier

v0.7 sweeps category, behavioral, and parent-category retrieval depths to identify validation Pareto-efficient candidate generators. The selection rule preserves a declared fraction of expanded-reference recall for every task while minimizing mean candidate-set size. See `docs/v0_7_retrieval_efficiency.md`.

```bash
python -m commercereclab.evaluation.efficiency \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --output-dir artifacts/v0_7_retrieval_efficiency
```

### v0.8 — Learned reranking on efficient retrieval

v0.8 revisits task-specific logistic reranking after freezing the v0.7-selected efficient candidate generator (`c300_b100_p0`). It compares learned reranking against deep category-only and fixed-RRF deterministic baselines on the same retrieval regime, with validation-only retain/reject logic. See `docs/v0_8_learned_reranking_efficient_retrieval.md`.

```bash
python -m commercereclab.evaluation.rerank \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --output-dir artifacts/v0_8_efficient_reranker
```

### v0.9 — Funnel-aware / stage-conditional ranking

v0.9 tests whether explicitly conditioning ranking on observed session stage resolves the task/context heterogeneity seen in v0.4, v0.5, and v0.8. It freezes the v0.7 efficient retriever (`c300_b100_p0`), compares stage-conditioned logistic rankers against category-only and pooled task-specific ranking, and uses validation only for retain/reject decisions. See `docs/v0_9_stage_aware_ranking.md`.

```bash
python -m commercereclab.evaluation.stage \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --output-dir artifacts/v0_9_stage_aware_ranking
```
