# v0.8 — Learned reranking on efficient retrieval

## Question

Does the v0.5 task-specific logistic reranker become worth retaining once candidate generation is fixed to the v0.7-selected efficient retriever (`c300_b100_p0`)?

## Frozen retrieval intervention

- category depth: 300
- behavioral depth: 100 for co-visitation, recency, and visitor history
- parent/sibling-category depth: 0
- prediction point and horizon remain frozen from v0.2

v0.8 does not retune retrieval on validation or test.

## Leakage control

The frozen training period is split internally by time. Behavioral source statistics for ranker training are fit on the earlier internal subwindow. Ranker examples come from later split-bounded sessions in the training period. Validation is used only for the retain/reject decision, and test is confirmation only.

## Ranker

Separate logistic-regression models are fit for future `view`, `addtocart`, and `transaction`. Features preserve the v0.5 design: reciprocal-rank features plus source-presence indicators for category, co-visitation, recency, and visitor history. The sampled-nonrelevant construction is intentionally held consistent with v0.5 so the main intervention is the candidate generator.

Sampled nonrelevant items are offline ranking labels. They are not interpreted as observed dislikes, passes, or recommendation impressions.

## Comparators

v0.8 reports:

1. `category_only`: deep same-category ranking at depth 300;
2. `efficient_rrf`: fixed equal-weight RRF over the same `c300_b100_p0` retrieval sources;
3. `learned_task_specific`: the task-specific logistic reranker over the same candidate union.

Candidate recall and candidate-set size are reported separately from top-K ranking metrics.

## Retain rule

Using validation only, retain the learned ranker only if:

1. its mean NDCG@K across view, cart, and transaction exceeds the best deterministic validation comparator (`category_only` or `efficient_rrf`); and
2. it beats that same comparator on at least two of the three task-specific validation NDCGs.

Test metrics do not affect the decision.

## Run

```bash
python -m commercereclab.evaluation.rerank \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --output-dir artifacts/v0_8_efficient_reranker
```
