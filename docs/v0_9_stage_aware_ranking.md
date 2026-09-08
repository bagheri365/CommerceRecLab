# v0.9 — Funnel-aware / stage-conditional ranking

## Hypothesis

The repeated failure of pooled learned rerankers may reflect heterogeneous intent across observable funnel stages rather than insufficient model capacity. v0.9 therefore conditions ranking on session prefix state while holding retrieval fixed.

## Frozen retrieval

Use the v0.7-selected `c300_b100_p0` candidate generator: category depth 300, behavioral depth 100, parent depth 0.

## Observable stages

- `view_stage`: prediction immediately after the first event only when that observed event is a `view`. Evaluate future `view`, `addtocart`, and `transaction` targets in the remainder of that session.
- `cart_stage`: prediction immediately after the first observed `addtocart`, only when later session events exist. Evaluate later `transaction` targets only.

Stage assignment uses only the observed session prefix. Future actions are labels, never stage features.

## Comparators

1. `category_only` — strongest deterministic baseline carried forward.
2. `pooled_task_specific` — logistic task ranker trained across all declared stages for that task.
3. `stage_conditioned` — separate logistic ranker for each declared stage-task cell.

All learned models use the same rank/presence feature family and deterministic sampled-nonrelevant construction as prior learned milestones.

## Retain rule

Using validation only, retain stage conditioning if its mean NDCG across the declared stage-task cells exceeds the best comparator (`category_only` or `pooled_task_specific`) and it wins on at least half of those cells. Test is confirmation only.

## Scientific guardrails

- Behavioral retrieval statistics for ranker training are fit only on the earlier internal training subwindow.
- Validation outcomes do not fit coefficients.
- Sampled nonrelevant candidates are offline ranking labels, not observed dislikes or impressions.
- Cart/transaction metrics remain conditional ranking metrics, not conversion probabilities.
- Category metadata remains point-in-time safe.
