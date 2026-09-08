# v0.5 — Learned Task-Specific Rankers

## Question

Can a lightweight learned combiner use the v0.4 behavioral signals more effectively than category-only ranking or equal-weight RRF, while preserving the frozen temporal evaluation contract?

## Design

- Keep the v0.2 prediction point and remainder-of-session horizon.
- Use category, co-visitation, recency, and visitor-history rankings as retrieval sources.
- Convert each source into reciprocal-rank and presence features.
- Fit separate logistic-regression rankers for `view`, `addtocart`, and `transaction`.
- Do not train on validation outcomes.
- Split the frozen training period internally by time: fit behavioral source statistics on the earlier subwindow and create ranker-training queries only from the later subwindow.
- Refit source statistics on the full frozen training period for validation/test scoring, while keeping learned coefficients fixed.
- Rank only the union of retrieved behavioral candidates and report candidate recall separately.

## Label semantics

A future target item inside the remainder of the current split-bounded session is relevant for its event task. Other retrieved items used during model fitting are sampled **nonrelevant candidates for the offline ranking objective**. They are not interpreted as observed dislikes, passes, recommendation impressions, or causal negatives.

## Retain rule

Retain the learned task-specific ranker only if its mean validation NDCG across view/cart/transaction exceeds category-only and it beats category-only on at least two of the three task-specific validation NDCGs. Test results are confirmation only.

## Run

```bash
python -m commercereclab.evaluation.learned \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --output-dir artifacts/v0_5_learned_ranker
```
