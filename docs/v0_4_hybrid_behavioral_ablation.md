# CommerceRecLab v0.4 — Hybrid Behavioral Scoring + Ablation

## Question

**Do the strongest simple behavioral signals contribute complementary ranking information, or does one signal already explain the useful gain?**

v0.4 does not introduce a learned ranker. It combines four v0.3 signals with an equal-weight reciprocal-rank fusion (RRF) rule and then removes one component at a time.

## Components

The full hybrid combines:

1. `category` — training-cutoff category-conditioned popularity;
2. `covisitation` — adjacent-item symmetric co-visitation from training sessions;
3. `recency` — time-decayed global popularity from training events;
4. `visitor_history` — visitor-specific repeated-item history from training.

Every component is fit only on the frozen v0.2 training period.

## Fusion rule

For each source ranking and candidate at rank `r`, v0.4 adds:

```text
1 / (RRF_OFFSET + r)
```

The default offset is `60`, every source has equal weight, and only the top `100` items from each source participate by default. Scores are summed across sources and sorted descending.

This is intentionally not tuned per task. The goal is to test complementarity before introducing learned weighting.

## Variants

v0.4 reports:

```text
category_only
covisitation_only
recency_only
visitor_history_only

hybrid_full
hybrid_minus_category
hybrid_minus_covisitation
hybrid_minus_recency
hybrid_minus_visitor_history
```

## Retain / reject rule

The full hybrid is retained only when both conditions hold on **validation**:

1. mean validation NDCG across view/cart/transaction exceeds the best single component's mean validation NDCG;
2. the full hybrid matches or beats the best single component on at least two of the three task-specific validation NDCGs.

The test split is reported only as confirmation and is not used to tune weights or make the retain decision.

## Ablation interpretation

For each component:

```text
ΔNDCG = NDCG(full hybrid) - NDCG(hybrid without component)
```

Positive `ΔNDCG` means the component contributed useful ranking signal for that task under this fixed fusion. Negative `ΔNDCG` means the full hybrid improved when that component was removed, so the component is harmful or redundant under this rule.

## Evaluation contract

v0.4 preserves the v0.2/v0.3 protocol:

- one query immediately after the first event of each multi-event split-bounded session;
- target horizon is the remainder of that same session;
- view/cart/transaction labels remain separate;
- cart and transaction Recall/NDCG remain conditional ranking metrics over queries that contain those targets;
- future catalog state is forbidden;
- category state is frozen at the training cutoff;
- the deterministic 50,000-session-per-split cap and seed are recorded.

## Run

```bash
python -m commercereclab.evaluation.hybrid \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --output-dir artifacts/v0_4_hybrid_ablation
```

For a quick smoke run:

```bash
python -m commercereclab.evaluation.hybrid \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --max-sessions-per-split 5000 \
  --output-dir artifacts/v0_4_hybrid_ablation_smoke
```

## Exit criteria

v0.4 is complete when:

1. all four components are verified train-only;
2. the full equal-weight fusion is compared against every single component;
3. leave-one-component-out ablations are reported;
4. the retain/reject decision is made from validation only;
5. test metrics are treated as confirmation rather than a tuning signal;
6. no learned ranker is introduced unless this experiment shows a remaining measurable failure worth targeting.
