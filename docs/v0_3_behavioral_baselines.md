# CommerceRecLab v0.3 — Behavioral Baselines

## Question

**How much ranking quality is available before adding learned personalization?**

v0.3 deliberately uses interpretable heuristics fitted only on the frozen v0.2 training period.

## Prediction example

The primary baseline query is placed **after the first event of each multi-event split-bounded session**.

```text
first session event
        ↓ prediction point
later events in same session
        ↓
view / addtocart / transaction positives
```

The prediction horizon remains the frozen v0.2 contract: the remainder of the current split-bounded session. The first event itself is context, not a positive label for that query.

This one-query-per-session design is intentionally simple. Later session models may evaluate richer prefix positions, but they must be reported as a different protocol rather than silently compared with this baseline table.

## Baselines

1. `most_view` — training item frequency among views.
2. `most_addtocart` — training item frequency among cart events.
3. `most_transaction` — training item frequency among transactions.
4. `time_decayed_popularity` — all training events with a 7-day half-life by default.
5. `visitor_history_repeat` — visitor's most frequent/recent training items, falling back to most-viewed items.
6. `last_item_covisitation` — symmetric adjacent-item co-occurrence in training sessions, conditioned on the first/last observed context item.
7. `last_item_transition` — directional adjacent-item transitions in training sessions.
8. `category_conditioned_popularity` — popular training items in the context item's training-cutoff category, falling back to most-viewed items.

The co-visitation definition is intentionally narrow in v0.3: adjacent items only. Wider windows are an architectural intervention for a later retrieval experiment.

## Leakage controls

- All popularity, transition, co-visitation, visitor-history, and category statistics are fitted on `train` only.
- Category state uses only `categoryid` property rows with timestamp at or before the frozen train cutoff.
- Validation/test labels never contribute to baseline statistics.
- Unseen future items are not injected into a baseline ranking simply because they later appear in the dataset.
- View, cart, and transaction tasks are evaluated separately.

## Metrics

Report `Recall@K` and binary `NDCG@K` separately for future:

- `view`,
- `addtocart`,
- `transaction`.

A query contributes to a task metric only when that session remainder contains at least one positive for the corresponding event type.

Therefore, cart and transaction Recall/NDCG are **conditional ranking metrics** over queries that contain at least one future cart or transaction target. They do not estimate the probability that an arbitrary session will add to cart or transact.

## Cohorts

Visitor-history cohorts are frozen as:

```text
new      0 training events
sparse   1–4 training events
repeat   >=5 training events
```

Target-item popularity cohorts are descriptive project definitions based on training event-frequency rank:

```text
head       top 10% of training items
mid_tail   next 40%
long_tail  bottom 50%
unseen     absent from training events
```

These item cohort thresholds are project conventions, not claims about universal retail head/tail definitions.

## Run

```bash
python -m commercereclab.evaluation.baselines \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --output-dir artifacts/v0_3_behavioral_baselines
```

By default, v0.3 deterministically samples up to **50,000 eligible multi-event sessions per split** with a recorded seed. This keeps the baseline suite laptop-friendly while preserving a large evaluation sample. Pass `--max-sessions-per-split 0` to evaluate every eligible session.

For a quick deterministic development run:

```bash
python -m commercereclab.evaluation.baselines \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --max-sessions-per-split 5000 \
  --output-dir artifacts/v0_3_behavioral_baselines_smoke
```

## Exit criteria

v0.3 is complete when:

1. all baseline statistics are provably train-only;
2. view/cart/transaction metrics are reported separately;
3. validation and test results use the frozen v0.2 temporal protocol;
4. new/sparse/repeat visitor cohorts are reported;
5. head/mid-tail/long-tail/unseen target coverage is inspectable;
6. failures of simple baselines are documented before adding learned ranking.
