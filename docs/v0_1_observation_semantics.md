# v0.1 — Observation Semantics

## Question

**What exactly is observed in Retailrocket, and what does missingness mean?**

v0.1 freezes the empirical meaning of the event log before any ranking labels, negative sampling, sessions, or models are introduced.

## Observation contract

A row such as:

```text
(visitor A, item B, view, time t)
```

means a view event was recorded for that visitor-item pair at that time.

It does **not** establish the recommendation slate or exposure mechanism that preceded the event.

Therefore:

```text
unobserved(A, B) != observed dislike
unobserved(A, B) != observed exposure-with-no-click
```

Missing visitor-item pairs remain unobserved behavior unless a later data source provides impression-level evidence.

## Funnel semantics

The observed event vocabulary has an intuitive strength ordering:

```text
view → addtocart → transaction
```

but this must not be treated as a guaranteed recorded path. A transaction can appear without a prior cart or view in the released log, and a cart can appear without a prior recorded view.

The v0.1 audit therefore measures actual visitor-item path coverage rather than assuming funnel completeness.

For path analysis only, exact duplicate rows are collapsed to one record. The raw source is never modified.

## Empirical tasks

The following are distinct estimands:

1. **Future interaction ranking** — rank items likely to receive a later recorded event.
2. **Cart-intent ranking** — rank items likely to receive a later `addtocart` event.
3. **Transaction ranking** — rank items likely to receive a later `transaction` event.
4. **Next-item/session ranking** — after sessionization is defined, rank the next interacted item.

Results for one task must not be silently reported as evidence for another.

## Command

```bash
python -m commercereclab.evaluation.observation \
  data/raw/retailrocket/events.csv \
  --output-dir artifacts/v0_1_observation_semantics
```

Outputs:

```text
artifacts/v0_1_observation_semantics/observation_semantics.json
artifacts/v0_1_observation_semantics/observation_semantics.md
```

## Exit criteria

v0.1 is complete when:

- the observation contract is documented;
- exact-duplicate handling for descriptive path analysis is explicit;
- observed visitor-item funnel patterns are measured;
- cart/transaction prior-event coverage is measured;
- missing pairs remain explicitly unobserved rather than negatives;
- the four empirical tasks above are kept separate.

Temporal train/validation/test splits, candidate sets, relevance construction, and point-in-time feature hydration belong to **v0.2**.
