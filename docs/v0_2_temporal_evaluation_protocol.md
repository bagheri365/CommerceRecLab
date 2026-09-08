# v0.2 — Temporal Evaluation Protocol

## Goal

Freeze an offline evaluation contract before training ranking models.

The protocol must approximate future recommendation while preventing time leakage and keeping distinct prediction tasks separate.

## Temporal split

The default split is event-time ordered:

```text
train       earliest 70%
validation  next 15%
test        final 15%
```

The percentages are converted into timestamp cutoffs from the deduplicated event log and written to a manifest. Downstream experiments must reuse the frozen cutoffs instead of recomputing convenient splits per model.

Events at the train cutoff remain in train; events after train and at/before the validation cutoff remain in validation; later events are test.

## Sessionization

A new session begins after more than 30 minutes of visitor inactivity. Exactly 30 minutes remains in the same session.

A temporal split boundary always breaks a session. If a visitor has events on both sides of the train/validation or validation/test cutoff, the post-boundary event begins a new session even when the inactivity gap is shorter than 30 minutes. This prevents a single evaluation session from straddling two data splits.

Sessionization is not silently substituted for a different prediction task; view, cart, transaction, and next-item outcomes remain separate labels.

## Prediction-time rule

For an example evaluated at time `t`, every feature must have an effective timestamp `<= t`.

For item state, use a backward as-of join:

```text
latest property value where property_timestamp <= t
```

Using the globally latest item property to hydrate historical examples is leakage.

## Prediction horizon

The primary ranking estimand uses the **remainder of the current split-bounded session**. For a prediction made at event time `t`, a positive target must occur strictly after `t` and before that session ends.

This prevents an interaction one minute later and another weeks later from being silently treated as the same prediction problem. Fixed clock-time horizons (for example, one hour or 24 hours) may be added later only as separately named experiments.

## Relevance tasks

Report these separately:

1. future interaction/view ranking,
2. add-to-cart ranking,
3. transaction ranking,
4. next-item/session ranking.

A weighted funnel gain may be added only as a separately named controlled objective. Its weights are project assumptions, not business values inferred from Retailrocket.

## Candidate regimes

### Full / large catalog

Rank against the eligible catalog or a documented large-catalog approximation available at prediction time. An item cannot enter the candidate universe solely because it appears later in the dataset: some observed catalog evidence for the item must exist at or before prediction time.

If availability is enforced, use the latest backward point-in-time `available` state. Missing availability is an unknown/missing state unless a separately documented experiment defines another treatment; it is not silently equivalent to unavailable.

### Sampled negatives

Sampling may be used for iteration speed, but reports must state:

- sampling distribution,
- number of negatives,
- seed,
- any popularity weighting,
- and that sampled metrics are not equivalent to full-catalog retrieval metrics.

Every positive target must remain in the evaluated candidate set.

## Missing metadata

Missing point-in-time metadata is itself part of the observed data condition. Do not fill historical examples from future snapshots. Models must either use an explicit missing/default representation or route those items to a feature-light path.

## Leakage prohibitions

Do not use:

- future events from validation/test to build training behavior features,
- future item-property snapshots,
- items introduced into a historical candidate set using only future catalog evidence,
- session context carried across train/validation/test boundaries,
- transaction IDs as predictive features before the transaction,
- test outcomes to choose candidate sources, hyperparameters, relevance thresholds, or sampling policy.

## Artifact

Run:

```bash
python -m commercereclab.evaluation.temporal \
  data/raw/retailrocket/events.csv \
  --output-dir artifacts/v0_2_temporal_protocol
```

The generated JSON manifest freezes cutoffs, event counts, and sessionization settings for later milestones.
