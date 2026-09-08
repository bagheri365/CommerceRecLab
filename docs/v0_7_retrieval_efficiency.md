# CommerceRecLab v0.7 — Retrieval Efficiency / Pareto Frontier

## Question

v0.6 showed that expanding category and hierarchy retrieval materially increases candidate recall, but the highest-recall configuration also grows the candidate set substantially. v0.7 asks a narrower systems question:

> How much of the expanded-retrieval recall can we preserve with a smaller candidate budget?

This milestone is upstream of ranking. It does not fit a learned ranker and does not reinterpret missing interactions as negatives.

## Frozen evaluation contract

The experiment reuses the v0.2 split-bounded sessions and prediction horizon. Retrieval statistics are fit on the frozen training period only. Validation selects the efficient configuration; test is confirmation only.

Candidate recall is measured before top-K ranking and remains conditional on queries that contain a future target for the corresponding task.

## Budget sweep

The default sweep crosses:

- same-category depth: `100, 200, 300`
- behavioral depth for co-visitation, recency, and visitor history: `100, 300`
- parent/sibling-category expansion depth: `0, 100, 300`

The maximum-depth combination is the **expanded reference**. It is a declared engineering reference point, not an oracle or an upper bound on achievable recall.

Each configuration reports task-specific candidate recall plus mean, median, and p95 candidate-set size.

## Pareto frontier

A validation configuration is dominated when another configuration has both:

- equal-or-higher mean candidate recall, and
- equal-or-smaller mean candidate-set size,

with at least one strict improvement.

The non-dominated configurations form the reported validation Pareto frontier. Mean recall is used only for the two-dimensional frontier display; task-specific recalls remain the scientific quantities used by the retention rule.

## Efficient-configuration rule

The default retention threshold is 95%.

A configuration is eligible only when **each** of view, cart, and transaction candidate recall on validation is at least 95% of the corresponding expanded-reference recall. Among eligible configurations, select the one with the smallest mean validation candidate set, breaking ties by higher mean candidate recall.

The test split does not change the selected configuration. It only reports whether the validation-selected efficiency tradeoff persists temporally.

## Run

```bash
python -m commercereclab.evaluation.efficiency \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --output-dir artifacts/v0_7_retrieval_efficiency
```

Outputs:

- `retrieval_efficiency.json`
- `retrieval_efficiency.md`

## Claim discipline

v0.7 supports claims about offline candidate coverage and candidate-set size under the declared Retailrocket protocol. Candidate count is a serving-cost proxy, not measured production latency, CPU utilization, or infrastructure cost. Those require separate systems experiments.
