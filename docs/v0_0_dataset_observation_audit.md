# v0.0 — Retailrocket Dataset + Observation Audit

## Goal

Before fitting a recommender, establish what the Retailrocket release actually contains, whether the four files are internally coherent, and which evaluation claims are scientifically supportable.

The audit is deliberately descriptive. It does not manufacture negative feedback from missing visitor-item pairs and it does not treat an observed event as proof that the item was recommended by a particular ranking system.

## Questions answered

1. Are all four canonical files present with the expected columns?
2. How many event rows, visitors, items, and event types exist?
3. What is the observed timestamp range?
4. Are duplicate or unexpected event records present?
5. Are transaction IDs consistent with transaction events?
6. How large is the time-varying item-property history?
7. How many event items have property, category, and availability history?
8. Do item properties change over time, requiring point-in-time joins?
9. Is the category tree structurally coherent (roots, parent references, self-links, cycles)?

## Local data layout

Keep the downloaded dataset outside Git tracking:

```text
data/
└── raw/
    └── retailrocket/
        ├── events.csv
        ├── item_properties_part1.csv
        ├── item_properties_part2.csv
        └── category_tree.csv
```

`data/raw/` is ignored by the repository.

## Run the audit

From the repository root:

```bash
python -m commercereclab.audit data/raw/retailrocket \
  --output-dir artifacts/v0_0_dataset_audit
```

The item-property tables contain tens of millions of rows, so the audit streams them in chunks. The default is 500,000 rows per chunk; it can be changed if needed:

```bash
python -m commercereclab.audit data/raw/retailrocket \
  --property-chunksize 250000 \
  --output-dir artifacts/v0_0_dataset_audit
```

## Outputs

```text
artifacts/v0_0_dataset_audit/
├── audit.json
└── audit.md
```

`audit.json` is machine-readable. `audit.md` is the human-readable milestone artifact.

## Observation rules

Use language such as:

- "visitor A generated a logged view event for item B";
- "item B was added to cart";
- "a transaction event was observed";
- "no logged interaction is present for this visitor-item pair."

Do not silently replace the last statement with:

- "the visitor disliked the item";
- "the visitor passed on the item";
- "the visitor saw the recommendation and ignored it."

Those require an impression/exposure mechanism that this release does not directly identify.

## Temporal leakage rule

Every downstream experiment using item properties must use point-in-time state. For an event at time `t`, a property value must come from a timestamp `<= t`.

This rule applies to category, availability, and every opaque property used as a feature.

## Exit criteria

v0.0 is complete when:

1. the canonical Retailrocket files pass schema inspection;
2. event type/count and time-range statistics are recorded;
3. transaction-field consistency is checked;
4. event-item metadata coverage is quantified;
5. time-varying property behavior is quantified;
6. category-tree integrity is checked;
7. observation and temporal-leakage semantics are documented;
8. the generated report has been reviewed before moving to v0.1 evaluation protocol and baselines.
