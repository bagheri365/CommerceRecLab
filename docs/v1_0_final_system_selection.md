# CommerceRecLab v1.0 — Final System Selection + Retrospective

v1.0 closes the initial Retailrocket research arc by selecting a conservative final system from the frozen v0.3–v0.9 evidence rather than introducing another model family.

## Decision rule

The retrospective preserves every milestone's original validation-only retain/reject decision. Test results are used as confirmation, not as a new tuning source. v1.0 therefore does **not** retroactively promote an intervention that failed its predeclared rule.

## Final system

The conservative final architecture is:

```text
session prefix
    ↓
c300_b100_p0 candidate generation
    ├─ same-category depth: 300
    ├─ behavioral depth: 100
    └─ parent/sibling depth: 0
    ↓
category-first deterministic ranking
    ↓
top-K recommendations
```

Why this architecture:

- v0.3 established category-conditioned popularity as the strongest simple baseline for all three objectives.
- v0.6 showed that retrieval expansion materially increased candidate recall.
- v0.7 selected `c300_b100_p0` as the smallest validation configuration retaining at least 95% of expanded-reference recall for every task.
- v0.4, v0.5, v0.8, and v0.9 all rejected more complex ranking interventions under their declared validation rules.

The v0.8 learned reranker improved the **view** objective in isolation, but it is treated as an exploratory follow-up rather than a retained final component because v0.8 failed its system-level retain rule. A future view-only model should be revalidated on a new holdout or dataset before being promoted.

## Reproduce the retrospective

First generate and preserve the frozen artifact JSONs from v0.3 through v0.9. Then run:

```bash
python -m commercereclab.evaluation.final \
  --artifacts-root artifacts \
  --output-dir artifacts/v1_0_final_system_selection
```

The command validates expected milestone decisions and writes:

```text
artifacts/v1_0_final_system_selection/final_system_selection.json
artifacts/v1_0_final_system_selection/final_system_selection.md
```

If an expected artifact is missing or a frozen decision no longer matches the retrospective assumptions, the command fails rather than silently generating a misleading summary.

## Claim discipline

The final architecture is selected for the defined Retailrocket offline tasks. It is not a causal claim about production conversion lift. Cart and transaction ranking metrics remain conditional on queries with corresponding future targets, missing events are not interpreted as dislikes or impressions, and all item/category state remains subject to the v0.2 point-in-time leakage contract.
