# CommerceRecLab

**An end-to-end e-commerce recommender-systems case study: from raw behavioral logs to a final retrieval + ranking design.**

> **Research question:** How should an e-commerce recommender balance user intent, product relevance, conversion opportunity, catalog coverage, freshness, and serving cost?

CommerceRecLab is a reproducible research project built on the Retailrocket e-commerce dataset. It does not start by assuming that a complex model is best. Instead, it tests increasingly sophisticated ideas under one frozen temporal evaluation protocol and keeps only the changes that earn their place on validation data.

## At a glance

| | |
|---|---|
| **Data** | Retailrocket visitor-item events: views, add-to-cart actions, transactions, and time-varying item metadata |
| **Problem** | Recommend useful products from the current session while balancing relevance, conversion-oriented intent, catalog coverage, and serving cost |
| **Final design** | `c300_b100_p0` candidate retrieval + category-first deterministic ranking |
| **Main result** | The selected retriever uses about **54% fewer candidates** than the expanded reference while retaining about **97% of its task-level recall** |
| **Main lesson** | Better retrieval and category context helped consistently; more elaborate reranking did not consistently beat the simple category-first baseline |
| **Evaluation** | Leakage-safe temporal splits, point-in-time metadata, validation-only model decisions, test-only confirmation |

If you only have a minute, read **At a glance**, **Final system**, and **What we tried**. The rest documents the research protocol and reproducibility details.

## Final system

The conservative v1.0 architecture is:

```text
current session context
        ↓
c300_b100_p0 candidate generation
(category depth 300, behavioral depth 100, parent depth 0)
        ↓
category-first deterministic ranking
        ↓
top-K recommendations
```

In plain English, the system first builds a manageable shortlist of plausible products, using category context most heavily and behavioral signals at a smaller depth. It then ranks that shortlist with a simple category-first policy.

### What we learned

> **Retrieval quality and category context earned their complexity; more elaborate reranking did not under the frozen offline protocol.**

The selected retriever preserves most of the expanded system's coverage while roughly halving the number of candidates that must be ranked:

| Metric | Selected `c300_b100_p0` | Expanded reference |
|---|---:|---:|
| Test mean candidates | **319.1** | 689.6 |
| Test view candidate recall | **0.6222** | 0.6441 |
| Test cart candidate recall | **0.6420** | 0.6616 |
| Test transaction candidate recall | **0.6366** | 0.6512 |

The final category-first ranking reference is:

| Task | Test NDCG@20 |
|---|---:|
| View | **0.1252** |
| Add-to-cart | **0.1497** |
| Transaction | **0.1542** |

**How to read these numbers:** candidate recall asks whether the true future item appears anywhere in the generated shortlist. NDCG@20 then asks whether relevant items are placed near the top of the final 20 recommendations. Higher is better for both.

A learned view reranker reached test NDCG@20 **0.1349**, but it remains exploratory follow-up evidence rather than part of the final architecture because the predeclared v0.8 system-level retain rule failed.

See [`docs/v1_0_final_system_selection.md`](docs/v1_0_final_system_selection.md) for the full retrospective.

## How the project was run

```text
baseline → measurable failure → targeted intervention → evaluate → retain / reject
```

Three rules guide the project:

1. **Define what the data actually observes.** A missing visitor-item event is unknown, not a dislike and not proof that a recommendation failed.
2. **Prevent future information from leaking backward.** Time splits, session boundaries, candidate eligibility, and item metadata are fixed before model comparison.
3. **Make complexity earn its place.** If a more complicated method does not improve the predeclared validation criteria, it is rejected and the negative result stays in the record.

## Data and what counts as feedback

CommerceRecLab uses Retailrocket's public e-commerce data:

```text
events.csv
item_properties_part1.csv
item_properties_part2.csv
category_tree.csv
```

The behavioral log contains timestamped:

```text
view → addtocart → transaction
```

An observed event means Retailrocket recorded a visitor-item action at a timestamp. It does **not** establish that unobserved items were shown and rejected.

```text
observed (visitor, item, event, time)
    = recorded behavior

missing visitor-item pair
    = no recorded event
    ≠ observed negative
```

Item properties are time-varying, so historical features use point-in-time joins only. Future catalog state must never leak into past recommendations. Cart and transaction ranking metrics are conditional on queries containing corresponding future targets; they are **not** conversion-probability estimates.

See [`docs/data_semantics.md`](docs/data_semantics.md) and [`docs/v0_2_temporal_evaluation_protocol.md`](docs/v0_2_temporal_evaluation_protocol.md).

## What we tried

| Milestone | Intervention | Decision | Main result |
|---|---|---|---|
| v0.3 | Behavioral baselines | Reference | Category-conditioned popularity was strongest across view/cart/transaction |
| v0.4 | Equal-weight behavioral fusion | Reject | Mean validation NDCG 0.1163 vs category-only 0.1692 |
| v0.5 | Task-specific logistic rankers | Reject | View improved, cart/transaction NDCG degraded |
| v0.6 | Expanded candidate retrieval | **Retain** | Mean validation candidate recall 0.6110 → 0.7223 |
| v0.7 | Retrieval efficiency sweep | **Select** | `c300_b100_p0` retained ≥95% of expanded-reference recall per task |
| v0.8 | Learned reranking on efficient retrieval | Reject | Stronger retrieval did not rescue cart/transaction learned ranking |
| v0.9 | Funnel-stage conditional ranking | Reject | Post-cart transaction improved over pooled learned ranking but remained below category-only |
| v1.0 | Final retrospective | Finalize | Select efficient retrieval + category-first ranking |

The table is the short version of the research story: retrieval improvements survived validation; most reranking complexity did not. Detailed milestone reports live under [`docs/`](docs/).

## How evaluation stayed fair

The frozen offline protocol uses:

- deterministic temporal train/validation/test splits;
- 30-minute inactivity sessionization;
- split boundaries that force new sessions;
- prediction horizon = remainder of the current split-bounded session;
- backward point-in-time item-property joins;
- candidate eligibility restricted to information available at prediction time;
- no transaction IDs as pre-outcome features;
- separate view, add-to-cart, and transaction objectives;
- validation-only retain/reject decisions and test-only confirmation.

These rules are implemented in `src/commercereclab/evaluation/` and summarized in [`docs/v0_2_temporal_evaluation_protocol.md`](docs/v0_2_temporal_evaluation_protocol.md).

## Run it yourself

### 1. Environment

CommerceRecLab requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest
```

Run lint checks:

```bash
ruff check .
```

### 2. Data

Place the Retailrocket files under:

```text
data/raw/retailrocket/
├── events.csv
├── item_properties_part1.csv
├── item_properties_part2.csv
└── category_tree.csv
```

Raw data are intentionally not committed to Git.

### 3. Freeze the temporal protocol

```bash
python -m commercereclab.evaluation.temporal \
  data/raw/retailrocket/events.csv \
  --output-dir artifacts/v0_2_temporal_protocol
```

### 4. Run the final retained retrieval/ranking evidence

```bash
python -m commercereclab.evaluation.efficiency \
  data/raw/retailrocket \
  --manifest artifacts/v0_2_temporal_protocol/temporal_split_manifest.json \
  --output-dir artifacts/v0_7_retrieval_efficiency
```

### 5. Generate the v1.0 retrospective

After generating the v0.3-v0.9 artifacts:

```bash
python -m commercereclab.evaluation.final \
  --artifacts-root artifacts \
  --output-dir artifacts/v1_0_final_system_selection
```

The retrospective CLI validates the expected retained/rejected decisions and fails if required milestone artifacts are missing or inconsistent.

## Detailed experiment reports

| Version | Topic | Report |
|---|---|---|
| v0.0 | Dataset + observation audit | [`docs/v0_0_dataset_observation_audit.md`](docs/v0_0_dataset_observation_audit.md) |
| v0.1 | Observation semantics | [`docs/v0_1_observation_semantics.md`](docs/v0_1_observation_semantics.md) |
| v0.2 | Temporal evaluation protocol | [`docs/v0_2_temporal_evaluation_protocol.md`](docs/v0_2_temporal_evaluation_protocol.md) |
| v0.3 | Behavioral baselines | [`docs/v0_3_behavioral_baselines.md`](docs/v0_3_behavioral_baselines.md) |
| v0.4 | Hybrid behavioral ablation | [`docs/v0_4_hybrid_behavioral_ablation.md`](docs/v0_4_hybrid_behavioral_ablation.md) |
| v0.5 | Learned task-specific rankers | [`docs/v0_5_learned_task_specific_rankers.md`](docs/v0_5_learned_task_specific_rankers.md) |
| v0.6 | Candidate retrieval | [`docs/v0_6_candidate_retrieval.md`](docs/v0_6_candidate_retrieval.md) |
| v0.7 | Retrieval efficiency frontier | [`docs/v0_7_retrieval_efficiency.md`](docs/v0_7_retrieval_efficiency.md) |
| v0.8 | Learned reranking on efficient retrieval | [`docs/v0_8_learned_reranking_efficient_retrieval.md`](docs/v0_8_learned_reranking_efficient_retrieval.md) |
| v0.9 | Stage-conditional ranking | [`docs/v0_9_stage_aware_ranking.md`](docs/v0_9_stage_aware_ranking.md) |
| v1.0 | Final system selection | [`docs/v1_0_final_system_selection.md`](docs/v1_0_final_system_selection.md) |

## Repository structure

```text
CommerceRecLab/
├── README.md
├── LICENSE
├── pyproject.toml
├── CITATION.cff
├── configs/
├── experiments/
├── artifacts/
├── docs/
├── src/
│   └── commercereclab/
│       ├── audit/
│       ├── data/
│       ├── eligibility/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       ├── policy/
│       ├── ranking/
│       ├── retrieval/
│       └── serving/
└── tests/
```

## What this project does — and does not — claim

CommerceRecLab is an offline research benchmark and portfolio project, not a claim to reproduce a retailer's production recommender. The results support conclusions about the defined Retailrocket tasks and frozen evaluation protocol. They do **not** establish causal sales lift, online conversion impact, or real production latency.

For the full research plan and system-design extensions, see [`docs/roadmap.md`](docs/roadmap.md).
