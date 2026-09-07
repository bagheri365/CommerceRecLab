# MatchLab

**Controlled experiments in reciprocal recommendation, personalization, candidate generation, and two-sided marketplace system design for dating platforms.**

> **Research question:** How should a two-sided recommendation platform allocate limited attention when both participants have preferences, eligibility constraints, uncertain reciprocal opportunity, changing availability, and unequal exposure?

MatchLab is a research-minded system-design project for studying dating recommendation systems. It combines **empirical preference modeling on public rating data** with a **separate, explicitly controlled simulation layer** for production-style signals that the public dataset does not contain.


## Scientific principle

```text
baseline → measurable failure → targeted intervention → evaluate → retain / reject
```

Complexity must earn its place through evidence. Negative results are part of the project, not something to hide.

## What is empirical vs. simulated?

### Layer A — empirical preference modeling

Uses observed profile ratings where the dataset supports the claim.

Planned experiments include:

- popularity and shrinkage baselines
- collaborative filtering
- matrix factorization
- pairwise ranking
- sparse-history evaluation
- reciprocal expressed-preference analysis, **only if reciprocal identity reconstruction is reliable**

### Layer B — controlled system-design environment

Uses clearly labeled synthetic or derived fields for signals unavailable in the public rating data, such as:

- location
- activity / freshness
- eligibility state
- market density
- exposure counters
- event timing
- serving latency
- expanded corpora for ANN scaling

Results from this layer are claims about the **controlled experiment**, not about observed real-world dating behavior.

## Observation semantics

An observed record means that user `A` provided an explicit rating for profile `B`.

```text
observed A → B rating
    = an explicit preference observation

unobserved A → B pair
    = no rating is available
    ≠ automatically a dislike, pass, or observed negative
```

The dataset does not provide a complete impression log showing every profile that was presented but left unrated. Ranking experiments therefore define their candidate sets and relevance construction explicitly rather than silently treating all missing pairs as negatives.

See [`docs/data_semantics.md`](docs/data_semantics.md).

## Evaluation discipline

Every empirical experiment must state:

1. its prediction estimand;
2. the train / validation / test unit;
3. the candidate set used for ranking;
4. how relevance is constructed;
5. how comparison candidates or negatives are formed;
6. what leakage is prohibited;
7. which claims are supported by the result.

Examples of distinct tasks include:

- **explicit-rating prediction:** predict a held-out 1–10 rating;
- **ranking among observed candidates:** rank a defined held-out set using an explicit relevance rule;
- **prospective reciprocal prediction:** hold out both `A → B` and `B → A` and predict both directions;
- **conditional reciprocation:** observe one direction and estimate the other direction.

These tasks are not interchangeable.

## Reciprocal recommendation

If reliable reciprocal pairs can be reconstructed, MatchLab will compare one-sided preference with reciprocal score fusion.

For directional outputs `a = score(A → B)` and `b = score(B → A)`, candidate fusion rules include:

```text
one-sided      a
product        a * b
minimum        min(a, b)
harmonic mean  2ab / (a + b)
weighted       αa + (1-α)b
```

When `a` and `b` are arbitrary ranking scores, these are treated as **score-fusion heuristics**. Probability language such as `P(A prefers B)` is used only when outputs have been evaluated for calibration; combining two directional probabilities also requires the relevant assumptions to be stated.

## Cold start

Cold-start evaluation is separated into distinct regimes:

```text
warm start       held-out interactions from known users
sparse history   deliberately restricted training history
new-user cold    entire behavioral history held out from model fitting
```

Profile-derived priors or synthetic metadata are kept separate from empirical behavioral claims when the real dataset does not contain the required features.

## Candidate generation and marketplace experiments

The controlled systems layer studies:

- multi-source candidate retrieval
- structured eligibility vs. semantic/vector retrieval
- small-market liquidity and preference relaxation
- activity/freshness
- exposure concentration control
- graceful degradation
- serving latency
- optional ANN scaling

ANN experiments use an explicitly expanded corpus only for systems scaling. They are not presented as the natural scale of the original public dataset.

## Planned metrics

Depending on the experiment and label semantics:

**Preference / ranking**
- RMSE / MAE
- AUC
- Recall@K
- NDCG@K
- MRR
- calibration diagnostics when probability interpretation is claimed

**Reciprocal analysis**
- reciprocal coverage
- mutual expressed-preference proxy
- reciprocal ranking quality
- one-sided vs. reciprocal quality frontier

**Candidate generation**
- Recall@100 / Recall@500
- source overlap
- marginal source contribution

**Marketplace / policy**
- exposure Gini
- catalog coverage
- fallback frequency
- market exhaustion rate

**Systems**
- p50 / p95 latency
- throughput
- cache hit rate
- index memory

## Statistical protocol

Empirical experiments will use effect sizes and uncertainty estimates such as paired bootstrap confidence intervals, with the resampling unit chosen to match the estimand (for example, user-level or pair-level resampling).

Controlled simulations will be repeated across random seeds and across scenario settings such as market density, activity heterogeneity, preference concentration, geography, and popularity skew. Simulation outcomes remain explicitly separate from empirical findings.

## Repository structure

```text
MatchLab/
├── README.md
├── LICENSE
├── pyproject.toml
├── CITATION.cff
├── configs/
├── experiments/
├── artifacts/
├── docs/
│   ├── roadmap.md
│   └── data_semantics.md
├── src/
│   └── matchlab/
│       ├── data/
│       ├── audit/
│       ├── eligibility/
│       ├── retrieval/
│       ├── features/
│       ├── models/
│       ├── ranking/
│       ├── reciprocity/
│       ├── policy/
│       ├── simulation/
│       ├── serving/
│       └── evaluation/
└── tests/
```

## Build order

The detailed roadmap lives in [`docs/roadmap.md`](docs/roadmap.md). The first implementation milestones are intentionally methodological:

```text
v0.0  Dataset identity / schema audit
v0.1  Observation & evaluation semantics
v0.2  Eligibility semantics
v0.3  One-sided preference baselines
v0.4  Reciprocal-pair audit and reciprocal scoring, if supported
...
```

The project does **not** begin with a large deep-ranking architecture.

## First milestone

The first executable milestone should produce a reproducible dataset audit answering questions such as:

- What does each identifier represent?
- Are user and profile identifiers in a shared identity space?
- How many ratings, raters, and rated profiles are present?
- What is the rating distribution?
- How sparse is the observed matrix?
- Can reciprocal `A → B` / `B → A` pairs be reconstructed reliably?
- How much reciprocal coverage exists?
- Are reciprocal pairs systematically different from one-directional observations?
- What can and cannot be inferred from missing ratings?

No recommendation model should be trusted before these questions are answered.

## Reproducibility

The project targets a CPU-first workflow and favors deterministic, inspectable experiments before expensive models. Seeds, dataset transformations, split definitions, experiment configs, and generated artifacts should be versioned or reproducibly generated.

## Status

**Early development.** The repository currently contains the scientific roadmap and project skeleton. Results will be added milestone by milestone rather than claimed in advance.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
