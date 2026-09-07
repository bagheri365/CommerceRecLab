# CommerceRecLab — E-Commerce Recommendation System Design Roadmap

## Final Project Thesis

**How should an e-commerce recommender balance user intent, product relevance, conversion opportunity, availability, catalog coverage, freshness, and serving cost?**

This project is designed as a research-minded system-design study for large-scale e-commerce recommendation.

The goal is not to build a generic recommender or to pretend a public clickstream dataset contains every signal available to a production retailer.

The goal is to:

1. use real behavioral and item data where scientifically valid,
2. define observation and evaluation semantics before modeling,
3. use controlled simulation only where production signals are unavailable,
4. justify architectural complexity experimentally,
5. preserve negative results and limitations.

---

# Core Research / Engineering Principle

> **baseline → measurable failure → targeted intervention → evaluate → retain / reject**

Every milestone should answer one narrow systems question.

Do not add architecture merely because it is common in industry.

---

# 0. Data Reality Check

## v0.0 — Retailrocket Dataset + Observation Audit

### Primary Dataset

Use the public **Retailrocket Recommender System Dataset** as the primary empirical source.

The local raw dataset contains:

```text
events.csv
item_properties_part1.csv
item_properties_part2.csv
category_tree.csv
```

### Observed Behavior Table

`events.csv` contains:

```text
timestamp
visitorid
event
itemid
transactionid
```

Observed event types are:

```text
view
addtocart
transaction
```

The dataset contains roughly 2.76 million behavioral events over about 4.5 months.

### Item Metadata

The item-property files contain:

```text
timestamp
itemid
property
value
```

Properties are time-dependent and represented as a change log.

Important scientific limitation:

- `categoryid` and `available` have interpretable semantics,
- most other property names / values are anonymized or hashed,
- hashed values may be useful for matching and representation learning,
- they should not be presented as semantically interpretable attributes.

### Category Hierarchy

`category_tree.csv` contains:

```text
categoryid
parentid
```

Use it to derive category ancestry, depth, siblings, and coarse hierarchical similarity.

### v0.0 Audit Questions

Before modeling, measure:

- event row counts,
- unique visitors,
- unique items,
- time range,
- event-type frequencies,
- null patterns,
- duplicate event patterns,
- visitor activity distribution,
- item interaction distribution,
- transaction-ID semantics,
- item-property coverage,
- temporal property updates,
- availability coverage,
- category coverage,
- category-tree integrity,
- orphan categories,
- event/property temporal overlap.

### Claims Supported by This Dataset

The empirical layer can study:

- implicit-feedback recommendation,
- view / cart / transaction behavior,
- next-item or future-interaction ranking,
- session-aware recommendation after sessionization,
- collaborative retrieval,
- item-to-item transitions,
- content / metadata-assisted recommendation,
- item cold start where metadata exists,
- temporal freshness,
- observed availability,
- category-aware retrieval,
- catalog concentration,
- candidate generation,
- offline serving experiments.

### Claims Not Directly Supported

The dataset does not directly identify:

- the complete set of products shown to a visitor,
- products shown but ignored,
- ranking position,
- recommendation source / widget,
- search queries,
- prices in directly interpretable currency,
- inventory quantity,
- margin,
- user demographics,
- user intent labels,
- causal effects of recommendations,
- real production latency.

Do not infer those as observed facts.

---

# 1. Observation and Exposure Semantics

## v0.1 — What Does an Event Mean?

### Question

**What exactly is observed, and what does missingness mean?**

An observed event means the visitor interacted with an item through a recorded event.

```text
(visitor A, item B, view)
```

means a view was logged.

It does **not** establish the full recommendation slate or exposure mechanism that produced the interaction.

Therefore:

```text
unobserved(A, B) != observed dislike
unobserved(A, B) != observed exposure-with-no-click
```

Do not automatically treat every unseen item as a true negative.

### Funnel Semantics

Treat the observed funnel as progressively stronger implicit feedback:

```text
view → addtocart → transaction
```

But do not assume every transaction must have a recorded cart event or that every cart must have a prior recorded view without checking the data.

Audit actual path frequencies first.

### Empirical Tasks

Define tasks separately.

#### Task A — Future Interaction Ranking

Given a visitor's history up to time `t`, rank items likely to receive a future event.

#### Task B — Cart Intent Ranking

Given history up to `t`, rank items likely to be added to cart.

#### Task C — Transaction Ranking

Given history up to `t`, rank items likely to be purchased.

#### Task D — Next-Item / Session Ranking

After defining sessions, predict the next interacted item.

Do not silently mix these estimands.

---

# 2. Evaluation Protocol

## v0.2 — Temporal Splits, Relevance, and Candidate Sets

### Question

**How should offline evaluation approximate future recommendation without leakage?**

### Temporal Split

Prefer time-respecting splits.

Example:

```text
train: earliest 70%
validation: next 15%
test: final 15%
```

Exact cut points should be chosen after auditing the timestamp distribution.

Never train on information whose effective timestamp is after the prediction time.

### Item-Property Time Travel

This is critical.

For a recommendation at time `t`, use the most recent item-property value with:

```text
property_timestamp <= t
```

Do not hydrate historical examples with future item properties.

### Relevance Definitions

Evaluate separate objectives:

```text
view relevance
cart relevance
transaction relevance
```

For a multi-stage objective, define weights explicitly rather than hiding them.

Example controlled objective:

```text
gain(view) = 1
gain(addtocart) = 3
gain(transaction) = 5
```

These weights are project assumptions, not observed business values.

### Candidate-Set Semantics

Report evaluation under clearly named candidate regimes.

#### Full / Large Catalog Retrieval

Rank against the eligible active catalog or a documented approximation.

#### Sampled-Negative Evaluation

If sampling is needed for speed, document:

- sampling distribution,
- number of negatives,
- seed,
- whether popular items are oversampled,
- how metrics differ from full-catalog evaluation.

Sampled evaluation must not be presented as equivalent to full-catalog retrieval.

### Primary Offline Metrics

Use metrics suited to ranking:

- Recall@K,
- NDCG@K,
- MRR,
- HitRate@K,
- MAP@K where appropriate.

Also track funnel-specific recall:

- CartRecall@K,
- TransactionRecall@K.

---

# 3. Baselines

## v0.3 — Non-Personalized and Behavioral Baselines

### Question

**How much value is obtained before adding complex personalization?**

Start with:

```text
global popularity
recent popularity
category popularity
visitor-history repeat
item co-occurrence
item-to-item transitions
```

### Required Baselines

1. most-viewed items,
2. most-carted items,
3. most-purchased items,
4. time-decayed popularity,
5. last-item co-visitation,
6. category-conditioned popularity.

### Evaluation

Report overall metrics and cohorts:

- new visitor,
- sparse-history visitor,
- repeat visitor,
- head items,
- mid-tail items,
- long-tail items.

---

# 4. User Intent and Session Modeling

## v0.4 — Long-Term Visitor History vs Short-Term Intent

### Question

**When does short-term session intent outperform long-term visitor history?**

Retailrocket provides visitor IDs and timestamps, but not explicit session IDs.

Sessionization is therefore a derived preprocessing decision.

Compare inactivity thresholds such as:

```text
15 minutes
30 minutes
60 minutes
```

Freeze the chosen rule before downstream model comparisons.

### Models

Compare:

- visitor-history popularity,
- last-N interactions,
- item co-visitation,
- Markov / transition model,
- matrix factorization,
- session-aware hybrid.

### Key Analysis

Measure performance as a function of:

- session length,
- visitor history depth,
- recency,
- category concentration.

---

# 5. Multi-Source Candidate Generation

## v0.5 — Retrieval Architecture

### Question

**How should the system reduce a large catalog to a few hundred high-recall candidates?**

Candidate sources:

```text
recent popularity
category popularity
co-visitation
item-to-item transitions
collaborative latent retrieval
metadata similarity
repeat / revisit candidates
exploration pool
```

Merge:

```text
source 1
source 2
source 3
source 4
   ↓
union
   ↓
deduplication
   ↓
source-aware features
   ↓
candidate pool
```

### Candidate-Source Ablation

Measure:

```text
source recall@100
source recall@500
unique hit contribution
source overlap
marginal recall contribution
```

Question:

> **Which source actually contributes useful candidates that the others miss?**

---

# 6. Ranking

## v0.6 — Funnel-Aware Ranking

### Question

**Should the ranker optimize generic interaction likelihood or downstream conversion intent?**

Candidate features may include:

### Visitor Features

- history length,
- recency,
- category preference distribution,
- event-type counts,
- repeat-item tendency.

### Item Features

- recent popularity,
- cart rate,
- transaction rate,
- category,
- category depth,
- availability,
- property embeddings / hashed-property representation,
- freshness of latest property update.

### Pair Features

- visitor-item history,
- co-visitation score,
- transition score,
- category affinity,
- latent similarity,
- source indicators.

### Models

CPU-first progression:

```text
logistic regression
LightGBM / gradient-boosted trees
matrix factorization score as feature
optional compact neural/session model later
```

Do not begin with a large deep architecture.

### Objectives

Compare separate rankers for:

```text
future view
future cart
future transaction
```

and an explicitly weighted multi-task / blended objective.

---

# 7. Availability and Eligibility

## v0.7 — Time-Valid Candidate Filtering

### Question

**Should unavailable products be filtered before retrieval, during retrieval, or during ranking?**

Retailrocket exposes an `available` property where present.

Construct historical availability using only information known at prediction time.

Compare:

### Post-Filter

```text
retrieve
   ↓
availability filter
```

### Pre-Filter

```text
available catalog
   ↓
retrieve
```

### Hybrid

Structured availability constraints plus similarity retrieval.

Measure:

- valid-candidate rate,
- candidate recall,
- wasted retrieval work,
- result-set exhaustion,
- latency.

Never use future availability states for historical recommendations.

---

# 8. Cold Start

## v0.8 — New Visitor and New Item

### Question

**How should recommendation behave before sufficient behavioral history exists?**

### New Visitor

With zero visitor history:

```text
recent popularity
category/context priors if available
exploration
```

With sparse history:

```text
recent session intent
+ category affinity
+ co-visitation
```

### New Item

Use metadata available at introduction time:

```text
category
availability
hashed properties
category hierarchy
```

Compare:

- popularity-only fallback,
- metadata similarity,
- category-conditioned retrieval,
- hybrid behavioral/content model.

### Scientific Split

A true new-item test must hold item behavior out prior to its simulated introduction point.

Do not call a warm item with withheld random interactions “cold start.”

---

# 9. Temporal Freshness and Drift

## v0.9 — Recency

### Question

**How quickly should the recommender react to changing demand and visitor intent?**

Compare:

- all-history popularity,
- exponential decay,
- rolling windows,
- session-only intent,
- hybrid long/short-term scores.

Evaluate by time period to detect drift.

Track:

- ranking quality,
- stale-item rate,
- newly active item coverage,
- category-shift responsiveness.

---

# 10. Catalog Exposure Concentration

## v0.10 — Attention Allocation

### Question

**How should the system respond when the same popular products dominate recommendations?**

Measure recommendation-side concentration, not just observed interaction concentration.

Metrics:

- catalog coverage,
- recommendation Gini,
- head / mid-tail / long-tail exposure,
- unique items recommended,
- category coverage.

Introduce a controlled reranker:

```text
final_score = relevance_score - lambda * exposure_penalty
```

Compare:

- pure relevance,
- exposure-aware reranking.

Call this **catalog exposure concentration control**.

Do not call it fairness unless a formal fairness objective is defined.

---

# 11. Controlled Marketplace Simulation

Retailrocket does not contain a complete impression/slate log or production serving environment.

Simulation is allowed only for explicitly labeled system questions such as:

- request arrival rate,
- recommendation exposure counters,
- cache behavior,
- feature-store delay,
- serving latency,
- inventory shocks beyond observed availability,
- large-corpus scaling.

### Simulation Protocol

For each synthetic factor, define a range rather than one arbitrary setting.

Example:

```text
traffic: low / medium / high
catalog skew: weak / medium / strong
cache hit rate: low / medium / high
feature delay: 0 / 5 / 30 minutes
```

Repeat across seeds and report variability.

Do not present simulation outcomes as observed retailer behavior.

---

# 12. Serving Architecture

## v0.11 — Online Recommendation Service

### Question

**How should candidate generation and ranking fit inside a latency and reliability budget?**

Architecture:

```text
Behavior Events ---------> Event Stream / Log
                              |
                              v
                        Feature Pipeline
                         /           \
                        v             v
               Online Feature     Offline Store
                   Store              |
                     |                v
                     |          Model Training
                     |                |
                     |                v
                     |           Model Registry
                     |                |
                     +-------+--------+
                             |
Request ----------------> Candidate Service
                             |
                             v
                      Feature Hydration
                             |
                             v
                        Ranking Service
                             |
                             v
                       Policy / Filters
                             |
                             v
                      Recommendation API
```

Candidate indexes may include:

- popularity cache,
- co-visitation index,
- category index,
- latent-vector index,
- availability-aware item set.

---

# 13. Event Architecture

Model events such as:

```text
view
addtocart
transaction
item_property_update
availability_update
```

System flow:

```text
event
  ↓
stream / queue
  ↓
feature aggregation
  ↓
online state
  ↓
offline training log
```

Discuss:

- event time vs processing time,
- duplicates,
- idempotency,
- late events,
- feature freshness,
- backfills,
- online/offline consistency,
- historical point-in-time joins.

---

# 14. Example Latency Budget

Use a hypothetical design target, not a claimed production number.

Example:

```text
Recommendation request: 120 ms p95

request/context parsing      5 ms
candidate retrieval         30 ms
feature lookup              20 ms
ranking                     30 ms
policy/filtering            10 ms
serialization/network       10 ms
headroom                    15 ms
---------------------------------
total                      120 ms
```

These are project design assumptions.

---

# 15. Graceful Degradation

Define fallbacks.

Examples:

```text
personalized retrieval unavailable
→ recent popularity / category popularity

online features unavailable
→ cached visitor state

metadata lookup unavailable
→ behavioral-only ranker

expensive reranker timeout
→ lightweight ranker

vector index unavailable
→ co-visitation + popularity
```

Measure both degraded latency and degraded quality.

---

# 16. ANN / Retrieval Scaling

## v0.12 — Approximate Retrieval

### Question

**At what catalog size and query load does approximate vector retrieval earn its complexity?**

Start with exact retrieval where practical.

If latent or metadata embeddings are introduced, compare:

- exact dot-product search,
- Faiss Flat,
- HNSW,
- IVF where justified.

Report:

- ANNRecall@K,
- p50 latency,
- p95 latency,
- throughput,
- memory,
- index build time.

ANN is retained only if latency/throughput gains justify quality and operational costs.

---

# Offline Evaluation

## Retrieval

- Recall@100,
- Recall@500,
- source overlap,
- marginal source contribution,
- ANNRecall@K where relevant.

## Ranking

- Recall@K,
- NDCG@K,
- MRR,
- HitRate@K,
- CartRecall@K,
- TransactionRecall@K.

## Cohorts

- new visitors,
- sparse-history visitors,
- repeat visitors,
- short sessions,
- long sessions,
- head items,
- mid-tail items,
- long-tail items,
- new items.

## Catalog / Policy

- catalog coverage,
- recommendation Gini,
- category coverage,
- unavailable-item rate,
- result-set exhaustion.

## Systems

- p50 latency,
- p95 latency,
- throughput,
- cache hit rate,
- index memory,
- feature freshness.

---

# Online Experiment Design — Conceptual Only

Document how a production experiment might be designed.

Potential primary metrics:

```text
click-through rate
add-to-cart rate
conversion rate
revenue / order value if available in production
```

Potential guardrails:

```text
latency
error rate
recommendation exhaustion
catalog concentration
out-of-stock recommendation rate
session abandonment
```

Do not claim these causal effects can be measured from Retailrocket's observational logs alone.

---

# Failure Taxonomy

Every major experiment should include manual failure analysis.

Possible categories:

```text
candidate miss
popular-item dominance
cold-start failure
stale intent
category mismatch
unavailable-item leak
metadata time-travel bug
sessionization failure
co-visitation collapse
long-tail starvation
ranker overweights views
conversion objective collapse
retrieval miss
policy overcorrection
```

Use a frozen sample.

Do not create categories only after selectively browsing interesting failures.

---

# Ablation Requirements

For successful complex mechanisms, remove components one at a time.

Example:

```text
full ranker
- session features
- category affinity
- co-visitation score
- availability feature
- recency feature
- exposure penalty
```

Question:

> **Which component actually caused the gain?**

---

# Statistical Protocol

For empirical experiments:

- use paired bootstrap confidence intervals where appropriate,
- resample at visitor or session level rather than individual rows when dependence matters,
- report effect sizes,
- report practical significance thresholds,
- preserve temporal split boundaries.

Example:

```text
ΔNDCG@10 = ...
95% paired bootstrap CI = [...]
```

For controlled simulations:

- repeat across multiple random seeds,
- report variability,
- vary important simulation parameters,
- distinguish simulation outcomes from empirical findings.

---

# Retain / Reject Criteria

Set thresholds after measuring baseline variance, then freeze them before intervention testing.

Examples:

Retain a new candidate source only if:

```text
candidate recall improves materially
AND
latency / memory cost remains acceptable
```

Retain exposure control only if:

```text
catalog concentration drops materially
AND
ranking-quality loss remains within tolerance
```

Retain ANN only if:

```text
latency or throughput improves materially
AND
retrieval-quality loss stays below a fixed threshold
```

Retain a conversion-aware ranker only if:

```text
cart / transaction ranking improves
AND
view-level relevance does not collapse beyond tolerance
```

---

# CPU-First Implementation

The first version should run on a Mac CPU.

Prefer:

- pandas / Polars where useful,
- NumPy / SciPy sparse matrices,
- logistic regression,
- LightGBM or equivalent gradient boosting,
- implicit matrix factorization,
- co-visitation tables,
- compact item embeddings,
- Faiss CPU only when justified,
- Parquet intermediate artifacts,
- deterministic subsets for fast tests.

Do not begin with a large deep-ranking architecture.

---

# Suggested Repository Structure

```text
CommerceRecLab/
├── README.md
├── pyproject.toml
├── CITATION.cff
├── LICENSE
│
├── data/
│   ├── raw/              # gitignored
│   ├── interim/          # gitignored
│   └── processed/        # gitignored
│
├── src/
│   └── commercereclab/
│       ├── data/
│       ├── audit/
│       ├── sessions/
│       ├── retrieval/
│       ├── features/
│       ├── models/
│       ├── ranking/
│       ├── policy/
│       ├── simulation/
│       ├── serving/
│       └── evaluation/
│
├── configs/
├── experiments/
├── artifacts/
├── tests/
└── docs/
```

---

# README Structure

```text
# CommerceRecLab

How should an e-commerce recommender balance
user intent, product relevance, conversion opportunity,
availability, catalog coverage, freshness, and serving cost?

## Why This Question Matters
## Dataset
## What This Project Is Not
## Observation Semantics
## Evaluation Protocol
## Experimental Design
## Baselines
## Session / Intent Modeling
## Candidate Generation
## Ranking
## Availability
## Cold Start
## Freshness
## Catalog Exposure
## Serving Architecture
## Failure Analysis
## Ablations
## Negative Results
## Statistical Protocol
## Limitations
## Reproducibility
## Conclusions
```

---

# Recommended Build Order

```text
v0.0  Dataset + observation audit
v0.1  Observation / exposure semantics
v0.2  Temporal evaluation protocol
v0.3  Behavioral baselines
v0.4  Session / intent modeling
v0.5  Multi-source candidate generation
v0.6  Funnel-aware ranking
v0.7  Availability filtering
v0.8  Cold start
v0.9  Temporal freshness / drift
v0.10 Catalog exposure concentration
v0.11 Serving / degradation
v0.12 ANN scaling if justified
```

---

# Interview Questions This Project Should Prepare You For

## Design an e-commerce recommendation system.

Discuss:

```text
request context
→ candidate generation
→ feature hydration
→ intent / relevance ranking
→ availability / policy filtering
→ serving
```

## How do you retrieve from a large catalog?

Discuss:

- popularity pools,
- category retrieval,
- co-visitation,
- collaborative retrieval,
- embedding retrieval,
- source union,
- candidate recall.

## How do you model short-term shopping intent?

Discuss:

- sessionization,
- recency,
- last-item transitions,
- session vs long-term history,
- intent drift.

## How do you handle a new visitor?

Discuss:

```text
recent popularity
→ exploration
→ first interaction
→ session intent
→ mature personalization
```

## How do you handle a new item?

Discuss:

- metadata,
- category hierarchy,
- content similarity,
- exploration,
- delayed behavioral signal.

## What should the ranking objective be?

Discuss the distinction between:

```text
view likelihood
cart likelihood
transaction likelihood
business value
```

and the risks of proxy optimization.

## How do you avoid showing only popular items?

Discuss:

- catalog exposure concentration,
- reranking penalties,
- coverage,
- long-tail tradeoffs.

## How do recommendations react quickly to new behavior?

Discuss:

- event streams,
- session state,
- online features,
- cache invalidation,
- incremental candidate updates.

## What if the ranker is too slow?

Discuss:

- retrieval/ranking separation,
- top-N reranking,
- precomputation,
- caching,
- lightweight fallbacks,
- graceful degradation.

## How would you evaluate the system?

Offline:

- temporal ranking quality,
- candidate recall,
- funnel-specific metrics,
- cohort behavior,
- catalog coverage,
- availability correctness,
- latency.

Online conceptually:

- CTR,
- cart rate,
- conversion,
- business metrics,
- latency and reliability guardrails.

---

# Final Design Principle

Do not reduce the problem to:

> **Which item will this visitor click?**

The stronger system-design question is:

> **How should an e-commerce recommendation platform allocate limited attention across a changing catalog when user intent is uncertain, interaction signals vary in strength, product state changes over time, and serving resources are finite?**

The guiding principle remains:

> **Complexity must earn its place through evidence.**
