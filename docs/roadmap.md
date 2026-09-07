# MatchLab — Bumble-Focused Dating Recommendation System Design Roadmap

## Final Project Thesis

**How should a two-sided dating recommender balance individual preference, reciprocal opportunity, eligibility, market liquidity, and serving cost?**

This project is designed as system-design preparation for dating / matching platforms.

The goal is not to build a generic recommender or to pretend a public dating dataset contains all of the signals used by a production dating app.

The goal is to:

1. use real interaction data where it is scientifically valid,
2. use controlled simulation where production-style signals are unavailable,
3. keep those two layers explicitly separate,
4. use experiments to justify architectural complexity.

---

# Core Research / Engineering Principle

> **baseline → measurable failure → targeted intervention → evaluate → retain / reject**

Every milestone should answer one narrow systems question.

Preserve negative results.

Do not add architecture merely because it is common in industry.

---

# 0. Data Reality Check

## v0.0 — Dataset Audit

Before modeling anything, audit the dating dataset.

### Primary Dataset

Use the public Rice / online-dating recommendation dataset.

It is appropriate for:

- user-to-profile preference modeling
- collaborative filtering
- matrix factorization
- rating prediction
- ranking from historical preference observations
- sparse-history analysis

It is not sufficient by itself for:

- precise geographic recommendation
- rich profile-text understanding
- real-time activity modeling
- message / reply optimization
- actual match outcomes
- relationship outcomes
- production-scale ANN evaluation

---

## Reciprocal-Pair Audit

A critical first question is:

> **Can the dataset reliably identify both A → B and B → A observations for the same real pair?**

The audit should determine:

- whether user IDs and profile IDs share the same identity space,
- whether reciprocal observations can be reconstructed,
- how many bidirectional pairs exist,
- whether bidirectional observations are dense enough for evaluation,
- whether those pairs are systematically different from one-directional observations.

### If Reliable Reciprocal Pairs Exist

Evaluate reciprocal preference directly on the observed subset.

Use language such as:

> mutual expressed preference among observed bidirectional pairs

### If They Do Not

Do not claim observed reciprocal-match evaluation.

Instead:

- keep one-sided preference modeling empirical,
- treat reciprocal scoring as a controlled architectural experiment,
- clearly mark any reciprocal counterparty signal as simulated or derived.

---

# 1. Separate Real Data From Controlled System Simulation

The project should contain two explicit experimental layers.

## Layer A — Empirical Preference Modeling

Uses real dating interaction data.

```text
observed ratings
    ↓
popularity baseline
    ↓
collaborative filtering
    ↓
matrix factorization
    ↓
pairwise ranking
    ↓
optional observed reciprocal analysis
```

Claims from this layer can concern:

- observed preference prediction,
- offline ranking,
- sparse-history behavior,
- reciprocal expressed preference if bidirectional observations support it.

---

## Layer B — Controlled System-Design Environment

Uses clearly labeled synthetic or derived fields to study production-style architecture.

```text
synthetic / controlled:
location
activity
eligibility
profile metadata
market density
exposure counters
event timing
serving latency
```

This layer is for studying:

- candidate generation,
- hard filtering,
- market liquidity,
- cold start,
- freshness,
- exposure,
- serving architecture,
- graceful degradation,
- optional ANN scaling.

Do not present Layer B outcomes as observed real-world dating behavior.

---

# 2. Observation and Evaluation Semantics

## v0.1 — Observation / Exposure Semantics

### Question

**What does an observed rating mean, what does an unobserved pair mean, and which prediction claims are identifiable from the dataset?**

The empirical dataset contains explicit profile ratings. Treat the observation unit precisely:

```text
observed(A → B)
= user A provided an explicit rating for profile B
```

Do not assume:

```text
unobserved(A → B)
= A saw B and disliked / passed on B
```

The exposure process is not directly observed. Therefore, unrated pairs should not automatically be treated as observed negatives.

### Empirical Tasks

Separate at least two tasks:

```text
Task A — explicit-rating prediction
Given that A rated B, predict rating(A, B).

Task B — ranking over a defined candidate set
Given a clearly specified candidate set and relevance rule, rank profiles for A.
```

Do not blur these tasks.

### Relevance Construction

For ranking metrics, define the relevance rule explicitly. Examples:

```text
fixed threshold:
relevant(A, B) = 1[rating(A, B) >= τ]

user-relative threshold:
relevant(A, B) = 1[rating(A, B) >= user-specific quantile estimated from training history only]

graded relevance:
gain = f(rating)
```

Report sensitivity to the relevance definition when conclusions depend on it.

### Candidate-Set Semantics

Every ranking experiment must specify:

- which profiles are eligible candidates,
- whether candidates come only from held-out observed ratings or include sampled / constructed alternatives,
- how comparison negatives are constructed,
- whether the same candidate policy is used across models,
- whether any candidate construction step uses test information.

If evaluation ranks only among held-out profiles that the user actually rated, describe the result as ranking among observed/held-out rated profiles. Do not treat that result as evidence that the system can retrieve relevant profiles from the full unseen population.

### Split Regimes

Define distinct evaluation regimes:

```text
warm-start preference
→ hold out interactions from users seen in training

sparse-history
→ retain only a controlled number of training observations per user

new-user cold start
→ hold the user's behavioral history out from model fitting

reciprocal-pair evaluation
→ use pair-level holdout appropriate to the reciprocal estimand
```

If reliable timestamps are available, add temporal evaluation for deployment-style claims.

If timestamps are not available, do not present random holdout results as prospective temporal performance.

### Leakage Rules

Freeze explicit leakage rules before experimentation.

Examples:

- no test labels in candidate construction,
- no held-out behavioral history in zero-history cold-start training,
- no reciprocal-direction leakage when evaluating prospective mutual preference,
- preprocessing parameters fit only on training data,
- user-relative thresholds or priors estimated only from training-side information,
- hyperparameters selected only on validation data.

### Statistical Unit

Match the resampling unit to the claim:

- user-level for user-centered ranking comparisons,
- pair-level for reciprocal-pair experiments,
- request-level only when requests are meaningfully independent.

---

# 3. Eligibility Semantics

## v0.2 — Eligibility and Preference Types

### Question

**Which conditions make a candidate invalid, and which merely affect ranking?**

Do not collapse all user settings into “hard constraints.”

Define four classes:

```text
1. mandatory eligibility
2. hard user preference
3. soft preference
4. ranking feature
```

Example:

| Signal | Type |
|---|---|
| blocked / excluded user | mandatory eligibility |
| incompatible orientation / preference | mandatory eligibility |
| safety / policy exclusion | mandatory eligibility |
| age range | configurable hard preference |
| distance | configurable hard preference |
| relationship goal | hard or soft depending on product semantics |
| interests | soft preference |
| activity recency | ranking feature |
| popularity / exposure | ranking or policy feature |

### Engineering Goal

The eligibility service should remove truly invalid candidates before ranking.

```text
request
  ↓
mandatory eligibility
  ↓
hard preference filtering
  ↓
candidate generation
```

Soft signals should generally remain available to ranking.

---

# 4. One-Sided Preference Modeling

## v0.3 — Preference Baselines

### Question

**How well can the system predict whom user A is likely to prefer?**

Start with:

- global profile popularity,
- shrinkage-adjusted average rating,
- user-user collaborative filtering,
- matrix factorization,
- BPR / pairwise ranking.

Do not introduce reciprocity yet.

### Core Outputs

Estimate:

```text
preference_score(A, B)
```

or, if calibrated appropriately:

```text
P(A prefers B)
```

### Metrics

Depending on the task and label construction:

- RMSE / MAE for explicit-rating prediction,
- AUC for a clearly defined binary preference target,
- Recall@K,
- NDCG@K,
- MRR.

For every ranking metric, report the relevance definition and candidate-set construction alongside the score.

Do not compare ranking metrics across experiments that silently use different candidate sets or different relevance thresholds.

---

# 5. Reciprocal Recommendation

## v0.4 — Reciprocal Scoring

### Question

**If both directional preference estimates are available, should ranking optimize one-sided preference or predicted reciprocity?**

Let:

```text
a = score(A → B)
b = score(B → A)
```

Compare:

### One-Sided

```text
score = a
```

### Product

```text
score = a * b
```

### Minimum

```text
score = min(a, b)
```

### Harmonic Mean

```text
score = 2ab / (a + b)
```

### Weighted Objective

```text
score =
alpha * a
+ (1 - alpha) * b
```

---

## Reciprocal Score Semantics

If `a` and `b` are arbitrary ranking scores, product / minimum / harmonic mean are score-fusion heuristics.

Do not interpret:

```text
a * b
```

as a mutual-preference probability unless both directional outputs are calibrated probabilities and the required dependence assumptions are stated.

Where possible, distinguish:

```text
raw_score(A → B)
calibrated P(A expresses positive preference for B)
```

A learned reciprocal combiner may also be compared against hand-designed fusion rules, but only if the added complexity is justified by held-out evaluation.

## Reciprocal Evaluation Rules

If real bidirectional observations are available:

- evaluate only on a frozen reciprocal subset,
- report reciprocal coverage,
- compare the reciprocal subset with the full population,
- explicitly discuss selection bias,
- define the reciprocal estimand before choosing the holdout policy.

For prospective mutual-preference prediction:

```text
test pair {A, B}
remove A → B and B → A from training
```

For conditional reciprocation prediction, observing one direction may be a legitimate input.

Do not mix these two tasks in one metric.

If reciprocal observations are not reliable:

- mark this milestone as controlled modeling,
- do not claim improvement in real match probability.

---

# 6. Cold Start

## v0.5 — Profile Priors to Behavioral Personalization

### Question

**How should recommendation transition from profile-derived priors to behavioral signals as interaction history accumulates?**

This is a strong system-design problem even if rich profile fields require a controlled benchmark.

Model stages:

```text
0 interactions
    → population / profile priors

1–10 interactions
    → profile + behavior

10–50 interactions
    → stronger behavioral model

50+
    → mature personalized model
```

### Possible Controlled Profile Fields

Clearly mark these as synthetic / derived if not in the real dataset:

- interests,
- relationship goal,
- lifestyle preferences,
- profile quality score,
- coarse location,
- profile embedding.

### Evaluation Cohorts

- zero history,
- sparse history,
- medium history,
- dense history.

Define the split logic explicitly:

```text
zero history
→ no behavioral interactions from the evaluation user are used to fit the behavioral model

sparse history
→ only a fixed small number of observations are retained for adaptation

medium / dense history
→ progressively more behavioral evidence is exposed
```

If profile priors depend on synthetic fields, report zero-history results as controlled-system results rather than empirical behavior from the public dataset.

Question:

> **At what history depth does behavioral personalization reliably outperform profile priors?**

---

# 7. Multi-Source Candidate Generation

## v0.6 — Candidate Retrieval Architecture

### Question

**How should the system reduce a very large eligible population to a few hundred candidates?**

Use the controlled system-design layer.

Candidate sources:

```text
geographic / local pool
collaborative retrieval
latent-factor retrieval
recently active pool
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
candidate pool
```

### Candidate-Source Ablation

Measure marginal contribution:

```text
local only                       Recall@K
+ collaborative                 ΔRecall
+ latent / embedding source     ΔRecall
+ active pool                   ΔRecall
+ exploration                   ΔRecall
```

Question:

> **Which source actually contributes unique useful candidates?**

---

# 8. Hard Filtering vs Semantic / Vector Retrieval

## v0.7 — Constraint Placement

### Question

**Where should structured eligibility constraints live relative to vector retrieval?**

Compare:

### Post-Filter

```text
ANN
 ↓
filter
```

### Pre-Filter

```text
eligible pool
 ↓
ANN
```

### Hybrid Structured Retrieval

```text
structured constraints
+
semantic similarity
```

Measure:

- valid-candidate rate,
- candidate recall,
- wasted retrieval work,
- p95 latency,
- result-set exhaustion.

### Important Scope Note

With the original dating dataset alone, ANN is not necessary at its natural scale.

This milestone should initially use exact retrieval or small-scale vector search.

Large ANN experiments belong later in the optional scaling appendix.

---

# 9. Market Liquidity

## v0.8 — Small-Market Behavior

### Question

**What should the recommender do when the eligible candidate pool becomes too small?**

Simulate markets such as:

```text
dense market
medium market
sparse market
```

Track:

```text
eligible candidate count
recommendation coverage
fallback frequency
```

Implement progressive relaxation only for explicitly soft or relaxable preferences.

Example:

```text
strict eligible pool
    ↓
too small?
    ↓
relax soft preference
    ↓
increase search radius if allowed
    ↓
widen freshness window
    ↓
inject exploration
```

Never silently relax:

- policy exclusions,
- safety constraints,
- explicit non-negotiable eligibility rules.

---

# 10. Activity and Freshness

## v0.9 — Opportunity-to-Interact

### Question

**Should a high-preference candidate be demoted if they are unlikely to be active?**

Controlled score:

```text
score =
preference
* reciprocal_opportunity
* activity_probability
```

Compare:

- preference only,
- preference + reciprocity,
- preference + reciprocity + activity.

If activity is simulated, say so explicitly.

Do not claim real engagement improvement.

### Metrics

- fraction of stale candidates,
- active-candidate coverage,
- reciprocal proxy,
- ranking quality.

---

# 11. Exposure Concentration

## v0.10 — Attention Allocation

### Question

**How should the system respond when the same highly popular profiles dominate recommendation exposure?**

Measure:

- profile exposure distribution,
- Gini coefficient,
- head / mid-tail / long-tail exposure,
- catalog coverage.

Introduce:

```text
final_score =
relevance_score
- lambda * exposure_penalty
```

Compare:

- pure relevance,
- exposure-aware reranking.

Where possible, evaluate exposure sequentially:

```text
request t
→ rank
→ serve
→ update exposure counters
→ request t+1
```

Report the relevance / concentration frontier, for example:

```text
ΔNDCG@K vs ΔGini
catalog coverage over time
```

Call this:

> **exposure concentration control**

Do not call it fairness unless a specific fairness objective is formally defined.

---

# Controlled Simulation Protocol

For Layer B experiments, define the synthetic data-generating assumptions and vary them systematically.

At minimum, vary scenarios such as:

```text
market density:             sparse / medium / dense
activity heterogeneity:     low / high
geographic dispersion:      compact / dispersed
preference concentration:   weak / strong
popularity skew:            low / high
```

For each scenario:

- run multiple random seeds,
- report variability,
- keep the simulation parameters in configuration files,
- avoid interpreting one chosen parameter setting as representative of real dating behavior,
- report where an intervention fails as well as where it helps.

The scientific claim should be:

> under these controlled assumptions, intervention X changes metric Y by Z

not:

> intervention X improves dating outcomes

---

# 12. Serving Architecture

## v0.11 — Online Recommendation Service

### Question

**How should the entire recommendation path fit within a latency and reliability budget?**

Architecture:

```text
                     +----------------+
Profile Updates ---->|  Profile Store |
                     +--------+-------+
                              |
                              v
                      Feature Pipeline
                              |
               +--------------+--------------+
               |                             |
               v                             v
        Online Feature Store          Candidate Index
               |                             |
               +--------------+--------------+
                              |
                              v
                      Candidate Service
                              |
                              v
                       Feature Hydration
                              |
                              v
                       Ranking Service
                              |
                              v
                         Policy Layer
                              |
                              v
                     Recommendation API
```

---

# 13. Event Architecture

Model events such as:

```text
profile_view
like
pass
match
message
reply
unmatch
profile_update
location_update
activity_ping
```

Flow:

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

- event-time vs processing-time,
- duplicate events,
- idempotency,
- late-arriving events,
- feature freshness,
- backfills,
- online / offline consistency.

---

# 14. Example Latency Budget

Use a hypothetical design target.

For example:

```text
Recommendation request: 150 ms p95

eligibility / filters       15 ms
candidate retrieval         30 ms
feature lookup              20 ms
ranking                     35 ms
policy reranking            10 ms
network / serialization     20 ms
headroom                    20 ms
--------------------------------
total                      150 ms
```

These are project design assumptions, not Bumble production numbers.

---

# 15. Graceful Degradation

Define fallbacks.

Examples:

```text
reciprocal model unavailable
→ one-sided ranker

candidate index unavailable
→ local / cached / popularity pool

online feature missing
→ population prior

expensive reranker timeout
→ lightweight ranker

feature store slow
→ cached features
```

Measure degraded quality and latency.

---

# 16. Optional ANN Scaling Appendix

## v0.12 — Expanded-Corpus Vector Retrieval

### Question

**At what scale does approximate retrieval become justified on a single node?**

The real Rice-style dataset is too small to make this a compelling ANN experiment by itself.

Therefore, if ANN is included:

- generate or construct an explicitly expanded candidate corpus,
- preserve the original real interactions for preference evaluation,
- use the expanded corpus only for systems scaling.

Possible corpus sizes:

```text
10k
100k
500k
1M+
```

Compare:

- exact vector search,
- Faiss Flat,
- HNSW,
- IVF.

Report:

- ANNRecall@K,
- p50 latency,
- p95 latency,
- throughput,
- memory,
- build time.

Call this:

> **controlled single-node ANN scaling**

Do not imply it reflects the natural scale of the original dataset.

---

# Offline Evaluation

## One-Sided Preference

- Recall@K
- NDCG@K
- MRR
- RMSE / MAE if appropriate
- AUC if preference is binarized

## Reciprocal Analysis

Only where scientifically supported:

- reciprocal coverage
- mutual expressed-preference proxy
- reciprocal NDCG
- one-sided vs reciprocal frontier

## Candidate Generation

- Recall@100
- Recall@500
- source overlap
- marginal source contribution

## Marketplace / Policy

- exposure Gini
- catalog coverage
- fallback frequency
- market exhaustion rate

## Systems

- p50 latency
- p95 latency
- throughput
- cache hit rate
- index memory

---

# Online Experiment Design — Conceptual Only

Document how a production experiment might be designed.

Potential primary metrics:

```text
mutual match rate
conversation initiation
reply rate
```

Potential guardrails:

```text
block / report rate
unmatch rate
session abandonment
latency
recommendation exhaustion
```

Do not claim these metrics can be measured from the public dataset.

---

# Two-Sided Experimentation Caveat

A dating platform has network interference.

Changing recommendations for user A can affect user B's experience.

Therefore standard independent-user A/B assumptions may be imperfect.

Discuss:

- naive user-level randomization,
- pair contamination,
- network effects,
- cluster / market randomization,
- two-sided randomization,
- switchback / temporal designs where appropriate,
- interference-aware estimands.

This is important system-design material.

---

# Failure Taxonomy

Every major experiment should include manual failure analysis.

Possible categories:

```text
eligibility mistake
candidate exhaustion
popular-profile dominance
cold-start failure
one-sided false positive
reciprocal-score collapse
sparse-history failure
stale candidate
activity over-penalty
candidate-source collapse
policy overcorrection
retrieval miss
```

Use a frozen sample.

Do not create categories after selectively browsing only interesting failures.

---

# Ablation Requirements

For successful complex mechanisms, remove components one at a time.

Example:

```text
full policy
- reciprocal term
- activity
- confidence gate
- exposure penalty
```

Question:

> **Which component actually caused the gain?**

---

# Statistical Protocol

For empirical experiments:

- paired bootstrap confidence intervals,
- user-level or pair-level resampling where appropriate,
- effect sizes,
- practical significance thresholds.

Report:

```text
ΔNDCG@10 = ...
95% paired bootstrap CI = [...]
```

For controlled simulations:

- repeat across multiple random seeds,
- report variability,
- distinguish simulation outcomes from empirical findings.

---

# Retain / Reject Criteria

Set thresholds after measuring baseline variance, then freeze them before intervention testing.

Examples:

Retain reciprocal scoring only if:

```text
reciprocal proxy improves materially
AND
one-sided quality loss remains within tolerance
```

Retain exposure control only if:

```text
exposure concentration drops materially
AND
ranking-quality loss remains acceptable
```

Retain ANN only if:

```text
latency gain is meaningful
AND
quality loss stays below a fixed threshold
```

---

# CPU-First Implementation

The first version should run on a Mac CPU.

Prefer:

- matrix factorization
- BPR
- logistic regression
- LightGBM
- TF-IDF
- compact embeddings
- Faiss CPU
- cached features
- deterministic subsets

Do not begin with a large deep-ranking architecture.

---

# Suggested Repository Structure

```text
matchlab/
├── README.md
├── pyproject.toml
├── CITATION.cff
├── LICENSE
│
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
# MatchLab

How should a two-sided dating recommender balance
individual preference, reciprocal opportunity,
eligibility, market liquidity, and serving cost?

## Why This Question Matters

## What This Project Is Not

## Dataset Audit

## Real vs Simulated Experimental Layers

## Observation Semantics

## Evaluation Protocol

## Eligibility Semantics

## Experimental Design

## Preference Baselines

## Reciprocal Analysis

## Cold Start

## Candidate Generation

## Liquidity

## Activity / Freshness

## Exposure Control

## Serving Architecture

## Latency Budget

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
v0.0 Dataset audit
v0.1 Observation / evaluation semantics
v0.2 Eligibility semantics
v0.3 One-sided preference baseline
v0.4 Reciprocal scoring, only if supported
v0.5 Cold-start transition
v0.6 Multi-source candidate generation
v0.7 Constraint placement
v0.8 Market liquidity
v0.9 Activity / freshness
v0.10 Exposure concentration
v0.11 Serving / degradation
v0.12 Optional ANN scaling
```

---

# Interview Questions This Project Should Prepare You For

## Design a dating recommendation system.

Discuss:

```text
eligibility
→ candidate generation
→ feature hydration
→ preference ranking
→ reciprocal opportunity
→ policy reranking
→ serving
```

## How do you retrieve candidates from a huge population?

Discuss:

- geo / structured eligibility,
- collaborative candidate generation,
- embedding retrieval,
- active pools,
- multi-source union,
- candidate recall.

## How do you handle a new user?

Discuss:

```text
profile / population priors
→ exploration
→ sparse behavior
→ mature personalization
```

## What if a market has too few candidates?

Discuss:

- liquidity,
- soft-preference relaxation,
- radius expansion,
- freshness expansion,
- hard-constraint preservation.

## What should the ranking objective be?

Discuss the distinction between:

```text
one-sided preference
mutual opportunity
activity
conversation likelihood
```

and the risks of proxy optimization.

## How do you avoid repeatedly showing the same popular profiles?

Discuss:

- exposure concentration,
- reranking penalties,
- coverage,
- marketplace attention allocation.

## How do recommendations react quickly to new behavior?

Discuss:

- event streams,
- online state,
- cache invalidation,
- asynchronous feature updates,
- embedding refresh.

## What if the ranker is too slow?

Discuss:

- top-N reranking,
- precomputation,
- lightweight fallbacks,
- candidate-source timeouts,
- graceful degradation.

## How would you evaluate the system?

Offline:

- ranking quality,
- reciprocal proxy where valid,
- candidate recall,
- cohort behavior,
- exposure,
- latency.

Online:

- mutual matches,
- conversations,
- replies,
- safety guardrails,
- latency.

---

# Final Design Principle

Do not reduce the problem to:

> **Who will user A like?**

The stronger system-design question is:

> **How should a two-sided recommendation platform allocate limited attention when both participants have preferences, eligibility constraints, uncertain reciprocal opportunity, changing availability, and unequal exposure?**

The guiding principle remains:

> **Complexity must earn its place through evidence.**

And the scientific order of operations should be:

```text
data semantics
→ estimand
→ evaluation design
→ baseline
→ measurable failure
→ targeted intervention
→ retain / reject
```
