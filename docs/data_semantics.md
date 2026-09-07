# Data and Evaluation Semantics

This document defines the minimum scientific contract for empirical experiments in MatchLab.

## 1. Observation unit

For an observed directed record:

```text
A → B = r
```

`A` is the rating user, `B` is the rated profile, and `r` is the observed explicit rating.

An unobserved pair means only that no rating is available in the dataset. It must not automatically be interpreted as an observed dislike, pass, impression without engagement, or negative preference.

## 2. Exposure limitation

The public rating data do not constitute a complete impression log. Unless additional evidence establishes the exposure process, MatchLab does not assume that every unobserved profile was presented to the user.

Therefore every ranking experiment must define its candidate universe explicitly.

## 3. Empirical estimands

### Explicit-rating prediction

Given an observed pair whose rating is held out, predict the numerical rating.

Suitable metrics may include RMSE and MAE.

### Ranking on a defined candidate set

Given a user and an explicitly defined evaluation candidate set, rank candidates according to a stated relevance rule.

The report must document:

- candidate-set construction;
- positive/relevance construction;
- comparison or negative construction;
- whether the task ranks only held-out rated profiles or attempts a broader retrieval proxy.

Ranking only among held-out rated profiles must not be described as retrieval over the full unseen population.

## 4. Relevance construction

If ratings are transformed into binary or graded relevance, the transformation is part of the experimental specification.

Possible controlled alternatives include:

```text
fixed threshold
user-relative threshold
graded relevance
```

A user-relative threshold must be estimated from that user's **training history only**. Validation/test ratings must not influence the threshold used to evaluate them.

Sensitivity to the chosen relevance construction should be reported when conclusions materially depend on it.

## 5. Split regimes

### Warm-start

Users appear during training and evaluation, but evaluation observations are held out.

### Sparse-history

Only a controlled number of training observations are retained for an evaluated user.

### New-user cold start

The evaluated user's behavioral history is excluded from behavioral model fitting. Any profile-derived prior must rely only on features legitimately available at recommendation time.

If such profile features are synthetic or derived, that experiment belongs to the controlled system-design layer.

## 6. Reciprocal tasks

Two reciprocal estimands are scientifically distinct.

### Prospective reciprocal prediction

Predict both directions for a pair when neither directional rating is available to the model.

For test pair `{A, B}`, exclude both:

```text
A → B
B → A
```

from training information used for that prediction.

### Conditional reciprocation

Estimate one direction given that the opposite direction is already observed. Here the observed opposite direction is part of the task definition, not leakage.

Reports must name which estimand is being evaluated.

## 7. Reciprocal score semantics

For:

```text
a = score(A → B)
b = score(B → A)
```

product, minimum, harmonic mean, or weighted combinations are score-fusion heuristics when `a` and `b` are arbitrary scores.

A model output may be written as `P(A prefers B)` only when the target event is precisely defined and probabilistic calibration has been evaluated on held-out data. Appropriate diagnostics may include reliability plots, Brier score, expected calibration error, or another justified calibration measure.

Even calibrated directional probabilities do not make their product an automatically valid estimate of mutual preference. Any probabilistic combination must state the dependence assumptions or be presented as a reciprocal proxy.

## 8. Leakage rules

At minimum, prohibit:

- fitting preprocessing statistics on validation/test targets;
- computing user-relative relevance thresholds from validation/test ratings;
- allowing held-out reciprocal directions into a prospective reciprocal prediction;
- using future information in a temporal experiment;
- training on behavioral history for a user labeled as zero-history cold start;
- tuning intervention thresholds on the final test set.

## 9. Statistical unit

The resampling unit should follow the estimand. Depending on the experiment, this may be a rating, user, unordered pair, market, or simulation seed/scenario.

Confidence intervals should not treat strongly dependent observations as independent merely to increase the nominal sample size.

## 10. Claim discipline

Use language such as:

- observed rating prediction;
- ranking quality on the defined candidate set;
- mutual expressed-preference proxy;
- controlled simulation result;
- controlled single-node ANN scaling.

Do not infer from the public dataset alone:

- real match probability;
- conversation or reply probability;
- relationship outcomes;
- production Bumble behavior;
- causal effects of a recommendation policy.
