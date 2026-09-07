# v0.0 — Dataset + Observation Audit

## Goal

Before fitting a recommender, establish what the rating table actually contains and what claims it can support.

The v0.0 audit is deliberately descriptive. It does **not** infer that an unrated user-profile pair was shown and rejected, and it does not treat observed bidirectional ratings as match outcomes.

## Questions answered

1. Are the required user, profile, and rating columns present?
2. How many rows, users, profiles, missing values, and duplicate directed pairs exist?
3. Are ratings numeric, and do they respect an explicitly supplied expected range?
4. How much literal ID overlap exists between the user and profile columns?
5. How many observed non-self directed pairs have an observed reverse direction?
6. How different is the reciprocal subset from the one-directional subset at a basic descriptive level?

## Run the audit

Install the package in development mode first:

```bash
python -m pip install -e ".[dev]"
```

Then run:

```bash
python -m matchlab.audit path/to/ratings.csv \
  --user-col UserID \
  --profile-col ProfileID \
  --rating-col Rating \
  --expected-rating-min 1 \
  --expected-rating-max 10 \
  --output-dir artifacts/v0_0_dataset_audit
```

For tab-delimited input, add:

```bash
--delimiter tab
```

The expected rating range is intentionally explicit rather than hard-coded. Omit the range flags if the local extract has not yet been verified.

## Outputs

The command writes:

```text
artifacts/v0_0_dataset_audit/
├── audit.json
└── audit.md
```

`audit.json` is intended for reproducible downstream checks. `audit.md` is the human-readable milestone artifact.

## Reciprocal-pair definition

For distinct IDs `A` and `B`, a reciprocal observed pair exists only when both directed pairs occur:

```text
A → B
B → A
```

Counts are computed from **unique directed pairs**. Duplicate rows are reported separately, and self-pairs such as `A → A` are never counted as reciprocal pairs.

This audit establishes only that both directed ratings are present. Before using those pairs as observations about the same real participants, independently validate the dataset's identity semantics.

## Interpretation rules

Use conclusions such as:

- "The table contains N observed directed rating pairs."
- "X% of unique non-self directed pairs have an observed reverse direction."
- "The user and profile ID columns have Y literal IDs in common."

Do not conclude from v0.0 alone:

- that every missing pair was exposed;
- that missing means dislike/pass;
- that numeric ID overlap proves a guaranteed shared identity namespace;
- that a bidirectional rating pair is a match;
- that reciprocal-subset differences are causal.

## Exit criteria

v0.0 is complete when:

1. the raw/local dataset can be audited deterministically;
2. schema and rating-range anomalies are documented;
3. duplicate/self-pair behavior is understood;
4. ID-space overlap is quantified;
5. reciprocal coverage is quantified;
6. the generated `audit.md` is committed or its key findings are summarized in an experiment record;
7. any claim about real reciprocal identity is supported by dataset documentation rather than inferred from numeric overlap alone.

Only then move to evaluation semantics and preference baselines.
