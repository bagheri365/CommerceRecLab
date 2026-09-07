"""Rendering helpers for MatchLab v0.0 dataset audits."""

from __future__ import annotations

import json
from pathlib import Path

from matchlab.audit.dataset import DatasetAudit


def _display(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(audit: DatasetAudit) -> str:
    """Render a compact human-readable audit report."""

    column_rows = "\n".join(
        f"| `{column.name}` | `{column.dtype}` | {column.null_count} | "
        f"{column.null_fraction:.4f} | {column.unique_non_null} |"
        for column in audit.columns
    )
    r = audit.reciprocal
    rating = audit.rating

    return f"""# MatchLab v0.0 — Dataset + Observation Audit

Source: `{audit.source_path}`

## Observation contract

This audit treats each non-null `(user, profile, rating)` record as an observed directed rating.
An unobserved user-profile pair is **not** interpreted as an observed dislike, pass, or exposure.

## Dataset shape

- Rows: **{audit.row_count}**
- Unique rating users: **{audit.unique_users}**
- Unique rated profiles: **{audit.unique_profiles}**

| Column | dtype | nulls | null fraction | unique non-null |
|---|---:|---:|---:|---:|
{column_rows}

## ID-space overlap

- IDs appearing as both user and profile: **{audit.user_profile_id_overlap_count}**
- Fraction of user IDs also seen as profile IDs: **{audit.user_id_overlap_fraction:.4f}**
- Fraction of profile IDs also seen as user IDs: **{audit.profile_id_overlap_fraction:.4f}**

These values are descriptive. Numeric/string overlap alone does not establish that the two columns share a guaranteed identity namespace.

## Rating diagnostics

- Numeric ratings: **{rating.numeric_count}**
- Non-numeric, non-null ratings: **{rating.non_numeric_non_null_count}**
- Minimum: **{_display(rating.minimum)}**
- Maximum: **{_display(rating.maximum)}**
- Mean: **{_display(rating.mean)}**
- Median: **{_display(rating.median)}**
- Expected range: **{_display(rating.expected_min)} to {_display(rating.expected_max)}**
- Numeric ratings outside expected range: **{_display(rating.outside_expected_range_count)}**

## Directed-pair diagnostics

- Unique directed pairs: **{r.unique_directed_pairs}**
- Duplicate directed-pair rows beyond first occurrence: **{r.duplicate_directed_pair_rows}**
- Unique self-pairs: **{r.self_pair_count}**
- Non-self unique directed pairs: **{r.nonself_unique_directed_pairs}**

## Reciprocal-pair diagnostics

- Reciprocal unordered pairs: **{r.reciprocal_unordered_pairs}**
- Fraction of non-self directed pairs with an observed reverse direction: **{r.reciprocal_directed_pair_fraction:.4f}**
- Rows belonging to reciprocal pairs: **{r.rows_on_reciprocal_pairs}**
- Rows belonging to one-directional non-self pairs: **{r.rows_on_one_directional_pairs}**
- Mean numeric rating on reciprocal-pair rows: **{_display(r.reciprocal_subset_numeric_rating_mean)}**
- Mean numeric rating on one-directional rows: **{_display(r.one_directional_subset_numeric_rating_mean)}**
- Descriptive mean difference (reciprocal − one-directional): **{_display(r.numeric_rating_mean_difference)}**

## Scientific interpretation

- Reciprocal-pair availability can justify later analysis of **observed bidirectional expressed preference** if identity semantics are independently validated.
- These records do not establish impressions, passes, matches, conversations, replies, or relationship outcomes.
- The reciprocal-subset comparison above is descriptive only; selection into that subset may be non-random.

## Audit notes

""" + "\n".join(f"- {note}" for note in audit.notes) + "\n"


def write_reports(audit: DatasetAudit, output_dir: str | Path) -> tuple[Path, Path]:
    """Write machine-readable JSON and human-readable Markdown reports."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    json_path = output / "audit.json"
    markdown_path = output / "audit.md"

    json_path.write_text(json.dumps(audit.to_dict(), indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")
    return json_path, markdown_path
