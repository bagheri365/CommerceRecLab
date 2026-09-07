"""Dataset and observation audit utilities for MatchLab v0.0.

The audit is intentionally descriptive. It reports what is present in an
explicit-rating table without treating missing user-profile pairs as observed
negatives or inferring an exposure process that is not recorded in the data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnAudit:
    """Basic completeness information for one required column."""

    name: str
    dtype: str
    null_count: int
    null_fraction: float
    unique_non_null: int


@dataclass(frozen=True)
class RatingAudit:
    """Numeric rating diagnostics after coercion."""

    numeric_count: int
    non_numeric_non_null_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    expected_min: float | None
    expected_max: float | None
    outside_expected_range_count: int | None


@dataclass(frozen=True)
class ReciprocalAudit:
    """Directed-pair and reciprocal-pair diagnostics."""

    unique_directed_pairs: int
    duplicate_directed_pair_rows: int
    self_pair_count: int
    nonself_unique_directed_pairs: int
    reciprocal_unordered_pairs: int
    reciprocal_directed_pair_fraction: float
    rows_on_reciprocal_pairs: int
    rows_on_one_directional_pairs: int
    reciprocal_subset_numeric_rating_mean: float | None
    one_directional_subset_numeric_rating_mean: float | None
    numeric_rating_mean_difference: float | None


@dataclass(frozen=True)
class DatasetAudit:
    """Serializable v0.0 audit result."""

    source_path: str
    row_count: int
    required_columns: tuple[str, str, str]
    columns: tuple[ColumnAudit, ...]
    unique_users: int
    unique_profiles: int
    user_profile_id_overlap_count: int
    user_id_overlap_fraction: float
    profile_id_overlap_fraction: float
    rating: RatingAudit
    reciprocal: ReciprocalAudit
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _finite_stat(series: pd.Series, statistic: str) -> float | None:
    if series.empty:
        return None
    value = getattr(series, statistic)()
    if pd.isna(value) or not np.isfinite(value):
        return None
    return float(value)


def audit_dataframe(
    frame: pd.DataFrame,
    *,
    user_col: str,
    profile_col: str,
    rating_col: str,
    source_path: str = "<dataframe>",
    expected_rating_min: float | None = None,
    expected_rating_max: float | None = None,
) -> DatasetAudit:
    """Audit one explicit-rating table.

    Parameters are column names rather than dataset-specific assumptions so the
    same audit can be run against a local Rice/LibimSeTi extract or a compatible
    derivative. Reciprocal calculations use unique directed pairs; duplicate
    rows are reported separately.
    """

    required = (user_col, profile_col, rating_col)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    columns = tuple(
        ColumnAudit(
            name=column,
            dtype=str(frame[column].dtype),
            null_count=int(frame[column].isna().sum()),
            null_fraction=float(frame[column].isna().mean()) if len(frame) else 0.0,
            unique_non_null=int(frame[column].nunique(dropna=True)),
        )
        for column in required
    )

    users = set(frame[user_col].dropna().unique().tolist())
    profiles = set(frame[profile_col].dropna().unique().tolist())
    overlap = users & profiles

    rating_numeric = pd.to_numeric(frame[rating_col], errors="coerce")
    numeric_non_null = rating_numeric.dropna()
    non_numeric_non_null_count = int((frame[rating_col].notna() & rating_numeric.isna()).sum())

    outside_range_count: int | None = None
    if expected_rating_min is not None or expected_rating_max is not None:
        outside = pd.Series(False, index=frame.index)
        if expected_rating_min is not None:
            outside |= rating_numeric < expected_rating_min
        if expected_rating_max is not None:
            outside |= rating_numeric > expected_rating_max
        outside_range_count = int(outside.fillna(False).sum())

    rating_audit = RatingAudit(
        numeric_count=int(numeric_non_null.size),
        non_numeric_non_null_count=non_numeric_non_null_count,
        minimum=_finite_stat(numeric_non_null, "min"),
        maximum=_finite_stat(numeric_non_null, "max"),
        mean=_finite_stat(numeric_non_null, "mean"),
        median=_finite_stat(numeric_non_null, "median"),
        expected_min=expected_rating_min,
        expected_max=expected_rating_max,
        outside_expected_range_count=outside_range_count,
    )

    pair_rows = frame[[user_col, profile_col]].dropna()
    duplicate_directed_pair_rows = int(pair_rows.duplicated(keep="first").sum())
    unique_pairs = pair_rows.drop_duplicates()
    unique_pair_tuples = set(unique_pairs.itertuples(index=False, name=None))

    self_pairs = {(a, b) for a, b in unique_pair_tuples if a == b}
    nonself_pairs = {(a, b) for a, b in unique_pair_tuples if a != b}
    reciprocal_pairs = {
        frozenset((a, b))
        for a, b in nonself_pairs
        if (b, a) in nonself_pairs
    }
    reciprocal_directed_pairs = {
        pair
        for pair in nonself_pairs
        if (pair[1], pair[0]) in nonself_pairs
    }

    row_pair_tuples = list(pair_rows.itertuples(index=False, name=None))
    reciprocal_row_mask = pd.Series(False, index=frame.index)
    non_null_pair_indices = pair_rows.index
    reciprocal_row_mask.loc[non_null_pair_indices] = [
        pair in reciprocal_directed_pairs for pair in row_pair_tuples
    ]
    one_directional_mask = (
        frame[user_col].notna()
        & frame[profile_col].notna()
        & ~reciprocal_row_mask
        & (frame[user_col] != frame[profile_col])
    )

    reciprocal_ratings = rating_numeric[reciprocal_row_mask].dropna()
    one_directional_ratings = rating_numeric[one_directional_mask].dropna()
    reciprocal_mean = _finite_stat(reciprocal_ratings, "mean")
    one_directional_mean = _finite_stat(one_directional_ratings, "mean")
    mean_difference = (
        reciprocal_mean - one_directional_mean
        if reciprocal_mean is not None and one_directional_mean is not None
        else None
    )

    reciprocal_audit = ReciprocalAudit(
        unique_directed_pairs=len(unique_pair_tuples),
        duplicate_directed_pair_rows=duplicate_directed_pair_rows,
        self_pair_count=len(self_pairs),
        nonself_unique_directed_pairs=len(nonself_pairs),
        reciprocal_unordered_pairs=len(reciprocal_pairs),
        reciprocal_directed_pair_fraction=_safe_fraction(
            len(reciprocal_directed_pairs), len(nonself_pairs)
        ),
        rows_on_reciprocal_pairs=int(reciprocal_row_mask.sum()),
        rows_on_one_directional_pairs=int(one_directional_mask.sum()),
        reciprocal_subset_numeric_rating_mean=reciprocal_mean,
        one_directional_subset_numeric_rating_mean=one_directional_mean,
        numeric_rating_mean_difference=mean_difference,
    )

    notes = (
        "An unobserved user-profile pair is missing data, not an observed dislike/pass.",
        "ID overlap is reported descriptively and does not by itself prove shared identity semantics.",
        "Reciprocal counts are based on observed directed pairs and do not imply match outcomes.",
        "Reciprocal-vs-one-directional rating differences are descriptive, not causal or inferential.",
    )

    return DatasetAudit(
        source_path=source_path,
        row_count=int(len(frame)),
        required_columns=required,
        columns=columns,
        unique_users=len(users),
        unique_profiles=len(profiles),
        user_profile_id_overlap_count=len(overlap),
        user_id_overlap_fraction=_safe_fraction(len(overlap), len(users)),
        profile_id_overlap_fraction=_safe_fraction(len(overlap), len(profiles)),
        rating=rating_audit,
        reciprocal=reciprocal_audit,
        notes=notes,
    )


def audit_csv(
    path: str | Path,
    *,
    user_col: str,
    profile_col: str,
    rating_col: str,
    delimiter: str = ",",
    expected_rating_min: float | None = None,
    expected_rating_max: float | None = None,
) -> DatasetAudit:
    """Load a delimited rating table and run :func:`audit_dataframe`."""

    source = Path(path)
    frame = pd.read_csv(source, sep=delimiter)
    return audit_dataframe(
        frame,
        user_col=user_col,
        profile_col=profile_col,
        rating_col=rating_col,
        source_path=str(source),
        expected_rating_min=expected_rating_min,
        expected_rating_max=expected_rating_max,
    )
