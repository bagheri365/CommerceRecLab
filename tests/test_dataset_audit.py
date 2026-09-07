from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from matchlab.audit.dataset import audit_dataframe
from matchlab.audit.report import render_markdown, write_reports


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 1, 3, 4, 4, 5, None],
            "profile_id": [2, 1, 3, 1, 4, 4, 6, 2],
            "rating": [9, 8, 5, 7, 10, 10, "bad", 6],
        }
    )


def test_audit_reconstructs_reciprocal_pairs_without_counting_self_pair() -> None:
    audit = audit_dataframe(
        sample_frame(),
        user_col="user_id",
        profile_col="profile_id",
        rating_col="rating",
        expected_rating_min=1,
        expected_rating_max=10,
    )

    reciprocal = audit.reciprocal
    assert audit.row_count == 8
    assert audit.unique_users == 5
    assert audit.unique_profiles == 5
    assert audit.user_profile_id_overlap_count == 4
    assert reciprocal.unique_directed_pairs == 6
    assert reciprocal.duplicate_directed_pair_rows == 1
    assert reciprocal.self_pair_count == 1
    assert reciprocal.nonself_unique_directed_pairs == 5
    assert reciprocal.reciprocal_unordered_pairs == 2
    assert reciprocal.reciprocal_directed_pair_fraction == pytest.approx(4 / 5)
    assert reciprocal.rows_on_reciprocal_pairs == 4
    assert reciprocal.rows_on_one_directional_pairs == 1


def test_audit_reports_non_numeric_ratings_and_expected_range() -> None:
    frame = sample_frame().copy()
    frame.loc[len(frame)] = [6, 7, 11]

    audit = audit_dataframe(
        frame,
        user_col="user_id",
        profile_col="profile_id",
        rating_col="rating",
        expected_rating_min=1,
        expected_rating_max=10,
    )

    assert audit.rating.non_numeric_non_null_count == 1
    assert audit.rating.outside_expected_range_count == 1
    assert audit.rating.minimum == 5.0
    assert audit.rating.maximum == 11.0


def test_missing_required_column_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Missing required columns: rating"):
        audit_dataframe(
            pd.DataFrame({"user_id": [1], "profile_id": [2]}),
            user_col="user_id",
            profile_col="profile_id",
            rating_col="rating",
        )


def test_reports_are_written_as_json_and_markdown(tmp_path) -> None:
    audit = audit_dataframe(
        sample_frame(),
        user_col="user_id",
        profile_col="profile_id",
        rating_col="rating",
    )

    json_path, markdown_path = write_reports(audit, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["reciprocal"]["reciprocal_unordered_pairs"] == 2
    assert "unobserved user-profile pair" in markdown.lower()
    assert "descriptive only" in markdown.lower()
    assert render_markdown(audit) == markdown


def test_cli_writes_reports(tmp_path) -> None:
    data_path = tmp_path / "ratings.csv"
    output_dir = tmp_path / "audit-output"
    sample_frame().to_csv(data_path, index=False)

    env = os.environ.copy()
    repo_src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(repo_src)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "matchlab.audit",
            str(data_path),
            "--user-col",
            "user_id",
            "--profile-col",
            "profile_id",
            "--rating-col",
            "rating",
            "--expected-rating-min",
            "1",
            "--expected-rating-max",
            "10",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert (output_dir / "audit.json").is_file()
    assert (output_dir / "audit.md").is_file()
    assert "Wrote" in result.stdout
