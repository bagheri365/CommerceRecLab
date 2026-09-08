import numpy as np
import pandas as pd

from commercereclab.evaluation.temporal import (
    assign_temporal_split,
    point_in_time_join,
    sampled_candidate_set,
    sessionize_events,
    time_valid_catalog_items,
    temporal_cutoffs,
)


def test_temporal_split_boundaries_are_time_respecting():
    ts = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    train_end, val_end = temporal_cutoffs(ts, train_fraction=0.7, validation_fraction=0.2)
    labels = assign_temporal_split(ts, train_end_ms=train_end, validation_end_ms=val_end)
    assert train_end == 70
    assert val_end == 90
    assert labels.tolist() == ["train"] * 7 + ["validation"] * 2 + ["test"]


def test_sessionization_starts_new_session_only_after_gap():
    events = pd.DataFrame(
        {
            "visitorid": [1, 1, 1, 2],
            "timestamp": [0, 30 * 60 * 1000, 61 * 60 * 1000, 5],
        }
    )
    out = sessionize_events(events, gap_minutes=30)
    visitor1 = out[out.visitorid == 1]
    assert visitor1.session_index.tolist() == [0, 0, 1]
    assert out[out.visitorid == 2].session_index.tolist() == [0]


def test_point_in_time_join_never_uses_future_state():
    events = pd.DataFrame({"itemid": [1, 1, 1], "timestamp": [5, 10, 15]})
    props = pd.DataFrame(
        {"itemid": [1, 1], "timestamp": [7, 12], "available": [0, 1]}
    )
    joined = point_in_time_join(events, props)
    assert pd.isna(joined.iloc[0]["available"])
    assert joined.iloc[1]["available"] == 0
    assert joined.iloc[2]["available"] == 1
    assert joined.iloc[2]["timestamp_property"] == 12


def test_sampled_candidates_are_deterministic_and_keep_positives():
    catalog = np.arange(1, 101)
    positives = [5, 9]
    first = sampled_candidate_set(catalog, positives, n_negatives=10, seed=42)
    second = sampled_candidate_set(catalog, positives, n_negatives=10, seed=42)
    assert np.array_equal(first, second)
    assert {5, 9}.issubset(set(first.tolist()))
    assert len(first) == 12


def test_sessionization_forces_break_at_temporal_split_boundary():
    events = pd.DataFrame(
        {
            "visitorid": [1, 1, 1],
            "timestamp": [0, 5 * 60 * 1000, 10 * 60 * 1000],
            "split": ["train", "validation", "validation"],
        }
    )
    out = sessionize_events(events, gap_minutes=30, split_col="split")
    assert out.session_index.tolist() == [0, 1, 1]
    assert out.split_boundary_break.tolist() == [False, True, False]
    assert out.split_boundary_forced_break.tolist() == [False, True, False]


def test_time_valid_catalog_excludes_future_items():
    first_seen = pd.Series({10: 1000, 20: 2000, 30: 3000})
    eligible = time_valid_catalog_items(first_seen, prediction_time_ms=2000)
    assert eligible.tolist() == [10, 20]


def test_split_boundary_not_counted_as_forced_when_gap_already_breaks_session():
    events = pd.DataFrame(
        {
            "visitorid": [1, 1],
            "timestamp": [0, 31 * 60 * 1000],
            "split": ["train", "validation"],
        }
    )
    out = sessionize_events(events, gap_minutes=30, split_col="split")
    assert out.session_index.tolist() == [0, 1]
    assert out.split_boundary_break.tolist() == [False, True]
    assert out.split_boundary_forced_break.tolist() == [False, False]
