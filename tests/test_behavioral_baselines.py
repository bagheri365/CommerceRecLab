import json

import numpy as np
import pandas as pd

from commercereclab.evaluation.baselines import (
    _recall_ndcg,
    _visitor_cohort,
    evaluate_baselines,
    write_report,
)


def _write_fixture(tmp_path):
    data = tmp_path / "retailrocket"
    data.mkdir()
    events = pd.DataFrame(
        [
            [0, 1, "view", 10, np.nan],
            [1_000, 1, "view", 11, np.nan],
            [2_000, 1, "addtocart", 11, np.nan],
            [3_000, 1, "transaction", 11, 1],
            [4_000, 2, "view", 10, np.nan],
            [5_000, 2, "view", 12, np.nan],
            [6_000, 2, "view", 10, np.nan],
            [7_000, 3, "view", 10, np.nan],
            [8_000, 3, "view", 11, np.nan],
            [9_000, 3, "transaction", 11, 2],
            [10_000, 4, "view", 10, np.nan],
            [11_000, 4, "addtocart", 11, np.nan],
            [12_000, 4, "transaction", 11, 3],
        ],
        columns=["timestamp", "visitorid", "event", "itemid", "transactionid"],
    )
    events.to_csv(data / "events.csv", index=False)
    properties = pd.DataFrame(
        [
            [0, 10, "categoryid", "A"],
            [0, 11, "categoryid", "A"],
            [0, 12, "categoryid", "B"],
        ],
        columns=["timestamp", "itemid", "property", "value"],
    )
    properties.iloc[:2].to_csv(data / "item_properties_part1.csv", index=False)
    properties.iloc[2:].to_csv(data / "item_properties_part2.csv", index=False)
    pd.DataFrame([["A", np.nan], ["B", np.nan]], columns=["categoryid", "parentid"]).to_csv(
        data / "category_tree.csv", index=False
    )
    manifest = {
        "train_end_ms": 6_000,
        "validation_end_ms": 9_000,
        "session_gap_minutes": 30,
        "session_boundary_policy": "split_boundary_breaks_session",
        "prediction_horizon": "remainder_of_current_split_bounded_session",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return data, manifest_path


def test_recall_ndcg_binary():
    recall, ndcg = _recall_ndcg([3, 2, 1], np.array([1, 3]), 3)
    assert recall == 1.0
    assert 0 < ndcg <= 1


def test_visitor_cohorts():
    assert _visitor_cohort(0) == "new"
    assert _visitor_cohort(1) == "sparse"
    assert _visitor_cohort(4) == "sparse"
    assert _visitor_cohort(5) == "repeat"


def test_evaluate_baselines_respects_frozen_manifest(tmp_path):
    data, manifest = _write_fixture(tmp_path)
    report = evaluate_baselines(data, manifest, k=2, neighbor_top_n=10)
    assert report.training_rows == 7
    assert report.prediction_horizon == "remainder_of_current_split_bounded_session"
    assert "most_view" in report.baselines
    assert "last_item_transition" in report.baselines
    assert report.metrics["validation"]["most_view"]["transaction"]["queries"] == 1
    assert report.metrics["test"]["most_view"]["transaction"]["queries"] == 1


def test_report_writes_json_and_markdown(tmp_path):
    data, manifest = _write_fixture(tmp_path)
    report = evaluate_baselines(data, manifest, k=2, neighbor_top_n=10)
    json_path, md_path = write_report(report, tmp_path / "out")
    payload = json.loads(json_path.read_text())
    assert payload["k"] == 2
    assert "Behavioral Baselines" in md_path.read_text()
