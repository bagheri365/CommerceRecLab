import json
from pathlib import Path

import numpy as np
import pandas as pd

from commercereclab.evaluation.learned import (
    _candidate_features,
    _fit_model,
    evaluate_learned_ranker,
    write_report,
)


def _write_dataset(root: Path) -> Path:
    data = root / "retailrocket"
    data.mkdir()
    rows = []
    # Early source-fit history.
    for v, start, target in [(1, 1000, 11), (2, 1200, 12), (3, 1400, 11), (4, 1600, 12)]:
        rows += [[start, v, "view", 10, None], [start + 50, v, "view", target, None]]
    # Later internal ranker-training sessions.
    rows += [
        [3000, 5, "view", 10, None], [3050, 5, "view", 11, None], [3060, 5, "addtocart", 11, None], [3070, 5, "transaction", 11, 1],
        [3200, 6, "view", 10, None], [3250, 6, "view", 12, None], [3260, 6, "addtocart", 12, None], [3270, 6, "transaction", 12, 2],
        # Validation.
        [5000, 7, "view", 10, None], [5050, 7, "view", 11, None], [5060, 7, "addtocart", 11, None], [5070, 7, "transaction", 11, 3],
        # Test.
        [7000, 8, "view", 10, None], [7050, 8, "view", 12, None], [7060, 8, "addtocart", 12, None], [7070, 8, "transaction", 12, 4],
    ]
    pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid", "transactionid"]).to_csv(data / "events.csv", index=False)
    props = pd.DataFrame(
        [[900, 10, "categoryid", "A"], [900, 11, "categoryid", "A"], [900, 12, "categoryid", "A"]],
        columns=["timestamp", "itemid", "property", "value"],
    )
    props.iloc[:2].to_csv(data / "item_properties_part1.csv", index=False)
    props.iloc[2:].to_csv(data / "item_properties_part2.csv", index=False)
    pd.DataFrame({"categoryid": [1], "parentid": [None]}).to_csv(data / "category_tree.csv", index=False)
    return data


def _write_manifest(root: Path) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps({
        "train_end_ms": 4000,
        "validation_end_ms": 6000,
        "session_gap_minutes": 30,
        "session_boundary_policy": "split_boundary_breaks_session",
        "prediction_horizon": "remainder_of_current_split_bounded_session",
    }))
    return path


def test_candidate_features_encode_rank_and_presence():
    items, x = _candidate_features(
        {"category": [2, 1], "covisitation": [1], "recency": [3], "visitor_history": []},
        source_depth=3,
        rrf_offset=1,
    )
    assert items == [1, 2, 3]
    assert x.shape == (3, 8)
    row1 = x[items.index(1)]
    assert row1[0] > 0 and row1[1] > 0
    assert row1[4] == 1 and row1[5] == 1


def test_logistic_model_fits_binary_examples():
    x = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    y = np.asarray([1, 0, 1, 0])
    model = _fit_model(x, y, seed=365)
    assert model.predict_proba([[1, 0]])[0, 1] > model.predict_proba([[0, 1]])[0, 1]


def test_learned_ranker_report_and_write(tmp_path: Path):
    dataset = _write_dataset(tmp_path)
    manifest = _write_manifest(tmp_path)
    report = evaluate_learned_ranker(
        dataset,
        manifest,
        k=2,
        source_depth=3,
        rrf_offset=1,
        internal_fit_fraction=0.5,
        max_internal_ranker_sessions=None,
        max_sessions_per_split=None,
        max_nonrelevant_per_query=3,
    )
    assert set(report.standardized_coefficients) == {"view", "addtocart", "transaction"}
    assert "learned_task_specific" in report.metrics["validation"]
    assert report.metrics["test"]["learned_task_specific"]["transaction"]["queries"] == 1
    json_path, md_path = write_report(report, tmp_path / "out")
    payload = json.loads(json_path.read_text())
    assert "validation_retain_learned_ranker" in payload
    assert "Retain learned ranker" in md_path.read_text()
