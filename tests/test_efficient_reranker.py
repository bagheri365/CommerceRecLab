import json
from pathlib import Path

import pandas as pd
import pytest

from commercereclab.evaluation.rerank import (
    _candidate_features,
    evaluate_efficient_reranker,
    write_report,
)


def _write_dataset(root: Path) -> Path:
    data = root / "retailrocket"
    data.mkdir()
    rows = []
    for v, start, target in [(1, 1000, 11), (2, 1200, 12), (3, 1400, 11), (4, 1600, 12)]:
        rows += [[start, v, "view", 10, None], [start + 50, v, "view", target, None]]
    rows += [
        [3000, 5, "view", 10, None], [3050, 5, "view", 11, None],
        [3060, 5, "addtocart", 11, None], [3070, 5, "transaction", 11, 1],
        [3200, 6, "view", 10, None], [3250, 6, "view", 12, None],
        [3260, 6, "addtocart", 12, None], [3270, 6, "transaction", 12, 2],
        [5000, 7, "view", 10, None], [5050, 7, "view", 11, None],
        [5060, 7, "addtocart", 11, None], [5070, 7, "transaction", 11, 3],
        [7000, 8, "view", 10, None], [7050, 8, "view", 12, None],
        [7060, 8, "addtocart", 12, None], [7070, 8, "transaction", 12, 4],
    ]
    pd.DataFrame(
        rows,
        columns=["timestamp", "visitorid", "event", "itemid", "transactionid"],
    ).to_csv(data / "events.csv", index=False)
    props = pd.DataFrame(
        [[900, 10, "categoryid", "1"], [900, 11, "categoryid", "1"], [900, 12, "categoryid", "1"]],
        columns=["timestamp", "itemid", "property", "value"],
    )
    props.iloc[:2].to_csv(data / "item_properties_part1.csv", index=False)
    props.iloc[2:].to_csv(data / "item_properties_part2.csv", index=False)
    pd.DataFrame({"categoryid": [1], "parentid": [None]}).to_csv(
        data / "category_tree.csv", index=False
    )
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


def test_candidate_features_use_asymmetric_depths():
    items, x = _candidate_features(
        {
            "category": [1, 2, 3],
            "covisitation": [4, 5, 6],
            "recency": [7, 8, 9],
            "visitor_history": [10, 11, 12],
        },
        category_depth=3,
        behavioral_depth=1,
        rrf_offset=1,
    )
    assert items == [1, 2, 3, 4, 7, 10]
    assert x.shape == (6, 8)


def test_parent_depth_is_frozen_to_zero(tmp_path: Path):
    dataset = _write_dataset(tmp_path)
    manifest = _write_manifest(tmp_path)
    with pytest.raises(ValueError, match="parent_depth"):
        evaluate_efficient_reranker(
            dataset,
            manifest,
            k=1,
            category_depth=2,
            behavioral_depth=1,
            parent_depth=1,
            internal_fit_fraction=0.5,
            max_internal_ranker_sessions=None,
            max_sessions_per_split=None,
        )


def test_efficient_reranker_report_and_write(tmp_path: Path):
    dataset = _write_dataset(tmp_path)
    manifest = _write_manifest(tmp_path)
    report = evaluate_efficient_reranker(
        dataset,
        manifest,
        k=1,
        category_depth=3,
        behavioral_depth=1,
        parent_depth=0,
        rrf_offset=1,
        internal_fit_fraction=0.5,
        max_internal_ranker_sessions=None,
        max_sessions_per_split=None,
        max_nonrelevant_per_query=3,
    )
    assert report.best_deterministic_validation_baseline in {"category_only", "efficient_rrf"}
    assert report.candidate_size["validation"]["mean"] > 0
    assert "learned_task_specific" in report.metrics["test"]
    json_path, md_path = write_report(report, tmp_path / "out")
    payload = json.loads(json_path.read_text())
    assert "validation_retain_learned_ranker" in payload
    assert "Learned Reranking on Efficient Retrieval" in md_path.read_text()
