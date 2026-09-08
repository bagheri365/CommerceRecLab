import json
from pathlib import Path

import pandas as pd
import pytest

from commercereclab.evaluation.hybrid import _rrf_fuse, evaluate_hybrid, write_report


def _write_dataset(root: Path) -> Path:
    data = root / "retailrocket"
    data.mkdir()
    events = pd.DataFrame(
        [
            # training
            [1000, 1, "view", 10, None],
            [1100, 1, "view", 11, None],
            [1200, 1, "addtocart", 11, None],
            [1300, 2, "view", 10, None],
            [1400, 2, "view", 12, None],
            [1500, 2, "transaction", 12, 1],
            # validation session
            [2100, 3, "view", 10, None],
            [2200, 3, "view", 11, None],
            [2300, 3, "addtocart", 11, None],
            # test session
            [3100, 1, "view", 10, None],
            [3200, 1, "transaction", 11, 2],
        ],
        columns=["timestamp", "visitorid", "event", "itemid", "transactionid"],
    )
    events.to_csv(data / "events.csv", index=False)
    props = pd.DataFrame(
        [
            [900, 10, "categoryid", "A"],
            [900, 11, "categoryid", "A"],
            [900, 12, "categoryid", "B"],
        ],
        columns=["timestamp", "itemid", "property", "value"],
    )
    props.iloc[:2].to_csv(data / "item_properties_part1.csv", index=False)
    props.iloc[2:].to_csv(data / "item_properties_part2.csv", index=False)
    pd.DataFrame({"categoryid": [1], "parentid": [None]}).to_csv(data / "category_tree.csv", index=False)
    return data


def _write_manifest(root: Path) -> Path:
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "train_end_ms": 1500,
                "validation_end_ms": 2500,
                "session_gap_minutes": 30,
                "session_boundary_policy": "split_boundary_breaks_session",
                "prediction_horizon": "remainder_of_current_split_bounded_session",
            }
        )
    )
    return path


def test_rrf_rewards_agreement_and_deduplicates():
    ranking = _rrf_fuse(
        [[1, 2, 3], [2, 4, 1]],
        fallback=[5, 6],
        k=4,
        source_depth=4,
        offset=0,
    )
    assert ranking[0] == 2
    assert len(ranking) == len(set(ranking)) == 4


def test_rrf_validates_parameters():
    with pytest.raises(ValueError):
        _rrf_fuse([[1]], fallback=[], k=2, source_depth=1, offset=60)
    with pytest.raises(ValueError):
        _rrf_fuse([[1]], fallback=[], k=1, source_depth=1, offset=-1)


def test_hybrid_report_has_ablation_and_validation_decision(tmp_path: Path):
    dataset = _write_dataset(tmp_path)
    manifest = _write_manifest(tmp_path)
    report = evaluate_hybrid(
        dataset,
        manifest,
        k=2,
        source_depth=3,
        rrf_offset=1,
        max_sessions_per_split=None,
    )
    assert "hybrid_full" in report.variants
    assert "hybrid_minus_category" in report.variants
    assert set(report.validation_ablation_delta_ndcg) == {
        "category",
        "covisitation",
        "recency",
        "visitor_history",
    }
    assert isinstance(report.validation_retain_full_hybrid, bool)
    assert report.metrics["validation"]["hybrid_full"]["view"]["queries"] == 1


def test_hybrid_report_writes_json_and_markdown(tmp_path: Path):
    dataset = _write_dataset(tmp_path)
    manifest = _write_manifest(tmp_path)
    report = evaluate_hybrid(dataset, manifest, k=2, source_depth=3, max_sessions_per_split=None)
    json_path, md_path = write_report(report, tmp_path / "out")
    payload = json.loads(json_path.read_text())
    assert "validation_retain_full_hybrid" in payload
    text = md_path.read_text()
    assert "Validation leave-one-component-out ablation" in text
    assert "Retain full hybrid" in text
