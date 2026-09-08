import json
from pathlib import Path

import pandas as pd

from commercereclab.evaluation.retrieval import (
    _candidate_set,
    _parent_category_items,
    evaluate_retrieval,
    write_report,
)


def _write_dataset(root: Path) -> Path:
    data = root / "retailrocket"
    data.mkdir()
    rows = [
        [1000, 1, "view", 10, None], [1050, 1, "view", 11, None],
        [1100, 2, "view", 10, None], [1150, 2, "view", 12, None],
        [1200, 3, "view", 20, None], [1250, 3, "view", 21, None],
        [5000, 4, "view", 10, None], [5050, 4, "view", 12, None], [5060, 4, "addtocart", 12, None], [5070, 4, "transaction", 12, 1],
        [7000, 5, "view", 10, None], [7050, 5, "view", 11, None], [7060, 5, "addtocart", 11, None], [7070, 5, "transaction", 11, 2],
    ]
    pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid", "transactionid"]).to_csv(data / "events.csv", index=False)
    props = pd.DataFrame(
        [[900, 10, "categoryid", "1"], [900, 11, "categoryid", "1"], [900, 12, "categoryid", "2"], [900, 20, "categoryid", "2"], [900, 21, "categoryid", "2"]],
        columns=["timestamp", "itemid", "property", "value"],
    )
    props.iloc[:3].to_csv(data / "item_properties_part1.csv", index=False)
    props.iloc[3:].to_csv(data / "item_properties_part2.csv", index=False)
    pd.DataFrame({"categoryid": [1, 2, 9], "parentid": [9, 9, None]}).to_csv(data / "category_tree.csv", index=False)
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


def test_candidate_set_deduplicates_sources():
    assert _candidate_set([[1, 2], [2, 3]]) == {1, 2, 3}


def test_parent_category_items_interleave_siblings():
    values = _parent_category_items(
        "1",
        parent_map={"1": "9", "2": "9"},
        children_map={"9": ["1", "2"]},
        category_rankings={"1": [10, 11], "2": [20, 21]},
        depth=4,
    )
    assert values == [10, 20, 11, 21]


def test_retrieval_report_and_write(tmp_path: Path):
    dataset = _write_dataset(tmp_path)
    manifest = _write_manifest(tmp_path)
    report = evaluate_retrieval(
        dataset,
        manifest,
        k=1,
        base_depth=1,
        expanded_depth=3,
        max_sessions_per_split=None,
    )
    assert set(report.candidate_recall["validation"]) == {
        "base_union", "deep_category", "deep_behavioral", "parent_category", "expanded_union"
    }
    assert report.candidate_size["validation"]["expanded_union"]["mean"] >= report.candidate_size["validation"]["base_union"]["mean"]
    json_path, md_path = write_report(report, tmp_path / "out")
    assert "validation_retain_expanded_union" in json.loads(json_path.read_text())
    assert "Candidate Generation / Retrieval" in md_path.read_text()
