import json
from pathlib import Path

import pandas as pd

from commercereclab.evaluation.efficiency import (
    RetrievalBudget,
    _budget_grid,
    _pareto_frontier,
    evaluate_efficiency,
    write_report,
)


def _write_dataset(root: Path) -> Path:
    data = root / "retailrocket"
    data.mkdir()
    rows = [
        [1000, 1, "view", 10, None], [1050, 1, "view", 11, None],
        [1100, 2, "view", 10, None], [1150, 2, "view", 12, None],
        [1200, 3, "view", 20, None], [1250, 3, "view", 21, None],
        [5000, 4, "view", 10, None], [5050, 4, "view", 12, None],
        [5060, 4, "addtocart", 12, None], [5070, 4, "transaction", 12, 1],
        [7000, 5, "view", 10, None], [7050, 5, "view", 11, None],
        [7060, 5, "addtocart", 11, None], [7070, 5, "transaction", 11, 2],
    ]
    pd.DataFrame(
        rows,
        columns=["timestamp", "visitorid", "event", "itemid", "transactionid"],
    ).to_csv(data / "events.csv", index=False)
    props = pd.DataFrame(
        [
            [900, 10, "categoryid", "1"], [900, 11, "categoryid", "1"],
            [900, 12, "categoryid", "2"], [900, 20, "categoryid", "2"],
            [900, 21, "categoryid", "2"],
        ],
        columns=["timestamp", "itemid", "property", "value"],
    )
    props.iloc[:3].to_csv(data / "item_properties_part1.csv", index=False)
    props.iloc[3:].to_csv(data / "item_properties_part2.csv", index=False)
    pd.DataFrame({"categoryid": [1, 2, 9], "parentid": [9, 9, None]}).to_csv(
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


def test_budget_grid_is_deterministic():
    grid = _budget_grid((1, 2), (1,), (0, 2))
    assert RetrievalBudget(1, 1, 0) in grid
    assert RetrievalBudget(2, 1, 2) in grid
    assert len(grid) == 4


def test_pareto_frontier_removes_dominated_configuration():
    recall = {"cheap": 0.5, "dominated": 0.4, "high": 0.7}
    size = {"cheap": 100.0, "dominated": 120.0, "high": 200.0}
    assert _pareto_frontier(recall, size) == ["cheap", "high"]


def test_efficiency_report_selects_valid_configuration(tmp_path: Path):
    dataset = _write_dataset(tmp_path)
    manifest = _write_manifest(tmp_path)
    report = evaluate_efficiency(
        dataset,
        manifest,
        category_depths=(1, 2),
        behavioral_depths=(1, 2),
        parent_depths=(0, 2),
        retention_threshold=0.5,
        max_sessions_per_split=None,
    )
    assert report.expanded_reference == "c2_b2_p2"
    assert report.selected_efficient_configuration in report.budgets
    assert report.validation_pareto_frontier
    json_path, md_path = write_report(report, tmp_path / "out")
    payload = json.loads(json_path.read_text())
    assert "selected_efficient_configuration" in payload
    assert "Retrieval Efficiency / Pareto Frontier" in md_path.read_text()
