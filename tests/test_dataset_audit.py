from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from commercereclab.audit.dataset import audit_retailrocket_dir
from commercereclab.audit.report import render_markdown, write_reports


def write_fixture(root: Path) -> Path:
    data = root / "retailrocket"
    data.mkdir()

    pd.DataFrame(
        {
            "timestamp": [1000, 2000, 3000, 4000, 4000],
            "visitorid": [1, 1, 2, 2, 2],
            "event": ["view", "addtocart", "view", "transaction", "transaction"],
            "itemid": [10, 10, 20, 20, 20],
            "transactionid": [None, None, None, 99, 99],
        }
    ).to_csv(data / "events.csv", index=False)

    pd.DataFrame(
        {
            "timestamp": [500, 1500, 500, 500],
            "itemid": [10, 10, 20, 30],
            "property": ["categoryid", "available", "categoryid", "available"],
            "value": ["1", "1", "2", "1"],
        }
    ).to_csv(data / "item_properties_part1.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": [2500, 3500, 3500],
            "itemid": [10, 20, 20],
            "property": ["available", "available", "colorhash"],
            "value": ["0", "1", "abc"],
        }
    ).to_csv(data / "item_properties_part2.csv", index=False)
    pd.DataFrame(
        {
            "categoryid": [1, 2, 3],
            "parentid": [None, 1, 1],
        }
    ).to_csv(data / "category_tree.csv", index=False)
    return data


def test_retailrocket_audit_counts_events_properties_and_coverage(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)
    audit = audit_retailrocket_dir(source, property_chunksize=2)

    assert audit.events.row_count == 5
    assert audit.events.unique_visitors == 2
    assert audit.events.unique_items == 2
    assert audit.events.event_counts == {"addtocart": 1, "transaction": 2, "view": 2}
    assert audit.events.exact_duplicate_rows == 1
    assert audit.events.transaction_rows == 2
    assert audit.events.transaction_rows_with_id == 2
    assert audit.events.nontransaction_rows_with_transaction_id == 0

    assert audit.properties.row_count == 7
    assert audit.properties.unique_items == 3
    assert audit.properties.unique_properties == 3
    assert audit.properties.items_with_multiple_property_timestamps == 2
    assert audit.properties.category_property_unique_items == 2
    assert audit.properties.available_property_unique_items == 3

    assert audit.coverage.event_item_property_coverage == pytest.approx(1.0)
    assert audit.coverage.event_item_category_coverage == pytest.approx(1.0)
    assert audit.coverage.event_item_available_coverage == pytest.approx(1.0)


def test_category_tree_integrity_is_reported(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)
    audit = audit_retailrocket_dir(source)

    assert audit.category_tree.unique_categories == 3
    assert audit.category_tree.root_categories == 1
    assert audit.category_tree.missing_parent_references == 0
    assert audit.category_tree.self_parent_rows == 0
    assert audit.category_tree.has_cycle is False


def test_missing_file_fails_clearly(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)
    (source / "category_tree.csv").unlink()

    with pytest.raises(FileNotFoundError, match="category_tree.csv"):
        audit_retailrocket_dir(source)


def test_bad_schema_fails_clearly(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)
    pd.DataFrame({"timestamp": [1], "visitorid": [1]}).to_csv(source / "events.csv", index=False)

    with pytest.raises(ValueError, match="events.csv: missing required columns"):
        audit_retailrocket_dir(source)


def test_reports_are_written_as_json_and_markdown(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)
    audit = audit_retailrocket_dir(source)

    output = tmp_path / "out"
    json_path, markdown_path = write_reports(audit, output)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["events"]["unique_visitors"] == 2
    assert payload["properties"]["row_count"] == 7
    assert "point-in-time" in markdown.lower()
    assert "not interpreted as a negative" in markdown.lower()
    assert render_markdown(audit) == markdown


def test_cli_writes_reports(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)
    output = tmp_path / "cli-output"
    env = os.environ.copy()
    repo_src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(repo_src)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "commercereclab.audit",
            str(source),
            "--property-chunksize",
            "2",
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert (output / "audit.json").is_file()
    assert (output / "audit.md").is_file()
    assert "Wrote" in result.stdout
