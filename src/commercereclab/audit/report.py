"""Rendering helpers for the CommerceRecLab Retailrocket v0.0 audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from commercereclab.audit.dataset import RetailrocketAudit


def _timestamp(value: int | None) -> str:
    if value is None:
        return "n/a"
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def render_markdown(audit: RetailrocketAudit) -> str:
    """Render a compact human-readable audit report."""

    file_rows = "\n".join(
        f"| `{file.name}` | {file.row_count:,} | `{', '.join(file.columns)}` |"
        for file in audit.files
    )
    event_rows = "\n".join(
        f"| `{event}` | {count:,} | {count / audit.events.row_count:.4%} |"
        for event, count in audit.events.event_counts.items()
    )
    e = audit.events
    p = audit.properties
    c = audit.category_tree
    v = audit.coverage

    return f"""# CommerceRecLab v0.0 — Retailrocket Dataset + Observation Audit

Source directory: `{audit.source_dir}`

## Observation contract

Each event is treated as a logged visitor-item action. The dataset is **not** assumed to contain a complete recommendation-impression log. An absent visitor-item pair is therefore not interpreted as a negative label.

## Files

| File | rows | columns |
|---|---:|---|
{file_rows}

## Event log

- Rows: **{e.row_count:,}**
- Unique visitors: **{e.unique_visitors:,}**
- Unique event items: **{e.unique_items:,}**
- Timestamp range: **{_timestamp(e.timestamp_min_ms)} → {_timestamp(e.timestamp_max_ms)}**
- Exact duplicate rows beyond first occurrence: **{e.exact_duplicate_rows:,}**
- Unexpected event-type rows: **{e.unexpected_event_count:,}**

| event | rows | fraction |
|---|---:|---:|
{event_rows}

### Transaction-field consistency

- Transaction events: **{e.transaction_rows:,}**
- Transaction events with a transaction ID: **{e.transaction_rows_with_id:,}**
- Non-transaction events carrying a transaction ID: **{e.nontransaction_rows_with_transaction_id:,}**

## Time-varying item properties

- Property rows across both parts: **{p.row_count:,}**
- Unique property-bearing items: **{p.unique_items:,}**
- Unique property identifiers: **{p.unique_properties:,}**
- Property timestamp range: **{_timestamp(p.timestamp_min_ms)} → {_timestamp(p.timestamp_max_ms)}**
- Items observed at more than one property timestamp: **{p.items_with_multiple_property_timestamps:,}**
- `categoryid` rows / unique items: **{p.category_property_rows:,} / {p.category_property_unique_items:,}**
- `available` rows / unique items: **{p.available_property_rows:,} / {p.available_property_unique_items:,}**

The presence of repeated item-property snapshots means downstream features must use point-in-time joins rather than future/latest metadata.

## Event-item metadata coverage

- Any property history: **{v.event_items_with_any_property:,} / {e.unique_items:,} ({v.event_item_property_coverage:.4%})**
- `categoryid`: **{v.event_items_with_category_property:,} / {e.unique_items:,} ({v.event_item_category_coverage:.4%})**
- `available`: **{v.event_items_with_available_property:,} / {e.unique_items:,} ({v.event_item_available_coverage:.4%})**

## Category tree integrity

- Rows: **{c.row_count:,}**
- Unique categories: **{c.unique_categories:,}**
- Root categories: **{c.root_categories:,}**
- Referenced parents missing from the table: **{c.missing_parent_references:,}**
- Self-parent rows: **{c.self_parent_rows:,}**
- Cycle detected: **{c.has_cycle}**

## Scientific interpretation

This release supports empirical work on observed behavioral sequences, temporal recommendation, conversion-funnel outcomes, item-state features, candidate generation, and catalog coverage. It does **not** by itself identify which products were recommended but ignored, so non-events must not be called observed dislikes or passes.

## Audit notes

""" + "\n".join(f"- {note}" for note in audit.notes) + "\n"


def write_reports(audit: RetailrocketAudit, output_dir: str | Path) -> tuple[Path, Path]:
    """Write machine-readable JSON and human-readable Markdown reports."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "audit.json"
    markdown_path = output / "audit.md"
    json_path.write_text(json.dumps(audit.to_dict(), indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")
    return json_path, markdown_path
