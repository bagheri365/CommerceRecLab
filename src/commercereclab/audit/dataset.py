"""Retailrocket dataset and observation audit utilities for CommerceRecLab v0.0.

The audit is deliberately descriptive. Logged events are treated as observed
visitor-item actions, not as a complete impression log. Missing visitor-item
pairs are never interpreted as negative feedback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

EXPECTED_FILES = (
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
    "category_tree.csv",
)
EXPECTED_EVENT_TYPES = ("view", "addtocart", "transaction")


@dataclass(frozen=True)
class FileAudit:
    name: str
    row_count: int
    columns: tuple[str, ...]
    null_counts: dict[str, int]


@dataclass(frozen=True)
class EventAudit:
    row_count: int
    unique_visitors: int
    unique_items: int
    timestamp_min_ms: int | None
    timestamp_max_ms: int | None
    event_counts: dict[str, int]
    unexpected_event_count: int
    exact_duplicate_rows: int
    transaction_rows: int
    transaction_rows_with_id: int
    nontransaction_rows_with_transaction_id: int


@dataclass(frozen=True)
class PropertyAudit:
    row_count: int
    unique_items: int
    unique_properties: int
    timestamp_min_ms: int | None
    timestamp_max_ms: int | None
    items_with_multiple_property_timestamps: int
    category_property_rows: int
    category_property_unique_items: int
    available_property_rows: int
    available_property_unique_items: int


@dataclass(frozen=True)
class CategoryTreeAudit:
    row_count: int
    unique_categories: int
    root_categories: int
    missing_parent_references: int
    self_parent_rows: int
    has_cycle: bool


@dataclass(frozen=True)
class CoverageAudit:
    event_items_with_any_property: int
    event_item_property_coverage: float
    event_items_with_category_property: int
    event_item_category_coverage: float
    event_items_with_available_property: int
    event_item_available_coverage: float


@dataclass(frozen=True)
class RetailrocketAudit:
    source_dir: str
    files: tuple[FileAudit, ...]
    events: EventAudit
    properties: PropertyAudit
    category_tree: CategoryTreeAudit
    coverage: CoverageAudit
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _int_or_none(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], filename: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{filename}: missing required columns: {', '.join(missing)}")


def _read_events(path: Path) -> tuple[pd.DataFrame, FileAudit, EventAudit, set[int]]:
    frame = pd.read_csv(path)
    required = ("timestamp", "visitorid", "event", "itemid", "transactionid")
    _require_columns(frame, required, path.name)

    event_counts = Counter(frame["event"].dropna().astype(str))
    unexpected = sum(
        count for event, count in event_counts.items() if event not in EXPECTED_EVENT_TYPES
    )
    transaction_mask = frame["event"].eq("transaction")
    nontransaction_mask = frame["event"].notna() & ~transaction_mask

    file_audit = FileAudit(
        name=path.name,
        row_count=int(len(frame)),
        columns=tuple(frame.columns.astype(str)),
        null_counts={column: int(frame[column].isna().sum()) for column in frame.columns},
    )
    event_audit = EventAudit(
        row_count=int(len(frame)),
        unique_visitors=int(frame["visitorid"].nunique(dropna=True)),
        unique_items=int(frame["itemid"].nunique(dropna=True)),
        timestamp_min_ms=_int_or_none(frame["timestamp"].min()),
        timestamp_max_ms=_int_or_none(frame["timestamp"].max()),
        event_counts={str(key): int(value) for key, value in sorted(event_counts.items())},
        unexpected_event_count=int(unexpected),
        exact_duplicate_rows=int(frame.duplicated().sum()),
        transaction_rows=int(transaction_mask.sum()),
        transaction_rows_with_id=int((transaction_mask & frame["transactionid"].notna()).sum()),
        nontransaction_rows_with_transaction_id=int(
            (nontransaction_mask & frame["transactionid"].notna()).sum()
        ),
    )
    event_items = set(frame["itemid"].dropna().astype("int64").unique().tolist())
    return frame, file_audit, event_audit, event_items


def _audit_properties(
    paths: tuple[Path, Path], *, chunksize: int
) -> tuple[tuple[FileAudit, FileAudit], PropertyAudit, set[int], set[int], set[int]]:
    total_rows = 0
    timestamp_min: int | None = None
    timestamp_max: int | None = None
    property_names: set[str] = set()
    property_items: set[int] = set()
    category_items: set[int] = set()
    available_items: set[int] = set()
    category_rows = 0
    available_rows = 0
    global_item_min_ts = pd.Series(dtype="int64")
    global_item_max_ts = pd.Series(dtype="int64")
    file_audits: list[FileAudit] = []

    for path in paths:
        file_rows = 0
        null_counts = Counter({"timestamp": 0, "itemid": 0, "property": 0, "value": 0})
        columns: tuple[str, ...] | None = None

        for chunk in pd.read_csv(path, chunksize=chunksize):
            required = ("timestamp", "itemid", "property", "value")
            _require_columns(chunk, required, path.name)
            if columns is None:
                columns = tuple(chunk.columns.astype(str))
            file_rows += len(chunk)
            total_rows += len(chunk)
            for column in chunk.columns:
                null_counts[column] += int(chunk[column].isna().sum())

            ts_min = _int_or_none(chunk["timestamp"].min())
            ts_max = _int_or_none(chunk["timestamp"].max())
            if ts_min is not None:
                timestamp_min = ts_min if timestamp_min is None else min(timestamp_min, ts_min)
            if ts_max is not None:
                timestamp_max = ts_max if timestamp_max is None else max(timestamp_max, ts_max)

            property_names.update(chunk["property"].dropna().astype(str).unique().tolist())
            valid = chunk[["itemid", "timestamp"]].dropna()
            if not valid.empty:
                grouped = valid.groupby("itemid", sort=False)["timestamp"].agg(["min", "max"])
                grouped.index = grouped.index.astype("int64")
                property_items.update(grouped.index.tolist())
                chunk_min = grouped["min"].astype("int64")
                chunk_max = grouped["max"].astype("int64")
                global_item_min_ts = pd.concat([global_item_min_ts, chunk_min], axis=1).min(axis=1).astype("int64")
                global_item_max_ts = pd.concat([global_item_max_ts, chunk_max], axis=1).max(axis=1).astype("int64")

            category_mask = chunk["property"].eq("categoryid")
            available_mask = chunk["property"].eq("available")
            category_rows += int(category_mask.sum())
            available_rows += int(available_mask.sum())
            category_items.update(
                chunk.loc[category_mask, "itemid"].dropna().astype("int64").unique().tolist()
            )
            available_items.update(
                chunk.loc[available_mask, "itemid"].dropna().astype("int64").unique().tolist()
            )

        file_audits.append(
            FileAudit(
                name=path.name,
                row_count=int(file_rows),
                columns=columns or (),
                null_counts={str(key): int(value) for key, value in sorted(null_counts.items())},
            )
        )

    multiple_timestamps = int(
        global_item_min_ts.ne(global_item_max_ts).sum()
    )
    audit = PropertyAudit(
        row_count=int(total_rows),
        unique_items=len(property_items),
        unique_properties=len(property_names),
        timestamp_min_ms=timestamp_min,
        timestamp_max_ms=timestamp_max,
        items_with_multiple_property_timestamps=int(multiple_timestamps),
        category_property_rows=int(category_rows),
        category_property_unique_items=len(category_items),
        available_property_rows=int(available_rows),
        available_property_unique_items=len(available_items),
    )
    return (
        (file_audits[0], file_audits[1]),
        audit,
        property_items,
        category_items,
        available_items,
    )


def _has_category_cycle(parent_by_category: dict[int, int]) -> bool:
    done: set[int] = set()
    for start in parent_by_category:
        if start in done:
            continue
        path: set[int] = set()
        current = start
        while current in parent_by_category:
            if current in path:
                return True
            if current in done:
                break
            path.add(current)
            current = parent_by_category[current]
        done.update(path)
    return False


def _audit_category_tree(path: Path) -> tuple[FileAudit, CategoryTreeAudit]:
    frame = pd.read_csv(path)
    required = ("categoryid", "parentid")
    _require_columns(frame, required, path.name)
    category_ids = set(frame["categoryid"].dropna().astype("int64").tolist())
    parent_ids = set(frame["parentid"].dropna().astype("int64").tolist())
    missing_parents = parent_ids - category_ids
    self_parent_mask = frame["categoryid"].eq(frame["parentid"]) & frame["parentid"].notna()
    parent_by_category = {
        int(category): int(parent)
        for category, parent in frame[["categoryid", "parentid"]].dropna().itertuples(index=False)
    }

    file_audit = FileAudit(
        name=path.name,
        row_count=int(len(frame)),
        columns=tuple(frame.columns.astype(str)),
        null_counts={column: int(frame[column].isna().sum()) for column in frame.columns},
    )
    audit = CategoryTreeAudit(
        row_count=int(len(frame)),
        unique_categories=len(category_ids),
        root_categories=int(frame["parentid"].isna().sum()),
        missing_parent_references=len(missing_parents),
        self_parent_rows=int(self_parent_mask.sum()),
        has_cycle=_has_category_cycle(parent_by_category),
    )
    return file_audit, audit


def audit_retailrocket_dir(
    source_dir: str | Path, *, property_chunksize: int = 500_000
) -> RetailrocketAudit:
    """Audit the canonical four-file Retailrocket release.

    Item-property tables are streamed in chunks so the full ~20M-row metadata
    history can be audited on a laptop without loading it all into memory.
    """

    source = Path(source_dir)
    missing = [name for name in EXPECTED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Retailrocket files: {', '.join(missing)}")
    if property_chunksize <= 0:
        raise ValueError("property_chunksize must be positive")

    _, events_file, events, event_items = _read_events(source / "events.csv")
    property_files, properties, property_items, category_items, available_items = _audit_properties(
        (source / "item_properties_part1.csv", source / "item_properties_part2.csv"),
        chunksize=property_chunksize,
    )
    category_file, category_tree = _audit_category_tree(source / "category_tree.csv")

    coverage = CoverageAudit(
        event_items_with_any_property=len(event_items & property_items),
        event_item_property_coverage=_safe_fraction(len(event_items & property_items), len(event_items)),
        event_items_with_category_property=len(event_items & category_items),
        event_item_category_coverage=_safe_fraction(len(event_items & category_items), len(event_items)),
        event_items_with_available_property=len(event_items & available_items),
        event_item_available_coverage=_safe_fraction(len(event_items & available_items), len(event_items)),
    )

    notes = (
        "A logged event is an observed visitor-item action, not proof of a recommendation impression.",
        "An absent visitor-item event is missing/unobserved behavior, not an observed negative preference.",
        "Event timestamps support temporal splits; random interaction splits should not be used for deployment-style claims.",
        "Item properties are time-varying. Downstream joins must use only property state available at or before the event time.",
        "Hashed property/value identifiers should be treated as opaque unless the dataset documentation gives semantic meaning.",
        "Transaction IDs are outcome identifiers, not product/user features and must not leak into pre-transaction ranking features.",
    )

    return RetailrocketAudit(
        source_dir=str(source),
        files=(events_file, *property_files, category_file),
        events=events,
        properties=properties,
        category_tree=category_tree,
        coverage=coverage,
        notes=notes,
    )
