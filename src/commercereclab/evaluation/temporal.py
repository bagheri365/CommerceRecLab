from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplitManifest:
    source: str
    deduplicate_exact_rows: bool
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    train_end_ms: int
    validation_end_ms: int
    timestamp_min_ms: int
    timestamp_max_ms: int
    split_row_counts: dict[str, int]
    split_event_counts: dict[str, dict[str, int]]
    session_gap_minutes: int
    session_boundary_policy: str
    prediction_horizon: str
    candidate_time_rule: str
    session_count: int
    singleton_session_count: int
    split_boundary_forced_sessions: int


def _validate_fractions(train_fraction: float, validation_fraction: float) -> float:
    test_fraction = 1.0 - train_fraction - validation_fraction
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be between 0 and 1")
    if test_fraction <= 0:
        raise ValueError("train + validation fractions must leave a positive test fraction")
    return test_fraction


def temporal_cutoffs(
    timestamps: pd.Series,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[int, int]:
    _validate_fractions(train_fraction, validation_fraction)
    values = pd.to_numeric(timestamps, errors="raise").to_numpy(dtype=np.int64)
    if len(values) == 0:
        raise ValueError("cannot derive temporal cutoffs from an empty event log")
    ordered = np.sort(values, kind="stable")
    train_idx = min(len(ordered) - 1, int(np.ceil(len(ordered) * train_fraction)) - 1)
    val_idx = min(
        len(ordered) - 1,
        int(np.ceil(len(ordered) * (train_fraction + validation_fraction))) - 1,
    )
    return int(ordered[train_idx]), int(ordered[val_idx])


def assign_temporal_split(
    timestamps: pd.Series,
    *,
    train_end_ms: int,
    validation_end_ms: int,
) -> pd.Series:
    ts = pd.to_numeric(timestamps, errors="raise")
    labels = np.where(
        ts <= train_end_ms,
        "train",
        np.where(ts <= validation_end_ms, "validation", "test"),
    )
    return pd.Series(labels, index=timestamps.index, dtype="string")


def sessionize_events(
    events: pd.DataFrame,
    *,
    gap_minutes: int = 30,
    split_col: str | None = None,
) -> pd.DataFrame:
    if gap_minutes <= 0:
        raise ValueError("gap_minutes must be positive")
    required = {"visitorid", "timestamp"}
    if split_col is not None:
        required.add(split_col)
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"missing sessionization columns: {sorted(missing)}")

    out = events.copy()
    out["_row_order"] = np.arange(len(out), dtype=np.int64)
    out = out.sort_values(["visitorid", "timestamp", "_row_order"], kind="stable")
    gap_ms = gap_minutes * 60 * 1000
    grouped = out.groupby("visitorid", sort=False)
    previous = grouped["timestamp"].shift()
    gap_break = (out["timestamp"] - previous) > gap_ms
    if split_col is None:
        split_break = pd.Series(False, index=out.index)
    else:
        previous_split = grouped[split_col].shift()
        split_break = previous_split.notna() & (out[split_col] != previous_split)
    split_forced_break = split_break & ~gap_break
    new_session = previous.isna() | gap_break | split_break
    out["split_boundary_break"] = split_break.astype(bool)
    out["split_boundary_forced_break"] = split_forced_break.astype(bool)
    local_id = new_session.groupby(out["visitorid"], sort=False).cumsum().astype("int64") - 1
    out["session_index"] = local_id
    out["session_id"] = out["visitorid"].astype(str) + ":" + local_id.astype(str)
    return out.drop(columns="_row_order")


def point_in_time_join(
    events: pd.DataFrame,
    properties: pd.DataFrame,
    *,
    by: str = "itemid",
    event_time: str = "timestamp",
    property_time: str = "timestamp",
) -> pd.DataFrame:
    """Backward as-of join: never attach property state from after an event."""
    left = events.copy()
    right = properties.copy()
    right_time = property_time
    if property_time == event_time:
        right_time = f"{property_time}_property"
        right = right.rename(columns={property_time: right_time})
    left = left.sort_values([event_time, by], kind="stable")
    right = right.sort_values([right_time, by], kind="stable")
    joined = pd.merge_asof(
        left,
        right,
        left_on=event_time,
        right_on=right_time,
        by=by,
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_property"),
    )
    return joined


def time_valid_catalog_items(
    item_first_seen_ms: pd.Series | dict[int, int],
    *,
    prediction_time_ms: int,
) -> np.ndarray:
    """Return only items with observed catalog evidence at or before prediction time."""
    first_seen = pd.Series(item_first_seen_ms, dtype="int64")
    eligible = first_seen[first_seen <= int(prediction_time_ms)].index.to_numpy(dtype=np.int64)
    return np.unique(eligible)


def sampled_candidate_set(
    catalog_items: np.ndarray | list[int],
    positives: np.ndarray | list[int],
    *,
    n_negatives: int,
    seed: int,
) -> np.ndarray:
    """Deterministic sampled candidate set; positives are always retained."""
    catalog = np.unique(np.asarray(catalog_items, dtype=np.int64))
    positive = np.unique(np.asarray(positives, dtype=np.int64))
    negative_pool = np.setdiff1d(catalog, positive, assume_unique=True)
    n = min(max(n_negatives, 0), len(negative_pool))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(negative_pool, size=n, replace=False) if n else np.array([], dtype=np.int64)
    return np.unique(np.concatenate([positive, sampled]))


def build_temporal_manifest(
    events_path: str | Path,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    session_gap_minutes: int = 30,
    deduplicate_exact_rows: bool = True,
) -> TemporalSplitManifest:
    events_path = Path(events_path)
    usecols = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
    events = pd.read_csv(events_path, usecols=usecols)
    if deduplicate_exact_rows:
        events = events.drop_duplicates(ignore_index=True)

    test_fraction = _validate_fractions(train_fraction, validation_fraction)
    train_end_ms, validation_end_ms = temporal_cutoffs(
        events["timestamp"],
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    events["split"] = assign_temporal_split(
        events["timestamp"],
        train_end_ms=train_end_ms,
        validation_end_ms=validation_end_ms,
    )

    split_row_counts = {
        split: int((events["split"] == split).sum())
        for split in ("train", "validation", "test")
    }
    split_event_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        counts = events.loc[events["split"] == split, "event"].value_counts().sort_index()
        split_event_counts[split] = {str(k): int(v) for k, v in counts.items()}

    sessions = sessionize_events(
        events[["visitorid", "timestamp", "split"]],
        gap_minutes=session_gap_minutes,
        split_col="split",
    )
    session_sizes = sessions.groupby("session_id", sort=False).size()
    forced_sessions = int(sessions["split_boundary_forced_break"].sum())

    return TemporalSplitManifest(
        source=str(events_path),
        deduplicate_exact_rows=deduplicate_exact_rows,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        train_end_ms=train_end_ms,
        validation_end_ms=validation_end_ms,
        timestamp_min_ms=int(events["timestamp"].min()),
        timestamp_max_ms=int(events["timestamp"].max()),
        split_row_counts=split_row_counts,
        split_event_counts=split_event_counts,
        session_gap_minutes=session_gap_minutes,
        session_boundary_policy="split_boundary_breaks_session",
        prediction_horizon="remainder_of_current_split_bounded_session",
        candidate_time_rule="candidate_item_must_have_observed_catalog_evidence_at_or_before_prediction_time",
        session_count=int(len(session_sizes)),
        singleton_session_count=int((session_sizes == 1).sum()),
        split_boundary_forced_sessions=forced_sessions,
    )


def _iso(ms: int) -> str:
    return pd.to_datetime(ms, unit="ms", utc=True).isoformat()


def write_manifest(manifest: TemporalSplitManifest, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "temporal_split_manifest.json"
    md_path = output_dir / "temporal_split_manifest.md"
    json_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")

    lines = [
        "# CommerceRecLab v0.2 — Temporal Evaluation Protocol",
        "",
        f"Source: `{manifest.source}`",
        "",
        "## Frozen split contract",
        "",
        f"- Train fraction: **{manifest.train_fraction:.2%}**",
        f"- Validation fraction: **{manifest.validation_fraction:.2%}**",
        f"- Test fraction: **{manifest.test_fraction:.2%}**",
        f"- Train end: **{_iso(manifest.train_end_ms)}**",
        f"- Validation end: **{_iso(manifest.validation_end_ms)}**",
        f"- Dataset end: **{_iso(manifest.timestamp_max_ms)}**",
        "",
        "| split | rows | views | addtocart | transactions |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        c = manifest.split_event_counts[split]
        lines.append(
            f"| {split} | {manifest.split_row_counts[split]:,} | "
            f"{c.get('view', 0):,} | {c.get('addtocart', 0):,} | {c.get('transaction', 0):,} |"
        )
    lines += [
        "",
        "## Session contract",
        "",
        f"A new session starts after more than **{manifest.session_gap_minutes} minutes** of visitor inactivity.",
        "A train/validation/test boundary also forces a new session, even when the inactivity gap is shorter.",
        f"Observed split-bounded sessions: **{manifest.session_count:,}**",
        f"Singleton sessions: **{manifest.singleton_session_count:,}**",
        f"Sessions forced by split boundaries: **{manifest.split_boundary_forced_sessions:,}**",
        "",
        "## Prediction horizon",
        "",
        "The primary ranking target is an event occurring strictly later in the **remainder of the current split-bounded session**.",
        "View, add-to-cart, transaction, and next-item targets remain separate tasks.",
        "",
        "## Evaluation semantics",
        "",
        "- Prediction features for an example at time `t` may use only information with effective timestamp `<= t`.",
        "- Item properties must use a backward point-in-time join; future/latest catalog state is forbidden.",
        "- View, cart, transaction, and next-item/session tasks are reported separately.",
        "- An unobserved visitor-item pair is not an observed negative or ignored impression.",
        "- Full/large-catalog and sampled-negative evaluation must be labeled separately.",
        "- Sampled candidate sets must document the sampling distribution, size, and random seed and must retain all positives.",
        "- A candidate item may enter an evaluation catalog only if some catalog evidence for that item was observed at or before prediction time.",
        "- Availability, when enforced, must come from the latest backward point-in-time `available` state; missing availability is not silently treated as unavailable.",
        "- Transaction IDs are outcome identifiers and cannot be used as pre-transaction features.",
        "",
        "## Relevance contract",
        "",
        "Primary tasks use separate binary relevance labels for strictly later `view`, `addtocart`, and `transaction` events within the remainder of the current split-bounded session.",
        "Any weighted multi-stage gain is a controlled project assumption and must be reported separately from those task-specific results.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the CommerceRecLab v0.2 temporal evaluation manifest.")
    parser.add_argument("events_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v0_2_temporal_protocol"))
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    args = parser.parse_args(argv)

    manifest = build_temporal_manifest(
        args.events_csv,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        session_gap_minutes=args.session_gap_minutes,
    )
    json_path, md_path = write_manifest(manifest, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
