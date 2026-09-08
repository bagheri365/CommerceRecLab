from __future__ import annotations

import argparse
import json
import math
from itertools import chain
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from commercereclab.evaluation.temporal import assign_temporal_split, sessionize_events

EVENT_TYPES = ("view", "addtocart", "transaction")


@dataclass(frozen=True)
class BaselineMetric:
    queries: int
    recall_at_k: float
    ndcg_at_k: float


@dataclass(frozen=True)
class BaselineReport:
    source: str
    manifest: str
    k: int
    prediction_point: str
    prediction_horizon: str
    training_rows: int
    validation_sessions: int
    test_sessions: int
    validation_evaluated_sessions: int
    test_evaluated_sessions: int
    evaluation_sample_seed: int
    visitor_cohort_definition: dict[str, str]
    item_cohort_definition: dict[str, str]
    baselines: list[str]
    metrics: dict[str, dict[str, dict[str, float | int]]]
    visitor_cohorts: dict[str, dict[str, dict[str, dict[str, float | int]]]]
    item_cohorts: dict[str, dict[str, dict[str, dict[str, float | int]]]]
    notes: list[str]


def _read_manifest(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    required = {
        "train_end_ms",
        "validation_end_ms",
        "session_gap_minutes",
        "session_boundary_policy",
        "prediction_horizon",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"manifest missing required fields: {sorted(missing)}")
    if data["session_boundary_policy"] != "split_boundary_breaks_session":
        raise ValueError("v0.3 requires split-bounded sessions from the frozen v0.2 contract")
    if data["prediction_horizon"] != "remainder_of_current_split_bounded_session":
        raise ValueError("v0.3 requires the frozen remainder-of-session prediction horizon")
    return data


def _deduplicated_events(path: str | Path) -> pd.DataFrame:
    events = pd.read_csv(
        path,
        usecols=["timestamp", "visitorid", "event", "itemid", "transactionid"],
    )
    return events.drop_duplicates(ignore_index=True)


def _event_rankings(train: pd.DataFrame, *, half_life_days: float) -> dict[str, list[int]]:
    rankings: dict[str, list[int]] = {}
    for event_type in EVENT_TYPES:
        counts = (
            train.loc[train["event"] == event_type]
            .groupby("itemid", sort=False)
            .size()
            .sort_values(ascending=False, kind="stable")
        )
        rankings[f"most_{event_type}"] = [int(x) for x in counts.index]

    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    max_ts = int(train["timestamp"].max())
    half_life_ms = half_life_days * 24 * 60 * 60 * 1000
    age = (max_ts - train["timestamp"].to_numpy(dtype=np.int64)) / half_life_ms
    weights = np.exp2(-age)
    weighted = pd.DataFrame({"itemid": train["itemid"].to_numpy(), "weight": weights})
    decayed = weighted.groupby("itemid", sort=False)["weight"].sum().sort_values(
        ascending=False, kind="stable"
    )
    rankings["time_decayed_popularity"] = [int(x) for x in decayed.index]
    return rankings


def _top_neighbors(
    source: pd.Series,
    target: pd.Series,
    *,
    symmetric: bool,
    top_n: int,
    source_filter: set[int] | None = None,
) -> dict[int, list[int]]:
    pairs = pd.DataFrame({"source": source.to_numpy(), "target": target.to_numpy()})
    pairs = pairs[pairs["source"] != pairs["target"]]
    if symmetric and len(pairs):
        reversed_pairs = pairs.rename(columns={"source": "target", "target": "source"})[
            ["source", "target"]
        ]
        pairs = pd.concat([pairs, reversed_pairs], ignore_index=True)
    if source_filter is not None:
        pairs = pairs[pairs["source"].isin(source_filter)]
    if pairs.empty:
        return {}
    counts = pairs.groupby(["source", "target"], sort=False).size().rename("count").reset_index()
    counts = counts.sort_values(
        ["source", "count", "target"], ascending=[True, False, True], kind="stable"
    )
    counts = counts.groupby("source", sort=False).head(top_n)
    return {
        int(source_id): [int(x) for x in group["target"].tolist()]
        for source_id, group in counts.groupby("source", sort=False)
    }


def _transition_and_covisitation(
    train: pd.DataFrame,
    *,
    gap_minutes: int,
    top_n: int,
    context_items: set[int] | None = None,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    sessions = sessionize_events(train[["visitorid", "timestamp", "itemid"]], gap_minutes=gap_minutes)
    sessions = sessions.sort_values(["visitorid", "session_index", "timestamp"], kind="stable")
    next_item = sessions.groupby("session_id", sort=False)["itemid"].shift(-1)
    valid = next_item.notna()
    source = sessions.loc[valid, "itemid"].astype("int64")
    target = next_item.loc[valid].astype("int64")
    transitions = _top_neighbors(
        source, target, symmetric=False, top_n=top_n, source_filter=context_items
    )
    covisitation = _top_neighbors(
        source, target, symmetric=True, top_n=top_n, source_filter=context_items
    )
    return transitions, covisitation


def _visitor_history(
    train: pd.DataFrame,
    *,
    top_n: int,
    visitors: set[int] | None = None,
) -> tuple[dict[int, list[int]], dict[int, int]]:
    working = train
    if visitors is not None:
        working = train[train["visitorid"].isin(visitors)]
    history_count = working.groupby("visitorid", sort=False).size().astype("int64").to_dict()
    grouped = (
        working.groupby(["visitorid", "itemid"], sort=False)
        .agg(count=("itemid", "size"), last_ts=("timestamp", "max"))
        .reset_index()
        .sort_values(
            ["visitorid", "count", "last_ts", "itemid"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .groupby("visitorid", sort=False)
        .head(top_n)
    )
    rankings = {
        int(visitor): [int(x) for x in group["itemid"].tolist()]
        for visitor, group in grouped.groupby("visitorid", sort=False)
    }
    return rankings, {int(k): int(v) for k, v in history_count.items()}


def _latest_category_map(
    dataset_dir: str | Path,
    *,
    cutoff_ms: int,
    chunksize: int = 500_000,
) -> dict[int, str]:
    dataset_dir = Path(dataset_dir)
    pieces: list[pd.DataFrame] = []
    for filename in ("item_properties_part1.csv", "item_properties_part2.csv"):
        path = dataset_dir / filename
        for chunk in pd.read_csv(
            path,
            usecols=["timestamp", "itemid", "property", "value"],
            chunksize=chunksize,
        ):
            cat = chunk[(chunk["property"] == "categoryid") & (chunk["timestamp"] <= cutoff_ms)]
            if len(cat):
                pieces.append(cat[["timestamp", "itemid", "value"]])
    if not pieces:
        return {}
    categories = pd.concat(pieces, ignore_index=True)
    categories = categories.sort_values(["itemid", "timestamp"], kind="stable")
    latest = categories.drop_duplicates("itemid", keep="last")
    return {int(row.itemid): str(row.value) for row in latest.itertuples(index=False)}


def _category_popularity(
    train: pd.DataFrame,
    category_map: dict[int, str],
    *,
    top_n: int,
) -> dict[str, list[int]]:
    frame = train[["itemid"]].copy()
    frame["category"] = frame["itemid"].map(category_map)
    frame = frame.dropna(subset=["category"])
    if frame.empty:
        return {}
    counts = frame.groupby(["category", "itemid"], sort=False).size().rename("count").reset_index()
    counts = counts.sort_values(
        ["category", "count", "itemid"], ascending=[True, False, True], kind="stable"
    )
    counts = counts.groupby("category", sort=False).head(top_n)
    return {
        str(category): [int(x) for x in group["itemid"].tolist()]
        for category, group in counts.groupby("category", sort=False)
    }


def _session_queries(
    events: pd.DataFrame,
    *,
    split: str,
    gap_minutes: int,
    max_sessions: int | None,
    seed: int,
) -> tuple[list[dict], int]:
    subset = events.loc[
        events["split"] == split,
        ["visitorid", "timestamp", "event", "itemid", "split"],
    ]
    sessions = sessionize_events(subset, gap_minutes=gap_minutes, split_col="split")
    sessions = sessions.sort_values(
        ["visitorid", "session_index", "timestamp"], kind="stable"
    )
    sizes = sessions.groupby("session_id", sort=False).size()
    eligible_ids = sizes.index[sizes >= 2].to_numpy()
    total_eligible = int(len(eligible_ids))

    if max_sessions is not None and max_sessions > 0 and total_eligible > max_sessions:
        rng = np.random.default_rng(seed)
        chosen_idx = np.sort(rng.choice(total_eligible, size=max_sessions, replace=False))
        chosen_ids = eligible_ids[chosen_idx]
        sessions = sessions[sessions["session_id"].isin(set(chosen_ids))]

    order = sessions.groupby("session_id", sort=False).cumcount()
    first = sessions.loc[order == 0].set_index("session_id")
    remainder = sessions.loc[order > 0]
    positive_maps: dict[str, dict[str, np.ndarray]] = {}
    for event_type in EVENT_TYPES:
        task_rows = remainder.loc[remainder["event"] == event_type, ["session_id", "itemid"]]
        grouped = task_rows.groupby("session_id", sort=False)["itemid"].unique()
        positive_maps[event_type] = {
            str(session_id): np.asarray(items, dtype=np.int64)
            for session_id, items in grouped.items()
        }

    empty = np.array([], dtype=np.int64)
    queries: list[dict] = []
    for session_id, row in first.iterrows():
        sid = str(session_id)
        queries.append(
            {
                "visitorid": int(row["visitorid"]),
                "timestamp": int(row["timestamp"]),
                "last_item": int(row["itemid"]),
                "positives": {
                    event_type: positive_maps[event_type].get(sid, empty)
                    for event_type in EVENT_TYPES
                },
            }
        )
    return queries, total_eligible


def _complete_ranking(primary: Iterable[int], fallback: list[int], k: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for item in chain(primary, fallback):
        item = int(item)
        if item not in seen:
            seen.add(item)
            out.append(item)
        if len(out) >= k:
            break
    return out


def _recall_ndcg(ranking: list[int], positives: np.ndarray, k: int) -> tuple[float, float]:
    if len(positives) == 0:
        return math.nan, math.nan
    positive_set = set(int(x) for x in positives)
    hits = [1 if item in positive_set else 0 for item in ranking[:k]]
    recall = sum(hits) / len(positive_set)
    dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))
    ideal_hits = min(len(positive_set), k)
    idcg = sum(1 / math.log2(idx + 2) for idx in range(ideal_hits))
    return float(recall), float(dcg / idcg if idcg else 0.0)


def _aggregate(values: list[tuple[float, float]]) -> BaselineMetric:
    valid = [(r, n) for r, n in values if not math.isnan(r)]
    if not valid:
        return BaselineMetric(queries=0, recall_at_k=math.nan, ndcg_at_k=math.nan)
    return BaselineMetric(
        queries=len(valid),
        recall_at_k=float(np.mean([x[0] for x in valid])),
        ndcg_at_k=float(np.mean([x[1] for x in valid])),
    )


def _metric_dict(metric: BaselineMetric) -> dict[str, float | int]:
    return asdict(metric)


def _visitor_cohort(history_count: int) -> str:
    if history_count == 0:
        return "new"
    if history_count < 5:
        return "sparse"
    return "repeat"


def _item_cohorts(train: pd.DataFrame) -> dict[int, str]:
    counts = train.groupby("itemid", sort=False).size().sort_values(ascending=False, kind="stable")
    if counts.empty:
        return {}
    rank_pct = np.arange(len(counts), dtype=float) / len(counts)
    labels = np.where(rank_pct < 0.10, "head", np.where(rank_pct < 0.50, "mid_tail", "long_tail"))
    return {int(item): str(label) for item, label in zip(counts.index, labels, strict=True)}


def evaluate_baselines(
    dataset_dir: str | Path,
    manifest_path: str | Path,
    *,
    k: int = 20,
    half_life_days: float = 7.0,
    neighbor_top_n: int = 50,
    max_sessions_per_split: int | None = 50_000,
    seed: int = 365,
) -> BaselineReport:
    if k <= 0:
        raise ValueError("k must be positive")
    dataset_dir = Path(dataset_dir)
    manifest = _read_manifest(manifest_path)
    events = _deduplicated_events(dataset_dir / "events.csv")
    events["split"] = assign_temporal_split(
        events["timestamp"],
        train_end_ms=int(manifest["train_end_ms"]),
        validation_end_ms=int(manifest["validation_end_ms"]),
    )
    train = events.loc[events["split"] == "train"].copy()
    gap = int(manifest["session_gap_minutes"])

    top_n = max(k, neighbor_top_n)

    # Freeze the deterministic evaluation sample first. Lookup tables that only serve query
    # contexts are then restricted to those visitors/items, while all counts still come from
    # the full training period. This keeps v0.3 practical on a laptop without changing labels.
    split_queries: dict[str, list[dict]] = {}
    split_session_counts: dict[str, int] = {}
    evaluated_session_counts: dict[str, int] = {}
    for split_index, split in enumerate(("validation", "test")):
        queries, total_eligible = _session_queries(
            events,
            split=split,
            gap_minutes=gap,
            max_sessions=max_sessions_per_split,
            seed=seed + split_index,
        )
        split_queries[split] = queries
        split_session_counts[split] = total_eligible
        evaluated_session_counts[split] = len(queries)

    query_visitors = {int(q["visitorid"]) for queries in split_queries.values() for q in queries}
    query_context_items = {int(q["last_item"]) for queries in split_queries.values() for q in queries}

    global_rankings = _event_rankings(train, half_life_days=half_life_days)
    fallback = global_rankings["most_view"]
    category_map = _latest_category_map(dataset_dir, cutoff_ms=int(manifest["train_end_ms"]))
    category_rankings = _category_popularity(train, category_map, top_n=top_n)
    transitions, covisitation = _transition_and_covisitation(
        train, gap_minutes=gap, top_n=top_n, context_items=query_context_items
    )
    visitor_rankings, visitor_counts = _visitor_history(
        train, top_n=top_n, visitors=query_visitors
    )
    item_cohort_map = _item_cohorts(train)

    baseline_names = [
        "most_view",
        "most_addtocart",
        "most_transaction",
        "time_decayed_popularity",
        "visitor_history_repeat",
        "last_item_covisitation",
        "last_item_transition",
        "category_conditioned_popularity",
    ]

    overall: dict[str, dict[str, dict[str, float | int]]] = {}
    visitor_cohorts: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    item_cohorts: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for split in ("validation", "test"):
        queries = split_queries[split]

        store: dict[str, dict[str, list[tuple[float, float]]]] = {
            name: {task: [] for task in EVENT_TYPES} for name in baseline_names
        }
        visitor_store: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = {
            cohort: {name: {task: [] for task in EVENT_TYPES} for name in baseline_names}
            for cohort in ("new", "sparse", "repeat")
        }
        item_store: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = {
            cohort: {name: {task: [] for task in EVENT_TYPES} for name in baseline_names}
            for cohort in ("head", "mid_tail", "long_tail", "unseen")
        }

        for query in queries:
            visitor = query["visitorid"]
            last_item = query["last_item"]
            history_count = visitor_counts.get(visitor, 0)
            vcohort = _visitor_cohort(history_count)
            category = category_map.get(last_item)
            rankings = {
                "most_view": global_rankings["most_view"][:k],
                "most_addtocart": global_rankings["most_addtocart"][:k],
                "most_transaction": global_rankings["most_transaction"][:k],
                "time_decayed_popularity": global_rankings["time_decayed_popularity"][:k],
                "visitor_history_repeat": _complete_ranking(
                    visitor_rankings.get(visitor, []), fallback, k
                ),
                "last_item_covisitation": _complete_ranking(
                    covisitation.get(last_item, []), fallback, k
                ),
                "last_item_transition": _complete_ranking(
                    transitions.get(last_item, []), fallback, k
                ),
                "category_conditioned_popularity": _complete_ranking(
                    category_rankings.get(category, []) if category is not None else [], fallback, k
                ),
            }
            for name, ranking in rankings.items():
                for task in EVENT_TYPES:
                    positives = query["positives"][task]
                    metric = _recall_ndcg(ranking, positives, k)
                    store[name][task].append(metric)
                    visitor_store[vcohort][name][task].append(metric)
                    if len(positives):
                        by_cohort: dict[str, list[int]] = {}
                        for item in positives:
                            cohort = item_cohort_map.get(int(item), "unseen")
                            by_cohort.setdefault(cohort, []).append(int(item))
                        for cohort, cohort_positives in by_cohort.items():
                            item_store[cohort][name][task].append(
                                _recall_ndcg(ranking, np.asarray(cohort_positives), k)
                            )

        overall[split] = {
            name: {task: _metric_dict(_aggregate(store[name][task])) for task in EVENT_TYPES}
            for name in baseline_names
        }
        visitor_cohorts[split] = {
            cohort: {
                name: {
                    task: _metric_dict(_aggregate(visitor_store[cohort][name][task]))
                    for task in EVENT_TYPES
                }
                for name in baseline_names
            }
            for cohort in visitor_store
        }
        item_cohorts[split] = {
            cohort: {
                name: {
                    task: _metric_dict(_aggregate(item_store[cohort][name][task]))
                    for task in EVENT_TYPES
                }
                for name in baseline_names
            }
            for cohort in item_store
        }

    return BaselineReport(
        source=str(dataset_dir / "events.csv"),
        manifest=str(manifest_path),
        k=k,
        prediction_point="after_first_event_of_each_multi_event_split_bounded_session",
        prediction_horizon=str(manifest["prediction_horizon"]),
        training_rows=int(len(train)),
        validation_sessions=split_session_counts["validation"],
        test_sessions=split_session_counts["test"],
        validation_evaluated_sessions=evaluated_session_counts["validation"],
        test_evaluated_sessions=evaluated_session_counts["test"],
        evaluation_sample_seed=seed,
        visitor_cohort_definition={"new": "0 training events", "sparse": "1-4 training events", "repeat": ">=5 training events"},
        item_cohort_definition={
            "head": "top 10% of training items by event count",
            "mid_tail": "next 40% of training items by event count",
            "long_tail": "bottom 50% of training items by event count",
            "unseen": "item absent from training events",
        },
        baselines=baseline_names,
        metrics=overall,
        visitor_cohorts=visitor_cohorts,
        item_cohorts=item_cohorts,
        notes=[
            "All baseline statistics are fitted on training events only.",
            "Each evaluation query is placed after the first event of a multi-event split-bounded session; labels are later items in that same session.",
            "When a session cap is used, eligible sessions are sampled deterministically before label aggregation and the sample seed is recorded.",
            "A baseline that cannot score an item falls back to training most-viewed popularity; future/unseen items are never injected into rankings.",
            "Category state is frozen at the training cutoff using only categoryid property rows available at or before that cutoff.",
            "The event tasks are reported separately; no weighted funnel objective is used in v0.3.",
            "Head/mid-tail/long-tail thresholds are project-defined descriptive cohorts, not universal retail categories.",
        ],
    )


def write_report(report: BaselineReport, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "behavioral_baselines.json"
    md_path = output_dir / "behavioral_baselines.md"
    json_path.write_text(json.dumps(asdict(report), indent=2, allow_nan=True) + "\n")

    lines = [
        "# CommerceRecLab v0.3 — Behavioral Baselines",
        "",
        f"Source: `{report.source}`",
        f"Frozen manifest: `{report.manifest}`",
        "",
        "## Evaluation contract",
        "",
        f"- K: **{report.k}**",
        f"- Prediction point: **{report.prediction_point}**",
        f"- Prediction horizon: **{report.prediction_horizon}**",
        f"- Training rows: **{report.training_rows:,}**",
        f"- Multi-event validation sessions: **{report.validation_sessions:,}** (evaluated: **{report.validation_evaluated_sessions:,}**)",
        f"- Multi-event test sessions: **{report.test_sessions:,}** (evaluated: **{report.test_evaluated_sessions:,}**)",
        f"- Evaluation sample seed: **{report.evaluation_sample_seed}**",
        "",
        "## Overall metrics",
        "",
    ]
    for split in ("validation", "test"):
        lines += [f"### {split.title()}", "", "| baseline | task | queries | Recall@K | NDCG@K |", "|---|---|---:|---:|---:|"]
        for baseline in report.baselines:
            for task in EVENT_TYPES:
                metric = report.metrics[split][baseline][task]
                lines.append(
                    f"| `{baseline}` | `{task}` | {metric['queries']:,} | {metric['recall_at_k']:.6f} | {metric['ndcg_at_k']:.6f} |"
                )
        lines.append("")
    lines += [
        "## Cohort definitions",
        "",
        "### Visitor history",
        "",
    ]
    lines += [f"- `{name}`: {definition}" for name, definition in report.visitor_cohort_definition.items()]
    lines += ["", "### Target item popularity", ""]
    lines += [f"- `{name}`: {definition}" for name, definition in report.item_cohort_definition.items()]
    lines += ["", "## Scientific notes", ""]
    lines += [f"- {note}" for note in report.notes]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate CommerceRecLab v0.3 behavioral baselines.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v0_3_behavioral_baselines"))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--half-life-days", type=float, default=7.0)
    parser.add_argument("--neighbor-top-n", type=int, default=50)
    parser.add_argument(
        "--max-sessions-per-split",
        type=int,
        default=50_000,
        help="Deterministic cap per split; use 0 to evaluate all eligible sessions.",
    )
    parser.add_argument("--seed", type=int, default=365)
    args = parser.parse_args(argv)
    report = evaluate_baselines(
        args.dataset_dir,
        args.manifest,
        k=args.k,
        half_life_days=args.half_life_days,
        neighbor_top_n=args.neighbor_top_n,
        max_sessions_per_split=args.max_sessions_per_split,
        seed=args.seed,
    )
    json_path, md_path = write_report(report, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
