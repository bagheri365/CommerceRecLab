from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from commercereclab.evaluation.baselines import (
    EVENT_TYPES,
    _aggregate,
    _category_popularity,
    _deduplicated_events,
    _event_rankings,
    _metric_dict,
    _read_manifest,
    _recall_ndcg,
    _session_queries,
    _transition_and_covisitation,
    _visitor_history,
)
from commercereclab.evaluation.hybrid import COMPONENTS, _rrf_fuse
from commercereclab.evaluation.temporal import assign_temporal_split

FEATURE_NAMES = tuple([f"rr_{x}" for x in COMPONENTS] + [f"present_{x}" for x in COMPONENTS])


def _category_maps_for_cutoffs(
    dataset_dir: Path,
    *,
    internal_cutoff_ms: int,
    full_cutoff_ms: int,
    chunksize: int = 500_000,
) -> tuple[dict[int, str], dict[int, str]]:
    pieces: list[pd.DataFrame] = []
    for filename in ("item_properties_part1.csv", "item_properties_part2.csv"):
        for chunk in pd.read_csv(
            dataset_dir / filename,
            usecols=["timestamp", "itemid", "property", "value"],
            chunksize=chunksize,
        ):
            cat = chunk[(chunk["property"] == "categoryid") & (chunk["timestamp"] <= full_cutoff_ms)]
            if len(cat):
                pieces.append(cat[["timestamp", "itemid", "value"]])
    if not pieces:
        return {}, {}
    categories = pd.concat(pieces, ignore_index=True).sort_values(["itemid", "timestamp"], kind="stable")
    full = categories.drop_duplicates("itemid", keep="last")
    internal_rows = categories[categories["timestamp"] <= internal_cutoff_ms]
    internal = internal_rows.drop_duplicates("itemid", keep="last")
    def as_map(frame: pd.DataFrame) -> dict[int, str]:
        return {int(row.itemid): str(row.value) for row in frame.itertuples(index=False)}
    return as_map(internal), as_map(full)


@dataclass(frozen=True)
class LearnedRankerReport:
    source: str
    manifest: str
    k: int
    source_depth: int
    rrf_offset: float
    internal_fit_fraction: float
    internal_fit_end_ms: int
    internal_ranker_sessions: int
    training_examples: dict[str, int]
    training_positive_examples: dict[str, int]
    prediction_point: str
    prediction_horizon: str
    validation_evaluated_sessions: int
    test_evaluated_sessions: int
    evaluation_sample_seed: int
    feature_names: list[str]
    standardized_coefficients: dict[str, dict[str, float]]
    candidate_recall: dict[str, dict[str, float]]
    metrics: dict[str, dict[str, dict[str, float | int]]]
    validation_mean_ndcg: dict[str, float]
    validation_retain_learned_ranker: bool
    retain_rule: str
    notes: list[str]


def _source_state(
    fit_events: pd.DataFrame,
    dataset_dir: Path,
    *,
    cutoff_ms: int,
    gap_minutes: int,
    source_depth: int,
    half_life_days: float,
    query_visitors: set[int],
    context_items: set[int],
    category_map: dict[int, str],
) -> dict:
    global_rankings = _event_rankings(fit_events, half_life_days=half_life_days)
    fallback = global_rankings["most_view"]
    category_rankings = _category_popularity(fit_events, category_map, top_n=source_depth)
    _, covisitation = _transition_and_covisitation(
        fit_events,
        gap_minutes=gap_minutes,
        top_n=source_depth,
        context_items=context_items,
    )
    visitor_rankings, _ = _visitor_history(
        fit_events,
        top_n=source_depth,
        visitors=query_visitors,
    )
    return {
        "fallback": fallback,
        "recency": global_rankings["time_decayed_popularity"][:source_depth],
        "category_map": category_map,
        "category_rankings": category_rankings,
        "covisitation": covisitation,
        "visitor_rankings": visitor_rankings,
    }


def _source_lists(query: dict, state: dict) -> dict[str, list[int]]:
    visitor = int(query["visitorid"])
    context_item = int(query["last_item"])
    category = state["category_map"].get(context_item)
    return {
        "category": state["category_rankings"].get(category, []) if category is not None else [],
        "covisitation": state["covisitation"].get(context_item, []),
        "recency": state["recency"],
        "visitor_history": state["visitor_rankings"].get(visitor, []),
    }


def _candidate_features(
    source_lists: dict[str, list[int]],
    *,
    source_depth: int,
    rrf_offset: float,
) -> tuple[list[int], np.ndarray]:
    ranks: dict[str, dict[int, int]] = {}
    candidates: set[int] = set()
    for component in COMPONENTS:
        mapping: dict[int, int] = {}
        for rank, item in enumerate(source_lists[component][:source_depth], start=1):
            item = int(item)
            if item not in mapping:
                mapping[item] = rank
                candidates.add(item)
        ranks[component] = mapping
    items = sorted(candidates)
    x = np.zeros((len(items), len(FEATURE_NAMES)), dtype=np.float64)
    for row, item in enumerate(items):
        for col, component in enumerate(COMPONENTS):
            rank = ranks[component].get(item)
            if rank is not None:
                x[row, col] = 1.0 / (rrf_offset + rank)
                x[row, col + len(COMPONENTS)] = 1.0
    return items, x


def _training_rows(
    queries: list[dict],
    state: dict,
    *,
    task: str,
    source_depth: int,
    rrf_offset: float,
    max_nonrelevant_per_query: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    usable_queries = 0
    positive_examples = 0
    for query in queries:
        positives = set(int(x) for x in query["positives"][task])
        if not positives:
            continue
        items, x = _candidate_features(
            _source_lists(query, state), source_depth=source_depth, rrf_offset=rrf_offset
        )
        if not items:
            continue
        y = np.asarray([1 if item in positives else 0 for item in items], dtype=np.int8)
        pos_idx = np.flatnonzero(y == 1)
        if len(pos_idx) == 0:
            continue
        neg_idx = np.flatnonzero(y == 0)[:max_nonrelevant_per_query]
        idx = np.concatenate([pos_idx, neg_idx])
        xs.append(x[idx])
        ys.append(y[idx])
        usable_queries += 1
        positive_examples += int(len(pos_idx))
    if not xs:
        raise ValueError(f"no usable internal training examples for task={task}")
    return np.vstack(xs), np.concatenate(ys), usable_queries, positive_examples


def _fit_model(x: np.ndarray, y: np.ndarray, *, seed: int) -> Pipeline:
    if len(np.unique(y)) < 2:
        raise ValueError("ranker training requires both relevant and sampled-nonrelevant examples")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def _rank_learned(
    query: dict,
    state: dict,
    model: Pipeline,
    *,
    k: int,
    source_depth: int,
    rrf_offset: float,
) -> tuple[list[int], set[int]]:
    source_lists = _source_lists(query, state)
    items, x = _candidate_features(source_lists, source_depth=source_depth, rrf_offset=rrf_offset)
    candidate_set = set(items)
    if items:
        scores = model.predict_proba(x)[:, 1]
        order = np.lexsort((np.asarray(items), -scores))
        primary = [items[i] for i in order]
    else:
        primary = []
    ranking: list[int] = []
    seen: set[int] = set()
    for item in [*primary, *state["fallback"]]:
        item = int(item)
        if item in seen:
            continue
        seen.add(item)
        ranking.append(item)
        if len(ranking) >= k:
            break
    return ranking, candidate_set


def evaluate_learned_ranker(
    dataset_dir: str | Path,
    manifest_path: str | Path,
    *,
    k: int = 20,
    source_depth: int = 100,
    rrf_offset: float = 60.0,
    half_life_days: float = 7.0,
    internal_fit_fraction: float = 0.80,
    max_internal_ranker_sessions: int | None = 20_000,
    max_sessions_per_split: int | None = 50_000,
    max_nonrelevant_per_query: int = 50,
    seed: int = 365,
) -> LearnedRankerReport:
    if not 0 < internal_fit_fraction < 1:
        raise ValueError("internal_fit_fraction must be between 0 and 1")
    if source_depth < k:
        raise ValueError("source_depth must be >= k")
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
    internal_fit_end_ms = int(train["timestamp"].quantile(internal_fit_fraction, interpolation="nearest"))
    source_train = train.loc[train["timestamp"] <= internal_fit_end_ms].copy()
    ranker_train = train.loc[train["timestamp"] > internal_fit_end_ms].copy()
    ranker_train["split"] = "ranker_train"
    ranker_queries, _ = _session_queries(
        ranker_train,
        split="ranker_train",
        gap_minutes=gap,
        max_sessions=max_internal_ranker_sessions,
        seed=seed + 10,
    )

    split_queries: dict[str, list[dict]] = {}
    for idx, split in enumerate(("validation", "test")):
        queries, _ = _session_queries(
            events,
            split=split,
            gap_minutes=gap,
            max_sessions=max_sessions_per_split,
            seed=seed + idx,
        )
        split_queries[split] = queries

    internal_category_map, full_category_map = _category_maps_for_cutoffs(
        dataset_dir,
        internal_cutoff_ms=internal_fit_end_ms,
        full_cutoff_ms=int(manifest["train_end_ms"]),
    )

    internal_visitors = {int(q["visitorid"]) for q in ranker_queries}
    internal_context = {int(q["last_item"]) for q in ranker_queries}
    internal_state = _source_state(
        source_train,
        dataset_dir,
        cutoff_ms=internal_fit_end_ms,
        gap_minutes=gap,
        source_depth=source_depth,
        half_life_days=half_life_days,
        query_visitors=internal_visitors,
        context_items=internal_context,
        category_map=internal_category_map,
    )

    models: dict[str, Pipeline] = {}
    training_examples: dict[str, int] = {}
    training_positive_examples: dict[str, int] = {}
    for task in EVENT_TYPES:
        x, y, _, positives = _training_rows(
            ranker_queries,
            internal_state,
            task=task,
            source_depth=source_depth,
            rrf_offset=rrf_offset,
            max_nonrelevant_per_query=max_nonrelevant_per_query,
        )
        models[task] = _fit_model(x, y, seed=seed)
        training_examples[task] = int(len(y))
        training_positive_examples[task] = int(positives)

    eval_visitors = {int(q["visitorid"]) for qs in split_queries.values() for q in qs}
    eval_context = {int(q["last_item"]) for qs in split_queries.values() for q in qs}
    eval_state = _source_state(
        train,
        dataset_dir,
        cutoff_ms=int(manifest["train_end_ms"]),
        gap_minutes=gap,
        source_depth=source_depth,
        half_life_days=half_life_days,
        query_visitors=eval_visitors,
        context_items=eval_context,
        category_map=full_category_map,
    )

    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    candidate_recall: dict[str, dict[str, float]] = {}
    for split in ("validation", "test"):
        stores = {
            variant: {task: [] for task in EVENT_TYPES}
            for variant in ("category_only", "equal_rrf", "learned_task_specific")
        }
        candidate_values = {task: [] for task in EVENT_TYPES}
        for query in split_queries[split]:
            source_lists = _source_lists(query, eval_state)
            category_ranking = _rrf_fuse(
                [source_lists["category"]],
                fallback=eval_state["fallback"],
                k=k,
                source_depth=source_depth,
                offset=rrf_offset,
            )
            equal_ranking = _rrf_fuse(
                [source_lists[c] for c in COMPONENTS],
                fallback=eval_state["fallback"],
                k=k,
                source_depth=source_depth,
                offset=rrf_offset,
            )
            learned_rankings = {}
            learned_candidates = None
            for task in EVENT_TYPES:
                learned_rankings[task], candidate_set = _rank_learned(
                    query,
                    eval_state,
                    models[task],
                    k=k,
                    source_depth=source_depth,
                    rrf_offset=rrf_offset,
                )
                learned_candidates = candidate_set
                positives = set(int(x) for x in query["positives"][task])
                if positives:
                    candidate_values[task].append(len(positives & candidate_set) / len(positives))
                stores["category_only"][task].append(
                    _recall_ndcg(category_ranking, query["positives"][task], k)
                )
                stores["equal_rrf"][task].append(
                    _recall_ndcg(equal_ranking, query["positives"][task], k)
                )
                stores["learned_task_specific"][task].append(
                    _recall_ndcg(learned_rankings[task], query["positives"][task], k)
                )
        metrics[split] = {
            variant: {task: _metric_dict(_aggregate(stores[variant][task])) for task in EVENT_TYPES}
            for variant in stores
        }
        candidate_recall[split] = {
            task: float(np.mean(candidate_values[task])) if candidate_values[task] else math.nan
            for task in EVENT_TYPES
        }

    validation_mean_ndcg = {
        variant: float(np.mean([metrics["validation"][variant][task]["ndcg_at_k"] for task in EVENT_TYPES]))
        for variant in metrics["validation"]
    }
    learned = "learned_task_specific"
    baseline = "category_only"
    task_wins = sum(
        metrics["validation"][learned][task]["ndcg_at_k"]
        > metrics["validation"][baseline][task]["ndcg_at_k"]
        for task in EVENT_TYPES
    )
    retain = bool(validation_mean_ndcg[learned] > validation_mean_ndcg[baseline] and task_wins >= 2)

    standardized_coefficients = {}
    for task, model in models.items():
        coefs = model.named_steps["logit"].coef_[0]
        standardized_coefficients[task] = {
            name: float(value) for name, value in zip(FEATURE_NAMES, coefs, strict=True)
        }

    return LearnedRankerReport(
        source=str(dataset_dir / "events.csv"),
        manifest=str(manifest_path),
        k=k,
        source_depth=source_depth,
        rrf_offset=rrf_offset,
        internal_fit_fraction=internal_fit_fraction,
        internal_fit_end_ms=internal_fit_end_ms,
        internal_ranker_sessions=len(ranker_queries),
        training_examples=training_examples,
        training_positive_examples=training_positive_examples,
        prediction_point="after_first_event_of_each_multi_event_split_bounded_session",
        prediction_horizon=str(manifest["prediction_horizon"]),
        validation_evaluated_sessions=len(split_queries["validation"]),
        test_evaluated_sessions=len(split_queries["test"]),
        evaluation_sample_seed=seed,
        feature_names=list(FEATURE_NAMES),
        standardized_coefficients=standardized_coefficients,
        candidate_recall=candidate_recall,
        metrics=metrics,
        validation_mean_ndcg=validation_mean_ndcg,
        validation_retain_learned_ranker=retain,
        retain_rule=(
            "Retain the learned task-specific ranker only if its mean validation NDCG across "
            "the three tasks exceeds category-only and it beats category-only on at least two "
            "of three task-specific validation NDCGs."
        ),
        notes=[
            "Behavioral source statistics for ranker training are fitted only on the earlier internal training subwindow.",
            "Ranker examples come only from later split-bounded sessions inside the frozen training period; validation labels are not used to fit coefficients.",
            "Separate logistic-regression rankers are fitted for view, addtocart, and transaction.",
            "Items without a future target event are sampled nonrelevant candidates for the offline ranking objective; they are not interpreted as observed dislikes or recommendation impressions.",
            "The learned ranker scores only the union of behavioral retrieval sources; candidate recall is reported separately so ranking gains cannot hide retrieval misses.",
            "Validation determines retain/reject; test is confirmation only.",
            "Category metadata is joined only at or before the relevant source-fit cutoff.",
        ],
    )


def write_report(report: LearnedRankerReport, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "learned_ranker.json"
    md_path = output_dir / "learned_ranker.md"
    json_path.write_text(json.dumps(asdict(report), indent=2, allow_nan=True) + "\n")
    lines = [
        "# CommerceRecLab v0.5 — Learned Task-Specific Rankers",
        "",
        f"Source: `{report.source}`",
        f"Frozen manifest: `{report.manifest}`",
        "",
        "## Evaluation contract",
        "",
        f"- K: **{report.k}**",
        f"- Source depth: **{report.source_depth}**",
        f"- Internal source-fit fraction of training time: **{report.internal_fit_fraction:.2f}**",
        f"- Internal source-fit end timestamp (ms): **{report.internal_fit_end_ms}**",
        f"- Internal ranker-training sessions: **{report.internal_ranker_sessions:,}**",
        f"- Validation sessions evaluated: **{report.validation_evaluated_sessions:,}**",
        f"- Test sessions evaluated: **{report.test_evaluated_sessions:,}**",
        "",
        "## Training examples",
        "",
        "| task | examples | retrieved positive examples |",
        "|---|---:|---:|",
    ]
    for task in EVENT_TYPES:
        lines.append(f"| `{task}` | {report.training_examples[task]:,} | {report.training_positive_examples[task]:,} |")
    lines += ["", "## Candidate recall", "", "| split | view | cart | transaction |", "|---|---:|---:|---:|"]
    for split in ("validation", "test"):
        c = report.candidate_recall[split]
        lines.append(f"| {split} | {c['view']:.6f} | {c['addtocart']:.6f} | {c['transaction']:.6f} |")
    lines += ["", "## Overall metrics", ""]
    for split in ("validation", "test"):
        lines += [f"### {split.title()}", "", "| variant | task | queries | Recall@K | NDCG@K |", "|---|---|---:|---:|---:|"]
        for variant in ("category_only", "equal_rrf", "learned_task_specific"):
            for task in EVENT_TYPES:
                m = report.metrics[split][variant][task]
                lines.append(f"| `{variant}` | `{task}` | {m['queries']:,} | {m['recall_at_k']:.6f} | {m['ndcg_at_k']:.6f} |")
        lines.append("")
    lines += ["## Standardized logistic coefficients", ""]
    for task in EVENT_TYPES:
        lines += [f"### {task}", "", "| feature | coefficient |", "|---|---:|"]
        for name, value in report.standardized_coefficients[task].items():
            lines.append(f"| `{name}` | {value:+.6f} |")
        lines.append("")
    lines += [
        "## Retain / reject decision",
        "",
        f"Rule: {report.retain_rule}",
        "",
        f"**Retain learned ranker: {'YES' if report.validation_retain_learned_ranker else 'NO'}**",
        "",
        "## Scientific notes",
        "",
    ]
    lines += [f"- {note}" for note in report.notes]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate CommerceRecLab v0.5 learned task-specific rankers.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v0_5_learned_ranker"))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--source-depth", type=int, default=100)
    parser.add_argument("--rrf-offset", type=float, default=60.0)
    parser.add_argument("--half-life-days", type=float, default=7.0)
    parser.add_argument("--internal-fit-fraction", type=float, default=0.80)
    parser.add_argument("--max-internal-ranker-sessions", type=int, default=20_000)
    parser.add_argument("--max-sessions-per-split", type=int, default=50_000)
    parser.add_argument("--max-nonrelevant-per-query", type=int, default=50)
    parser.add_argument("--seed", type=int, default=365)
    args = parser.parse_args(argv)
    report = evaluate_learned_ranker(
        args.dataset_dir,
        args.manifest,
        k=args.k,
        source_depth=args.source_depth,
        rrf_offset=args.rrf_offset,
        half_life_days=args.half_life_days,
        internal_fit_fraction=args.internal_fit_fraction,
        max_internal_ranker_sessions=args.max_internal_ranker_sessions,
        max_sessions_per_split=args.max_sessions_per_split,
        max_nonrelevant_per_query=args.max_nonrelevant_per_query,
        seed=args.seed,
    )
    json_path, md_path = write_report(report, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
