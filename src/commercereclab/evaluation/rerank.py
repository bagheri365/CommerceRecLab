from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from commercereclab.evaluation.baselines import (
    EVENT_TYPES,
    _aggregate,
    _deduplicated_events,
    _metric_dict,
    _read_manifest,
    _recall_ndcg,
    _session_queries,
)
from commercereclab.evaluation.hybrid import COMPONENTS, _rrf_fuse
from commercereclab.evaluation.learned import FEATURE_NAMES, _fit_model
from commercereclab.evaluation.retrieval import _fit_retrieval_state
from commercereclab.evaluation.temporal import assign_temporal_split

DETERMINISTIC_VARIANTS = ("category_only", "efficient_rrf")
LEARNED_VARIANT = "learned_task_specific"


@dataclass(frozen=True)
class EfficientRerankReport:
    source: str
    manifest: str
    k: int
    category_depth: int
    behavioral_depth: int
    parent_depth: int
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
    candidate_size: dict[str, dict[str, float]]
    metrics: dict[str, dict[str, dict[str, dict[str, float | int]]]]
    validation_mean_ndcg: dict[str, float]
    best_deterministic_validation_baseline: str
    validation_retain_learned_ranker: bool
    retain_rule: str
    notes: list[str]


def _source_lists(query: dict, state: dict) -> dict[str, list[int]]:
    visitor = int(query["visitorid"])
    context_item = int(query["last_item"])
    category = state["category_map"].get(context_item)
    return {
        "category": state["category_rankings"].get(category, []) if category is not None else [],
        "covisitation": state["covisitation"].get(context_item, []),
        "recency": state["recency"],
        "visitor_history": state["visitor_history"].get(visitor, []),
    }


def _component_depth(component: str, *, category_depth: int, behavioral_depth: int) -> int:
    return category_depth if component == "category" else behavioral_depth


def _candidate_features(
    source_lists: dict[str, list[int]],
    *,
    category_depth: int,
    behavioral_depth: int,
    rrf_offset: float,
) -> tuple[list[int], np.ndarray]:
    ranks: dict[str, dict[int, int]] = {}
    candidates: set[int] = set()
    for component in COMPONENTS:
        depth = _component_depth(
            component,
            category_depth=category_depth,
            behavioral_depth=behavioral_depth,
        )
        mapping: dict[int, int] = {}
        for rank, item in enumerate(source_lists[component][:depth], start=1):
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
    category_depth: int,
    behavioral_depth: int,
    rrf_offset: float,
    max_nonrelevant_per_query: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    positive_examples = 0
    for query in queries:
        positives = set(int(x) for x in query["positives"][task])
        if not positives:
            continue
        items, x = _candidate_features(
            _source_lists(query, state),
            category_depth=category_depth,
            behavioral_depth=behavioral_depth,
            rrf_offset=rrf_offset,
        )
        if not items:
            continue
        y = np.asarray([1 if item in positives else 0 for item in items], dtype=np.int8)
        pos_idx = np.flatnonzero(y == 1)
        if len(pos_idx) == 0:
            continue
        # Preserve v0.5's deterministic sampled-nonrelevant construction so v0.8 isolates
        # the effect of the stronger candidate generator rather than changing two variables.
        neg_idx = np.flatnonzero(y == 0)[:max_nonrelevant_per_query]
        idx = np.concatenate([pos_idx, neg_idx])
        xs.append(x[idx])
        ys.append(y[idx])
        positive_examples += int(len(pos_idx))
    if not xs:
        raise ValueError(f"no usable internal training examples for task={task}")
    return np.vstack(xs), np.concatenate(ys), positive_examples


def _learned_ranking(
    query: dict,
    state: dict,
    model: Pipeline,
    *,
    k: int,
    category_depth: int,
    behavioral_depth: int,
    rrf_offset: float,
) -> tuple[list[int], set[int]]:
    items, x = _candidate_features(
        _source_lists(query, state),
        category_depth=category_depth,
        behavioral_depth=behavioral_depth,
        rrf_offset=rrf_offset,
    )
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


def evaluate_efficient_reranker(
    dataset_dir: str | Path,
    manifest_path: str | Path,
    *,
    k: int = 20,
    category_depth: int = 300,
    behavioral_depth: int = 100,
    parent_depth: int = 0,
    rrf_offset: float = 60.0,
    half_life_days: float = 7.0,
    internal_fit_fraction: float = 0.80,
    max_internal_ranker_sessions: int | None = 20_000,
    max_sessions_per_split: int | None = 50_000,
    max_nonrelevant_per_query: int = 50,
    seed: int = 365,
) -> EfficientRerankReport:
    if k <= 0:
        raise ValueError("k must be positive")
    if category_depth < k or behavioral_depth < k:
        raise ValueError("category_depth and behavioral_depth must be >= k")
    if parent_depth != 0:
        raise ValueError("v0.8 freezes the v0.7-selected parent_depth at 0")
    if not 0 < internal_fit_fraction < 1:
        raise ValueError("internal_fit_fraction must be between 0 and 1")

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

    internal_fit_end_ms = int(
        train["timestamp"].quantile(internal_fit_fraction, interpolation="nearest")
    )
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

    max_depth = max(category_depth, behavioral_depth)
    internal_visitors = {int(q["visitorid"]) for q in ranker_queries}
    internal_contexts = {int(q["last_item"]) for q in ranker_queries}
    internal_state = _fit_retrieval_state(
        source_train,
        dataset_dir,
        train_end_ms=internal_fit_end_ms,
        gap_minutes=gap,
        expanded_depth=max_depth,
        half_life_days=half_life_days,
        query_visitors=internal_visitors,
        context_items=internal_contexts,
    )

    models: dict[str, Pipeline] = {}
    training_examples: dict[str, int] = {}
    training_positive_examples: dict[str, int] = {}
    for task in EVENT_TYPES:
        x, y, positives = _training_rows(
            ranker_queries,
            internal_state,
            task=task,
            category_depth=category_depth,
            behavioral_depth=behavioral_depth,
            rrf_offset=rrf_offset,
            max_nonrelevant_per_query=max_nonrelevant_per_query,
        )
        models[task] = _fit_model(x, y, seed=seed)
        training_examples[task] = int(len(y))
        training_positive_examples[task] = int(positives)

    eval_visitors = {int(q["visitorid"]) for qs in split_queries.values() for q in qs}
    eval_contexts = {int(q["last_item"]) for qs in split_queries.values() for q in qs}
    eval_state = _fit_retrieval_state(
        train,
        dataset_dir,
        train_end_ms=int(manifest["train_end_ms"]),
        gap_minutes=gap,
        expanded_depth=max_depth,
        half_life_days=half_life_days,
        query_visitors=eval_visitors,
        context_items=eval_contexts,
    )

    metrics: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    candidate_recall: dict[str, dict[str, float]] = {}
    candidate_size: dict[str, dict[str, float]] = {}
    for split in ("validation", "test"):
        stores = {
            variant: {task: [] for task in EVENT_TYPES}
            for variant in (*DETERMINISTIC_VARIANTS, LEARNED_VARIANT)
        }
        candidate_values = {task: [] for task in EVENT_TYPES}
        candidate_sizes: list[int] = []
        for query in split_queries[split]:
            source_lists = _source_lists(query, eval_state)
            category_ranking = _rrf_fuse(
                [source_lists["category"]],
                fallback=eval_state["fallback"],
                k=k,
                source_depth=category_depth,
                offset=rrf_offset,
            )
            efficient_lists = [
                source_lists["category"][:category_depth],
                source_lists["covisitation"][:behavioral_depth],
                source_lists["recency"][:behavioral_depth],
                source_lists["visitor_history"][:behavioral_depth],
            ]
            efficient_ranking = _rrf_fuse(
                efficient_lists,
                fallback=eval_state["fallback"],
                k=k,
                source_depth=category_depth,
                offset=rrf_offset,
            )
            candidate_set: set[int] | None = None
            learned_rankings: dict[str, list[int]] = {}
            for task in EVENT_TYPES:
                learned_rankings[task], task_candidates = _learned_ranking(
                    query,
                    eval_state,
                    models[task],
                    k=k,
                    category_depth=category_depth,
                    behavioral_depth=behavioral_depth,
                    rrf_offset=rrf_offset,
                )
                if candidate_set is None:
                    candidate_set = task_candidates
                    candidate_sizes.append(len(task_candidates))
                positives = set(int(x) for x in query["positives"][task])
                if positives:
                    candidate_values[task].append(
                        len(positives & task_candidates) / len(positives)
                    )
                stores["category_only"][task].append(
                    _recall_ndcg(category_ranking, query["positives"][task], k)
                )
                stores["efficient_rrf"][task].append(
                    _recall_ndcg(efficient_ranking, query["positives"][task], k)
                )
                stores[LEARNED_VARIANT][task].append(
                    _recall_ndcg(learned_rankings[task], query["positives"][task], k)
                )
        metrics[split] = {
            variant: {
                task: _metric_dict(_aggregate(stores[variant][task])) for task in EVENT_TYPES
            }
            for variant in stores
        }
        candidate_recall[split] = {
            task: float(np.mean(candidate_values[task])) if candidate_values[task] else math.nan
            for task in EVENT_TYPES
        }
        candidate_size[split] = {
            "mean": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
            "median": float(np.median(candidate_sizes)) if candidate_sizes else 0.0,
            "p95": float(np.percentile(candidate_sizes, 95)) if candidate_sizes else 0.0,
        }

    validation_mean_ndcg = {
        variant: float(
            np.mean([metrics["validation"][variant][task]["ndcg_at_k"] for task in EVENT_TYPES])
        )
        for variant in metrics["validation"]
    }
    best_deterministic = max(
        DETERMINISTIC_VARIANTS,
        key=lambda variant: (validation_mean_ndcg[variant], variant),
    )
    task_wins = sum(
        metrics["validation"][LEARNED_VARIANT][task]["ndcg_at_k"]
        > metrics["validation"][best_deterministic][task]["ndcg_at_k"]
        for task in EVENT_TYPES
    )
    retain = bool(
        validation_mean_ndcg[LEARNED_VARIANT] > validation_mean_ndcg[best_deterministic]
        and task_wins >= 2
    )

    standardized_coefficients: dict[str, dict[str, float]] = {}
    for task, model in models.items():
        coefs = model.named_steps["logit"].coef_[0]
        standardized_coefficients[task] = {
            name: float(value) for name, value in zip(FEATURE_NAMES, coefs, strict=True)
        }

    return EfficientRerankReport(
        source=str(dataset_dir / "events.csv"),
        manifest=str(manifest_path),
        k=k,
        category_depth=category_depth,
        behavioral_depth=behavioral_depth,
        parent_depth=parent_depth,
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
        candidate_size=candidate_size,
        metrics=metrics,
        validation_mean_ndcg=validation_mean_ndcg,
        best_deterministic_validation_baseline=best_deterministic,
        validation_retain_learned_ranker=retain,
        retain_rule=(
            "Retain the learned ranker only if its mean validation NDCG across the three tasks "
            "exceeds the best deterministic validation baseline (deep category-only or fixed "
            "RRF on the same efficient candidate generator) and it beats that baseline on at "
            "least two of three task-specific validation NDCGs."
        ),
        notes=[
            f"The candidate generator uses c{category_depth}_b{behavioral_depth}_p{parent_depth}; the v0.8 default freezes the v0.7-selected c300_b100_p0 configuration.",
            "Behavioral source statistics for ranker training are fitted only on the earlier internal training subwindow.",
            "Ranker examples come only from later split-bounded sessions inside the frozen training period; validation labels are not used to fit coefficients.",
            "Separate logistic-regression rankers are fitted for view, addtocart, and transaction.",
            "The sampled-nonrelevant construction is held consistent with v0.5 so the v0.8 intervention isolates the candidate-generator change.",
            "Sampled nonrelevant candidates are offline ranking labels, not observed dislikes or recommendation impressions.",
            "Candidate recall and candidate-set size are reported separately from top-K ranking metrics.",
            "Validation determines retain/reject; test is confirmation only.",
            "Category metadata is joined only at or before the relevant source-fit cutoff.",
        ],
    )


def write_report(report: EfficientRerankReport, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "efficient_reranker.json"
    md_path = output_dir / "efficient_reranker.md"
    json_path.write_text(json.dumps(asdict(report), indent=2, allow_nan=True) + "\n")

    lines = [
        "# CommerceRecLab v0.8 — Learned Reranking on Efficient Retrieval",
        "",
        f"Source: `{report.source}`",
        f"Frozen manifest: `{report.manifest}`",
        "",
        "## Evaluation contract",
        "",
        f"- K: **{report.k}**",
        f"- Frozen retriever: **c{report.category_depth}_b{report.behavioral_depth}_p{report.parent_depth}**",
        f"- Internal source-fit fraction of training time: **{report.internal_fit_fraction:.2f}**",
        f"- Internal source-fit end timestamp (ms): **{report.internal_fit_end_ms}**",
        f"- Internal ranker-training sessions: **{report.internal_ranker_sessions:,}**",
        f"- Validation sessions evaluated: **{report.validation_evaluated_sessions:,}**",
        f"- Test sessions evaluated: **{report.test_evaluated_sessions:,}**",
        f"- Seed: **{report.evaluation_sample_seed}**",
        "",
        "## Training examples",
        "",
        "| task | examples | retrieved positive examples |",
        "|---|---:|---:|",
    ]
    for task in EVENT_TYPES:
        lines.append(
            f"| `{task}` | {report.training_examples[task]:,} | "
            f"{report.training_positive_examples[task]:,} |"
        )

    lines += [
        "",
        "## Candidate retrieval",
        "",
        "| split | view recall | cart recall | transaction recall | mean candidates | median | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        c = report.candidate_recall[split]
        s = report.candidate_size[split]
        lines.append(
            f"| {split} | {c['view']:.6f} | {c['addtocart']:.6f} | "
            f"{c['transaction']:.6f} | {s['mean']:.1f} | {s['median']:.1f} | {s['p95']:.1f} |"
        )

    for split in ("validation", "test"):
        lines += [
            "",
            f"## {split.title()} ranking metrics",
            "",
            "| variant | task | queries | Recall@K | NDCG@K |",
            "|---|---|---:|---:|---:|",
        ]
        for variant in ("category_only", "efficient_rrf", LEARNED_VARIANT):
            for task in EVENT_TYPES:
                m = report.metrics[split][variant][task]
                lines.append(
                    f"| `{variant}` | `{task}` | {m['queries']:,} | "
                    f"{m['recall_at_k']:.6f} | {m['ndcg_at_k']:.6f} |"
                )

    lines += [
        "",
        "## Standardized logistic coefficients",
        "",
    ]
    for task in EVENT_TYPES:
        lines += [f"### {task}", "", "| feature | coefficient |", "|---|---:|"]
        for feature in report.feature_names:
            lines.append(
                f"| `{feature}` | {report.standardized_coefficients[task][feature]:+.6f} |"
            )
        lines.append("")

    lines += [
        "## Retain / reject decision",
        "",
        f"Best deterministic validation baseline: `{report.best_deterministic_validation_baseline}`",
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
    parser = argparse.ArgumentParser(
        description="Evaluate CommerceRecLab v0.8 learned reranking on the efficient retriever."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/v0_8_efficient_reranker")
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--category-depth", type=int, default=300)
    parser.add_argument("--behavioral-depth", type=int, default=100)
    parser.add_argument("--parent-depth", type=int, default=0)
    parser.add_argument("--rrf-offset", type=float, default=60.0)
    parser.add_argument("--half-life-days", type=float, default=7.0)
    parser.add_argument("--internal-fit-fraction", type=float, default=0.80)
    parser.add_argument("--max-internal-ranker-sessions", type=int, default=20_000)
    parser.add_argument("--max-sessions-per-split", type=int, default=50_000)
    parser.add_argument("--max-nonrelevant-per-query", type=int, default=50)
    parser.add_argument("--seed", type=int, default=365)
    args = parser.parse_args(argv)
    report = evaluate_efficient_reranker(
        args.dataset_dir,
        args.manifest,
        k=args.k,
        category_depth=args.category_depth,
        behavioral_depth=args.behavioral_depth,
        parent_depth=args.parent_depth,
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
