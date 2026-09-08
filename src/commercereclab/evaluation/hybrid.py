from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from itertools import chain
from typing import Iterable

import numpy as np

from commercereclab.evaluation.baselines import (
    EVENT_TYPES,
    _aggregate,
    _category_popularity,
    _deduplicated_events,
    _event_rankings,
    _latest_category_map,
    _metric_dict,
    _read_manifest,
    _recall_ndcg,
    _session_queries,
    _transition_and_covisitation,
    _visitor_history,
)
from commercereclab.evaluation.temporal import assign_temporal_split

COMPONENTS = ("category", "covisitation", "recency", "visitor_history")


@dataclass(frozen=True)
class HybridReport:
    source: str
    manifest: str
    k: int
    source_depth: int
    rrf_offset: float
    prediction_point: str
    prediction_horizon: str
    validation_evaluated_sessions: int
    test_evaluated_sessions: int
    evaluation_sample_seed: int
    components: list[str]
    variants: dict[str, list[str]]
    metrics: dict[str, dict[str, dict[str, float | int]]]
    validation_ablation_delta_ndcg: dict[str, dict[str, float]]
    validation_mean_ndcg: dict[str, float]
    validation_retain_full_hybrid: bool
    retain_rule: str
    notes: list[str]


def _rrf_fuse(
    sources: Iterable[Iterable[int]],
    *,
    fallback: Iterable[int],
    k: int,
    source_depth: int,
    offset: float,
) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    if source_depth < k:
        raise ValueError("source_depth must be >= k")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for source in sources:
        seen_source: set[int] = set()
        for rank, raw_item in enumerate(source, start=1):
            if rank > source_depth:
                break
            item = int(raw_item)
            if item in seen_source:
                continue
            seen_source.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (offset + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)

    ranked = sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))
    out: list[int] = []
    seen: set[int] = set()
    for raw_item in chain(ranked, fallback):
        item = int(raw_item)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= k:
            break
    return out


def _variant_map() -> dict[str, list[str]]:
    return {
        "category_only": ["category"],
        "covisitation_only": ["covisitation"],
        "recency_only": ["recency"],
        "visitor_history_only": ["visitor_history"],
        "hybrid_full": list(COMPONENTS),
        "hybrid_minus_category": ["covisitation", "recency", "visitor_history"],
        "hybrid_minus_covisitation": ["category", "recency", "visitor_history"],
        "hybrid_minus_recency": ["category", "covisitation", "visitor_history"],
        "hybrid_minus_visitor_history": ["category", "covisitation", "recency"],
    }


def evaluate_hybrid(
    dataset_dir: str | Path,
    manifest_path: str | Path,
    *,
    k: int = 20,
    source_depth: int = 100,
    rrf_offset: float = 60.0,
    half_life_days: float = 7.0,
    max_sessions_per_split: int | None = 50_000,
    seed: int = 365,
) -> HybridReport:
    if k <= 0:
        raise ValueError("k must be positive")
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

    split_queries: dict[str, list[dict]] = {}
    for split_index, split in enumerate(("validation", "test")):
        queries, _ = _session_queries(
            events,
            split=split,
            gap_minutes=gap,
            max_sessions=max_sessions_per_split,
            seed=seed + split_index,
        )
        split_queries[split] = queries

    query_visitors = {int(q["visitorid"]) for qs in split_queries.values() for q in qs}
    context_items = {int(q["last_item"]) for qs in split_queries.values() for q in qs}

    global_rankings = _event_rankings(train, half_life_days=half_life_days)
    fallback = global_rankings["most_view"]
    recency = global_rankings["time_decayed_popularity"][:source_depth]

    category_map = _latest_category_map(dataset_dir, cutoff_ms=int(manifest["train_end_ms"]))
    category_rankings = _category_popularity(train, category_map, top_n=source_depth)
    _, covisitation = _transition_and_covisitation(
        train,
        gap_minutes=gap,
        top_n=source_depth,
        context_items=context_items,
    )
    visitor_rankings, _ = _visitor_history(
        train,
        top_n=source_depth,
        visitors=query_visitors,
    )

    variants = _variant_map()
    metrics: dict[str, dict[str, dict[str, float | int]]] = {}

    for split in ("validation", "test"):
        store: dict[str, dict[str, list[tuple[float, float]]]] = {
            variant: {task: [] for task in EVENT_TYPES} for variant in variants
        }
        for query in split_queries[split]:
            visitor = int(query["visitorid"])
            context_item = int(query["last_item"])
            category = category_map.get(context_item)
            source_lists = {
                "category": category_rankings.get(category, []) if category is not None else [],
                "covisitation": covisitation.get(context_item, []),
                "recency": recency,
                "visitor_history": visitor_rankings.get(visitor, []),
            }
            rankings = {
                name: _rrf_fuse(
                    [source_lists[component] for component in components],
                    fallback=fallback,
                    k=k,
                    source_depth=source_depth,
                    offset=rrf_offset,
                )
                for name, components in variants.items()
            }
            for name, ranking in rankings.items():
                for task in EVENT_TYPES:
                    store[name][task].append(
                        _recall_ndcg(ranking, query["positives"][task], k)
                    )

        metrics[split] = {
            name: {task: _metric_dict(_aggregate(store[name][task])) for task in EVENT_TYPES}
            for name in variants
        }

    validation_mean_ndcg = {
        name: float(np.mean([metrics["validation"][name][task]["ndcg_at_k"] for task in EVENT_TYPES]))
        for name in variants
    }
    best_single_mean = max(validation_mean_ndcg[f"{component}_only"] for component in COMPONENTS)
    full_mean = validation_mean_ndcg["hybrid_full"]
    task_wins = sum(
        metrics["validation"]["hybrid_full"][task]["ndcg_at_k"]
        >= max(metrics["validation"][f"{component}_only"][task]["ndcg_at_k"] for component in COMPONENTS)
        for task in EVENT_TYPES
    )
    retain = bool(full_mean > best_single_mean and task_wins >= 2)

    ablation_delta = {
        component: {
            task: float(
                metrics["validation"]["hybrid_full"][task]["ndcg_at_k"]
                - metrics["validation"][f"hybrid_minus_{component}"][task]["ndcg_at_k"]
            )
            for task in EVENT_TYPES
        }
        for component in COMPONENTS
    }

    return HybridReport(
        source=str(dataset_dir / "events.csv"),
        manifest=str(manifest_path),
        k=k,
        source_depth=source_depth,
        rrf_offset=rrf_offset,
        prediction_point="after_first_event_of_each_multi_event_split_bounded_session",
        prediction_horizon=str(manifest["prediction_horizon"]),
        validation_evaluated_sessions=len(split_queries["validation"]),
        test_evaluated_sessions=len(split_queries["test"]),
        evaluation_sample_seed=seed,
        components=list(COMPONENTS),
        variants=variants,
        metrics=metrics,
        validation_ablation_delta_ndcg=ablation_delta,
        validation_mean_ndcg=validation_mean_ndcg,
        validation_retain_full_hybrid=retain,
        retain_rule=(
            "Retain the equal-weight full hybrid only if its mean validation NDCG across the "
            "three tasks exceeds the best single component and it matches or beats the best "
            "single component on at least two of three task-specific validation NDCGs."
        ),
        notes=[
            "All component statistics are fitted on the frozen training period only.",
            "The fusion rule is equal-weight reciprocal-rank fusion; no task-specific weights are tuned in v0.4.",
            "Validation determines the retain/reject decision; test results are reported as confirmation and are not used to tune the fusion.",
            "A positive ablation delta means removing that component reduced validation NDCG for the task.",
            "Cart and transaction metrics are conditional ranking metrics over queries containing corresponding future targets, not arbitrary-session conversion probabilities.",
            "Category state is frozen at the training cutoff; no future item metadata enters the fusion.",
        ],
    )


def write_report(report: HybridReport, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hybrid_ablation.json"
    md_path = output_dir / "hybrid_ablation.md"
    json_path.write_text(json.dumps(asdict(report), indent=2, allow_nan=True) + "\n")

    lines = [
        "# CommerceRecLab v0.4 — Hybrid Behavioral Scoring + Ablation",
        "",
        f"Source: `{report.source}`",
        f"Frozen manifest: `{report.manifest}`",
        "",
        "## Evaluation contract",
        "",
        f"- K: **{report.k}**",
        f"- Source depth: **{report.source_depth}**",
        f"- RRF offset: **{report.rrf_offset:g}**",
        f"- Prediction point: **{report.prediction_point}**",
        f"- Prediction horizon: **{report.prediction_horizon}**",
        f"- Validation sessions evaluated: **{report.validation_evaluated_sessions:,}**",
        f"- Test sessions evaluated: **{report.test_evaluated_sessions:,}**",
        f"- Evaluation sample seed: **{report.evaluation_sample_seed}**",
        "",
        "## Fusion variants",
        "",
    ]
    for name, components in report.variants.items():
        lines.append(f"- `{name}`: {', '.join(components)}")
    lines += ["", "## Overall metrics", ""]
    for split in ("validation", "test"):
        lines += [
            f"### {split.title()}",
            "",
            "| variant | task | queries | Recall@K | NDCG@K |",
            "|---|---|---:|---:|---:|",
        ]
        for variant in report.variants:
            for task in EVENT_TYPES:
                metric = report.metrics[split][variant][task]
                lines.append(
                    f"| `{variant}` | `{task}` | {metric['queries']:,} | "
                    f"{metric['recall_at_k']:.6f} | {metric['ndcg_at_k']:.6f} |"
                )
        lines.append("")

    lines += [
        "## Validation leave-one-component-out ablation",
        "",
        "Positive values mean the full hybrid performed better than the corresponding ablation.",
        "",
        "| component removed | view ΔNDCG | cart ΔNDCG | transaction ΔNDCG |",
        "|---|---:|---:|---:|",
    ]
    for component in report.components:
        delta = report.validation_ablation_delta_ndcg[component]
        lines.append(
            f"| `{component}` | {delta['view']:+.6f} | {delta['addtocart']:+.6f} | "
            f"{delta['transaction']:+.6f} |"
        )

    lines += [
        "",
        "## Retain / reject decision",
        "",
        f"Rule: {report.retain_rule}",
        "",
        f"**Retain full hybrid: {'YES' if report.validation_retain_full_hybrid else 'NO'}**",
        "",
        "## Scientific notes",
        "",
    ]
    lines += [f"- {note}" for note in report.notes]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate CommerceRecLab v0.4 equal-weight hybrid behavioral scoring and ablations."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v0_4_hybrid_ablation"))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--source-depth", type=int, default=100)
    parser.add_argument("--rrf-offset", type=float, default=60.0)
    parser.add_argument("--half-life-days", type=float, default=7.0)
    parser.add_argument("--max-sessions-per-split", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=365)
    args = parser.parse_args(argv)

    report = evaluate_hybrid(
        args.dataset_dir,
        args.manifest,
        k=args.k,
        source_depth=args.source_depth,
        rrf_offset=args.rrf_offset,
        half_life_days=args.half_life_days,
        max_sessions_per_split=args.max_sessions_per_split,
        seed=args.seed,
    )
    json_path, md_path = write_report(report, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
