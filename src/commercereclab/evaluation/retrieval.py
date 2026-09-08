from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
from commercereclab.evaluation.hybrid import _rrf_fuse
from commercereclab.evaluation.temporal import assign_temporal_split

VARIANTS = (
    "base_union",
    "deep_category",
    "deep_behavioral",
    "parent_category",
    "expanded_union",
)


@dataclass(frozen=True)
class RetrievalReport:
    source: str
    manifest: str
    k: int
    base_depth: int
    expanded_depth: int
    prediction_point: str
    prediction_horizon: str
    validation_evaluated_sessions: int
    test_evaluated_sessions: int
    evaluation_sample_seed: int
    variants: dict[str, str]
    candidate_recall: dict[str, dict[str, dict[str, float]]]
    candidate_size: dict[str, dict[str, dict[str, float]]]
    ranking_metrics: dict[str, dict[str, dict[str, dict[str, float | int]]]]
    validation_mean_candidate_recall: dict[str, float]
    validation_retain_expanded_union: bool
    retain_rule: str
    notes: list[str]


def _category_parent_children(dataset_dir: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    tree = pd.read_csv(dataset_dir / "category_tree.csv")
    parent: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    for row in tree.itertuples(index=False):
        category = str(int(row.categoryid))
        if pd.isna(row.parentid):
            continue
        parent_id = str(int(row.parentid))
        parent[category] = parent_id
        children.setdefault(parent_id, []).append(category)
    for values in children.values():
        values.sort(key=lambda x: int(x) if x.isdigit() else x)
    return parent, children


def _parent_category_items(
    category: str | None,
    *,
    parent_map: dict[str, str],
    children_map: dict[str, list[str]],
    category_rankings: dict[str, list[int]],
    depth: int,
) -> list[int]:
    if category is None or category not in parent_map:
        return []
    siblings = children_map.get(parent_map[category], [])
    out: list[int] = []
    seen: set[int] = set()
    # Interleave sibling-category popularity so one large sibling cannot monopolize the pool.
    lists = [category_rankings.get(sibling, []) for sibling in siblings]
    for rank in range(depth):
        added = False
        for values in lists:
            if rank >= len(values):
                continue
            item = int(values[rank])
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
            added = True
            if len(out) >= depth:
                return out
        if not added and all(rank >= len(values) for values in lists):
            break
    return out


def _fit_retrieval_state(
    train: pd.DataFrame,
    dataset_dir: Path,
    *,
    train_end_ms: int,
    gap_minutes: int,
    expanded_depth: int,
    half_life_days: float,
    query_visitors: set[int],
    context_items: set[int],
) -> dict:
    category_map = _latest_category_map(dataset_dir, cutoff_ms=train_end_ms)
    category_rankings = _category_popularity(train, category_map, top_n=expanded_depth)
    rankings = _event_rankings(train, half_life_days=half_life_days)
    _, covisitation = _transition_and_covisitation(
        train,
        gap_minutes=gap_minutes,
        top_n=expanded_depth,
        context_items=context_items,
    )
    visitor_history, _ = _visitor_history(train, top_n=expanded_depth, visitors=query_visitors)
    parent_map, children_map = _category_parent_children(dataset_dir)
    return {
        "fallback": rankings["most_view"],
        "recency": rankings["time_decayed_popularity"][:expanded_depth],
        "category_map": category_map,
        "category_rankings": category_rankings,
        "covisitation": covisitation,
        "visitor_history": visitor_history,
        "parent_map": parent_map,
        "children_map": children_map,
    }


def _variant_lists(query: dict, state: dict, *, base_depth: int, expanded_depth: int) -> dict[str, list[list[int]]]:
    visitor = int(query["visitorid"])
    context = int(query["last_item"])
    category = state["category_map"].get(context)
    category_items = state["category_rankings"].get(category, []) if category is not None else []
    covisit = state["covisitation"].get(context, [])
    recency = state["recency"]
    history = state["visitor_history"].get(visitor, [])
    parent_items = _parent_category_items(
        category,
        parent_map=state["parent_map"],
        children_map=state["children_map"],
        category_rankings=state["category_rankings"],
        depth=expanded_depth,
    )
    return {
        "base_union": [category_items[:base_depth], covisit[:base_depth], recency[:base_depth], history[:base_depth]],
        "deep_category": [category_items[:expanded_depth], covisit[:base_depth], recency[:base_depth], history[:base_depth]],
        "deep_behavioral": [category_items[:base_depth], covisit[:expanded_depth], recency[:expanded_depth], history[:expanded_depth]],
        "parent_category": [category_items[:base_depth], parent_items[:expanded_depth], covisit[:base_depth], recency[:base_depth], history[:base_depth]],
        "expanded_union": [category_items[:expanded_depth], parent_items[:expanded_depth], covisit[:expanded_depth], recency[:expanded_depth], history[:expanded_depth]],
    }


def _candidate_set(lists: list[list[int]]) -> set[int]:
    return {int(item) for values in lists for item in values}


def evaluate_retrieval(
    dataset_dir: str | Path,
    manifest_path: str | Path,
    *,
    k: int = 20,
    base_depth: int = 100,
    expanded_depth: int = 300,
    half_life_days: float = 7.0,
    max_sessions_per_split: int | None = 50_000,
    seed: int = 365,
) -> RetrievalReport:
    if k <= 0:
        raise ValueError("k must be positive")
    if base_depth < k:
        raise ValueError("base_depth must be >= k")
    if expanded_depth <= base_depth:
        raise ValueError("expanded_depth must be greater than base_depth")

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
    for idx, split in enumerate(("validation", "test")):
        queries, _ = _session_queries(
            events,
            split=split,
            gap_minutes=gap,
            max_sessions=max_sessions_per_split,
            seed=seed + idx,
        )
        split_queries[split] = queries

    visitors = {int(q["visitorid"]) for qs in split_queries.values() for q in qs}
    contexts = {int(q["last_item"]) for qs in split_queries.values() for q in qs}
    state = _fit_retrieval_state(
        train,
        dataset_dir,
        train_end_ms=int(manifest["train_end_ms"]),
        gap_minutes=gap,
        expanded_depth=expanded_depth,
        half_life_days=half_life_days,
        query_visitors=visitors,
        context_items=contexts,
    )

    candidate_recall: dict[str, dict[str, dict[str, float]]] = {}
    candidate_size: dict[str, dict[str, dict[str, float]]] = {}
    ranking_metrics: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}

    for split in ("validation", "test"):
        recalls = {v: {t: [] for t in EVENT_TYPES} for v in VARIANTS}
        sizes = {v: [] for v in VARIANTS}
        ranking = {v: {t: [] for t in EVENT_TYPES} for v in VARIANTS}
        for query in split_queries[split]:
            variants = _variant_lists(query, state, base_depth=base_depth, expanded_depth=expanded_depth)
            for variant, lists in variants.items():
                candidates = _candidate_set(lists)
                sizes[variant].append(len(candidates))
                ranking_items = _rrf_fuse(
                    lists,
                    fallback=state["fallback"],
                    k=k,
                    source_depth=expanded_depth,
                    offset=60.0,
                )
                for task in EVENT_TYPES:
                    positives = set(int(x) for x in query["positives"][task])
                    if positives:
                        recalls[variant][task].append(len(positives & candidates) / len(positives))
                    ranking[variant][task].append(_recall_ndcg(ranking_items, query["positives"][task], k))

        candidate_recall[split] = {
            variant: {
                task: float(np.mean(recalls[variant][task])) if recalls[variant][task] else math.nan
                for task in EVENT_TYPES
            }
            for variant in VARIANTS
        }
        candidate_size[split] = {
            variant: {
                "mean": float(np.mean(sizes[variant])) if sizes[variant] else 0.0,
                "median": float(np.median(sizes[variant])) if sizes[variant] else 0.0,
                "p95": float(np.percentile(sizes[variant], 95)) if sizes[variant] else 0.0,
            }
            for variant in VARIANTS
        }
        ranking_metrics[split] = {}
        for variant in VARIANTS:
            ranking_metrics[split][variant] = {}
            for task in EVENT_TYPES:
                ranking_metrics[split][variant][task] = _metric_dict(
                    _aggregate(ranking[variant][task])
                )

    validation_mean = {
        variant: float(np.mean([candidate_recall["validation"][variant][task] for task in EVENT_TYPES]))
        for variant in VARIANTS
    }
    improvement = validation_mean["expanded_union"] - validation_mean["base_union"]
    no_task_degrades = all(
        candidate_recall["validation"]["expanded_union"][task]
        >= candidate_recall["validation"]["base_union"][task]
        for task in EVENT_TYPES
    )
    retain = bool(improvement >= 0.03 and no_task_degrades)

    return RetrievalReport(
        source=str(dataset_dir / "events.csv"),
        manifest=str(manifest_path),
        k=k,
        base_depth=base_depth,
        expanded_depth=expanded_depth,
        prediction_point="after_first_event_of_each_multi_event_split_bounded_session",
        prediction_horizon=str(manifest["prediction_horizon"]),
        validation_evaluated_sessions=len(split_queries["validation"]),
        test_evaluated_sessions=len(split_queries["test"]),
        evaluation_sample_seed=seed,
        variants={
            "base_union": "category/covisitation/recency/visitor-history at base depth",
            "deep_category": "deeper same-category retrieval; other sources at base depth",
            "deep_behavioral": "deeper covisitation/recency/visitor-history; category at base depth",
            "parent_category": "base union plus popular items from sibling categories sharing the context category parent",
            "expanded_union": "deep category + deep behavioral + parent/sibling-category expansion",
        },
        candidate_recall=candidate_recall,
        candidate_size=candidate_size,
        ranking_metrics=ranking_metrics,
        validation_mean_candidate_recall=validation_mean,
        validation_retain_expanded_union=retain,
        retain_rule=(
            "Retain expanded_union only if its mean validation candidate recall across view, "
            "addtocart, and transaction improves over base_union by at least 0.03 absolute and "
            "no task-specific validation candidate recall decreases."
        ),
        notes=[
            "All retrieval statistics are fitted on the frozen training period only.",
            "Category state is frozen at or before the training cutoff; category-tree expansion never uses future item metadata.",
            "Candidate recall is measured before top-K ranking and is the primary v0.6 endpoint.",
            "Candidate-set mean, median, and p95 are reported to expose recall-versus-serving-cost tradeoffs.",
            "RRF Recall@K/NDCG@K are secondary diagnostics; v0.6 does not tune a learned ranker.",
            "Cart and transaction metrics are conditional on queries containing corresponding future targets; they are not conversion probabilities.",
            "Absent events remain unobserved, not observed dislikes or recommendation impressions.",
            "Validation determines retain/reject; test is confirmation only.",
        ],
    )


def write_report(report: RetrievalReport, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "candidate_retrieval.json"
    md_path = output_dir / "candidate_retrieval.md"
    json_path.write_text(json.dumps(asdict(report), indent=2, allow_nan=True) + "\n")
    lines = [
        "# CommerceRecLab v0.6 — Candidate Generation / Retrieval",
        "",
        f"Source: `{report.source}`",
        f"Frozen manifest: `{report.manifest}`",
        "",
        "## Evaluation contract",
        "",
        f"- K: **{report.k}**",
        f"- Base source depth: **{report.base_depth}**",
        f"- Expanded source depth: **{report.expanded_depth}**",
        f"- Validation sessions evaluated: **{report.validation_evaluated_sessions:,}**",
        f"- Test sessions evaluated: **{report.test_evaluated_sessions:,}**",
        f"- Seed: **{report.evaluation_sample_seed}**",
        "",
        "## Retrieval variants",
        "",
    ]
    for variant in VARIANTS:
        lines.append(f"- `{variant}`: {report.variants[variant]}")
    for split in ("validation", "test"):
        lines += ["", f"## {split.title()} candidate recall", "", "| variant | view | cart | transaction | mean candidates | median | p95 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for variant in VARIANTS:
            c = report.candidate_recall[split][variant]
            s = report.candidate_size[split][variant]
            lines.append(
                f"| `{variant}` | {c['view']:.6f} | {c['addtocart']:.6f} | {c['transaction']:.6f} | "
                f"{s['mean']:.1f} | {s['median']:.1f} | {s['p95']:.1f} |"
            )
    lines += ["", "## Secondary top-K ranking diagnostics", ""]
    for split in ("validation", "test"):
        lines += [f"### {split.title()}", "", "| variant | task | queries | Recall@K | NDCG@K |", "|---|---|---:|---:|---:|"]
        for variant in VARIANTS:
            for task in EVENT_TYPES:
                m = report.ranking_metrics[split][variant][task]
                lines.append(f"| `{variant}` | `{task}` | {m['queries']:,} | {m['recall_at_k']:.6f} | {m['ndcg_at_k']:.6f} |")
        lines.append("")
    lines += [
        "## Retain / reject decision",
        "",
        f"Rule: {report.retain_rule}",
        "",
        f"**Retain expanded union: {'YES' if report.validation_retain_expanded_union else 'NO'}**",
        "",
        "## Scientific notes",
        "",
    ]
    lines += [f"- {note}" for note in report.notes]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate CommerceRecLab v0.6 candidate retrieval variants.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v0_6_candidate_retrieval"))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--base-depth", type=int, default=100)
    parser.add_argument("--expanded-depth", type=int, default=300)
    parser.add_argument("--half-life-days", type=float, default=7.0)
    parser.add_argument("--max-sessions-per-split", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=365)
    args = parser.parse_args(argv)
    report = evaluate_retrieval(
        args.dataset_dir,
        args.manifest,
        k=args.k,
        base_depth=args.base_depth,
        expanded_depth=args.expanded_depth,
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
