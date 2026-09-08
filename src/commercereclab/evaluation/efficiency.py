from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

from commercereclab.evaluation.baselines import (
    EVENT_TYPES,
    _deduplicated_events,
    _read_manifest,
    _session_queries,
)
from commercereclab.evaluation.retrieval import (
    _candidate_set,
    _fit_retrieval_state,
    _parent_category_items,
)
from commercereclab.evaluation.temporal import assign_temporal_split


@dataclass(frozen=True)
class RetrievalBudget:
    category_depth: int
    behavioral_depth: int
    parent_depth: int

    @property
    def key(self) -> str:
        return f"c{self.category_depth}_b{self.behavioral_depth}_p{self.parent_depth}"


@dataclass(frozen=True)
class EfficiencyReport:
    source: str
    manifest: str
    prediction_point: str
    prediction_horizon: str
    validation_evaluated_sessions: int
    test_evaluated_sessions: int
    evaluation_sample_seed: int
    retention_threshold: float
    budgets: dict[str, dict[str, int]]
    candidate_recall: dict[str, dict[str, dict[str, float]]]
    candidate_size: dict[str, dict[str, dict[str, float]]]
    mean_candidate_recall: dict[str, dict[str, float]]
    validation_pareto_frontier: list[str]
    expanded_reference: str
    selected_efficient_configuration: str
    selection_rule: str
    selected_validation_retention: dict[str, float]
    selected_test_retention: dict[str, float]
    notes: list[str]


def _parse_depths(value: str) -> tuple[int, ...]:
    try:
        depths = tuple(sorted({int(x.strip()) for x in value.split(",") if x.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("depths must be comma-separated integers") from exc
    if not depths or any(depth <= 0 for depth in depths):
        raise argparse.ArgumentTypeError("depths must contain positive integers")
    return depths


def _parse_parent_depths(value: str) -> tuple[int, ...]:
    try:
        depths = tuple(sorted({int(x.strip()) for x in value.split(",") if x.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("parent depths must be comma-separated integers") from exc
    if not depths or any(depth < 0 for depth in depths):
        raise argparse.ArgumentTypeError("parent depths must contain non-negative integers")
    return depths


def _budget_grid(
    category_depths: tuple[int, ...],
    behavioral_depths: tuple[int, ...],
    parent_depths: tuple[int, ...],
) -> list[RetrievalBudget]:
    budgets = [
        RetrievalBudget(c, b, p)
        for c, b, p in product(category_depths, behavioral_depths, parent_depths)
    ]
    return sorted(
        budgets,
        key=lambda x: (
            x.category_depth + x.behavioral_depth + x.parent_depth,
            x.category_depth,
            x.behavioral_depth,
            x.parent_depth,
        ),
    )


def _query_sources(query: dict, state: dict, *, max_parent_depth: int) -> dict[str, list[int]]:
    visitor = int(query["visitorid"])
    context = int(query["last_item"])
    category = state["category_map"].get(context)
    category_items = state["category_rankings"].get(category, []) if category is not None else []
    parent_items = _parent_category_items(
        category,
        parent_map=state["parent_map"],
        children_map=state["children_map"],
        category_rankings=state["category_rankings"],
        depth=max_parent_depth,
    )
    return {
        "category": category_items,
        "covisitation": state["covisitation"].get(context, []),
        "recency": state["recency"],
        "visitor_history": state["visitor_history"].get(visitor, []),
        "parent": parent_items,
    }


def _budget_candidates(sources: dict[str, list[int]], budget: RetrievalBudget) -> set[int]:
    lists = [
        sources["category"][: budget.category_depth],
        sources["covisitation"][: budget.behavioral_depth],
        sources["recency"][: budget.behavioral_depth],
        sources["visitor_history"][: budget.behavioral_depth],
    ]
    if budget.parent_depth:
        lists.append(sources["parent"][: budget.parent_depth])
    return _candidate_set(lists)


def _pareto_frontier(mean_recall: dict[str, float], mean_size: dict[str, float]) -> list[str]:
    frontier: list[str] = []
    keys = list(mean_recall)
    for key in keys:
        dominated = False
        for other in keys:
            if other == key:
                continue
            no_worse = mean_recall[other] >= mean_recall[key] and mean_size[other] <= mean_size[key]
            strictly_better = (
                mean_recall[other] > mean_recall[key]
                or mean_size[other] < mean_size[key]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(key)
    return sorted(frontier, key=lambda key: (mean_size[key], -mean_recall[key], key))


def _retention(
    candidate_recall: dict[str, dict[str, dict[str, float]]],
    *,
    split: str,
    candidate: str,
    reference: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for task in EVENT_TYPES:
        denom = candidate_recall[split][reference][task]
        out[task] = float(candidate_recall[split][candidate][task] / denom) if denom > 0 else 1.0
    return out


def evaluate_efficiency(
    dataset_dir: str | Path,
    manifest_path: str | Path,
    *,
    category_depths: tuple[int, ...] = (100, 200, 300),
    behavioral_depths: tuple[int, ...] = (100, 300),
    parent_depths: tuple[int, ...] = (0, 100, 300),
    retention_threshold: float = 0.95,
    half_life_days: float = 7.0,
    max_sessions_per_split: int | None = 50_000,
    seed: int = 365,
) -> EfficiencyReport:
    if not 0 < retention_threshold <= 1:
        raise ValueError("retention_threshold must be in (0, 1]")
    if not category_depths or any(depth <= 0 for depth in category_depths):
        raise ValueError("category_depths must be positive")
    if not behavioral_depths or any(depth <= 0 for depth in behavioral_depths):
        raise ValueError("behavioral_depths must be positive")
    if not parent_depths or any(depth < 0 for depth in parent_depths):
        raise ValueError("parent_depths must be non-negative")

    budgets = _budget_grid(category_depths, behavioral_depths, parent_depths)
    max_category = max(category_depths)
    max_behavioral = max(behavioral_depths)
    max_parent = max(parent_depths)
    expanded_reference_budget = RetrievalBudget(max_category, max_behavioral, max_parent)
    expanded_reference = expanded_reference_budget.key

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
        expanded_depth=max(max_category, max_behavioral, max_parent),
        half_life_days=half_life_days,
        query_visitors=visitors,
        context_items=contexts,
    )

    candidate_recall: dict[str, dict[str, dict[str, float]]] = {}
    candidate_size: dict[str, dict[str, dict[str, float]]] = {}
    mean_candidate_recall: dict[str, dict[str, float]] = {}

    for split in ("validation", "test"):
        recalls = {budget.key: {task: [] for task in EVENT_TYPES} for budget in budgets}
        sizes = {budget.key: [] for budget in budgets}
        for query in split_queries[split]:
            sources = _query_sources(query, state, max_parent_depth=max_parent)
            positive_sets = {
                task: set(int(x) for x in query["positives"][task]) for task in EVENT_TYPES
            }
            for budget in budgets:
                candidates = _budget_candidates(sources, budget)
                sizes[budget.key].append(len(candidates))
                for task in EVENT_TYPES:
                    positives = positive_sets[task]
                    if positives:
                        recalls[budget.key][task].append(
                            len(positives & candidates) / len(positives)
                        )

        candidate_recall[split] = {
            budget.key: {
                task: (
                    float(np.mean(recalls[budget.key][task]))
                    if recalls[budget.key][task]
                    else float("nan")
                )
                for task in EVENT_TYPES
            }
            for budget in budgets
        }
        candidate_size[split] = {
            budget.key: {
                "mean": float(np.mean(sizes[budget.key])) if sizes[budget.key] else 0.0,
                "median": float(np.median(sizes[budget.key])) if sizes[budget.key] else 0.0,
                "p95": float(np.percentile(sizes[budget.key], 95)) if sizes[budget.key] else 0.0,
            }
            for budget in budgets
        }
        mean_candidate_recall[split] = {
            budget.key: float(
                np.mean([candidate_recall[split][budget.key][task] for task in EVENT_TYPES])
            )
            for budget in budgets
        }

    validation_sizes = {key: value["mean"] for key, value in candidate_size["validation"].items()}
    frontier = _pareto_frontier(mean_candidate_recall["validation"], validation_sizes)
    eligible = [
        budget.key
        for budget in budgets
        if all(
            candidate_recall["validation"][budget.key][task]
            >= retention_threshold * candidate_recall["validation"][expanded_reference][task]
            for task in EVENT_TYPES
        )
    ]
    selected = min(
        eligible or [expanded_reference],
        key=lambda key: (
            validation_sizes[key],
            -mean_candidate_recall["validation"][key],
            key,
        ),
    )

    return EfficiencyReport(
        source=str(dataset_dir / "events.csv"),
        manifest=str(manifest_path),
        prediction_point="after_first_event_of_each_multi_event_split_bounded_session",
        prediction_horizon=str(manifest["prediction_horizon"]),
        validation_evaluated_sessions=len(split_queries["validation"]),
        test_evaluated_sessions=len(split_queries["test"]),
        evaluation_sample_seed=seed,
        retention_threshold=retention_threshold,
        budgets={
            budget.key: {
                "category_depth": budget.category_depth,
                "behavioral_depth": budget.behavioral_depth,
                "parent_depth": budget.parent_depth,
            }
            for budget in budgets
        },
        candidate_recall=candidate_recall,
        candidate_size=candidate_size,
        mean_candidate_recall=mean_candidate_recall,
        validation_pareto_frontier=frontier,
        expanded_reference=expanded_reference,
        selected_efficient_configuration=selected,
        selection_rule=(
            "Using validation only, require every task-specific candidate recall to retain at "
            f"least {retention_threshold:.0%} of the expanded-reference recall, then choose the "
            "eligible configuration with the smallest mean candidate set (breaking ties by "
            "higher mean recall)."
        ),
        selected_validation_retention=_retention(
            candidate_recall,
            split="validation",
            candidate=selected,
            reference=expanded_reference,
        ),
        selected_test_retention=_retention(
            candidate_recall,
            split="test",
            candidate=selected,
            reference=expanded_reference,
        ),
        notes=[
            "All retrieval statistics are fitted on the frozen training period only.",
            "Category state and hierarchy expansion use no item metadata after the training "
            "cutoff.",
            "Candidate recall is measured before top-K ranking; v0.7 does not fit or tune a "
            "ranker.",
            "The expanded reference is the maximum category, behavioral, and parent depth in "
            "the declared sweep, not an oracle upper bound.",
            "Pareto dominance uses mean candidate recall versus mean candidate-set size on "
            "validation; task-specific recalls remain reported separately.",
            "The efficient configuration is selected on validation only; test retention is "
            "confirmation, not a tuning signal.",
            "Cart and transaction candidate recall are conditional on queries containing "
            "corresponding future targets, not conversion probabilities.",
            "Absent events remain unobserved rather than observed dislikes or recommendation "
            "impressions.",
        ],
    )


def write_report(report: EfficiencyReport, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval_efficiency.json"
    md_path = output_dir / "retrieval_efficiency.md"
    json_path.write_text(json.dumps(asdict(report), indent=2, allow_nan=True) + "\n")

    lines = [
        "# CommerceRecLab v0.7 — Retrieval Efficiency / Pareto Frontier",
        "",
        f"Source: `{report.source}`",
        f"Frozen manifest: `{report.manifest}`",
        "",
        "## Evaluation contract",
        "",
        f"- Validation sessions evaluated: **{report.validation_evaluated_sessions:,}**",
        f"- Test sessions evaluated: **{report.test_evaluated_sessions:,}**",
        f"- Seed: **{report.evaluation_sample_seed}**",
        (
            f"- Per-task retention threshold: **{report.retention_threshold:.0%}** of "
            "expanded-reference candidate recall"
        ),
        f"- Expanded reference: `{report.expanded_reference}`",
        "",
        "## Validation efficiency sweep",
        "",
        (
            "| configuration | cat depth | behavioral depth | parent depth | view recall | "
            "cart recall | transaction recall | mean recall | mean candidates | p95 | Pareto |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    validation_order = sorted(
        report.budgets,
        key=lambda key: (
            report.candidate_size["validation"][key]["mean"],
            -report.mean_candidate_recall["validation"][key],
            key,
        ),
    )
    frontier = set(report.validation_pareto_frontier)
    for key in validation_order:
        budget = report.budgets[key]
        recall = report.candidate_recall["validation"][key]
        size = report.candidate_size["validation"][key]
        lines.append(
            f"| `{key}` | {budget['category_depth']} | {budget['behavioral_depth']} | "
            f"{budget['parent_depth']} | "
            f"{recall['view']:.6f} | {recall['addtocart']:.6f} | {recall['transaction']:.6f} | "
            f"{report.mean_candidate_recall['validation'][key]:.6f} | "
            f"{size['mean']:.1f} | {size['p95']:.1f} | "
            f"{'YES' if key in frontier else ''} |"
        )

    lines += [
        "",
        "## Selected efficient configuration",
        "",
        f"Rule: {report.selection_rule}",
        "",
        f"**Selected: `{report.selected_efficient_configuration}`**",
        "",
        "| split | view retention | cart retention | transaction retention | mean candidates |",
        "|---|---:|---:|---:|---:|",
    ]
    selected = report.selected_efficient_configuration
    for split, retention in (
        ("validation", report.selected_validation_retention),
        ("test", report.selected_test_retention),
    ):
        lines.append(
            f"| {split} | {retention['view']:.3f} | {retention['addtocart']:.3f} | "
            f"{retention['transaction']:.3f} | "
            f"{report.candidate_size[split][selected]['mean']:.1f} |"
        )

    lines += [
        "",
        "## Validation Pareto frontier",
        "",
        ", ".join(f"`{key}`" for key in report.validation_pareto_frontier),
        "",
        "## Scientific notes",
        "",
    ]
    lines += [f"- {note}" for note in report.notes]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate CommerceRecLab v0.7 retrieval efficiency frontier."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v0_7_retrieval_efficiency"),
    )
    parser.add_argument("--category-depths", type=_parse_depths, default=(100, 200, 300))
    parser.add_argument("--behavioral-depths", type=_parse_depths, default=(100, 300))
    parser.add_argument("--parent-depths", type=_parse_parent_depths, default=(0, 100, 300))
    parser.add_argument("--retention-threshold", type=float, default=0.95)
    parser.add_argument("--half-life-days", type=float, default=7.0)
    parser.add_argument("--max-sessions-per-split", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=365)
    args = parser.parse_args(argv)
    report = evaluate_efficiency(
        args.dataset_dir,
        args.manifest,
        category_depths=args.category_depths,
        behavioral_depths=args.behavioral_depths,
        parent_depths=args.parent_depths,
        retention_threshold=args.retention_threshold,
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
