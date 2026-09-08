"""v1.0 retrospective system selection from frozen milestone artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MilestoneRow:
    milestone: str
    intervention: str
    validation_result: str
    decision: str
    test_confirmation: str


@dataclass(frozen=True)
class FinalSelection:
    retrieval_configuration: str
    ranking_policy: str
    optional_follow_up: str
    benchmark: list[MilestoneRow]
    evidence_summary: dict[str, Any]
    notes: list[str]


ARTIFACTS = {
    "v0.3": ("v0_3_behavioral_baselines", "behavioral_baselines.json"),
    "v0.4": ("v0_4_hybrid_ablation", "hybrid_ablation.json"),
    "v0.5": ("v0_5_learned_ranker", "learned_ranker.json"),
    "v0.6": ("v0_6_candidate_retrieval", "candidate_retrieval.json"),
    "v0.7": ("v0_7_retrieval_efficiency", "retrieval_efficiency.json"),
    "v0.8": ("v0_8_efficient_reranker", "efficient_reranker.json"),
    "v0.9": ("v0_9_stage_aware_ranking", "stage_aware_ranking.json"),
}


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required frozen artifact not found: {path}")
    return json.loads(path.read_text())


def _load_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for version, (directory, filename) in ARTIFACTS.items():
        loaded[version] = _load(root / directory / filename)
    return loaded


def _metric(payload: dict[str, Any], split: str, variant: str, task: str, key: str) -> float:
    return float(payload["metrics"][split][variant][task][key])


def build_final_selection(artifacts_root: Path) -> FinalSelection:
    a = _load_artifacts(artifacts_root)

    selected = a["v0.7"].get("selected_efficient_configuration")
    if selected != "c300_b100_p0":
        raise ValueError(f"unexpected v0.7 selected configuration: {selected!r}")
    if not a["v0.6"].get("validation_retain_expanded_union"):
        raise ValueError("v0.6 expanded retrieval was expected to be retained")
    for version, key in [
        ("v0.4", "validation_retain_full_hybrid"),
        ("v0.5", "validation_retain_learned_ranker"),
        ("v0.8", "validation_retain_learned_ranker"),
        ("v0.9", "validation_retain_stage_conditioning"),
    ]:
        if a[version].get(key):
            raise ValueError(f"{version} frozen decision no longer matches the retrospective")

    v03_test = a["v0.3"]["metrics"]["test"]
    cat_test = v03_test["category_conditioned_popularity"]
    v06 = a["v0.6"]
    v07 = a["v0.7"]
    v08 = a["v0.8"]
    v09 = a["v0.9"]

    rows = [
        MilestoneRow(
            "v0.3",
            "behavioral baselines",
            "category-conditioned popularity was the strongest simple baseline across view/cart/transaction",
            "reference baseline",
            (
                "test NDCG@20 category-only: "
                f"view {cat_test['view']['ndcg_at_k']:.4f}, "
                f"cart {cat_test['addtocart']['ndcg_at_k']:.4f}, "
                f"transaction {cat_test['transaction']['ndcg_at_k']:.4f}"
            ),
        ),
        MilestoneRow(
            "v0.4",
            "equal-weight behavioral fusion",
            f"mean validation NDCG {a['v0.4']['validation_mean_ndcg']['hybrid_full']:.4f} vs category-only {a['v0.4']['validation_mean_ndcg']['category_only']:.4f}",
            "reject",
            "test confirmed category-only superiority for cart and transaction",
        ),
        MilestoneRow(
            "v0.5",
            "task-specific logistic rankers",
            f"mean validation NDCG {a['v0.5']['validation_mean_ndcg']['learned_task_specific']:.4f} vs category-only {a['v0.5']['validation_mean_ndcg']['category_only']:.4f}",
            "reject",
            "view improved, but cart and transaction NDCG degraded",
        ),
        MilestoneRow(
            "v0.6",
            "expanded candidate retrieval",
            f"mean validation candidate recall {a['v0.6']['validation_mean_candidate_recall']['base_union']:.4f} → {a['v0.6']['validation_mean_candidate_recall']['expanded_union']:.4f}",
            "retain",
            "test candidate recall improved for all three tasks",
        ),
        MilestoneRow(
            "v0.7",
            "retrieval efficiency sweep",
            "selected c300_b100_p0 while retaining at least 95% of expanded-reference recall per task",
            "retain / select",
            (
                "test retention: "
                f"view {v07['selected_test_retention']['view']:.3f}, "
                f"cart {v07['selected_test_retention']['addtocart']:.3f}, "
                f"transaction {v07['selected_test_retention']['transaction']:.3f}"
            ),
        ),
        MilestoneRow(
            "v0.8",
            "learned reranking on efficient retrieval",
            f"mean validation NDCG {v08['validation_mean_ndcg']['learned_task_specific']:.4f} vs category-only {v08['validation_mean_ndcg']['category_only']:.4f}",
            "reject",
            "stronger retrieval did not rescue cart/transaction learned ranking",
        ),
        MilestoneRow(
            "v0.9",
            "stage-conditional ranking",
            f"mean validation NDCG {v09['validation_mean_ndcg']['stage_conditioned']:.4f} vs category-only {v09['validation_mean_ndcg']['category_only']:.4f}",
            "reject",
            "cart-stage transaction improved over pooled learned ranking but remained below category-only",
        ),
    ]

    evidence = {
        "selected_retriever": selected,
        "selected_retriever_validation_candidate_recall": {
            task: v07["candidate_recall"]["validation"][selected][task]
            for task in ("view", "addtocart", "transaction")
        },
        "selected_retriever_test_candidate_recall": {
            task: v07["candidate_recall"]["test"][selected][task]
            for task in ("view", "addtocart", "transaction")
        },
        "selected_retriever_test_mean_candidates": v07["candidate_size"]["test"][selected]["mean"],
        "expanded_reference_test_mean_candidates": v07["candidate_size"]["test"]["c300_b300_p300"]["mean"],
        "category_only_test_ndcg_at_20": {
            "view": cat_test["view"]["ndcg_at_k"],
            "addtocart": cat_test["addtocart"]["ndcg_at_k"],
            "transaction": cat_test["transaction"]["ndcg_at_k"],
        },
        "exploratory_view_reranker_test_ndcg_at_20": _metric(
            v08, "test", "learned_task_specific", "view", "ndcg_at_k"
        ),
    }

    notes = [
        "The conservative v1.0 system recommendation is c300_b100_p0 retrieval plus category-first ranking.",
        "The learned view reranker is recorded as exploratory follow-up evidence, not a retained system component, because the v0.8 system-level retain rule failed.",
        "Cart and transaction metrics remain conditional ranking metrics over queries with corresponding future targets; they are not conversion probabilities.",
        "Absent visitor-item events remain unobserved rather than observed dislikes or recommendation impressions.",
        "All retrospective statements inherit the frozen temporal and point-in-time leakage rules of v0.2.",
    ]
    return FinalSelection(
        retrieval_configuration="c300_b100_p0 (category depth 300, behavioral depth 100, parent depth 0)",
        ranking_policy="category-first deterministic ranking for the conservative final system",
        optional_follow_up=(
            "revalidate a view-only learned reranker in a new holdout or future dataset before deployment; "
            "v0.8 showed a consistent view NDCG gain but failed the predeclared system-level retain rule"
        ),
        benchmark=rows,
        evidence_summary=evidence,
        notes=notes,
    )


def _render_markdown(result: FinalSelection) -> str:
    e = result.evidence_summary
    lines = [
        "# CommerceRecLab v1.0 — Final System Selection + Retrospective",
        "",
        "## Final recommendation",
        "",
        f"- Retrieval: **{result.retrieval_configuration}**",
        f"- Ranking: **{result.ranking_policy}**",
        f"- Follow-up: {result.optional_follow_up}",
        "",
        "## Selected retrieval operating point",
        "",
        "| split | view candidate recall | cart candidate recall | transaction candidate recall | mean candidates |",
        "|---|---:|---:|---:|---:|",
        (
            "| validation | "
            f"{e['selected_retriever_validation_candidate_recall']['view']:.6f} | "
            f"{e['selected_retriever_validation_candidate_recall']['addtocart']:.6f} | "
            f"{e['selected_retriever_validation_candidate_recall']['transaction']:.6f} | — |"
        ),
        (
            "| test | "
            f"{e['selected_retriever_test_candidate_recall']['view']:.6f} | "
            f"{e['selected_retriever_test_candidate_recall']['addtocart']:.6f} | "
            f"{e['selected_retriever_test_candidate_recall']['transaction']:.6f} | "
            f"{e['selected_retriever_test_mean_candidates']:.1f} |"
        ),
        "",
        (
            "The expanded reference used "
            f"{e['expanded_reference_test_mean_candidates']:.1f} mean test candidates; the selected "
            "retriever preserves most of its recall at materially lower candidate-set cost."
        ),
        "",
        "## Final ranking reference",
        "",
        "| task | category-first test NDCG@20 |",
        "|---|---:|",
        f"| view | {e['category_only_test_ndcg_at_20']['view']:.6f} |",
        f"| addtocart | {e['category_only_test_ndcg_at_20']['addtocart']:.6f} |",
        f"| transaction | {e['category_only_test_ndcg_at_20']['transaction']:.6f} |",
        "",
        (
            "Exploratory note: the v0.8 view-only learned reranker reached test NDCG@20 "
            f"{e['exploratory_view_reranker_test_ndcg_at_20']:.6f}, but it is not retained in the "
            "conservative final system because the predeclared v0.8 system-level rule failed."
        ),
        "",
        "## Retrospective benchmark",
        "",
        "| milestone | intervention | validation result | decision | test confirmation |",
        "|---|---|---|---|---|",
    ]
    for row in result.benchmark:
        lines.append(
            f"| {row.milestone} | {row.intervention} | {row.validation_result} | "
            f"**{row.decision}** | {row.test_confirmation} |"
        )
    lines.extend(["", "## Scientific conclusion", ""])
    lines.extend(f"- {note}" for note in result.notes)
    lines.extend(
        [
            "",
            "The central result is that **retrieval and category context earned their complexity; more elaborate reranking did not** under the frozen offline protocol. This is a retained/rejected intervention record, not a claim about production causal lift.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(result: FinalSelection, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "final_system_selection.json"
    md_path = output_dir / "final_system_selection.md"
    json_path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    md_path.write_text(_render_markdown(result))
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v1_0_final_system_selection"))
    args = parser.parse_args(argv)
    result = build_final_selection(args.artifacts_root)
    json_path, md_path = write_report(result, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
