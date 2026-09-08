from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from commercereclab.evaluation.baselines import (
    EVENT_TYPES,
    _aggregate,
    _deduplicated_events,
    _metric_dict,
    _read_manifest,
    _recall_ndcg,
)
from commercereclab.evaluation.hybrid import COMPONENTS, _rrf_fuse
from commercereclab.evaluation.learned import FEATURE_NAMES, _fit_model
from commercereclab.evaluation.rerank import _candidate_features, _source_lists
from commercereclab.evaluation.retrieval import _fit_retrieval_state
from commercereclab.evaluation.temporal import assign_temporal_split, sessionize_events

STAGE_TASKS = {
    "view_stage": EVENT_TYPES,
    "cart_stage": ("transaction",),
}
VARIANTS = ("category_only", "pooled_task_specific", "stage_conditioned")


@dataclass(frozen=True)
class StageAwareReport:
    source: str
    manifest: str
    k: int
    category_depth: int
    behavioral_depth: int
    parent_depth: int
    internal_fit_fraction: float
    internal_fit_end_ms: int
    internal_ranker_sessions: int
    prediction_points: dict[str, str]
    stage_tasks: dict[str, list[str]]
    training_examples: dict[str, dict[str, int]]
    training_positive_examples: dict[str, dict[str, int]]
    validation_queries: dict[str, int]
    test_queries: dict[str, int]
    candidate_recall: dict[str, dict[str, dict[str, float]]]
    candidate_size: dict[str, dict[str, float]]
    metrics: dict[str, dict[str, dict[str, dict[str, dict[str, float | int]]]]]
    validation_mean_ndcg: dict[str, float]
    best_validation_comparator: str
    validation_retain_stage_conditioning: bool
    retain_rule: str
    notes: list[str]


def _stage_queries(
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
    sessions = sessions.sort_values(["visitorid", "session_index", "timestamp"], kind="stable")
    sizes = sessions.groupby("session_id", sort=False).size()
    eligible_ids = sizes.index[sizes >= 2].to_numpy()
    total_eligible = int(len(eligible_ids))
    if max_sessions is not None and max_sessions > 0 and total_eligible > max_sessions:
        rng = np.random.default_rng(seed)
        chosen = eligible_ids[np.sort(rng.choice(total_eligible, size=max_sessions, replace=False))]
        sessions = sessions[sessions["session_id"].isin(set(chosen))]

    empty = np.array([], dtype=np.int64)
    queries: list[dict] = []
    for _, group in sessions.groupby("session_id", sort=False):
        rows = group.reset_index(drop=True)
        if len(rows) < 2:
            continue
        first = rows.iloc[0]
        remainder = rows.iloc[1:]
        if first["event"] == "view":
            positives = {
                task: remainder.loc[remainder["event"] == task, "itemid"].unique().astype(np.int64)
                for task in EVENT_TYPES
            }
            queries.append({
                "stage": "view_stage",
                "visitorid": int(first["visitorid"]),
                "timestamp": int(first["timestamp"]),
                "last_item": int(first["itemid"]),
                "positives": positives,
            })

        cart_idx = rows.index[rows["event"] == "addtocart"].tolist()
        if cart_idx:
            idx = int(cart_idx[0])
            if idx < len(rows) - 1:
                cart = rows.iloc[idx]
                after_cart = rows.iloc[idx + 1 :]
                trans = after_cart.loc[after_cart["event"] == "transaction", "itemid"].unique()
                queries.append({
                    "stage": "cart_stage",
                    "visitorid": int(cart["visitorid"]),
                    "timestamp": int(cart["timestamp"]),
                    "last_item": int(cart["itemid"]),
                    "positives": {
                        "view": empty,
                        "addtocart": empty,
                        "transaction": trans.astype(np.int64),
                    },
                })
    return queries, total_eligible


def _training_rows(queries, state, *, stage, task, category_depth, behavioral_depth, rrf_offset, max_nonrelevant_per_query):
    xs, ys, pos_count = [], [], 0
    for query in queries:
        if query["stage"] != stage:
            continue
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
        neg_idx = np.flatnonzero(y == 0)[:max_nonrelevant_per_query]
        idx = np.concatenate([pos_idx, neg_idx])
        xs.append(x[idx]); ys.append(y[idx]); pos_count += len(pos_idx)
    if not xs:
        raise ValueError(f"no usable training examples for stage={stage}, task={task}")
    return np.vstack(xs), np.concatenate(ys), int(pos_count)


def _pooled_training_rows(queries, state, *, task, category_depth, behavioral_depth, rrf_offset, max_nonrelevant_per_query):
    xs, ys, pos_count = [], [], 0
    for stage, tasks in STAGE_TASKS.items():
        if task not in tasks:
            continue
        try:
            x, y, p = _training_rows(
                queries, state, stage=stage, task=task,
                category_depth=category_depth, behavioral_depth=behavioral_depth,
                rrf_offset=rrf_offset, max_nonrelevant_per_query=max_nonrelevant_per_query,
            )
        except ValueError:
            continue
        xs.append(x); ys.append(y); pos_count += p
    if not xs:
        raise ValueError(f"no pooled training examples for task={task}")
    return np.vstack(xs), np.concatenate(ys), pos_count


def _model_ranking(query, state, model: Pipeline, *, k, category_depth, behavioral_depth, rrf_offset):
    source_lists = _source_lists(query, state)
    items, x = _candidate_features(source_lists, category_depth=category_depth, behavioral_depth=behavioral_depth, rrf_offset=rrf_offset)
    candidate_set = set(items)
    if items:
        scores = model.predict_proba(x)[:, 1]
        order = np.lexsort((np.asarray(items), -scores))
        primary = [items[i] for i in order]
    else:
        primary = []
    out, seen = [], set()
    for item in [*primary, *state["fallback"]]:
        item = int(item)
        if item in seen: continue
        seen.add(item); out.append(item)
        if len(out) >= k: break
    return out, candidate_set


def _category_ranking(query, state, *, k, category_depth):
    primary = _source_lists(query, state)["category"][:category_depth]
    out, seen = [], set()
    for item in [*primary, *state["fallback"]]:
        item = int(item)
        if item in seen: continue
        seen.add(item); out.append(item)
        if len(out) >= k: break
    return out


def evaluate_stage_aware(
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
) -> StageAwareReport:
    if k <= 0: raise ValueError("k must be positive")
    if category_depth < k or behavioral_depth < k: raise ValueError("depths must be >= k")
    if parent_depth != 0: raise ValueError("v0.9 freezes parent_depth at 0")
    if not 0 < internal_fit_fraction < 1: raise ValueError("internal_fit_fraction must be between 0 and 1")

    dataset_dir = Path(dataset_dir)
    manifest = _read_manifest(manifest_path)
    events = _deduplicated_events(dataset_dir / "events.csv")
    events["split"] = assign_temporal_split(events["timestamp"], train_end_ms=int(manifest["train_end_ms"]), validation_end_ms=int(manifest["validation_end_ms"]))
    train = events.loc[events["split"] == "train"].copy()
    gap = int(manifest["session_gap_minutes"])
    internal_fit_end_ms = int(train["timestamp"].quantile(internal_fit_fraction, interpolation="nearest"))
    source_train = train.loc[train["timestamp"] <= internal_fit_end_ms].copy()
    ranker_train = train.loc[train["timestamp"] > internal_fit_end_ms].copy(); ranker_train["split"] = "ranker_train"
    ranker_queries, _ = _stage_queries(ranker_train, split="ranker_train", gap_minutes=gap, max_sessions=max_internal_ranker_sessions, seed=seed+10)

    split_queries = {}
    for idx, split in enumerate(("validation", "test")):
        split_queries[split], _ = _stage_queries(events, split=split, gap_minutes=gap, max_sessions=max_sessions_per_split, seed=seed+idx)

    all_queries = ranker_queries + split_queries["validation"] + split_queries["test"]
    query_visitors = {int(q["visitorid"]) for q in all_queries}
    context_items = {int(q["last_item"]) for q in all_queries}
    internal_state = _fit_retrieval_state(source_train, dataset_dir, train_end_ms=internal_fit_end_ms, gap_minutes=gap, expanded_depth=max(category_depth, behavioral_depth), half_life_days=half_life_days, query_visitors={int(q["visitorid"]) for q in ranker_queries}, context_items={int(q["last_item"]) for q in ranker_queries})
    full_state = _fit_retrieval_state(train, dataset_dir, train_end_ms=int(manifest["train_end_ms"]), gap_minutes=gap, expanded_depth=max(category_depth, behavioral_depth), half_life_days=half_life_days, query_visitors=query_visitors, context_items=context_items)

    stage_models, pooled_models = {}, {}
    training_examples, training_positive = {}, {}
    for stage, tasks in STAGE_TASKS.items():
        training_examples[stage] = {}; training_positive[stage] = {}
        for task in tasks:
            x, y, p = _training_rows(ranker_queries, internal_state, stage=stage, task=task, category_depth=category_depth, behavioral_depth=behavioral_depth, rrf_offset=rrf_offset, max_nonrelevant_per_query=max_nonrelevant_per_query)
            stage_models[(stage, task)] = _fit_model(x, y, seed=seed)
            training_examples[stage][task] = int(len(y)); training_positive[stage][task] = p
    for task in EVENT_TYPES:
        x, y, _ = _pooled_training_rows(ranker_queries, internal_state, task=task, category_depth=category_depth, behavioral_depth=behavioral_depth, rrf_offset=rrf_offset, max_nonrelevant_per_query=max_nonrelevant_per_query)
        pooled_models[task] = _fit_model(x, y, seed=seed)

    metrics, candidate_recall, candidate_size = {}, {}, {}
    validation_queries, test_queries = {}, {}
    for split in ("validation", "test"):
        stores = {v: {s: {t: [] for t in tasks} for s, tasks in STAGE_TASKS.items()} for v in VARIANTS}
        cand_vals = {s: {t: [] for t in tasks} for s, tasks in STAGE_TASKS.items()}
        sizes = {s: [] for s in STAGE_TASKS}
        counts = {s: 0 for s in STAGE_TASKS}
        for q in split_queries[split]:
            stage = q["stage"]; counts[stage] += 1
            source_lists = _source_lists(q, full_state)
            _, x = _candidate_features(source_lists, category_depth=category_depth, behavioral_depth=behavioral_depth, rrf_offset=rrf_offset)
            items, _ = _candidate_features(source_lists, category_depth=category_depth, behavioral_depth=behavioral_depth, rrf_offset=rrf_offset)
            candidate_set = set(items); sizes[stage].append(len(candidate_set))
            category_rank = _category_ranking(q, full_state, k=k, category_depth=category_depth)
            for task in STAGE_TASKS[stage]:
                positives = q["positives"][task]
                if len(positives): cand_vals[stage][task].append(len(candidate_set & set(map(int, positives))) / len(set(map(int, positives))))
                pooled_rank, _ = _model_ranking(q, full_state, pooled_models[task], k=k, category_depth=category_depth, behavioral_depth=behavioral_depth, rrf_offset=rrf_offset)
                stage_rank, _ = _model_ranking(q, full_state, stage_models[(stage, task)], k=k, category_depth=category_depth, behavioral_depth=behavioral_depth, rrf_offset=rrf_offset)
                stores["category_only"][stage][task].append(_recall_ndcg(category_rank, positives, k))
                stores["pooled_task_specific"][stage][task].append(_recall_ndcg(pooled_rank, positives, k))
                stores["stage_conditioned"][stage][task].append(_recall_ndcg(stage_rank, positives, k))
        metrics[split] = {v: {s: {t: _metric_dict(_aggregate(vals)) for t, vals in td.items()} for s, td in sd.items()} for v, sd in stores.items()}
        candidate_recall[split] = {s: {t: float(np.mean(vals)) if vals else math.nan for t, vals in td.items()} for s, td in cand_vals.items()}
        candidate_size[split] = {s: float(np.mean(vals)) if vals else 0.0 for s, vals in sizes.items()}
        (validation_queries if split == "validation" else test_queries).update(counts)

    cells = [(s, t) for s, tasks in STAGE_TASKS.items() for t in tasks]
    validation_mean_ndcg = {v: float(np.mean([metrics["validation"][v][s][t]["ndcg_at_k"] for s, t in cells])) for v in VARIANTS}
    comparator = max(("category_only", "pooled_task_specific"), key=lambda v: (validation_mean_ndcg[v], v))
    wins = sum(metrics["validation"]["stage_conditioned"][s][t]["ndcg_at_k"] > metrics["validation"][comparator][s][t]["ndcg_at_k"] for s, t in cells)
    retain = bool(validation_mean_ndcg["stage_conditioned"] > validation_mean_ndcg[comparator] and wins >= math.ceil(len(cells)/2))

    return StageAwareReport(
        source=str(dataset_dir / "events.csv"), manifest=str(manifest_path), k=k,
        category_depth=category_depth, behavioral_depth=behavioral_depth, parent_depth=parent_depth,
        internal_fit_fraction=internal_fit_fraction, internal_fit_end_ms=internal_fit_end_ms,
        internal_ranker_sessions=len(ranker_queries),
        prediction_points={"view_stage": "after_first_event_only_when_first_event_is_view", "cart_stage": "immediately_after_first_observed_addtocart_with_later_session_events"},
        stage_tasks={s: list(t) for s, t in STAGE_TASKS.items()},
        training_examples=training_examples, training_positive_examples=training_positive,
        validation_queries=validation_queries, test_queries=test_queries,
        candidate_recall=candidate_recall, candidate_size=candidate_size, metrics=metrics,
        validation_mean_ndcg=validation_mean_ndcg, best_validation_comparator=comparator,
        validation_retain_stage_conditioning=retain,
        retain_rule="Retain stage conditioning only if its mean validation NDCG across declared stage-task cells exceeds the best comparator (category-only or pooled task-specific logistic ranking) and it wins on at least half of those cells.",
        notes=[
            f"The candidate generator uses c{category_depth}_b{behavioral_depth}_p{parent_depth}; the v0.9 default freezes c300_b100_p0 so stage conditioning is the intended intervention.",
            "Stage is defined only from actions observed in the session prefix; no latent funnel state or future event is used as a feature.",
            "view_stage is defined only when the first observed session event is a view; it evaluates later view, cart, and transaction targets in the session remainder.",
            "cart_stage is created only after the first observed addtocart and evaluates later transaction targets; the addtocart itself is not a label.",
            "Behavioral source statistics for ranker training are fitted only on the earlier internal training subwindow.",
            "Validation determines retain/reject; test is confirmation only.",
            "Sampled nonrelevant candidates are offline labels, not observed dislikes or recommendation impressions.",
        ],
    )


def write_report(report: StageAwareReport, output_dir: str | Path):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    jp = output_dir / "stage_aware_ranking.json"; mp = output_dir / "stage_aware_ranking.md"
    jp.write_text(json.dumps(asdict(report), indent=2, allow_nan=True) + "\n")
    lines = ["# CommerceRecLab v0.9 — Funnel-Aware / Stage-Conditional Ranking", "", f"Source: `{report.source}`", f"Frozen manifest: `{report.manifest}`", "", "## Evaluation contract", "", f"- K: **{report.k}**", f"- Frozen retriever: **c{report.category_depth}_b{report.behavioral_depth}_p{report.parent_depth}**", f"- Internal source-fit fraction: **{report.internal_fit_fraction:.2f}**", f"- Internal ranker-training queries: **{report.internal_ranker_sessions:,}**", "", "## Stage definitions", "", "- `view_stage`: after the first event only when that observed first event is a view; evaluates future view/cart/transaction in the remainder of the split-bounded session.", "- `cart_stage`: immediately after the first observed add-to-cart with later session events; evaluates later transaction only.", ""]
    for split in ("validation", "test"):
        lines += [f"## {split.title()} metrics", "", "| variant | stage | task | queries | Recall@K | NDCG@K |", "|---|---|---|---:|---:|---:|"]
        for variant in VARIANTS:
            for stage, tasks in STAGE_TASKS.items():
                for task in tasks:
                    m = report.metrics[split][variant][stage][task]
                    lines.append(f"| `{variant}` | `{stage}` | `{task}` | {m['queries']:,} | {m['recall_at_k']:.6f} | {m['ndcg_at_k']:.6f} |")
        lines.append("")
    lines += ["## Retain / reject decision", "", f"Best validation comparator: `{report.best_validation_comparator}`", "", f"Rule: {report.retain_rule}", "", f"**Retain stage conditioning: {'YES' if report.validation_retain_stage_conditioning else 'NO'}**", "", "## Scientific notes", ""]
    lines += [f"- {n}" for n in report.notes]
    mp.write_text("\n".join(lines) + "\n")
    return jp, mp


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate CommerceRecLab v0.9 stage-conditional ranking.")
    p.add_argument("dataset_dir", type=Path); p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/v0_9_stage_aware_ranking"))
    p.add_argument("--k", type=int, default=20); p.add_argument("--category-depth", type=int, default=300); p.add_argument("--behavioral-depth", type=int, default=100); p.add_argument("--parent-depth", type=int, default=0)
    p.add_argument("--rrf-offset", type=float, default=60.0); p.add_argument("--half-life-days", type=float, default=7.0); p.add_argument("--internal-fit-fraction", type=float, default=0.80)
    p.add_argument("--max-internal-ranker-sessions", type=int, default=20_000); p.add_argument("--max-sessions-per-split", type=int, default=50_000); p.add_argument("--max-nonrelevant-per-query", type=int, default=50); p.add_argument("--seed", type=int, default=365)
    a = p.parse_args(argv)
    r = evaluate_stage_aware(a.dataset_dir, a.manifest, k=a.k, category_depth=a.category_depth, behavioral_depth=a.behavioral_depth, parent_depth=a.parent_depth, rrf_offset=a.rrf_offset, half_life_days=a.half_life_days, internal_fit_fraction=a.internal_fit_fraction, max_internal_ranker_sessions=a.max_internal_ranker_sessions, max_sessions_per_split=a.max_sessions_per_split, max_nonrelevant_per_query=a.max_nonrelevant_per_query, seed=a.seed)
    jp, mp = write_report(r, a.output_dir); print(f"Wrote {jp}"); print(f"Wrote {mp}"); return 0

if __name__ == "__main__": raise SystemExit(main())
