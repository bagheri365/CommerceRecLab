"""Empirical observation-semantics audit for Retailrocket events."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = {"timestamp", "visitorid", "event", "itemid", "transactionid"}
EVENT_ORDER = ("view", "addtocart", "transaction")


@dataclass(frozen=True)
class ObservationSemanticsAudit:
    source: str
    raw_rows: int
    exact_duplicate_rows: int
    deduplicated_rows: int
    unique_visitor_item_pairs: int
    pair_pattern_counts: dict[str, int]
    cart_pairs: int
    cart_pairs_with_prior_view: int
    transaction_pairs: int
    transaction_pairs_with_prior_view: int
    transaction_pairs_with_prior_cart: int
    transaction_pairs_with_prior_view_and_cart: int
    transaction_pairs_with_ordered_view_cart: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _pattern_label(events: set[str]) -> str:
    return "+".join(event for event in EVENT_ORDER if event in events)


def audit_observation_semantics(events_csv: str | Path) -> ObservationSemanticsAudit:
    """Audit which visitor-item funnel paths are actually observed.

    Exact duplicate rows are removed for path analysis only. The source file is never modified.
    Pair-path statements are descriptive: absence of an earlier event is not evidence that the
    earlier action did not occur outside the recorded log.
    """

    source = Path(events_csv)
    frame = pd.read_csv(source)
    missing = EXPECTED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"events.csv missing required columns: {', '.join(sorted(missing))}")

    raw_rows = len(frame)
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise")

    first = (
        frame.groupby(["visitorid", "itemid", "event"], observed=True)["timestamp"]
        .min()
        .unstack("event")
    )

    for event in EVENT_ORDER:
        if event not in first.columns:
            first[event] = pd.NA

    present = first[list(EVENT_ORDER)].notna()
    codes = (
        present["view"].astype("int8")
        + 2 * present["addtocart"].astype("int8")
        + 4 * present["transaction"].astype("int8")
    )
    code_counts = codes.value_counts().to_dict()
    labels = {
        1: "view",
        2: "addtocart",
        3: "view+addtocart",
        4: "transaction",
        5: "view+transaction",
        6: "addtocart+transaction",
        7: "view+addtocart+transaction",
    }
    pattern_counts = {labels[int(code)]: int(count) for code, count in code_counts.items()}

    has_view = present["view"]
    has_cart = present["addtocart"]
    has_tx = present["transaction"]

    cart_prior_view = has_cart & has_view & (first["view"] <= first["addtocart"])
    tx_prior_view = has_tx & has_view & (first["view"] <= first["transaction"])
    tx_prior_cart = has_tx & has_cart & (first["addtocart"] <= first["transaction"])
    tx_prior_view_and_cart = tx_prior_view & tx_prior_cart
    tx_ordered_view_cart = (
        has_tx
        & has_cart
        & has_view
        & (first["view"] <= first["addtocart"])
        & (first["addtocart"] <= first["transaction"])
    )

    return ObservationSemanticsAudit(
        source=str(source),
        raw_rows=raw_rows,
        exact_duplicate_rows=duplicate_rows,
        deduplicated_rows=len(frame),
        unique_visitor_item_pairs=len(first),
        pair_pattern_counts=dict(sorted(pattern_counts.items())),
        cart_pairs=int(has_cart.sum()),
        cart_pairs_with_prior_view=int(cart_prior_view.sum()),
        transaction_pairs=int(has_tx.sum()),
        transaction_pairs_with_prior_view=int(tx_prior_view.sum()),
        transaction_pairs_with_prior_cart=int(tx_prior_cart.sum()),
        transaction_pairs_with_prior_view_and_cart=int(tx_prior_view_and_cart.sum()),
        transaction_pairs_with_ordered_view_cart=int(tx_ordered_view_cart.sum()),
    )


def render_markdown(audit: ObservationSemanticsAudit) -> str:
    def pct(n: int, d: int) -> str:
        return "n/a" if d == 0 else f"{100 * n / d:.4f}%"

    lines = [
        "# CommerceRecLab v0.1 — Observation Semantics",
        "",
        f"Source: `{audit.source}`",
        "",
        "## Duplicate handling",
        "",
        f"- Raw rows: **{audit.raw_rows:,}**",
        f"- Exact duplicate rows beyond first occurrence: **{audit.exact_duplicate_rows:,}**",
        f"- Rows used for path analysis: **{audit.deduplicated_rows:,}**",
        "",
        "Exact duplicates are removed for this descriptive path audit only. The raw file is not modified.",
        "",
        "## Visitor-item event patterns",
        "",
        f"Unique visitor-item pairs: **{audit.unique_visitor_item_pairs:,}**",
        "",
        "| observed event set | pairs |",
        "|---|---:|",
    ]
    for pattern, count in audit.pair_pattern_counts.items():
        lines.append(f"| `{pattern}` | {count:,} |")

    lines.extend(
        [
            "",
            "## Recorded funnel-path coverage",
            "",
            f"- Cart pairs: **{audit.cart_pairs:,}**",
            f"- Cart pairs with a recorded prior/equal-time view: **{audit.cart_pairs_with_prior_view:,} ({pct(audit.cart_pairs_with_prior_view, audit.cart_pairs)})**",
            f"- Transaction pairs: **{audit.transaction_pairs:,}**",
            f"- Transaction pairs with a recorded prior/equal-time view: **{audit.transaction_pairs_with_prior_view:,} ({pct(audit.transaction_pairs_with_prior_view, audit.transaction_pairs)})**",
            f"- Transaction pairs with a recorded prior/equal-time cart: **{audit.transaction_pairs_with_prior_cart:,} ({pct(audit.transaction_pairs_with_prior_cart, audit.transaction_pairs)})**",
            f"- Transaction pairs with both recorded prior view and cart: **{audit.transaction_pairs_with_prior_view_and_cart:,} ({pct(audit.transaction_pairs_with_prior_view_and_cart, audit.transaction_pairs)})**",
            f"- Transaction pairs with recorded `view <= addtocart <= transaction`: **{audit.transaction_pairs_with_ordered_view_cart:,} ({pct(audit.transaction_pairs_with_ordered_view_cart, audit.transaction_pairs)})**",
            "",
            "## Scientific interpretation",
            "",
            "These frequencies describe what appears in the released log. Missing earlier funnel events do not prove that the visitor skipped those actions in reality; the release is not assumed to be a complete exposure or instrumentation log.",
            "",
            "The empirical tasks remain separate: future interaction ranking, cart-intent ranking, transaction ranking, and later session/next-item ranking. Results for one task must not be silently relabeled as results for another.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(audit: ObservationSemanticsAudit, output_dir: str | Path) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "observation_semantics.json"
    md_path = out / "observation_semantics.md"
    json_path.write_text(json.dumps(audit.to_dict(), indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(audit), encoding="utf-8")
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Retailrocket observation/funnel semantics.")
    parser.add_argument("events_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v0_1_observation_semantics"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_observation_semantics(args.events_csv)
    json_path, md_path = write_outputs(audit, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
