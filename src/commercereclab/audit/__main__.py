"""Command-line entry point for ``python -m commercereclab.audit``."""

from __future__ import annotations

import argparse
from pathlib import Path

from commercereclab.audit.dataset import audit_csv
from commercereclab.audit.report import write_reports


def _delimiter(value: str) -> str:
    aliases = {"tab": "\t", "\\t": "\t", "comma": ",", "pipe": "|"}
    return aliases.get(value, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit an explicit-rating dataset without treating missing pairs as negatives."
    )
    parser.add_argument("path", type=Path, help="Path to a CSV/TSV-style rating table.")
    parser.add_argument("--user-col", required=True, help="Column containing the rating user ID.")
    parser.add_argument("--profile-col", required=True, help="Column containing the rated profile ID.")
    parser.add_argument("--rating-col", required=True, help="Column containing the explicit rating.")
    parser.add_argument(
        "--delimiter",
        default=",",
        type=_delimiter,
        help="Field delimiter. Supports ',', 'tab', '\\t', 'comma', or 'pipe'.",
    )
    parser.add_argument("--expected-rating-min", type=float)
    parser.add_argument("--expected-rating-max", type=float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v0_0_dataset_audit"),
        help="Directory for audit.json and audit.md.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audit = audit_csv(
        args.path,
        user_col=args.user_col,
        profile_col=args.profile_col,
        rating_col=args.rating_col,
        delimiter=args.delimiter,
        expected_rating_min=args.expected_rating_min,
        expected_rating_max=args.expected_rating_max,
    )
    json_path, markdown_path = write_reports(audit, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
