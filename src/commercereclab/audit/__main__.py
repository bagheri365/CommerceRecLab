"""Command-line entry point for ``python -m commercereclab.audit``."""

from __future__ import annotations

import argparse
from pathlib import Path

from commercereclab.audit.dataset import audit_retailrocket_dir
from commercereclab.audit.report import write_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the canonical four-file Retailrocket e-commerce dataset."
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing events.csv, both item-property files, and category_tree.csv.",
    )
    parser.add_argument(
        "--property-chunksize",
        type=int,
        default=500_000,
        help="Rows per chunk when streaming the large item-property tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v0_0_dataset_audit"),
        help="Directory for audit.json and audit.md.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audit = audit_retailrocket_dir(
        args.source_dir,
        property_chunksize=args.property_chunksize,
    )
    json_path, markdown_path = write_reports(audit, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
