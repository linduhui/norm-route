"""Audit Stage 2 outputs before Stage 3 routing-matrix construction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normroute.routing.join_predictions import audit_stage2_output_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage 2 output artifacts for Stage 3.")
    parser.add_argument(
        "output_root",
        nargs="?",
        default="outputs/stage2",
        help="Stage 2 output root or one run output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_stage2_output_tree(args.output_root)
    if audit.errors:
        for error in audit.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Audited {len(audit.run_dirs)} Stage 2 run directories under {args.output_root}")


if __name__ == "__main__":
    main()

