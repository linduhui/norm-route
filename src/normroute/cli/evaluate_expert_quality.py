"""Evaluate Stage 3 expert quality from evaluator-only routing matrices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ..routing.quality_metrics import (
    QualityMetricError,
    evaluate_expert_quality,
    write_expert_quality_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Stage 3 evaluator-only expert quality metrics."
    )
    parser.add_argument(
        "--routing-matrix-long",
        default="outputs/stage3/evaluator_only/routing_matrix_long.csv",
    )
    parser.add_argument("--output-dir", default="outputs/stage3/evaluator_only")
    parser.add_argument("--reports-dir", default="reports/stage3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = evaluate_expert_quality(args.routing_matrix_long)
        by_run_path, summary_path, warnings_path = write_expert_quality_outputs(
            result,
            output_dir=args.output_dir,
            reports_dir=args.reports_dir,
        )
    except QualityMetricError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Wrote {by_run_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {warnings_path}")
    if result.warnings:
        print(f"Wrote {len(result.warnings)} warnings to {warnings_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
