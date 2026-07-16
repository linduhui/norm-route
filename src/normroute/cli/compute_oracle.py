"""Compute Stage 3 evaluator-only Oracle and complementarity analyses."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ..routing.complementarity import (
    compute_complementarity_summary,
    write_complementarity_summary,
)
from ..routing.oracle import (
    OracleError,
    compute_oracle_outputs,
    write_oracle_outputs,
)
from ..routing.join_predictions import RoutingMatrixError
from ..routing.quality_metrics import QualityMetricError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Stage 3 evaluator-only Oracle upper bounds."
    )
    parser.add_argument(
        "--routing-matrix-long",
        "--routing-matrix",
        dest="routing_matrix_long",
        default="outputs/stage3/evaluator_only/routing_matrix_long.csv",
    )
    parser.add_argument(
        "--expert-quality-by-run",
        "--expert-quality",
        dest="expert_quality_by_run",
        default="outputs/stage3/quality/expert_quality_by_run.csv",
    )
    parser.add_argument("--output-dir", default="outputs/stage3/oracle")
    parser.add_argument("--report-dir", default="reports/stage3")
    parser.add_argument(
        "--selection-metric",
        "--metric",
        dest="selection_metric",
        choices=("image_auroc", "image_ap"),
        default="image_auroc",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        oracle_result = compute_oracle_outputs(
            routing_matrix_long=args.routing_matrix_long,
            expert_quality_by_run=args.expert_quality_by_run,
            selection_metric=args.selection_metric,
        )
        oracle_summary_path, oracle_counts_path = write_oracle_outputs(
            oracle_result,
            output_dir=args.output_dir,
        )
        complementarity_rows = compute_complementarity_summary(
            routing_matrix_long=args.routing_matrix_long,
            expert_quality_by_run=args.expert_quality_by_run,
            selection_metric=args.selection_metric,
        )
        complementarity_path = write_complementarity_summary(
            complementarity_rows,
            output_dir=args.output_dir,
        )
        report_paths = _copy_reports(
            (oracle_summary_path, oracle_counts_path, complementarity_path),
            args.report_dir,
        )
    except (OracleError, QualityMetricError, RoutingMatrixError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Wrote {oracle_summary_path}")
    print(f"Wrote {oracle_counts_path}")
    print(f"Wrote {complementarity_path}")
    for report_path in report_paths:
        print(f"Wrote {report_path}")


def _copy_reports(paths: tuple[Path, ...], report_dir: str | Path | None) -> tuple[Path, ...]:
    if report_dir is None:
        return ()
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in paths:
        destination = report_path / path.name
        shutil.copyfile(path, destination)
        copied.append(destination)
    return tuple(copied)


if __name__ == "__main__":
    main()
