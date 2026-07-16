"""CLI for full evaluator-only Stage 4 summaries."""

from __future__ import annotations

import argparse
import sys

from src.normroute.evaluation.stage4_summary import (
    Stage4SummaryError,
    discover_selected_predictions,
    summarize_stage4,
    write_stage4_summary,
    write_stage4_summary_failure,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize all Stage 4 test policy/fold outputs and compute fold-local "
            "best-single and run-level Oracle evaluator references."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=False)
    inputs.add_argument(
        "--runs-root",
        nargs="+",
        default=None,
        help="One or more trees containing selected_predictions.csv files.",
    )
    inputs.add_argument(
        "--selected-predictions",
        nargs="+",
        help="Explicit selected_predictions.csv files.",
    )
    parser.add_argument("--evaluator-csv", default="data/manifests/mvtec_evaluator.csv")
    parser.add_argument(
        "--expert-predictions",
        help=(
            "Optional evaluator-only all-expert CSV. If omitted, expert predictions "
            "are reconstructed from fixed-policy replay outputs."
        ),
    )
    parser.add_argument(
        "--fold-manifest",
        default="outputs/stage4/splits/fold_manifest.csv",
        help="Frozen Stage 4 manifest used to audit complete test-fold coverage.",
    )
    parser.add_argument("--selection-metric", choices=["auroc", "ap"], default="auroc")
    parser.add_argument("--output-dir", default="outputs/stage4/evaluation")
    parser.add_argument("--reports-dir", default="reports/stage4")
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Do not copy compact CSV summaries to reports/stage4.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        selected_paths = (
            tuple(args.selected_predictions)
            if args.selected_predictions
            else discover_selected_predictions(args.runs_root or ("outputs/stage4/runs",))
        )
        result = summarize_stage4(
            selected_predictions=selected_paths,
            evaluator_csv=args.evaluator_csv,
            expert_predictions_csv=args.expert_predictions,
            fold_manifest_csv=args.fold_manifest,
            split="test",
            selection_metric=args.selection_metric,
        )
        paths = write_stage4_summary(
            result=result,
            output_dir=args.output_dir,
            reports_dir=None if args.no_reports else args.reports_dir,
        )
    except (Stage4SummaryError, OSError, ValueError) as exc:
        try:
            failure_path = write_stage4_summary_failure(
                output_dir=args.output_dir, error=exc
            )
            suffix = f"; wrote {failure_path}"
        except (Stage4SummaryError, OSError):
            suffix = ""
        print(f"ERROR: {exc}{suffix}", file=sys.stderr)
        raise SystemExit(1) from exc

    for path in paths.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
