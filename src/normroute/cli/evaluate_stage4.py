"""CLI for evaluator-only Stage 4 selected-prediction metrics."""

from __future__ import annotations

import argparse
import sys

from src.normroute.evaluation.stage4 import (
    Stage4EvaluationError,
    evaluate_selected_predictions,
    write_stage4_evaluation,
    write_stage4_evaluation_failure,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 4 selected_predictions.csv after routing is complete."
    )
    parser.add_argument(
        "--selected-predictions",
        nargs="+",
        required=True,
        help="One or more selected_predictions.csv files.",
    )
    parser.add_argument("--evaluator-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/stage4/evaluator_only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = evaluate_selected_predictions(
            selected_predictions=args.selected_predictions,
            evaluator_csv=args.evaluator_csv,
        )
        paths = write_stage4_evaluation(
            result=result,
            output_dir=args.output_dir,
            selected_predictions=args.selected_predictions,
            evaluator_csv=args.evaluator_csv,
        )
    except (Stage4EvaluationError, OSError, ValueError) as exc:
        failure_path = write_stage4_evaluation_failure(output_dir=args.output_dir, error=exc)
        print(f"ERROR: {exc}; wrote {failure_path}", file=sys.stderr)
        raise SystemExit(1) from exc

    for path in paths.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
