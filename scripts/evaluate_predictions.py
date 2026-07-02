"""Evaluate saved prediction JSONL against evaluator-only CSV labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.join_predictions import EvaluationJoinError, join_predictions_with_evaluator
from src.evaluation.metrics_stub import compute_metrics_stub
from src.evaluation.reporting import write_evaluation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write count-only evaluation artifacts for saved predictions.")
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--evaluator-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--agent-input-csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        joined = join_predictions_with_evaluator(
            predictions_jsonl=args.predictions_jsonl,
            evaluator_csv=args.evaluator_csv,
            agent_input_csv=args.agent_input_csv,
        )
        metrics = compute_metrics_stub(joined)
        metrics_path, joined_path, failures_path = write_evaluation_outputs(
            output_dir=args.output_dir,
            joined=joined,
            metrics=metrics,
        )
    except EvaluationJoinError as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc

    print(f"Wrote {metrics_path}")
    print(f"Wrote {joined_path}")
    print(f"Wrote {failures_path}")

    if joined.missing_predictions or joined.extra_predictions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
