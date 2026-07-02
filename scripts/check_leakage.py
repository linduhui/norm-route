"""Run Stage 1 NORM-Route leakage checks on manifests and support sets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.leakage_checker import check_leakage, write_json_report, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-input-csv", required=True, help="Path to agent_input CSV.")
    parser.add_argument("--evaluator-csv", required=True, help="Path to evaluator CSV.")
    parser.add_argument(
        "--support",
        required=True,
        help="Path to one support-set CSV or a directory containing support-set CSV files.",
    )
    parser.add_argument("--json-report", required=True, help="Path for machine-readable JSON.")
    parser.add_argument("--summary-output", help="Optional path for human-readable summary text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = check_leakage(
        agent_input_csv=Path(args.agent_input_csv),
        evaluator_csv=Path(args.evaluator_csv),
        support=Path(args.support),
    )
    write_json_report(args.json_report, report)
    if args.summary_output:
        write_summary(args.summary_output, report)
    print(report["summary"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
