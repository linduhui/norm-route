"""Summarize Stage 2 metrics.json files into one CSV report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "expert",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "metrics_path",
    "num_predictions",
    "num_success",
    "num_failed",
    "average_tool_calls",
    "average_runtime_ms",
    "abstention_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 2 metrics files.")
    parser.add_argument("--input-root", default="outputs/stage2")
    parser.add_argument("--output-csv", default="reports/stage2/stage2_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = summarize_stage2(
        input_root=Path(args.input_root),
        output_csv=Path(args.output_csv),
    )
    print(f"Wrote {output_csv}")


def summarize_stage2(*, input_root: Path, output_csv: Path) -> Path:
    """Collect metrics from an output tree and write the Stage 2 summary CSV."""

    rows = [_summary_row(metrics_path, input_root) for metrics_path in sorted(input_root.rglob("metrics.json"))]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def _summary_row(metrics_path: Path, input_root: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    combo = _parse_combo(metrics_path.parent, input_root)
    return {
        **combo,
        "metrics_path": str(metrics_path),
        "num_predictions": metrics.get("num_predictions", 0),
        "num_success": metrics.get("num_success", 0),
        "num_failed": metrics.get("num_failed", 0),
        "average_tool_calls": metrics.get("average_tool_calls", 0.0),
        "average_runtime_ms": metrics.get("average_runtime_ms", 0.0),
        "abstention_rate": metrics.get("abstention_rate", 0.0),
    }


def _parse_combo(output_dir: Path, input_root: Path) -> dict[str, Any]:
    try:
        parts = output_dir.relative_to(input_root).parts
    except ValueError:
        parts = output_dir.parts
    if len(parts) < 6:
        return {
            "expert": "",
            "dataset": "",
            "category": "",
            "k_shot": "",
            "seed": "",
            "support_set_id": "",
        }
    return {
        "expert": parts[-6],
        "dataset": parts[-5],
        "category": parts[-4],
        "k_shot": parts[-3].removeprefix("k"),
        "seed": parts[-2].removeprefix("seed"),
        "support_set_id": parts[-1],
    }


if __name__ == "__main__":
    main()
