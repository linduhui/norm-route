"""Write Stage 2 expert run artifacts without evaluator-only inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.normroute.experts.base import ExpertPrediction


PREDICTION_COLUMNS = [
    "image_id",
    "expert_name",
    "dataset",
    "category",
    "support_set_id",
    "k_shot",
    "seed",
    "final_score",
    "final_decision",
    "anomaly_map_path",
    "pixel_score_path",
    "actions",
    "tool_calls",
    "runtime_ms",
    "status",
    "error_message",
]

METRICS_FIELDS = [
    "num_predictions",
    "num_success",
    "num_failed",
    "average_tool_calls",
    "average_runtime_ms",
    "abstention_rate",
]

FAILURES_COLUMNS = ["image_id", "status", "error_message"]


def export_run_outputs(
    *,
    output_dir: str | Path,
    predictions: list[ExpertPrediction],
    run_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    """Write standard Stage 2 artifacts for one expert run."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "anomaly_maps").mkdir(exist_ok=True)

    prediction_rows = [prediction.to_dict() for prediction in predictions]
    predictions_path = output_path / "predictions.csv"
    metrics_path = output_path / "metrics.json"
    failures_path = output_path / "failures.json"
    metadata_path = output_path / "run_metadata.json"

    _write_csv(predictions_path, PREDICTION_COLUMNS, prediction_rows)
    metrics_path.write_text(
        json.dumps(_metrics(prediction_rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failures_path.write_text(
        json.dumps(_failures(prediction_rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(run_metadata or {}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return predictions_path, metrics_path, failures_path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    failed = [row for row in rows if row["status"] != "ok"]
    tool_calls = sum(int(row["tool_calls"]) for row in rows)
    runtime_ms = sum(float(row["runtime_ms"]) for row in rows)
    abstained = [row for row in rows if str(row["final_decision"]).lower() == "abstain"]
    return {
        "num_predictions": count,
        "num_success": count - len(failed),
        "num_failed": len(failed),
        "average_tool_calls": tool_calls / count if count else 0.0,
        "average_runtime_ms": runtime_ms / count if count else 0.0,
        "abstention_rate": len(abstained) / count if count else 0.0,
    }


def _failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "failed_predictions": [
            {column: row[column] for column in FAILURES_COLUMNS}
            for row in rows
            if row["status"] != "ok"
        ]
    }
