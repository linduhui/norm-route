"""Evaluation artifact writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.evaluation.join_predictions import JoinedEvaluation


JOINED_CSV_NAME = "per_sample_joined.csv"
METRICS_JSON_NAME = "metrics_stub.json"
FAILURES_JSON_NAME = "evaluation_failures.json"


def write_evaluation_outputs(
    *,
    output_dir: str | Path,
    joined: JoinedEvaluation,
    metrics: dict[str, int],
) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_path = output_path / METRICS_JSON_NAME
    joined_path = output_path / JOINED_CSV_NAME
    failures_path = output_path / FAILURES_JSON_NAME

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_joined_csv(joined_path, joined.joined_rows)
    failures_path.write_text(
        json.dumps(_failure_report(joined), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return metrics_path, joined_path, failures_path


def write_joined_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _failure_report(joined: JoinedEvaluation) -> dict[str, Any]:
    failed_predictions = [
        {
            "image_id": prediction["image_id"],
            "status": prediction.get("status", ""),
            "error_message": prediction.get("error_message", ""),
        }
        for prediction in joined.predictions
        if prediction.get("status") != "ok"
    ]
    return {
        "missing_predictions": joined.missing_predictions,
        "extra_predictions": joined.extra_predictions,
        "failed_predictions": failed_predictions,
    }


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "schema_version",
        "expert_name",
        "image_id",
        "support_set_id",
        "raw_score",
        "normalized_score",
        "anomaly_map_path",
        "runtime_ms",
        "peak_memory_mb",
        "model_version",
        "weight_hash",
        "status",
        "error_message",
        "label",
        "mask_path",
        "defect_type",
        "has_evaluator_label",
    ]
    remaining = sorted({key for row in rows for key in row}.difference(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining
