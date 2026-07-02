"""Count-only metric stub for early evaluation pipeline tests."""

from __future__ import annotations

from typing import Any

from src.evaluation.join_predictions import JoinedEvaluation


METRIC_STUB_KEYS = [
    "num_predictions",
    "num_joined",
    "num_missing_labels",
    "num_anomaly",
    "num_normal",
    "num_failed_predictions",
]


def compute_metrics_stub(joined: JoinedEvaluation) -> dict[str, int]:
    """Return count-only metrics without computing formal AUROC or reading masks."""

    metrics = {
        "num_predictions": len(joined.predictions),
        "num_joined": _num_joined(joined.joined_rows),
        "num_missing_labels": _num_missing_labels(joined.joined_rows),
        "num_anomaly": _num_label(joined.joined_rows, "1"),
        "num_normal": _num_label(joined.joined_rows, "0"),
        "num_failed_predictions": _num_failed_predictions(joined.predictions),
    }
    return {key: metrics[key] for key in METRIC_STUB_KEYS}


def _num_joined(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("has_evaluator_label") == "1")


def _num_missing_labels(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("has_evaluator_label") != "1")


def _num_label(rows: list[dict[str, Any]], label: str) -> int:
    return sum(1 for row in rows if str(row.get("label", "")) == label)


def _num_failed_predictions(predictions: list[dict[str, Any]]) -> int:
    return sum(1 for prediction in predictions if prediction.get("status") != "ok")
