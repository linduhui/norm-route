"""Join expert predictions with evaluator-only labels by image_id."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experts.results import ExpertResult, ExpertResultValidationError


EVALUATOR_REQUIRED_COLUMNS = ["image_id", "label", "mask_path", "defect_type"]
AGENT_INPUT_REQUIRED_COLUMNS = ["image_id"]


class EvaluationJoinError(ValueError):
    """Raised when prediction/evaluator inputs cannot be safely joined."""


@dataclass(frozen=True)
class JoinedEvaluation:
    predictions: list[dict[str, Any]]
    evaluator_rows: list[dict[str, str]]
    joined_rows: list[dict[str, Any]]
    expected_image_ids: list[str]
    missing_predictions: list[str]
    extra_predictions: list[str]


def join_predictions_with_evaluator(
    *,
    predictions_jsonl: str | Path,
    evaluator_csv: str | Path,
    agent_input_csv: str | Path | None = None,
) -> JoinedEvaluation:
    """Load and join predictions to evaluator rows using only image_id."""

    predictions = read_predictions_jsonl(predictions_jsonl)
    evaluator_rows = read_evaluator_csv(evaluator_csv)
    expected_image_ids = read_expected_image_ids(agent_input_csv) if agent_input_csv else _image_ids(evaluator_rows)

    _reject_duplicate_ids(expected_image_ids, "expected samples")
    _reject_duplicate_ids(_image_ids(evaluator_rows), "evaluator rows")
    _reject_duplicate_ids([str(row["image_id"]) for row in predictions], "predictions")

    evaluator_by_id = {row["image_id"]: row for row in evaluator_rows}
    prediction_ids = [str(row["image_id"]) for row in predictions]
    prediction_id_set = set(prediction_ids)
    expected_id_set = set(expected_image_ids)

    missing_predictions = [image_id for image_id in expected_image_ids if image_id not in prediction_id_set]
    extra_predictions = [image_id for image_id in prediction_ids if image_id not in expected_id_set]

    joined_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        image_id = str(prediction["image_id"])
        evaluator = evaluator_by_id.get(image_id)
        joined_rows.append(_joined_row(prediction, evaluator))

    return JoinedEvaluation(
        predictions=predictions,
        evaluator_rows=evaluator_rows,
        joined_rows=joined_rows,
        expected_image_ids=expected_image_ids,
        missing_predictions=missing_predictions,
        extra_predictions=extra_predictions,
    )


def read_predictions_jsonl(path: str | Path) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                result = ExpertResult.from_jsonl_line(line)
            except ExpertResultValidationError as exc:
                raise EvaluationJoinError(f"Invalid prediction at line {line_number}: {exc}") from exc
            predictions.append(result.to_dict())
    return predictions


def read_evaluator_csv(path: str | Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    _require_columns(rows.fieldnames, EVALUATOR_REQUIRED_COLUMNS, "evaluator CSV")
    return rows.rows


def read_expected_image_ids(path: str | Path) -> list[str]:
    rows = _read_csv(path)
    _require_columns(rows.fieldnames, AGENT_INPUT_REQUIRED_COLUMNS, "agent input CSV")
    return _image_ids(rows.rows)


def _joined_row(prediction: dict[str, Any], evaluator: dict[str, str] | None) -> dict[str, Any]:
    row = dict(prediction)
    if evaluator is None:
        row["label"] = ""
        row["mask_path"] = ""
        row["defect_type"] = ""
        row["has_evaluator_label"] = "0"
        return row

    row["label"] = evaluator["label"]
    row["mask_path"] = evaluator["mask_path"]
    row["defect_type"] = evaluator["defect_type"]
    row["has_evaluator_label"] = "1" if evaluator["label"] != "" else "0"
    return row


@dataclass(frozen=True)
class _CsvRows:
    fieldnames: list[str]
    rows: list[dict[str, str]]


def _read_csv(path: str | Path) -> _CsvRows:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        return _CsvRows(fieldnames=fieldnames, rows=[dict(row) for row in reader])


def _require_columns(fieldnames: list[str], required: list[str], name: str) -> None:
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise EvaluationJoinError(f"{name} is missing required columns: {missing}")


def _reject_duplicate_ids(image_ids: list[str], name: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for image_id in image_ids:
        if image_id in seen and image_id not in duplicates:
            duplicates.append(image_id)
        seen.add(image_id)
    if duplicates:
        raise EvaluationJoinError(f"Duplicate image_id values in {name}: {duplicates}")


def _image_ids(rows: list[dict[str, str]]) -> list[str]:
    return [row["image_id"] for row in rows]
