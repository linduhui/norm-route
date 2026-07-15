"""Evaluator-only image-level metrics for Stage 4 selected predictions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

from src.normroute.agent.replay_executor import SELECTED_PREDICTION_COLUMNS
from src.normroute.routing.quality_metrics import compute_auroc, compute_average_precision


STAGE4_EVALUATION_VERSION = "stage4.evaluator_only.v1"
STAGE4_METRIC_COLUMNS = (
    "policy_name",
    "fold",
    "split",
    "num_samples",
    "num_normal",
    "num_anomaly",
    "auroc",
    "ap",
    "f1",
    "estimated_runtime_ms",
    "average_estimated_runtime_ms",
    "tool_calls",
    "average_tool_calls",
)
EVALUATED_SAMPLE_COLUMNS = (
    *SELECTED_PREDICTION_COLUMNS,
    "label",
)
_EVALUATOR_REQUIRED_COLUMNS = ("image_id", "label")
_FORBIDDEN_SELECTED_COLUMNS = frozenset(
    {"label", "mask", "mask_path", "defect_type", "anomaly_type", "ground_truth"}
)


class Stage4EvaluationError(ValueError):
    """Raised when Stage 4 metrics cannot be computed without dropping samples."""


@dataclass(frozen=True)
class Stage4EvaluationResult:
    metric_rows: list[dict[str, Any]]
    evaluated_rows: list[dict[str, Any]]
    warnings: list[str]


def evaluate_selected_predictions(
    *,
    selected_predictions: Sequence[str | Path] | str | Path,
    evaluator_csv: str | Path,
) -> Stage4EvaluationResult:
    """Join labels after routing, then report one row per policy/fold/split."""

    paths = _coerce_paths(selected_predictions)
    evaluator_by_id = _read_evaluator_labels(evaluator_csv)
    selected_rows: list[dict[str, str]] = []
    for path in paths:
        selected_rows.extend(_read_selected_predictions(path))
    if not selected_rows:
        raise Stage4EvaluationError("No selected predictions were provided for evaluation")

    seen: set[tuple[str, str, str, str]] = set()
    evaluated_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        key = (row["policy_name"], row["fold"], row["split"], row["task_id"])
        if key in seen:
            raise Stage4EvaluationError(
                "Duplicate selected prediction for policy/fold/split/task: " + repr(key)
            )
        seen.add(key)
        image_id = row["image_id"]
        if image_id not in evaluator_by_id:
            raise Stage4EvaluationError(
                f"Selected prediction row {index + 2} image_id={image_id!r} is absent from "
                f"evaluator CSV {evaluator_csv}"
            )
        if row["status"].lower() != "ok":
            raise Stage4EvaluationError(
                f"Selected prediction task_id={row['task_id']!r} has status={row['status']!r}; "
                "failed samples cannot be silently excluded from metrics"
            )
        _finite_float(row["final_score"], "final_score", row["task_id"])
        _finite_nonnegative(row["runtime_ms"], "runtime_ms", row["task_id"])
        _nonnegative_int(row["tool_calls"], "tool_calls", row["task_id"])
        _decision_to_binary(row["final_decision"], row["task_id"])
        evaluated_rows.append({**row, "label": evaluator_by_id[image_id]})

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in evaluated_rows:
        group_key = (row["policy_name"], row["fold"], row["split"])
        grouped.setdefault(group_key, []).append(row)

    warnings: list[str] = []
    metric_rows = [
        _compute_group_metrics(key, rows, warnings)
        for key, rows in sorted(grouped.items())
    ]
    return Stage4EvaluationResult(
        metric_rows=metric_rows,
        evaluated_rows=evaluated_rows,
        warnings=warnings,
    )


def write_stage4_evaluation(
    *,
    result: Stage4EvaluationResult,
    output_dir: str | Path,
    selected_predictions: Sequence[str | Path] | str | Path,
    evaluator_csv: str | Path,
) -> dict[str, Path]:
    """Write metrics, evaluator-only joined rows, failures, config, and provenance."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_paths = _coerce_paths(selected_predictions)
    metrics_path = output_path / "stage4_metrics.csv"
    samples_path = output_path / "per_sample_evaluator_only.csv"
    failures_path = output_path / "evaluation_failures.json"
    config_path = output_path / "evaluation_config.json"
    metadata_path = output_path / "evaluation_metadata.json"
    _write_csv(metrics_path, STAGE4_METRIC_COLUMNS, result.metric_rows)
    _write_csv(samples_path, EVALUATED_SAMPLE_COLUMNS, result.evaluated_rows)
    _write_json(
        failures_path,
        {
            "protocol_version": STAGE4_EVALUATION_VERSION,
            "failures": [],
            "warnings": result.warnings,
        },
    )
    config = {
        "protocol_version": STAGE4_EVALUATION_VERSION,
        "selected_predictions": [str(path) for path in selected_paths],
        "evaluator_csv": str(evaluator_csv),
        "f1_source": "saved_final_decision",
        "runtime_source": "estimated_runtime",
    }
    _write_json(config_path, config)
    _write_json(
        metadata_path,
        {
            "protocol_version": STAGE4_EVALUATION_VERSION,
            "config": config,
            "seed": None,
            "seeds": sorted({int(row["seed"]) for row in result.evaluated_rows}),
            "git_commit": _git_commit(),
            "environment": {
                "python_executable": sys.executable,
                "python_version": sys.version,
                "platform": platform.platform(),
            },
            "input_sha256": {
                str(path): _sha256(path) for path in (*selected_paths, Path(evaluator_csv))
            },
            "num_metric_rows": len(result.metric_rows),
            "num_evaluated_samples": len(result.evaluated_rows),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "metrics_path": metrics_path,
        "samples_path": samples_path,
        "failures_path": failures_path,
        "config_path": config_path,
        "metadata_path": metadata_path,
    }


def write_stage4_evaluation_failure(
    *, output_dir: str | Path, error: Exception
) -> Path:
    """Persist an evaluator failure instead of silently losing the run."""

    path = Path(output_dir) / "evaluation_failures.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "protocol_version": STAGE4_EVALUATION_VERSION,
            "failures": [
                {
                    "failure_type": "evaluation_error",
                    "error_message": str(error),
                }
            ],
            "warnings": [],
        },
    )
    return path


def _compute_group_metrics(
    key: tuple[str, str, str],
    rows: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["final_score"]) for row in rows]
    decisions = [
        _decision_to_binary(row["final_decision"], row["task_id"]) for row in rows
    ]
    num_anomaly = sum(labels)
    num_samples = len(labels)
    num_normal = num_samples - num_anomaly
    if num_anomaly == 0 or num_normal == 0:
        auroc: float | None = None
        ap: float | None = None
        warnings.append(
            f"Undefined AUROC/AP for policy={key[0]}, fold={key[1]}, split={key[2]}: "
            f"num_normal={num_normal}, num_anomaly={num_anomaly}"
        )
    else:
        auroc = compute_auroc(labels, scores)
        ap = compute_average_precision(labels, scores)
    # Replay reuses historical Stage 2 runtime as an estimate; it does not
    # measure the current replay's wall-clock duration.
    runtime_ms = sum(float(row["runtime_ms"]) for row in rows)
    tool_calls = sum(int(row["tool_calls"]) for row in rows)
    return {
        "policy_name": key[0],
        "fold": key[1],
        "split": key[2],
        "num_samples": num_samples,
        "num_normal": num_normal,
        "num_anomaly": num_anomaly,
        "auroc": auroc,
        "ap": ap,
        "f1": _binary_f1(labels, decisions),
        "estimated_runtime_ms": runtime_ms,
        "average_estimated_runtime_ms": runtime_ms / num_samples,
        "tool_calls": tool_calls,
        "average_tool_calls": tool_calls / num_samples,
    }


def _binary_f1(labels: Sequence[int], decisions: Sequence[int]) -> float:
    tp = sum(1 for label, predicted in zip(labels, decisions) if label == predicted == 1)
    fp = sum(1 for label, predicted in zip(labels, decisions) if label == 0 and predicted == 1)
    fn = sum(1 for label, predicted in zip(labels, decisions) if label == 1 and predicted == 0)
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2 * tp) / denominator


def _decision_to_binary(value: str, task_id: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"anomaly", "anomalous", "1", "true", "stop_anomaly"}:
        return 1
    if normalized in {"normal", "good", "0", "false", "stop_normal", "abstain"}:
        return 0
    raise Stage4EvaluationError(
        f"task_id={task_id!r} has unsupported final_decision={value!r}"
    )


def _read_selected_predictions(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise Stage4EvaluationError(f"Selected predictions do not exist: {source}")
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in SELECTED_PREDICTION_COLUMNS if column not in fieldnames]
        forbidden = sorted(_FORBIDDEN_SELECTED_COLUMNS.intersection(fieldnames))
        if missing or forbidden:
            raise Stage4EvaluationError(
                f"{source} has invalid selected-prediction columns; "
                f"missing={missing}, forbidden={forbidden}"
            )
        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            clean = {key: (value or "").strip() for key, value in row.items()}
            required_values = (
                "task_id",
                "fold",
                "split",
                "policy_name",
                "selected_expert",
                "image_id",
                "final_score",
                "final_decision",
                "runtime_ms",
                "tool_calls",
                "status",
            )
            empty = [column for column in required_values if not clean.get(column)]
            if empty:
                raise Stage4EvaluationError(f"{source}:{line_number} has empty values: {empty}")
            runtime_source = clean.get("runtime_source")
            if runtime_source and runtime_source != "estimated_runtime":
                raise Stage4EvaluationError(
                    f"{source}:{line_number} runtime_source must be 'estimated_runtime'"
                )
            rows.append(clean)
    return rows


def _read_evaluator_labels(path: str | Path) -> dict[str, str]:
    source = Path(path)
    if not source.is_file():
        raise Stage4EvaluationError(f"Evaluator CSV does not exist: {source}")
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in _EVALUATOR_REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise Stage4EvaluationError(f"{source} is missing evaluator columns: {missing}")
        labels: dict[str, str] = {}
        for line_number, row in enumerate(reader, start=2):
            image_id = (row.get("image_id") or "").strip()
            label = (row.get("label") or "").strip()
            if not image_id or label not in {"0", "1", "0.0", "1.0"}:
                raise Stage4EvaluationError(
                    f"{source}:{line_number} has invalid image_id/label: "
                    f"{image_id!r}/{label!r}"
                )
            if image_id in labels:
                raise Stage4EvaluationError(f"{source} has duplicate image_id={image_id!r}")
            labels[image_id] = "1" if label in {"1", "1.0"} else "0"
    if not labels:
        raise Stage4EvaluationError(f"Evaluator CSV is empty: {source}")
    return labels


def _coerce_paths(
    value: Sequence[str | Path] | str | Path,
) -> tuple[Path, ...]:
    if isinstance(value, (str, Path)):
        paths = (Path(value),)
    else:
        paths = tuple(Path(path) for path in value)
    if not paths:
        raise Stage4EvaluationError("At least one selected_predictions.csv is required")
    return paths


def _finite_float(value: str, field: str, task_id: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise Stage4EvaluationError(
            f"task_id={task_id!r} field {field!r} must be numeric"
        ) from exc
    if not math.isfinite(parsed):
        raise Stage4EvaluationError(
            f"task_id={task_id!r} field {field!r} must be finite"
        )
    return parsed


def _finite_nonnegative(value: str, field: str, task_id: str) -> float:
    parsed = _finite_float(value, field, task_id)
    if parsed < 0:
        raise Stage4EvaluationError(
            f"task_id={task_id!r} field {field!r} must be >= 0"
        )
    return parsed


def _nonnegative_int(value: str, field: str, task_id: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise Stage4EvaluationError(
            f"task_id={task_id!r} field {field!r} must be an integer"
        ) from exc
    if parsed < 0:
        raise Stage4EvaluationError(
            f"task_id={task_id!r} field {field!r} must be >= 0"
        )
    return parsed


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: "" if row.get(column) is None else row.get(column, "")
                    for column in fieldnames
                }
            )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    project_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"
