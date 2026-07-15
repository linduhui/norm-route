"""Compute evaluator-only Stage 3 expert quality metrics."""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GROUP_COLUMNS = ("expert", "dataset", "category", "k_shot", "seed", "support_set_id")
SUMMARY_GROUP_COLUMNS = ("expert", "category", "k_shot")
BY_RUN_COLUMNS = (
    *GROUP_COLUMNS,
    "image_auroc",
    "image_ap",
    "image_f1_max",
    "best_threshold",
    "average_runtime_ms",
    "num_samples",
    "num_normal",
    "num_anomaly",
)
SUMMARY_COLUMNS = (
    *SUMMARY_GROUP_COLUMNS,
    "num_runs",
    "num_samples",
    "num_normal",
    "num_anomaly",
    "image_auroc_mean",
    "image_ap_mean",
    "image_f1_max_mean",
    "average_runtime_ms_mean",
)
REQUIRED_COLUMNS = (
    "dataset",
    "category",
    "support_set_id",
    "k_shot",
    "seed",
    "label",
)


class QualityMetricError(ValueError):
    """Raised when Stage 3 quality metric inputs are invalid."""


@dataclass(frozen=True)
class ExpertQualityResult:
    """In-memory evaluator-only expert quality outputs."""

    by_run_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    warnings: list[str]


def evaluate_expert_quality(path: str | Path) -> ExpertQualityResult:
    """Compute per-run and aggregate quality metrics from a routing long matrix."""

    rows = read_routing_matrix_long(path)
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[column] for column in GROUP_COLUMNS)
        grouped.setdefault(key, []).append(row)

    warnings: list[str] = []
    by_run_rows = [
        compute_group_metrics(key, group_rows, warnings)
        for key, group_rows in sorted(grouped.items())
    ]
    summary_rows = summarize_quality_rows(by_run_rows)
    return ExpertQualityResult(
        by_run_rows=by_run_rows,
        summary_rows=summary_rows,
        warnings=warnings,
    )


def read_routing_matrix_long(path: str | Path) -> list[dict[str, str]]:
    """Read and validate Stage 3 evaluator-only routing_matrix_long.csv rows."""

    matrix_path = Path(path)
    with matrix_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        expert_column = _expert_column(fieldnames)
        score_column = _score_column(fieldnames)
        has_runtime = "runtime_ms" in fieldnames
        missing = [
            column
            for column in (*REQUIRED_COLUMNS, expert_column, score_column)
            if column not in fieldnames
        ]
        if missing:
            raise QualityMetricError(f"{matrix_path} is missing columns: {missing}")

        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            clean = {key: (value or "").strip() for key, value in row.items()}
            clean["expert"] = clean[expert_column]
            clean["image_score"] = clean[score_column]
            missing_values = [
                column
                for column in (*GROUP_COLUMNS, "label", "image_score")
                if not clean.get(column)
            ]
            if missing_values:
                raise QualityMetricError(
                    f"{matrix_path}:{line_number} is missing required values: {missing_values}"
                )
            _parse_label(clean["label"], matrix_path, line_number)
            _parse_float(clean["image_score"], matrix_path, line_number, "image_score")
            if has_runtime:
                if not clean.get("runtime_ms"):
                    raise QualityMetricError(
                        f"{matrix_path}:{line_number} is missing required value: runtime_ms"
                    )
                runtime_ms = _parse_float(
                    clean["runtime_ms"], matrix_path, line_number, "runtime_ms"
                )
                if runtime_ms < 0:
                    raise QualityMetricError(
                        f"{matrix_path}:{line_number} has negative runtime_ms"
                    )
            rows.append(clean)
    return rows


def compute_group_metrics(
    key: tuple[str, ...],
    rows: list[dict[str, str]],
    warnings: list[str],
) -> dict[str, Any]:
    """Compute image-level metrics for one expert/dataset/category/run group."""

    labels = [_parse_label(row["label"]) for row in rows]
    scores = [_parse_float(row["image_score"]) for row in rows]
    num_anomaly = sum(labels)
    num_samples = len(labels)
    num_normal = num_samples - num_anomaly

    image_auroc: float | None
    image_ap: float | None
    if num_normal == 0 or num_anomaly == 0:
        warning = (
            "Single-label group has undefined AUROC/AP: "
            + ", ".join(f"{column}={value}" for column, value in zip(GROUP_COLUMNS, key))
            + f", num_normal={num_normal}, num_anomaly={num_anomaly}"
        )
        warnings.append(warning)
        image_auroc = None
        image_ap = None
    else:
        image_auroc = compute_auroc(labels, scores)
        image_ap = compute_average_precision(labels, scores)

    image_f1_max, best_threshold = compute_f1_max(labels, scores)
    runtime_values = [row.get("runtime_ms", "") for row in rows]
    if any(runtime_values) and not all(runtime_values):
        raise QualityMetricError(
            "Runtime coverage is inconsistent within run group: "
            + ", ".join(f"{column}={value}" for column, value in zip(GROUP_COLUMNS, key))
        )
    average_runtime_ms = (
        sum(float(value) for value in runtime_values) / len(runtime_values)
        if runtime_values and all(runtime_values)
        else None
    )
    return {
        **dict(zip(GROUP_COLUMNS, key)),
        "image_auroc": image_auroc,
        "image_ap": image_ap,
        "image_f1_max": image_f1_max,
        "best_threshold": best_threshold,
        "average_runtime_ms": average_runtime_ms,
        "num_samples": num_samples,
        "num_normal": num_normal,
        "num_anomaly": num_anomaly,
    }


def compute_auroc(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC with average ranks for tied image scores."""

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    rank = 1
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][1] == ranked[index][1]:
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        for ranked_index in range(index, end):
            original_index = ranked[ranked_index][0]
            ranks[original_index] = average_rank
        rank += end - index
        index = end

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_average_precision(labels: list[int], scores: list[float]) -> float:
    """Compute non-interpolated image average precision for anomaly labels."""

    n_pos = sum(labels)
    sorted_pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(sorted_pairs, start=1):
        if label == 1:
            true_positives += 1
            precision_sum += true_positives / index
    return precision_sum / n_pos


def compute_f1_max(labels: list[int], scores: list[float]) -> tuple[float, float]:
    """Return the maximum F1 and threshold, predicting anomaly when score >= threshold."""

    best_f1 = -1.0
    best_threshold = max(scores)
    for threshold in sorted(set(scores), reverse=True):
        tp = fp = fn = 0
        for label, score in zip(labels, scores):
            predicted = score >= threshold
            if predicted and label == 1:
                tp += 1
            elif predicted and label == 0:
                fp += 1
            elif not predicted and label == 1:
                fn += 1
        denominator = 2 * tp + fp + fn
        f1 = 0.0 if denominator == 0 else (2 * tp) / denominator
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_f1, best_threshold


def summarize_quality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate run-level expert quality by expert/category/k_shot."""

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[column]) for column in SUMMARY_GROUP_COLUMNS)
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        summaries.append(
            {
                **dict(zip(SUMMARY_GROUP_COLUMNS, key)),
                "num_runs": len(group_rows),
                "num_samples": sum(int(row["num_samples"]) for row in group_rows),
                "num_normal": sum(int(row["num_normal"]) for row in group_rows),
                "num_anomaly": sum(int(row["num_anomaly"]) for row in group_rows),
                "image_auroc_mean": _mean_defined(row["image_auroc"] for row in group_rows),
                "image_ap_mean": _mean_defined(row["image_ap"] for row in group_rows),
                "image_f1_max_mean": _mean_defined(row["image_f1_max"] for row in group_rows),
                "average_runtime_ms_mean": _mean_defined(
                    row.get("average_runtime_ms") for row in group_rows
                ),
            }
        )
    return summaries


def write_expert_quality_outputs(
    result: ExpertQualityResult,
    output_dir: str | Path,
    reports_dir: str | Path | None = "reports/stage3",
) -> tuple[Path, Path, Path]:
    """Write by-run, summary, and warning artifacts, then sync summary to reports."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    by_run_path = output_path / "expert_quality_by_run.csv"
    summary_path = output_path / "expert_quality_summary.csv"
    warnings_path = output_path / "expert_quality_warnings.json"
    write_csv(by_run_path, list(BY_RUN_COLUMNS), result.by_run_rows)
    write_csv(summary_path, list(SUMMARY_COLUMNS), result.summary_rows)
    warnings_path.write_text(
        json.dumps({"warnings": result.warnings}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if reports_dir is not None:
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(summary_path, reports_path / "expert_quality_summary.csv")

    return by_run_path, summary_path, warnings_path


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(fieldnames=fieldnames, f=handle, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_csv_value(row.get(column)) for column in fieldnames})


def _expert_column(fieldnames: Iterable[str]) -> str:
    if "expert" in fieldnames:
        return "expert"
    if "expert_name" in fieldnames:
        return "expert_name"
    return "expert"


def _score_column(fieldnames: Iterable[str]) -> str:
    if "image_score" in fieldnames:
        return "image_score"
    if "final_score" in fieldnames:
        return "final_score"
    return "image_score"


def _parse_label(value: str, path: Path | None = None, line_number: int | None = None) -> int:
    if value in {"0", "0.0"}:
        return 0
    if value in {"1", "1.0"}:
        return 1
    location = f"{path}:{line_number} " if path is not None and line_number is not None else ""
    raise QualityMetricError(f"{location}has invalid binary label: {value!r}")


def _parse_float(
    value: str,
    path: Path | None = None,
    line_number: int | None = None,
    column: str = "value",
) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        location = f"{path}:{line_number} " if path is not None and line_number is not None else ""
        raise QualityMetricError(f"{location}has invalid numeric {column}: {value!r}") from exc
    if not math.isfinite(parsed):
        location = f"{path}:{line_number} " if path is not None and line_number is not None else ""
        raise QualityMetricError(f"{location}has non-finite numeric {column}: {value!r}")
    return parsed


def _mean_defined(values: Iterable[Any]) -> float | None:
    defined = [float(value) for value in values if value is not None and value != ""]
    if not defined:
        return None
    return sum(defined) / len(defined)


def _format_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)
