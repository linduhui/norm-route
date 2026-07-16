"""Evaluator-only Stage 3 Oracle upper-bound computations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .join_predictions import RoutingMatrixError
from .quality_metrics import (
    GROUP_COLUMNS,
    QualityMetricError,
    compute_group_metrics,
    read_routing_matrix_long,
    write_csv,
)


RUN_KEY_COLUMNS = ("dataset", "category", "k_shot", "seed")
ORACLE_SUMMARY_COLUMNS = (
    "evaluator_only",
    "method",
    "selection_metric",
    "selected_expert",
    "num_runs",
    "num_samples",
    "image_auroc_mean",
    "image_ap_mean",
    "image_f1_max_mean",
)
ORACLE_SELECTION_COUNT_COLUMNS = (
    "evaluator_only",
    "selection_metric",
    "dataset",
    "category",
    "k_shot",
    "expert",
    "selection_count",
)


class OracleError(ValueError):
    """Raised when evaluator-only Oracle inputs are invalid."""


@dataclass(frozen=True)
class OracleResult:
    """In-memory evaluator-only Oracle outputs."""

    summary_rows: list[dict[str, Any]]
    selection_count_rows: list[dict[str, Any]]


def compute_oracle_outputs(
    *,
    routing_matrix_long: str | Path,
    expert_quality_by_run: str | Path,
    selection_metric: str = "image_auroc",
) -> OracleResult:
    """Compute best-single, run-level Oracle, and sample-level Oracle summaries."""

    if selection_metric not in {"image_auroc", "image_ap"}:
        raise OracleError("selection_metric must be one of: image_auroc, image_ap")

    quality_rows = read_expert_quality_by_run(expert_quality_by_run)
    routing_rows = read_routing_matrix_long(routing_matrix_long)

    best_single_rows = _best_single_rows(quality_rows, selection_metric)
    run_oracle_rows, selection_count_rows = _run_level_oracle_rows(
        quality_rows,
        selection_metric,
    )
    sample_oracle_rows = _sample_level_oracle_rows(routing_rows)

    summary_rows = [
        _summarize_method("best_single", selection_metric, best_single_rows),
        _summarize_method("run_level_oracle", selection_metric, run_oracle_rows),
        _summarize_method("sample_level_oracle", selection_metric, sample_oracle_rows),
    ]
    return OracleResult(
        summary_rows=summary_rows,
        selection_count_rows=selection_count_rows,
    )


def read_expert_quality_by_run(path: str | Path) -> list[dict[str, Any]]:
    """Read expert_quality_by_run.csv without silently skipping malformed rows."""

    quality_path = Path(path)
    with quality_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = (
            *GROUP_COLUMNS,
            "image_auroc",
            "image_ap",
            "image_f1_max",
            "num_samples",
        )
        missing = [column for column in required if column not in fieldnames]
        if missing:
            raise OracleError(f"{quality_path} is missing columns: {missing}")

        rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(reader, start=2):
            clean = {key: (value or "").strip() for key, value in row.items()}
            missing_values = [column for column in GROUP_COLUMNS if not clean.get(column)]
            if missing_values:
                raise OracleError(
                    f"{quality_path}:{line_number} is missing required values: {missing_values}"
                )
            for column in ("image_auroc", "image_ap", "image_f1_max"):
                clean[column] = _parse_optional_float(
                    clean[column],
                    quality_path,
                    line_number,
                    column,
                )
            clean["num_samples"] = _parse_int(
                clean["num_samples"],
                quality_path,
                line_number,
                "num_samples",
            )
            rows.append(clean)
    return rows


def write_oracle_outputs(result: OracleResult, output_dir: str | Path) -> tuple[Path, Path]:
    """Write evaluator-only Oracle summary and selection count artifacts."""

    output_path = Path(output_dir)
    ensure_oracle_output(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "oracle_summary.csv"
    counts_path = output_path / "oracle_selection_counts.csv"
    write_csv(summary_path, list(ORACLE_SUMMARY_COLUMNS), result.summary_rows)
    write_csv(counts_path, list(ORACLE_SELECTION_COUNT_COLUMNS), result.selection_count_rows)
    return summary_path, counts_path


def ensure_oracle_output(output_dir: Path) -> None:
    """Keep evaluator-only Oracle aggregates out of agent-visible directories."""

    parts = {part.lower() for part in output_dir.parts}
    leaf = output_dir.name.lower()
    if "agent_visible" in parts:
        raise RoutingMatrixError("Oracle outputs are evaluator-only and cannot be agent_visible.")
    if "evaluator_only" not in parts and not leaf.startswith("oracle"):
        raise RoutingMatrixError(
            "Oracle outputs are evaluator-only aggregates and must be written under an "
            "evaluator_only directory or an oracle* directory."
        )


def _best_single_rows(
    quality_rows: list[dict[str, Any]],
    selection_metric: str,
) -> list[dict[str, Any]]:
    by_expert: dict[str, list[dict[str, Any]]] = {}
    for row in quality_rows:
        by_expert.setdefault(str(row["expert"]), []).append(row)
    if not by_expert:
        raise OracleError("expert_quality_by_run.csv has no rows")

    selected_expert = max(
        sorted(by_expert),
        key=lambda expert: _mean_for_selection(
            (row[selection_metric] for row in by_expert[expert]),
            selection_metric,
        ),
    )
    return [_with_method(row, "best_single", selected_expert) for row in by_expert[selected_expert]]


def _run_level_oracle_rows(
    quality_rows: list[dict[str, Any]],
    selection_metric: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in quality_rows:
        key = tuple(str(row[column]) for column in RUN_KEY_COLUMNS)
        grouped.setdefault(key, []).append(row)

    selected_rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, ...], int] = {}
    for key, rows in sorted(grouped.items()):
        selected = max(
            sorted(rows, key=lambda row: str(row["expert"])),
            key=lambda row: _metric_value(row[selection_metric]),
        )
        if _metric_value(selected[selection_metric]) == float("-inf"):
            raise OracleError(
                "Run-level Oracle cannot select an expert because "
                f"{selection_metric} is undefined for dataset={key[0]}, "
                f"category={key[1]}, k_shot={key[2]}, seed={key[3]}"
            )
        selected_rows.append(_with_method(selected, "run_level_oracle", str(selected["expert"])))
        count_key = (selection_metric, key[0], key[1], key[2], str(selected["expert"]))
        counts[count_key] = counts.get(count_key, 0) + 1

    count_rows = [
        {
            "evaluator_only": True,
            "selection_metric": metric,
            "dataset": dataset,
            "category": category,
            "k_shot": k_shot,
            "expert": expert,
            "selection_count": count,
        }
        for (metric, dataset, category, k_shot, expert), count in sorted(counts.items())
    ]
    return selected_rows, count_rows


def _sample_level_oracle_rows(routing_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in routing_rows:
        key = tuple(row[column] for column in RUN_KEY_COLUMNS)
        grouped.setdefault(key, []).append(row)

    by_sample: dict[tuple[str, ...], list[dict[str, str]]]
    oracle_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run_key, rows in sorted(grouped.items()):
        by_sample = {}
        for row in rows:
            sample_key = tuple(row[column] for column in ("image_id", *RUN_KEY_COLUMNS))
            by_sample.setdefault(sample_key, []).append(row)

        synthetic_rows: list[dict[str, str]] = []
        support_set_ids = sorted({row["support_set_id"] for row in rows})
        for sample_rows in by_sample.values():
            label = sample_rows[0]["label"]
            selected = _sample_oracle_row(sample_rows)
            synthetic_rows.append(
                {
                    **selected,
                    "expert": "sample_level_oracle",
                    "image_score": selected["image_score"],
                    "label": label,
                    "support_set_id": "+".join(support_set_ids),
                }
            )

        key = (
            "sample_level_oracle",
            run_key[0],
            run_key[1],
            run_key[2],
            run_key[3],
            "+".join(support_set_ids),
        )
        oracle_rows.append(
            _with_method(compute_group_metrics(key, synthetic_rows, warnings), "sample_level_oracle", "")
        )

    if warnings:
        raise OracleError("Sample-level Oracle has undefined metrics:\n" + "\n".join(warnings))
    return oracle_rows


def _sample_oracle_row(rows: list[dict[str, str]]) -> dict[str, str]:
    label = rows[0]["label"]
    if any(row["label"] != label for row in rows):
        raise OracleError(f"Conflicting labels for sample-level Oracle image_id={rows[0].get('image_id')}")
    reverse = label in {"1", "1.0"}
    return sorted(rows, key=lambda row: float(row["image_score"]), reverse=reverse)[0]


def _summarize_method(
    method: str,
    selection_metric: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_experts = sorted({str(row.get("selected_expert", "")) for row in rows if row.get("selected_expert")})
    return {
        "evaluator_only": True,
        "method": method,
        "selection_metric": selection_metric,
        "selected_expert": selected_experts[0] if len(selected_experts) == 1 else "",
        "num_runs": len(rows),
        "num_samples": sum(int(row["num_samples"]) for row in rows),
        "image_auroc_mean": _mean_optional(row["image_auroc"] for row in rows),
        "image_ap_mean": _mean_optional(row["image_ap"] for row in rows),
        "image_f1_max_mean": _mean_optional(row["image_f1_max"] for row in rows),
    }


def _with_method(row: dict[str, Any], method: str, selected_expert: str) -> dict[str, Any]:
    return {**row, "method": method, "selected_expert": selected_expert}


def _mean_optional(values: Iterable[Any]) -> float | None:
    defined = [float(value) for value in values if value is not None and value != ""]
    if not defined:
        return None
    return sum(defined) / len(defined)


def _mean_for_selection(values: Iterable[Any], selection_metric: str) -> float:
    defined = [_metric_value(value) for value in values]
    finite = [value for value in defined if value != float("-inf")]
    if not finite:
        raise OracleError(f"No defined {selection_metric} values are available for selection")
    return sum(finite) / len(finite)


def _metric_value(value: Any) -> float:
    if value is None or value == "":
        return float("-inf")
    return float(value)


def _parse_optional_float(value: str, path: Path, line_number: int, column: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise OracleError(f"{path}:{line_number} has invalid numeric {column}: {value!r}") from exc


def _parse_int(value: str, path: Path, line_number: int, column: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise OracleError(f"{path}:{line_number} has invalid integer {column}: {value!r}") from exc


__all__ = [
    "OracleError",
    "OracleResult",
    "compute_oracle_outputs",
    "ensure_oracle_output",
    "read_expert_quality_by_run",
    "write_oracle_outputs",
]
