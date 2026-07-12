"""Evaluator-only Stage 3 expert complementarity summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.normroute.routing.oracle import OracleError, ensure_oracle_output, read_expert_quality_by_run
from src.normroute.routing.quality_metrics import read_routing_matrix_long, write_csv


COMPLEMENTARITY_COLUMNS = (
    "evaluator_only",
    "analysis",
    "selection_metric",
    "dataset",
    "category",
    "k_shot",
    "expert",
    "comparison_expert",
    "value",
    "num_runs",
    "num_samples",
)


def compute_complementarity_summary(
    *,
    routing_matrix_long: str | Path,
    expert_quality_by_run: str | Path,
    selection_metric: str = "image_auroc",
) -> list[dict[str, Any]]:
    """Compute evaluator-only complementarity analysis rows."""

    if selection_metric not in {"image_auroc", "image_ap"}:
        raise OracleError("selection_metric must be one of: image_auroc, image_ap")

    quality_rows = read_expert_quality_by_run(expert_quality_by_run)
    routing_rows = read_routing_matrix_long(routing_matrix_long)
    return [
        *_best_expert_by_group(
            quality_rows,
            analysis="best_expert_by_category",
            group_columns=("dataset", "category"),
            selection_metric=selection_metric,
        ),
        *_best_expert_by_group(
            quality_rows,
            analysis="best_expert_by_k_shot",
            group_columns=("dataset", "k_shot"),
            selection_metric=selection_metric,
        ),
        *_expert_score_margin_rows(quality_rows, selection_metric),
        *_expert_disagreement_rate_rows(routing_rows, selection_metric),
    ]


def write_complementarity_summary(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Write evaluator-only complementarity_summary.csv."""

    output_path = Path(output_dir)
    ensure_oracle_output(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "complementarity_summary.csv"
    write_csv(summary_path, list(COMPLEMENTARITY_COLUMNS), rows)
    return summary_path


def _best_expert_by_group(
    quality_rows: list[dict[str, Any]],
    *,
    analysis: str,
    group_columns: tuple[str, ...],
    selection_metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in quality_rows:
        key = tuple(str(row[column]) for column in group_columns)
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        by_expert: dict[str, list[dict[str, Any]]] = {}
        for row in group_rows:
            by_expert.setdefault(str(row["expert"]), []).append(row)
        selected_expert, selected_value = max(
            (
                (
                    expert,
                    _mean_for_selection(row[selection_metric] for row in rows_for_expert),
                )
                for expert, rows_for_expert in by_expert.items()
            ),
            key=lambda item: (item[1], item[0]),
        )
        rows.append(
            _analysis_row(
                analysis=analysis,
                selection_metric=selection_metric,
                key_columns=dict(zip(group_columns, key)),
                expert=selected_expert,
                value=selected_value,
                num_runs=len(by_expert[selected_expert]),
                num_samples=sum(int(row["num_samples"]) for row in by_expert[selected_expert]),
            )
        )
    return rows


def _expert_score_margin_rows(
    quality_rows: list[dict[str, Any]],
    selection_metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in quality_rows:
        key = tuple(str(row[column]) for column in ("dataset", "category", "k_shot"))
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        by_expert: dict[str, list[dict[str, Any]]] = {}
        for row in group_rows:
            by_expert.setdefault(str(row["expert"]), []).append(row)
        ranked = sorted(
            (
                (
                    expert,
                    _mean_for_selection(row[selection_metric] for row in rows_for_expert),
                )
                for expert, rows_for_expert in by_expert.items()
            ),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        if not ranked:
            continue
        best_expert, best_value = ranked[0]
        second_expert = ranked[1][0] if len(ranked) > 1 else ""
        second_value = ranked[1][1] if len(ranked) > 1 else best_value
        rows.append(
            _analysis_row(
                analysis="expert_score_margin",
                selection_metric=selection_metric,
                key_columns=dict(zip(("dataset", "category", "k_shot"), key)),
                expert=best_expert,
                comparison_expert=second_expert,
                value=best_value - second_value,
                num_runs=len(group_rows),
                num_samples=sum(int(row["num_samples"]) for row in group_rows),
            )
        )
    return rows


def _expert_disagreement_rate_rows(
    routing_rows: list[dict[str, str]],
    selection_metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[tuple[str, ...], list[dict[str, str]]]] = {}
    for row in routing_rows:
        group_key = tuple(row[column] for column in ("dataset", "category", "k_shot"))
        sample_key = tuple(
            row[column] for column in ("image_id", "dataset", "category", "k_shot", "seed")
        )
        grouped.setdefault(group_key, {}).setdefault(sample_key, []).append(row)

    rows: list[dict[str, Any]] = []
    for key, samples in sorted(grouped.items()):
        total = len(samples)
        disagreements = 0
        for sample_rows in samples.values():
            decisions = {_decision(row) for row in sample_rows}
            if len(decisions) > 1:
                disagreements += 1
        rows.append(
            _analysis_row(
                analysis="expert_disagreement_rate",
                selection_metric=selection_metric,
                key_columns=dict(zip(("dataset", "category", "k_shot"), key)),
                value=0.0 if total == 0 else disagreements / total,
                num_runs=len({row["seed"] for sample_rows in samples.values() for row in sample_rows}),
                num_samples=total,
            )
        )
    return rows


def _decision(row: dict[str, str]) -> str:
    decision = (row.get("final_decision") or "").strip().lower()
    if decision:
        return decision
    return "anomaly" if float(row["image_score"]) >= 0.5 else "normal"


def _analysis_row(
    *,
    analysis: str,
    selection_metric: str,
    key_columns: dict[str, str],
    value: float,
    expert: str = "",
    comparison_expert: str = "",
    num_runs: int,
    num_samples: int,
) -> dict[str, Any]:
    return {
        "evaluator_only": True,
        "analysis": analysis,
        "selection_metric": selection_metric,
        "dataset": key_columns.get("dataset", ""),
        "category": key_columns.get("category", ""),
        "k_shot": key_columns.get("k_shot", ""),
        "expert": expert,
        "comparison_expert": comparison_expert,
        "value": value,
        "num_runs": num_runs,
        "num_samples": num_samples,
    }


def _mean_for_selection(values: Iterable[Any]) -> float:
    defined = [float(value) for value in values if value is not None and value != ""]
    if not defined:
        raise OracleError("No defined metric values are available for complementarity selection")
    return sum(defined) / len(defined)


__all__ = [
    "compute_complementarity_summary",
    "write_complementarity_summary",
]
