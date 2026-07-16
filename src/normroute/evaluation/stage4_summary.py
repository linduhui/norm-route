"""Evaluator-only full-grid summaries for Stage 4 routing experiments.

This module joins labels only after replay has completed.  Its Oracle is a
run-level evaluator reference: it chooses one expert for an entire
dataset/category/K/seed/support-set run.  It deliberately does not construct
or export a sample-level Oracle policy.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from src.normroute.routing.quality_metrics import compute_average_precision, compute_auroc


STAGE4_SUMMARY_VERSION = "stage4.evaluator_summary.v1"
AGGREGATIONS = ("micro", "macro_category", "macro_run")
FORBIDDEN_REALIZABLE_POLICIES = frozenset(
    {"best_single_expert", "run_level_oracle", "sample_level_oracle"}
)
RUN_COLUMNS = ("dataset", "category", "k_shot", "seed", "support_set_id")
TASK_COLUMNS = ("image_id", *RUN_COLUMNS)
FOLD_METRIC_COLUMNS = (
    "evaluator_only",
    "policy_name",
    "policy_kind",
    "fold",
    "split",
    "aggregation",
    "status",
    "selected_expert",
    "num_samples",
    "num_normal",
    "num_anomaly",
    "num_categories",
    "num_runs",
    "auroc",
    "ap",
    "f1",
    "oracle_auroc",
    "oracle_regret",
    "selection_agreement_with_oracle",
    "estimated_runtime_ms",
    "average_estimated_runtime_ms",
    "tool_calls",
    "average_tool_calls",
    "num_abstained",
    "abstention_rate",
)
POLICY_SUMMARY_COLUMNS = (
    "evaluator_only",
    "policy_name",
    "policy_kind",
    "aggregation",
    "num_folds",
    "expected_num_folds",
    "auroc_mean",
    "auroc_std",
    "ap_mean",
    "ap_std",
    "f1_mean",
    "f1_std",
    "oracle_regret_mean",
    "oracle_regret_std",
    "selection_agreement_with_oracle",
    "selection_agreement_with_oracle_std",
    "estimated_runtime_ms_mean",
    "estimated_runtime_ms_std",
    "average_estimated_runtime_ms_mean",
    "average_estimated_runtime_ms_std",
    "tool_calls_mean",
    "tool_calls_std",
    "average_tool_calls_mean",
    "average_tool_calls_std",
    "abstention_rate_mean",
    "abstention_rate_std",
)
ORACLE_REGRET_COLUMNS = (
    "evaluator_only",
    "policy_name",
    "policy_kind",
    "fold",
    "aggregation",
    "policy_auroc",
    "oracle_auroc",
    "oracle_regret",
    "policy_ap",
    "oracle_ap",
    "ap_regret",
    "policy_f1",
    "oracle_f1",
    "f1_regret",
)
SELECTION_COUNT_COLUMNS = (
    "evaluator_only",
    "policy_name",
    "policy_kind",
    "fold",
    "selected_expert",
    "selection_count",
    "selection_rate",
    "num_runs_with_selection",
    "total_runs",
    "oracle_agreement_count",
    "oracle_agreement_rate",
)
COST_QUALITY_COLUMNS = (
    "evaluator_only",
    "policy_name",
    "policy_kind",
    "fold",
    "aggregation",
    "auroc",
    "ap",
    "f1",
    "oracle_regret",
    "estimated_runtime_ms",
    "average_estimated_runtime_ms",
    "tool_calls",
    "average_tool_calls",
    "abstention_rate",
)
REPORT_SUMMARY_FILENAMES = (
    "stage4_fold_metrics.csv",
    "stage4_policy_summary.csv",
    "stage4_oracle_regret.csv",
    "stage4_selection_counts.csv",
    "stage4_cost_quality.csv",
)
_SELECTED_REQUIRED = (
    "task_id",
    "fold",
    "split",
    "policy_name",
    "selected_expert",
    "image_id",
    "expert_name",
    "dataset",
    "category",
    "support_set_id",
    "k_shot",
    "seed",
    "final_score",
    "final_decision",
    "tool_calls",
    "runtime_ms",
    "status",
)
_FORBIDDEN_SELECTED = frozenset(
    {"label", "mask", "mask_path", "defect_type", "anomaly_type", "ground_truth"}
)


class Stage4SummaryError(ValueError):
    """Raised when a full Stage 4 summary would be ambiguous or incomplete."""


@dataclass(frozen=True)
class Stage4SummaryResult:
    fold_metric_rows: list[dict[str, Any]]
    policy_summary_rows: list[dict[str, Any]]
    oracle_regret_rows: list[dict[str, Any]]
    selection_count_rows: list[dict[str, Any]]
    cost_quality_rows: list[dict[str, Any]]
    failure_cases: list[dict[str, Any]]
    warnings: list[str]
    selected_prediction_paths: tuple[Path, ...]
    evaluator_csv: Path
    expert_predictions_csv: Path | None
    fold_manifest_csv: Path | None
    split: str
    selection_metric: str
    seeds: tuple[int, ...]


def discover_selected_predictions(run_roots: Sequence[str | Path] | str | Path) -> tuple[Path, ...]:
    """Find replay outputs while excluding evaluator/report trees."""

    roots = (run_roots,) if isinstance(run_roots, (str, Path)) else tuple(run_roots)
    if not roots:
        raise Stage4SummaryError("At least one Stage 4 run root is required")
    paths: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root)
        if root.is_file():
            if root.name != "selected_predictions.csv":
                raise Stage4SummaryError(f"Not a selected_predictions.csv file: {root}")
            candidates = (root,)
        elif root.is_dir():
            candidates = tuple(root.rglob("selected_predictions.csv"))
        else:
            raise Stage4SummaryError(f"Stage 4 run root does not exist: {root}")
        for path in candidates:
            lowered = {part.lower() for part in path.parts}
            if lowered.intersection({"evaluation", "evaluator_only", "reports", "oracle"}):
                continue
            paths.add(path)
    if not paths:
        raise Stage4SummaryError("No selected_predictions.csv files were discovered")
    return tuple(sorted(paths, key=lambda path: str(path)))


def summarize_stage4(
    *,
    selected_predictions: Sequence[str | Path] | str | Path,
    evaluator_csv: str | Path,
    expert_predictions_csv: str | Path | None = None,
    fold_manifest_csv: str | Path | None = None,
    split: str = "test",
    selection_metric: str = "auroc",
) -> Stage4SummaryResult:
    """Summarize every policy/fold and compute fold-local evaluator references.

    Selected-prediction files are scanned in bounded, file-at-a-time passes.
    Evaluator references are built one fold at a time, and each policy/fold
    aggregate immediately releases its sample rows.  This keeps a full 50-file
    grid from expanding into hundreds of thousands of simultaneous Python
    dictionaries.
    """

    if split != "test":
        raise Stage4SummaryError("Stage 4 full summaries must use split='test'")
    if selection_metric not in {"auroc", "ap"}:
        raise Stage4SummaryError("selection_metric must be 'auroc' or 'ap'")
    paths = _coerce_paths(selected_predictions)
    labels = _read_evaluator_labels(Path(evaluator_csv))
    manifest_expected = (
        _read_fold_manifest(Path(fold_manifest_csv), split)
        if fold_manifest_csv is not None
        else None
    )

    source_combos: dict[tuple[str, str, str], set[Path]] = {}
    failure_cases: list[dict[str, Any]] = []
    inferred_expected: dict[str, dict[str, tuple[str, ...]]] = {}
    seeds: set[int] = set()
    num_successful_rows = 0
    for path in paths:
        file_rows, file_combos, file_failures = _read_selected_rows(
            (path,), labels, split
        )
        failure_cases.extend(file_failures)
        for combo, combo_paths in file_combos.items():
            combined = source_combos.setdefault(combo, set())
            combined.update(combo_paths)
            if len(combined) > 1:
                raise Stage4SummaryError(
                    f"Multiple selected-prediction files claim policy/fold/split={combo}: "
                    f"{sorted(str(item) for item in combined)}"
                )
        num_successful_rows += len(file_rows)
        for row in file_rows:
            fold = str(row["fold"])
            task_id = str(row["task_id"])
            task_key = _task_key(row)
            prior_task = inferred_expected.setdefault(fold, {}).get(task_id)
            if prior_task is not None and prior_task != task_key:
                raise Stage4SummaryError(
                    f"Policies disagree on task provenance for fold/task={fold}/{task_id}"
                )
            inferred_expected[fold][task_id] = task_key
            seeds.add(int(row["seed"]))
    if num_successful_rows == 0:
        raise Stage4SummaryError("No successful test selected predictions were found")

    policies = sorted({combo[0] for combo in source_combos})
    forbidden = sorted(set(policies).intersection(FORBIDDEN_REALIZABLE_POLICIES))
    if forbidden:
        raise Stage4SummaryError(
            "Evaluator references cannot be supplied as realizable policy outputs: "
            + repr(forbidden)
        )
    folds = sorted({combo[1] for combo in source_combos})
    expected = manifest_expected or inferred_expected
    if not expected:
        raise Stage4SummaryError("No expected test-fold tasks are available")
    unexpected_folds = sorted(set(folds) - set(expected))
    if unexpected_folds:
        raise Stage4SummaryError(f"Selected predictions use folds absent from manifest: {unexpected_folds}")
    folds = sorted(expected)

    warnings: list[str] = []
    for policy in policies:
        for fold in folds:
            if (policy, fold, split) not in source_combos:
                failure_cases.append(
                    _failure("missing_policy_fold", policy, fold, "No selected prediction file/rows")
                )
    _collect_replay_failures(paths, source_combos, failure_cases, warnings)

    expert_path = Path(expert_predictions_csv) if expert_predictions_csv is not None else None
    policy_kinds = {policy: "realizable_policy" for policy in policies}
    policy_kinds.update(
        {
            "best_single_expert": "evaluator_only_reference",
            "run_level_oracle": "evaluator_only_reference",
        }
    )
    fold_metric_rows: list[dict[str, Any]] = []
    metrics_by_combo_aggregation: dict[tuple[str, str, str], dict[str, Any]] = {}
    selection_count_rows: list[dict[str, Any]] = []

    # Build evaluator references one fold at a time.  At most one fold's three
    # expert matrices and two synthetic references are resident simultaneously.
    oracle_expert_by_run: dict[str, dict[tuple[str, ...], str]] = {}
    for fold in folds:
        fold_source: dict[tuple[str, str], dict[str, Any]] = {}
        fold_paths = sorted(
            {
                path
                for combo, combo_paths in source_combos.items()
                if combo[1] == fold
                for path in combo_paths
            },
            key=str,
        )
        for path in fold_paths:
            current, _, _ = _read_selected_rows((path,), labels, split)
            for row in current:
                key = (
                    str(row["task_id"]),
                    _canonical_expert(str(row["selected_expert"])),
                )
                prior = fold_source.get(key)
                if prior is not None and (
                    float(prior["final_score"]) != float(row["final_score"])
                    or str(prior["final_decision"]) != str(row["final_decision"])
                ):
                    raise Stage4SummaryError(
                        f"Policies copied conflicting expert prediction for fold/task/expert={fold}/{key}"
                    )
                fold_source.setdefault(key, row)
        expert_rows = _build_expert_rows(
            policy_rows=list(fold_source.values()),
            expected={fold: expected[fold]},
            labels=labels,
            expert_predictions_csv=expert_path,
        )
        fold_references, fold_oracle = _build_references(
            expert_rows=expert_rows,
            expected={fold: expected[fold]},
            selection_metric=selection_metric,
            warnings=warnings,
        )
        oracle_expert_by_run[fold] = fold_oracle[fold]
        for policy, _, selected_expert, reference_rows in fold_references:
            annotated = _annotate_oracle_agreement(
                reference_rows, oracle_expert_by_run[fold]
            )
            for aggregation in AGGREGATIONS:
                metric = _metric_row(
                    policy=policy,
                    policy_kind=policy_kinds[policy],
                    fold=fold,
                    split=split,
                    aggregation=aggregation,
                    rows=annotated,
                    status="ok",
                    selected_expert=selected_expert,
                    warnings=warnings,
                )
                fold_metric_rows.append(metric)
                metrics_by_combo_aggregation[(policy, fold, aggregation)] = metric
            selection_count_rows.extend(
                _selection_counts_for(
                    policy=policy,
                    fold=fold,
                    rows=reference_rows,
                    policy_kind=policy_kinds[policy],
                    oracle=oracle_expert_by_run[fold],
                )
            )
        del fold_source, expert_rows, fold_references

    # Second pass: aggregate and release one policy/fold file at a time.
    for path in paths:
        current, file_combos, _ = _read_selected_rows((path,), labels, split)
        target_combos = list(file_combos)
        if not target_combos:
            continue
        combo = target_combos[0]
        policy, fold, _ = combo
        if not current:
            continue
        _check_combo_coverage(
            policy=policy,
            fold=fold,
            rows=current,
            expected=expected[fold],
            failure_cases=failure_cases,
        )
        annotated = _annotate_oracle_agreement(current, oracle_expert_by_run[fold])
        for aggregation in AGGREGATIONS:
            metric = _metric_row(
                policy=policy,
                policy_kind=policy_kinds[policy],
                fold=fold,
                split=split,
                aggregation=aggregation,
                rows=annotated,
                status="ok",
                selected_expert="",
                warnings=warnings,
            )
            fold_metric_rows.append(metric)
            metrics_by_combo_aggregation[(policy, fold, aggregation)] = metric
        selection_count_rows.extend(
            _selection_counts_for(
                policy=policy,
                fold=fold,
                rows=current,
                policy_kind=policy_kinds[policy],
                oracle=oracle_expert_by_run[fold],
            )
        )

    incomplete = {
        (str(case.get("policy_name", "")), str(case.get("fold", "")))
        for case in failure_cases
        if case.get("policy_name") and case.get("fold")
    }
    for metric in fold_metric_rows:
        if (str(metric["policy_name"]), str(metric["fold"])) in incomplete:
            metric["status"] = "incomplete"

    oracle_regret_rows: list[dict[str, Any]] = []
    for metric in fold_metric_rows:
        oracle = metrics_by_combo_aggregation[("run_level_oracle", metric["fold"], metric["aggregation"])]
        metric["oracle_auroc"] = oracle["auroc"]
        metric["oracle_regret"] = _difference(oracle["auroc"], metric["auroc"])
        oracle_regret_rows.append(_regret_row(metric, oracle))

    policy_summary_rows = _summarize_folds(fold_metric_rows, len(folds))
    cost_quality_rows = [
        {column: row.get(column) for column in COST_QUALITY_COLUMNS}
        for row in fold_metric_rows
    ]
    return Stage4SummaryResult(
        fold_metric_rows=sorted(
            fold_metric_rows,
            key=lambda row: (str(row["policy_name"]), str(row["fold"]), AGGREGATIONS.index(str(row["aggregation"]))),
        ),
        policy_summary_rows=policy_summary_rows,
        oracle_regret_rows=sorted(
            oracle_regret_rows,
            key=lambda row: (str(row["policy_name"]), str(row["fold"]), AGGREGATIONS.index(str(row["aggregation"]))),
        ),
        selection_count_rows=selection_count_rows,
        cost_quality_rows=sorted(
            cost_quality_rows,
            key=lambda row: (str(row["policy_name"]), str(row["fold"]), AGGREGATIONS.index(str(row["aggregation"]))),
        ),
        failure_cases=failure_cases,
        warnings=sorted(set(warnings)),
        selected_prediction_paths=paths,
        evaluator_csv=Path(evaluator_csv),
        expert_predictions_csv=expert_path,
        fold_manifest_csv=Path(fold_manifest_csv) if fold_manifest_csv is not None else None,
        split=split,
        selection_metric=selection_metric,
        seeds=tuple(sorted(seeds)),
    )


def write_stage4_summary(
    *,
    result: Stage4SummaryResult,
    output_dir: str | Path = "outputs/stage4/evaluation",
    reports_dir: str | Path | None = "reports/stage4",
) -> dict[str, Path]:
    """Write full evaluator artifacts and copy only compact CSV summaries to reports."""

    output = Path(output_dir)
    _ensure_evaluation_path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "fold_metrics": output / "stage4_fold_metrics.csv",
        "policy_summary": output / "stage4_policy_summary.csv",
        "oracle_regret": output / "stage4_oracle_regret.csv",
        "selection_counts": output / "stage4_selection_counts.csv",
        "cost_quality": output / "stage4_cost_quality.csv",
        "failure_cases": output / "stage4_failure_cases.json",
        "config": output / "stage4_summary_config.json",
        "metadata": output / "stage4_summary_metadata.json",
    }
    _write_csv(paths["fold_metrics"], FOLD_METRIC_COLUMNS, result.fold_metric_rows)
    _write_csv(paths["policy_summary"], POLICY_SUMMARY_COLUMNS, result.policy_summary_rows)
    _write_csv(paths["oracle_regret"], ORACLE_REGRET_COLUMNS, result.oracle_regret_rows)
    _write_csv(paths["selection_counts"], SELECTION_COUNT_COLUMNS, result.selection_count_rows)
    _write_csv(paths["cost_quality"], COST_QUALITY_COLUMNS, result.cost_quality_rows)
    _write_json(
        paths["failure_cases"],
        {
            "protocol_version": STAGE4_SUMMARY_VERSION,
            "evaluator_only": True,
            "num_failure_cases": len(result.failure_cases),
            "failure_cases": result.failure_cases,
            "warnings": result.warnings,
        },
    )
    config = {
        "protocol_version": STAGE4_SUMMARY_VERSION,
        "evaluator_only": True,
        "selected_predictions": [str(path) for path in result.selected_prediction_paths],
        "evaluator_csv": str(result.evaluator_csv),
        "expert_predictions_csv": str(result.expert_predictions_csv or ""),
        "fold_manifest_csv": str(result.fold_manifest_csv or ""),
        "split": result.split,
        "selection_metric": result.selection_metric,
        "aggregation_levels": list(AGGREGATIONS),
        "std_ddof": 0,
        "oracle_scope": "one expert per complete test run; evaluator-only reference",
        "sample_level_oracle_exported": False,
    }
    _write_json(paths["config"], config)
    inputs = (*result.selected_prediction_paths, result.evaluator_csv)
    optional_inputs = tuple(
        path for path in (result.expert_predictions_csv, result.fold_manifest_csv) if path is not None
    )
    _write_json(
        paths["metadata"],
        {
            "protocol_version": STAGE4_SUMMARY_VERSION,
            "evaluator_only": True,
            "config": config,
            "seed": None,
            "seeds": list(result.seeds),
            "git_commit": _git_commit(),
            "environment": {
                "python_executable": sys.executable,
                "python_version": sys.version,
                "platform": platform.platform(),
            },
            "input_sha256": {str(path): _sha256(path) for path in (*inputs, *optional_inputs)},
            "num_fold_metric_rows": len(result.fold_metric_rows),
            "num_policy_summary_rows": len(result.policy_summary_rows),
            "num_failure_cases": len(result.failure_cases),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    if reports_dir is not None:
        reports = Path(reports_dir)
        reports.mkdir(parents=True, exist_ok=True)
        for filename in REPORT_SUMMARY_FILENAMES:
            shutil.copyfile(output / filename, reports / filename)
    return paths


def write_stage4_summary_failure(*, output_dir: str | Path, error: Exception) -> Path:
    """Persist a fatal summarization error under the evaluator-only output tree."""

    output = Path(output_dir)
    _ensure_evaluation_path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "stage4_failure_cases.json"
    _write_json(
        path,
        {
            "protocol_version": STAGE4_SUMMARY_VERSION,
            "evaluator_only": True,
            "num_failure_cases": 1,
            "failure_cases": [
                {"failure_type": "summary_error", "error_message": str(error)}
            ],
            "warnings": [],
        },
    )
    return path


def _read_selected_rows(
    paths: tuple[Path, ...], labels: Mapping[str, int], split: str
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], set[Path]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    source_combos: dict[tuple[str, str, str], set[Path]] = {}
    failures: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for path in paths:
        if not path.is_file():
            raise Stage4SummaryError(f"Selected predictions do not exist: {path}")
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            missing = [column for column in _SELECTED_REQUIRED if column not in fields]
            forbidden = sorted(_FORBIDDEN_SELECTED.intersection(fields))
            if missing or forbidden:
                raise Stage4SummaryError(
                    f"{path} has invalid selected-prediction columns; missing={missing}, forbidden={forbidden}"
                )
            file_combos: set[tuple[str, str, str]] = set()
            for line_number, raw in enumerate(reader, start=2):
                clean = {key: (value or "").strip() for key, value in raw.items()}
                empty = [column for column in _SELECTED_REQUIRED if not clean.get(column)]
                if empty:
                    raise Stage4SummaryError(f"{path}:{line_number} has empty values: {empty}")
                combo = (clean["policy_name"], clean["fold"], clean["split"])
                file_combos.add(combo)
                if clean["split"] != split:
                    continue
                key = (*combo, clean["task_id"])
                if key in seen:
                    raise Stage4SummaryError(f"Duplicate policy/fold/split/task selected prediction: {key}")
                seen.add(key)
                if clean["image_id"] not in labels:
                    raise Stage4SummaryError(
                        f"{path}:{line_number} image_id={clean['image_id']!r} is absent from evaluator CSV"
                    )
                if clean["status"].lower() != "ok":
                    failures.append(
                        _failure(
                            "selected_prediction_failure",
                            clean["policy_name"],
                            clean["fold"],
                            clean.get("error_message", "") or f"status={clean['status']}",
                            task_id=clean["task_id"],
                            source_path=str(path),
                        )
                    )
                    continue
                if _canonical_expert(clean["selected_expert"]) != _canonical_expert(clean["expert_name"]):
                    raise Stage4SummaryError(
                        f"{path}:{line_number} selected_expert and expert_name disagree"
                    )
                # Keep only fields consumed by aggregation.  A full grid contains
                # hundreds of thousands of rows, so retaining action strings and
                # artifact paths here needlessly multiplies memory use.  Interning
                # repeated provenance values also shares them across policies.
                row: dict[str, Any] = {
                    "task_id": sys.intern(clean["task_id"]),
                    "fold": sys.intern(clean["fold"]),
                    "split": sys.intern(clean["split"]),
                    "policy_name": sys.intern(clean["policy_name"]),
                    "selected_expert": sys.intern(clean["selected_expert"]),
                    "image_id": sys.intern(clean["image_id"]),
                    "dataset": sys.intern(clean["dataset"]),
                    "category": sys.intern(clean["category"]),
                    "support_set_id": sys.intern(clean["support_set_id"]),
                    "k_shot": _positive_int(clean["k_shot"], path, line_number),
                    "seed": _nonnegative_int(clean["seed"], path, line_number),
                    "final_score": _finite_float(clean["final_score"], path, line_number),
                    "final_decision": sys.intern(clean["final_decision"].lower()),
                    "runtime_ms": _nonnegative_float(clean["runtime_ms"], path, line_number),
                    "tool_calls": _nonnegative_int(clean["tool_calls"], path, line_number),
                    "label": labels[clean["image_id"]],
                }
                _decision_binary(clean["final_decision"], path, line_number)
                rows.append(row)
            for combo in file_combos:
                if combo[2] == split:
                    source_combos.setdefault(combo, set()).add(path)
            target_combos = {combo for combo in file_combos if combo[2] == split}
            if len(target_combos) > 1:
                raise Stage4SummaryError(
                    f"{path} mixes multiple test policy/fold combinations: {sorted(target_combos)}"
                )
    return rows, source_combos, failures


def _read_evaluator_labels(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise Stage4SummaryError(f"Evaluator CSV does not exist: {path}")
    labels: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "image_id" not in fields or "label" not in fields:
            raise Stage4SummaryError(f"{path} must contain image_id and label")
        for line_number, row in enumerate(reader, start=2):
            image_id = (row.get("image_id") or "").strip()
            label = (row.get("label") or "").strip()
            if not image_id or label not in {"0", "0.0", "1", "1.0"}:
                raise Stage4SummaryError(f"{path}:{line_number} has invalid image_id/label")
            parsed = 1 if label in {"1", "1.0"} else 0
            if image_id in labels and labels[image_id] != parsed:
                raise Stage4SummaryError(f"{path} has conflicting label for image_id={image_id!r}")
            if image_id in labels:
                raise Stage4SummaryError(f"{path} has duplicate image_id={image_id!r}")
            labels[image_id] = parsed
    if not labels:
        raise Stage4SummaryError(f"Evaluator CSV is empty: {path}")
    return labels


def _read_fold_manifest(path: Path, split: str) -> dict[str, dict[str, tuple[str, ...]]]:
    if not path.is_file():
        raise Stage4SummaryError(f"Fold manifest does not exist: {path}")
    required = ("fold", "split", "task_id", "sample_id", *RUN_COLUMNS)
    expected: dict[str, dict[str, tuple[str, ...]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [column for column in required if column not in fields]
        forbidden = sorted(_FORBIDDEN_SELECTED.intersection(fields))
        if missing or forbidden:
            raise Stage4SummaryError(f"{path} has invalid manifest columns; missing={missing}, forbidden={forbidden}")
        for line_number, raw in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw.items()}
            if row["split"] != split:
                continue
            values = (row["sample_id"], *(row[column] for column in RUN_COLUMNS))
            if not row["fold"] or not row["task_id"] or any(not value for value in values):
                raise Stage4SummaryError(f"{path}:{line_number} has empty test assignment values")
            fold_rows = expected.setdefault(row["fold"], {})
            if row["task_id"] in fold_rows:
                raise Stage4SummaryError(f"{path} duplicates fold/task_id={row['fold']}/{row['task_id']}")
            fold_rows[row["task_id"]] = values
    return expected


def _check_combo_coverage(
    *,
    policy: str,
    fold: str,
    rows: Sequence[Mapping[str, Any]],
    expected: Mapping[str, tuple[str, ...]],
    failure_cases: list[dict[str, Any]],
) -> None:
    actual = {str(row["task_id"]): _task_key(row) for row in rows}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(task_id for task_id in set(actual).intersection(expected) if actual[task_id] != expected[task_id])
    if missing or extra or mismatched:
        failure_cases.append(
            _failure(
                "coverage_mismatch",
                policy,
                fold,
                f"missing={len(missing)}, extra={len(extra)}, provenance_mismatch={len(mismatched)}",
                missing_task_ids=missing[:20],
                extra_task_ids=extra[:20],
                mismatched_task_ids=mismatched[:20],
            )
        )


def _collect_replay_failures(
    paths: Sequence[Path],
    source_combos: Mapping[tuple[str, str, str], set[Path]],
    failure_cases: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    combo_by_path = {
        path: combo for combo, combo_paths in source_combos.items() for path in combo_paths
    }
    for selected_path in paths:
        combo = combo_by_path.get(selected_path)
        if combo is None:
            continue
        failure_path = selected_path.with_name("failures.json")
        if not failure_path.is_file():
            failure_cases.append(
                _failure(
                    "missing_failure_artifact",
                    combo[0],
                    combo[1],
                    f"Missing required replay failure artifact: {failure_path}",
                    source_path=str(failure_path),
                )
            )
            continue
        try:
            payload = json.loads(failure_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            failure_cases.append(
                _failure("invalid_failure_artifact", combo[0], combo[1], str(exc), source_path=str(failure_path))
            )
            continue
        entries: Any = None
        for key in ("failed_tasks", "failures", "failed_predictions"):
            if key in payload:
                entries = payload[key]
                break
        if entries is None:
            if int(payload.get("num_failed", 0) or 0) != 0:
                failure_cases.append(
                    _failure("unlisted_replay_failures", combo[0], combo[1], "num_failed is nonzero but no failure list is present", source_path=str(failure_path))
                )
            continue
        if not isinstance(entries, list):
            failure_cases.append(
                _failure("invalid_failure_artifact", combo[0], combo[1], "failure list is not an array", source_path=str(failure_path))
            )
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                warnings.append(f"Non-object replay failure in {failure_path}")
                continue
            failure_cases.append(
                {
                    "failure_type": "replay_failure",
                    "policy_name": combo[0],
                    "fold": combo[1],
                    "source_path": str(failure_path),
                    "replay_failure": dict(entry),
                }
            )


def _build_expert_rows(
    *,
    policy_rows: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, tuple[str, ...]]],
    labels: Mapping[str, int],
    expert_predictions_csv: Path | None,
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    observed: dict[str, dict[tuple[str, str], dict[str, Any]]] = {fold: {} for fold in expected}
    for row in policy_rows:
        key = (str(row["task_id"]), _canonical_expert(str(row["selected_expert"])))
        fold_rows = observed[str(row["fold"])]
        prior = fold_rows.get(key)
        if prior is not None and (
            float(prior["final_score"]) != float(row["final_score"])
            or str(prior["final_decision"]) != str(row["final_decision"])
        ):
            raise Stage4SummaryError(f"Policies copied conflicting expert prediction for fold/task/expert={row['fold']}/{key}")
        fold_rows.setdefault(key, row)  # type: ignore[arg-type]
    if expert_predictions_csv is None:
        return observed
    if not expert_predictions_csv.is_file():
        raise Stage4SummaryError(f"Expert predictions CSV does not exist: {expert_predictions_csv}")

    task_lookup: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    for fold, assignments in expected.items():
        for task_id, task_key in assignments.items():
            task_lookup.setdefault(task_key, []).append((fold, task_id))
    with expert_predictions_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        expert_column = "expert_name" if "expert_name" in fields else "expert"
        score_column = "final_score" if "final_score" in fields else "image_score"
        required = (*TASK_COLUMNS, expert_column, score_column, "final_decision", "status")
        missing = [column for column in required if column not in fields]
        if missing:
            raise Stage4SummaryError(f"{expert_predictions_csv} is missing expert columns: {missing}")
        for line_number, raw in enumerate(reader, start=2):
            clean = {key: (value or "").strip() for key, value in raw.items()}
            task_key = tuple(clean.get(column, "") for column in TASK_COLUMNS)
            matches = task_lookup.get(task_key, [])
            if not matches:
                continue
            if clean["status"].lower() != "ok":
                raise Stage4SummaryError(f"{expert_predictions_csv}:{line_number} has failed expert prediction for a test task")
            expert = _canonical_expert(clean[expert_column])
            score = _finite_float(clean[score_column], expert_predictions_csv, line_number)
            _decision_binary(clean["final_decision"], expert_predictions_csv, line_number)
            if clean["image_id"] not in labels:
                raise Stage4SummaryError(f"{expert_predictions_csv}:{line_number} image_id is absent from evaluator CSV")
            if clean.get("label"):
                parsed_label = 1 if clean["label"] in {"1", "1.0"} else 0 if clean["label"] in {"0", "0.0"} else -1
                if parsed_label != labels[clean["image_id"]]:
                    raise Stage4SummaryError(f"{expert_predictions_csv}:{line_number} conflicts with evaluator label")
            for fold, task_id in matches:
                fallback = observed[fold].get((task_id, expert))
                runtime = (
                    _nonnegative_float(clean["runtime_ms"], expert_predictions_csv, line_number)
                    if clean.get("runtime_ms")
                    else (fallback.get("runtime_ms") if fallback else None)
                )
                tool_calls = (
                    _nonnegative_int(clean["tool_calls"], expert_predictions_csv, line_number)
                    if clean.get("tool_calls")
                    else (fallback.get("tool_calls") if fallback else None)
                )
                row = {
                    "task_id": task_id,
                    "fold": fold,
                    "split": "test",
                    "policy_name": "",
                    "selected_expert": clean[expert_column],
                    "expert_name": clean[expert_column],
                    "image_id": clean["image_id"],
                    "dataset": clean["dataset"],
                    "category": clean["category"],
                    "support_set_id": clean["support_set_id"],
                    "k_shot": int(clean["k_shot"]),
                    "seed": int(clean["seed"]),
                    "final_score": score,
                    "final_decision": clean["final_decision"],
                    "runtime_ms": runtime,
                    "tool_calls": tool_calls,
                    "status": "ok",
                    "label": labels[clean["image_id"]],
                    "source_path": str(expert_predictions_csv),
                }
                key = (task_id, expert)
                prior = observed[fold].get(key)
                if prior is not None and (
                    float(prior["final_score"]) != score
                    or str(prior["final_decision"]).lower() != clean["final_decision"].lower()
                ):
                    raise Stage4SummaryError(f"Expert source conflicts with replay for fold/task/expert={fold}/{key}")
                observed[fold][key] = row
    return observed


def _build_references(
    *,
    expert_rows: Mapping[str, Mapping[tuple[str, str], dict[str, Any]]],
    expected: Mapping[str, Mapping[str, tuple[str, ...]]],
    selection_metric: str,
    warnings: list[str],
) -> tuple[list[tuple[str, str, str, list[dict[str, Any]]]], dict[str, dict[tuple[str, ...], str]]]:
    references: list[tuple[str, str, str, list[dict[str, Any]]]] = []
    oracle_by_fold: dict[str, dict[tuple[str, ...], str]] = {}
    for fold in sorted(expected):
        rows = expert_rows[fold]
        experts = sorted({expert for _, expert in rows})
        if not experts:
            raise Stage4SummaryError(f"Fold {fold} has no expert predictions for evaluator references")
        task_ids = set(expected[fold])
        for expert in experts:
            missing = sorted(task_id for task_id in task_ids if (task_id, expert) not in rows)
            if missing:
                raise Stage4SummaryError(
                    f"Fold {fold} cannot compute complete Oracle for expert={expert}; missing {len(missing)} task(s)"
                )
        run_tasks: dict[tuple[str, ...], list[str]] = {}
        for task_id, task_key in expected[fold].items():
            run_tasks.setdefault(task_key[1:], []).append(task_id)
        metric_by_expert_run: dict[tuple[str, tuple[str, ...]], float] = {}
        for expert in experts:
            for run_key, run_task_ids in run_tasks.items():
                quality = _quality([rows[(task_id, expert)] for task_id in run_task_ids])
                value = quality[selection_metric]
                if value is None:
                    raise Stage4SummaryError(
                        f"Fold {fold} run={run_key} has undefined {selection_metric} for expert={expert}"
                    )
                metric_by_expert_run[(expert, run_key)] = float(value)
        best_single = max(
            experts,
            key=lambda expert: (
                sum(metric_by_expert_run[(expert, run)] for run in run_tasks) / len(run_tasks),
                _reverse_lexical_tiebreak(expert),
            ),
        )
        best_rows = [_reference_row(rows[(task_id, best_single)], "best_single_expert") for task_id in sorted(task_ids)]
        references.append(("best_single_expert", fold, str(best_rows[0]["selected_expert"]), best_rows))

        oracle_map: dict[tuple[str, ...], str] = {}
        oracle_rows: list[dict[str, Any]] = []
        for run_key in sorted(run_tasks):
            selected = max(
                experts,
                key=lambda expert: (
                    metric_by_expert_run[(expert, run_key)],
                    _reverse_lexical_tiebreak(expert),
                ),
            )
            oracle_map[run_key] = selected
            oracle_rows.extend(
                _reference_row(rows[(task_id, selected)], "run_level_oracle")
                for task_id in sorted(run_tasks[run_key])
            )
        oracle_by_fold[fold] = oracle_map
        references.append(("run_level_oracle", fold, "", oracle_rows))
    return references, oracle_by_fold


def _reference_row(row: Mapping[str, Any], policy: str) -> dict[str, Any]:
    return {**row, "policy_name": policy, "evaluator_only": True}


def _annotate_oracle_agreement(
    rows: Sequence[Mapping[str, Any]], oracle_by_run: Mapping[tuple[str, ...], str]
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        run = _run_key(row)
        if run not in oracle_by_run:
            raise Stage4SummaryError(f"No run-level Oracle selection for run={run}")
        annotated.append(
            {
                **row,
                "oracle_agreement": _canonical_expert(str(row["selected_expert"])) == oracle_by_run[run],
            }
        )
    return annotated


def _metric_row(
    *,
    policy: str,
    policy_kind: str,
    fold: str,
    split: str,
    aggregation: str,
    rows: Sequence[Mapping[str, Any]],
    status: str,
    selected_expert: str,
    warnings: list[str],
) -> dict[str, Any]:
    groups = _aggregation_groups(rows, aggregation)
    qualities = []
    agreements = []
    for group_key, group_rows in groups:
        quality = _quality(group_rows)
        qualities.append(quality)
        agreements.append(sum(bool(row["oracle_agreement"]) for row in group_rows) / len(group_rows))
        for metric in ("auroc", "ap"):
            if quality[metric] is None:
                warnings.append(
                    f"Undefined {metric} for policy={policy}, fold={fold}, aggregation={aggregation}, group={group_key}"
                )
    auroc = _mean_defined(quality["auroc"] for quality in qualities)
    ap = _mean_defined(quality["ap"] for quality in qualities)
    f1 = _mean_defined(quality["f1"] for quality in qualities)
    runtimes = [row.get("runtime_ms") for row in rows]
    calls = [row.get("tool_calls") for row in rows]
    runtime_total = sum(float(value) for value in runtimes) if all(value is not None for value in runtimes) else None
    calls_total = sum(int(value) for value in calls) if all(value is not None for value in calls) else None
    num_samples = len(rows)
    num_abstained = sum(str(row["final_decision"]).strip().lower() == "abstain" for row in rows)
    return {
        "evaluator_only": True,
        "policy_name": policy,
        "policy_kind": policy_kind,
        "fold": fold,
        "split": split,
        "aggregation": aggregation,
        "status": status,
        "selected_expert": selected_expert,
        "num_samples": num_samples,
        "num_normal": sum(int(row["label"]) == 0 for row in rows),
        "num_anomaly": sum(int(row["label"]) == 1 for row in rows),
        "num_categories": len({_category_key(row) for row in rows}),
        "num_runs": len({_run_key(row) for row in rows}),
        "auroc": auroc,
        "ap": ap,
        "f1": f1,
        "oracle_auroc": None,
        "oracle_regret": None,
        "selection_agreement_with_oracle": sum(agreements) / len(agreements),
        "estimated_runtime_ms": runtime_total,
        "average_estimated_runtime_ms": runtime_total / num_samples if runtime_total is not None else None,
        "tool_calls": calls_total,
        "average_tool_calls": calls_total / num_samples if calls_total is not None else None,
        "num_abstained": num_abstained,
        "abstention_rate": num_abstained / num_samples,
    }


def _aggregation_groups(
    rows: Sequence[Mapping[str, Any]], aggregation: str
) -> list[tuple[Any, list[Mapping[str, Any]]]]:
    if aggregation == "micro":
        return [("all", list(rows))]
    columns = ("dataset", "category") if aggregation == "macro_category" else RUN_COLUMNS
    grouped = _group(rows, columns)
    return [(key, grouped[key]) for key in sorted(grouped)]


def _quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["final_score"]) for row in rows]
    decisions = [_decision_binary(str(row["final_decision"])) for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    return {
        "auroc": compute_auroc(labels, scores) if positives and negatives else None,
        "ap": compute_average_precision(labels, scores) if positives and negatives else None,
        "f1": _binary_f1(labels, decisions),
    }


def _binary_f1(labels: Sequence[int], decisions: Sequence[int]) -> float:
    tp = sum(label == predicted == 1 for label, predicted in zip(labels, decisions))
    fp = sum(label == 0 and predicted == 1 for label, predicted in zip(labels, decisions))
    fn = sum(label == 1 and predicted == 0 for label, predicted in zip(labels, decisions))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2 * tp) / denominator


def _regret_row(metric: Mapping[str, Any], oracle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evaluator_only": True,
        "policy_name": metric["policy_name"],
        "policy_kind": metric["policy_kind"],
        "fold": metric["fold"],
        "aggregation": metric["aggregation"],
        "policy_auroc": metric["auroc"],
        "oracle_auroc": oracle["auroc"],
        "oracle_regret": _difference(oracle["auroc"], metric["auroc"]),
        "policy_ap": metric["ap"],
        "oracle_ap": oracle["ap"],
        "ap_regret": _difference(oracle["ap"], metric["ap"]),
        "policy_f1": metric["f1"],
        "oracle_f1": oracle["f1"],
        "f1_regret": _difference(oracle["f1"], metric["f1"]),
    }


def _summarize_folds(rows: Sequence[Mapping[str, Any]], expected_folds: int) -> list[dict[str, Any]]:
    grouped = _group(rows, ("policy_name", "policy_kind", "aggregation"))
    summaries: list[dict[str, Any]] = []
    for (policy, kind, aggregation), group_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "evaluator_only": True,
            "policy_name": policy,
            "policy_kind": kind,
            "aggregation": aggregation,
            "num_folds": len(group_rows),
            "expected_num_folds": expected_folds,
        }
        for source, prefix in (
            ("auroc", "auroc"),
            ("ap", "ap"),
            ("f1", "f1"),
            ("oracle_regret", "oracle_regret"),
            ("selection_agreement_with_oracle", "selection_agreement_with_oracle"),
            ("estimated_runtime_ms", "estimated_runtime_ms"),
            ("average_estimated_runtime_ms", "average_estimated_runtime_ms"),
            ("tool_calls", "tool_calls"),
            ("average_tool_calls", "average_tool_calls"),
            ("abstention_rate", "abstention_rate"),
        ):
            mean, std = _mean_std(row.get(source) for row in group_rows)
            summary[f"{prefix}_mean"] = mean
            summary[f"{prefix}_std"] = std
        # The requested agreement name is the across-fold mean without a redundant suffix.
        summary["selection_agreement_with_oracle"] = summary.pop(
            "selection_agreement_with_oracle_mean"
        )
        summaries.append(summary)
    return summaries


def _selection_counts_for(
    *,
    policy: str,
    fold: str,
    rows: Sequence[Mapping[str, Any]],
    policy_kind: str,
    oracle: Mapping[tuple[str, ...], str],
) -> list[dict[str, Any]]:
    total = len(rows)
    total_runs = len({_run_key(row) for row in rows})
    by_expert: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_expert.setdefault(str(row["selected_expert"]), []).append(row)
    output: list[dict[str, Any]] = []
    for expert, expert_rows in sorted(by_expert.items()):
        agreement = sum(
            _canonical_expert(expert) == oracle[_run_key(row)] for row in expert_rows
        )
        output.append(
            {
                "evaluator_only": True,
                "policy_name": policy,
                "policy_kind": policy_kind,
                "fold": fold,
                "selected_expert": expert,
                "selection_count": len(expert_rows),
                "selection_rate": len(expert_rows) / total,
                "num_runs_with_selection": len({_run_key(row) for row in expert_rows}),
                "total_runs": total_runs,
                "oracle_agreement_count": agreement,
                "oracle_agreement_rate": agreement / len(expert_rows),
            }
        )
    return output


def _failure(failure_type: str, policy: str, fold: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "failure_type": failure_type,
        "policy_name": policy,
        "fold": fold,
        "error_message": message,
        **extra,
    }


def _group(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[column]) for column in columns)
        grouped.setdefault(key, []).append(row)
    return grouped


def _task_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in TASK_COLUMNS)


def _run_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in RUN_COLUMNS)


def _category_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["category"])


def _canonical_expert(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", value.lower())
    if not normalized:
        raise Stage4SummaryError("Expert name must be non-empty")
    return normalized


def _reverse_lexical_tiebreak(value: str) -> tuple[int, ...]:
    # max(..., key=...) with this component chooses lexical ascending on ties.
    return tuple(-ord(character) for character in value)


def _decision_binary(value: str, path: Path | None = None, line_number: int | None = None) -> int:
    normalized = value.strip().lower()
    if normalized in {"anomaly", "anomalous", "1", "true", "stop_anomaly"}:
        return 1
    if normalized in {"normal", "good", "0", "false", "stop_normal", "abstain"}:
        return 0
    location = f"{path}:{line_number} " if path is not None else ""
    raise Stage4SummaryError(f"{location}has unsupported final_decision={value!r}")


def _finite_float(value: Any, path: Path, line_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage4SummaryError(f"{path}:{line_number} has invalid numeric value={value!r}") from exc
    if not math.isfinite(parsed):
        raise Stage4SummaryError(f"{path}:{line_number} has non-finite numeric value={value!r}")
    return parsed


def _nonnegative_float(value: Any, path: Path, line_number: int) -> float:
    parsed = _finite_float(value, path, line_number)
    if parsed < 0:
        raise Stage4SummaryError(f"{path}:{line_number} has negative numeric value={value!r}")
    return parsed


def _nonnegative_int(value: Any, path: Path, line_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Stage4SummaryError(f"{path}:{line_number} has invalid integer value={value!r}") from exc
    if parsed < 0:
        raise Stage4SummaryError(f"{path}:{line_number} has negative integer value={value!r}")
    return parsed


def _positive_int(value: Any, path: Path, line_number: int) -> int:
    parsed = _nonnegative_int(value, path, line_number)
    if parsed < 1:
        raise Stage4SummaryError(f"{path}:{line_number} has non-positive integer value={value!r}")
    return parsed


def _mean_defined(values: Iterable[Any]) -> float | None:
    defined = [float(value) for value in values if value is not None and value != ""]
    return sum(defined) / len(defined) if defined else None


def _mean_std(values: Iterable[Any]) -> tuple[float | None, float | None]:
    defined = [float(value) for value in values if value is not None and value != ""]
    if not defined:
        return None, None
    return statistics.fmean(defined), statistics.pstdev(defined)


def _difference(left: Any, right: Any) -> float | None:
    if left is None or right is None or left == "" or right == "":
        return None
    return float(left) - float(right)


def _coerce_paths(value: Sequence[str | Path] | str | Path) -> tuple[Path, ...]:
    paths = (Path(value),) if isinstance(value, (str, Path)) else tuple(Path(path) for path in value)
    if not paths:
        raise Stage4SummaryError("At least one selected_predictions.csv is required")
    if len(paths) != len(set(paths)):
        raise Stage4SummaryError("Duplicate selected_predictions.csv paths were provided")
    return paths


def _ensure_evaluation_path(path: Path) -> None:
    parts = [part.lower() for part in path.parts]
    if not any(parts[index : index + 3] == ["outputs", "stage4", "evaluation"] for index in range(max(len(parts) - 2, 0))):
        raise Stage4SummaryError(
            "Stage 4 evaluator-only summaries must be written under outputs/stage4/evaluation"
        )


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format(row.get(column)) for column in columns})


def _format(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


__all__ = [
    "AGGREGATIONS",
    "REPORT_SUMMARY_FILENAMES",
    "Stage4SummaryError",
    "Stage4SummaryResult",
    "discover_selected_predictions",
    "summarize_stage4",
    "write_stage4_summary",
    "write_stage4_summary_failure",
]
