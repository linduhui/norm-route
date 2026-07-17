"""Validate the frozen Stage 4 routing, evaluation, and live-smoke artifacts.

The validator is deliberately read-only.  It checks the Agent-visible boundary,
the seed-CV split, learned-policy features, one-expert routing, one-call budget,
selected-prediction provenance, failure counts, and evaluator ordering.  It does
not rerun experts or recompute experiment results.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from ..agent.protocol import (
    CANDIDATE_EXPERTS,
    Stage4ProtocolError,
    validate_pre_route_task,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_VERSION = "stage4.gate.v1"
SPLITS = ("train", "val", "test")
REQUIRED_RUN_FILES = (
    "route_decisions.csv",
    "selected_predictions.csv",
    "failures.json",
    "run_metadata.json",
    "budget_summary.json",
)
FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "seed",
        "support_set_id",
        "label",
        "labels",
        "ground_truth",
        "expert_score",
        "expert_scores",
        "patchcore_score",
        "winclip_score",
        "anomalydino_score",
    }
)
CANONICAL_CANDIDATE_EXPERTS = frozenset(
    re.sub(r"[^a-z0-9]+", "", name.lower()) for name in CANDIDATE_EXPERTS
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-root", default="outputs/stage4")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--evaluation-root", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--fold-manifest", default=None)
    parser.add_argument("--policy-root", default=None)
    parser.add_argument("--live-root", default=None)
    parser.add_argument("--output", default="reports/stage4/stage4_gate.json")
    parser.add_argument(
        "--no-live-smoke",
        action="store_true",
        help="Do not require a live smoke run (intended only for isolated fixtures).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = inspect_stage4_outputs(
        stage4_root=Path(args.stage4_root),
        runs_root=Path(args.runs_root) if args.runs_root else None,
        evaluation_root=Path(args.evaluation_root) if args.evaluation_root else None,
        tasks_path=Path(args.tasks) if args.tasks else None,
        fold_manifest_path=Path(args.fold_manifest) if args.fold_manifest else None,
        policy_root=Path(args.policy_root) if args.policy_root else None,
        live_root=Path(args.live_root) if args.live_root else None,
        require_live=not args.no_live_smoke,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for check in result["checks"]:
        print(f"{check['status']}: {check['name']} - {check['summary']}")
    print(f"Wrote {output}")
    if result["status"] != "PASS":
        raise SystemExit(1)


def validate_stage4_outputs(**kwargs: Any) -> list[str]:
    """Return all Stage 4 validation errors.

    ``inspect_stage4_outputs`` exposes the richer, report-ready result.  This
    compatibility helper follows the earlier Stage validators' list-of-errors
    API and is convenient for tests and callers that only need pass/fail.
    """

    return list(inspect_stage4_outputs(**kwargs)["errors"])


def inspect_stage4_outputs(
    *,
    stage4_root: Path = Path("outputs/stage4"),
    runs_root: Path | None = None,
    evaluation_root: Path | None = None,
    tasks_path: Path | None = None,
    fold_manifest_path: Path | None = None,
    policy_root: Path | None = None,
    live_root: Path | None = None,
    require_live: bool = True,
) -> dict[str, Any]:
    """Return a structured Stage 4 gate result without changing experiment data."""

    stage4_root = Path(stage4_root)
    tasks_path = Path(tasks_path) if tasks_path else stage4_root / "tasks" / "pre_route_tasks.jsonl"
    fold_manifest_path = (
        Path(fold_manifest_path)
        if fold_manifest_path
        else stage4_root / "splits" / "fold_manifest.csv"
    )
    runs_root = Path(runs_root) if runs_root else _discover_latest_grid(stage4_root / "runs")
    evaluation_root = (
        Path(evaluation_root)
        if evaluation_root
        else _discover_evaluation_root(stage4_root / "evaluation", runs_root)
    )
    policy_root = (
        Path(policy_root)
        if policy_root
        else _policy_root_from_grid(runs_root, stage4_root / "policies")
    )
    live_root = Path(live_root) if live_root else stage4_root / "live"

    checks = [
        _check_pre_route_tasks(tasks_path),
        _check_policy_feature_manifests(policy_root),
        _check_split_isolation(fold_manifest_path, stage4_root / "splits" / "split_audit.json"),
    ]
    run_checks, run_context = _check_run_tree(runs_root)
    checks.extend(run_checks)
    evaluation_checks = _check_evaluation_tree(
        evaluation_root=evaluation_root,
        run_context=run_context,
    )
    checks.extend(evaluation_checks)
    checks.append(
        _check_live_smoke(live_root) if require_live else _skip_check("live_smoke", "not required")
    )

    errors = [
        f"{check['name']}: {error}"
        for check in checks
        for error in check.get("errors", [])
        if check["status"] == "FAIL"
    ]
    resolved_inputs = {
        "stage4_root": str(stage4_root),
        "tasks_path": str(tasks_path),
        "fold_manifest_path": str(fold_manifest_path),
        "runs_root": str(runs_root) if runs_root else "",
        "policy_root": str(policy_root) if policy_root else "",
        "evaluation_root": str(evaluation_root) if evaluation_root else "",
        "live_root": str(live_root),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "generated_at_utc": _utc_now(),
        "git_commit": _git_commit(),
        "environment": _environment(),
        "config": {
            "require_live": require_live,
            **resolved_inputs,
        },
        "input_sha256": _input_hashes(
            [
                tasks_path,
                fold_manifest_path,
                stage4_root / "splits" / "split_audit.json",
                *(path for path in (runs_root, evaluation_root, policy_root) if path is not None),
            ]
        ),
        "checks": checks,
        "num_checks": len(checks),
        "num_passed": sum(check["status"] == "PASS" for check in checks),
        "num_failed": sum(check["status"] == "FAIL" for check in checks),
        "num_skipped": sum(check["status"] == "SKIP" for check in checks),
        "errors": errors,
    }


def _check_pre_route_tasks(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    task_ids: set[str] = set()
    rows = 0
    if not path.is_file():
        return _fail_check("pre_route_tasks_no_forbidden_fields", f"Missing {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    errors.append(f"{path}:{line_number} is blank; failed rows must not be skipped")
                    continue
                rows += 1
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number} is invalid JSON: {exc}")
                    continue
                if not isinstance(payload, Mapping):
                    errors.append(f"{path}:{line_number} must be a JSON object")
                    continue
                try:
                    validate_pre_route_task(payload)
                except (Stage4ProtocolError, TypeError, ValueError) as exc:
                    errors.append(f"{path}:{line_number} violates the frozen schema: {exc}")
                    continue
                task_id = str(payload["task_id"])
                if task_id in task_ids:
                    errors.append(f"{path}:{line_number} duplicates task_id={task_id!r}")
                task_ids.add(task_id)
    except OSError as exc:
        errors.append(f"Could not read {path}: {exc}")
    if rows == 0:
        errors.append(f"{path} contains no tasks")
    return _make_check(
        "pre_route_tasks_no_forbidden_fields",
        errors,
        f"validated {rows} pre-route tasks",
        {"path": str(path), "num_tasks": rows, "num_unique_task_ids": len(task_ids)},
    )


def _check_policy_feature_manifests(policy_root: Path | None) -> dict[str, Any]:
    if policy_root is None or not policy_root.exists():
        return _fail_check("policy_feature_manifest_no_leakage", f"Missing policy root {policy_root}")
    manifests = sorted(policy_root.rglob("feature_manifest.json"))
    if not manifests:
        return _fail_check(
            "policy_feature_manifest_no_leakage",
            f"No feature_manifest.json found under {policy_root}",
        )
    errors: list[str] = []
    for path in manifests:
        payload = _read_json(path, errors)
        if payload is None:
            continue
        for field in ("raw_feature_allowlist", "encoded_feature_names"):
            values = payload.get(field)
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                errors.append(f"{path} must contain a string list at {field}")
                continue
            forbidden = sorted(value for value in values if _forbidden_feature_name(value))
            if forbidden:
                errors.append(f"{path} {field} contains forbidden policy features: {forbidden}")
        scaling = payload.get("numeric_scaling", {})
        if not isinstance(scaling, Mapping):
            errors.append(f"{path} numeric_scaling must be an object")
        else:
            forbidden_scaling = sorted(
                str(name) for name in scaling if _forbidden_feature_name(str(name))
            )
            if forbidden_scaling:
                errors.append(
                    f"{path} numeric_scaling contains forbidden policy features: {forbidden_scaling}"
                )
    return _make_check(
        "policy_feature_manifest_no_leakage",
        errors,
        f"validated {len(manifests)} learned-policy feature manifests",
        {"policy_root": str(policy_root), "num_feature_manifests": len(manifests)},
    )


def _check_split_isolation(manifest_path: Path, audit_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    rows = _read_csv(manifest_path, errors)
    if rows is None:
        return _make_check("train_val_test_isolation", errors, "split manifest unavailable", {})
    required = {"fold", "split", "task_id", "seed"}
    if rows:
        missing = sorted(required - set(rows[0]))
        if missing:
            errors.append(f"{manifest_path} is missing columns: {missing}")
    elif not errors:
        errors.append(f"{manifest_path} contains no split rows")

    seeds: dict[str, dict[str, set[str]]] = {}
    assignment_counts: dict[tuple[str, str], int] = {}
    for row_number, row in enumerate(rows, start=2):
        fold = row.get("fold", "").strip()
        split = row.get("split", "").strip()
        task_id = row.get("task_id", "").strip()
        seed = row.get("seed", "").strip()
        if not fold or split not in SPLITS or not task_id or not seed:
            errors.append(
                f"{manifest_path}:{row_number} has invalid fold/split/task_id/seed values"
            )
            continue
        seeds.setdefault(fold, {name: set() for name in SPLITS})[split].add(seed)
        assignment_counts[(fold, task_id)] = assignment_counts.get((fold, task_id), 0) + 1
    for fold, split_seeds in sorted(seeds.items()):
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sorted(split_seeds[left].intersection(split_seeds[right]))
            if overlap:
                errors.append(f"{fold} {left}/{right} seed overlap: {overlap}")
        missing_splits = [split for split in SPLITS if not split_seeds[split]]
        if missing_splits:
            errors.append(f"{fold} has empty splits: {missing_splits}")
    duplicates = sorted(key for key, count in assignment_counts.items() if count != 1)
    if duplicates:
        errors.append(f"Task assignments are not unique within fold; examples={duplicates[:5]}")

    if audit_path.is_file():
        audit = _read_json(audit_path, errors)
        if audit is not None:
            if audit.get("all_folds_valid") is not True:
                errors.append(f"{audit_path} all_folds_valid is not true")
            configured_folds = audit.get("config", {}).get("folds", {})
            if isinstance(configured_folds, Mapping) and set(seeds) != set(configured_folds):
                errors.append(
                    f"Manifest folds {sorted(seeds)} do not match configured folds "
                    f"{sorted(configured_folds)}"
                )
    stats = {
        "manifest_path": str(manifest_path),
        "num_rows": len(rows),
        "num_folds": len(seeds),
        "fold_seed_sets": {
            fold: {split: sorted(values, key=_sortable_int) for split, values in split_seeds.items()}
            for fold, split_seeds in sorted(seeds.items())
        },
    }
    return _make_check(
        "train_val_test_isolation",
        errors,
        f"validated {len(rows)} assignments across {len(seeds)} folds",
        stats,
    )


def _check_run_tree(runs_root: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    categories: dict[str, list[str]] = {
        "route_decisions_one_expert_per_run": [],
        "tool_calls_at_most_one": [],
        "selected_predictions_match_selected_expert": [],
        "failures_match_metrics": [],
        "reproducibility_artifacts": [],
    }
    stats = {
        "runs_root": str(runs_root) if runs_root else "",
        "num_run_directories": 0,
        "num_routing_tasks": 0,
        "num_route_decisions": 0,
        "num_selected_predictions": 0,
        "num_failed_tasks": 0,
        "run_completed_at_utc": {},
        "selected_prediction_paths": [],
    }
    if runs_root is None or not runs_root.exists():
        message = f"Missing Stage 4 runs root {runs_root}"
        for errors in categories.values():
            errors.append(message)
        return _run_checks(categories, stats), stats

    run_dirs = sorted({path.parent for path in runs_root.rglob("route_decisions.csv")})
    if not run_dirs:
        message = f"No route_decisions.csv found under {runs_root}"
        for errors in categories.values():
            errors.append(message)
        return _run_checks(categories, stats), stats
    stats["num_run_directories"] = len(run_dirs)

    for run_dir in run_dirs:
        relative = _relative_display(run_dir, runs_root)
        missing_files = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
        if missing_files:
            categories["reproducibility_artifacts"].append(
                f"{relative} is missing required artifacts: {missing_files}"
            )
        route_errors: list[str] = []
        decisions = _read_csv(run_dir / "route_decisions.csv", route_errors) or []
        predictions = _read_csv(run_dir / "selected_predictions.csv", route_errors) or []
        if route_errors:
            categories["route_decisions_one_expert_per_run"].extend(
                f"{relative}: {error}" for error in route_errors
            )
        stats["num_route_decisions"] += len(decisions)
        stats["num_selected_predictions"] += len(predictions)
        stats["selected_prediction_paths"].append(str(run_dir / "selected_predictions.csv"))

        decision_by_task: dict[str, list[dict[str, str]]] = {}
        for row_number, row in enumerate(decisions, start=2):
            task_id = row.get("task_id", "").strip()
            expert = row.get("selected_expert", "").strip()
            if not task_id or not expert:
                categories["route_decisions_one_expert_per_run"].append(
                    f"{relative}/route_decisions.csv:{row_number} lacks task_id or selected_expert"
                )
                continue
            decision_by_task.setdefault(task_id, []).append(row)
            if _canonical_expert(expert) not in CANONICAL_CANDIDATE_EXPERTS:
                categories["route_decisions_one_expert_per_run"].append(
                    f"{relative}/route_decisions.csv:{row_number} task_id={task_id!r} "
                    f"selects invalid or multiple expert value {expert!r}; expected exactly "
                    f"one of {list(CANDIDATE_EXPERTS)!r}"
                )
            _check_tool_calls_value(
                row.get("tool_calls"),
                f"{relative}/route_decisions.csv:{row_number}",
                categories["tool_calls_at_most_one"],
            )
        stats["num_routing_tasks"] += len(decision_by_task)
        duplicate_tasks = sorted(task for task, values in decision_by_task.items() if len(values) != 1)
        if duplicate_tasks:
            categories["route_decisions_one_expert_per_run"].append(
                f"{relative} has task_ids with != 1 route decision; examples={duplicate_tasks[:5]}"
            )

        predictions_by_task: dict[str, list[dict[str, str]]] = {}
        for row_number, row in enumerate(predictions, start=2):
            task_id = row.get("task_id", "").strip()
            predictions_by_task.setdefault(task_id, []).append(row)
            _check_tool_calls_value(
                row.get("tool_calls"),
                f"{relative}/selected_predictions.csv:{row_number}",
                categories["tool_calls_at_most_one"],
            )
            decisions_for_task = decision_by_task.get(task_id, [])
            if len(decisions_for_task) != 1:
                continue
            selected = _canonical_expert(decisions_for_task[0].get("selected_expert", ""))
            copied_selected = _canonical_expert(row.get("selected_expert", ""))
            source_expert = _canonical_expert(row.get("expert_name", ""))
            if selected != copied_selected or selected != source_expert:
                categories["selected_predictions_match_selected_expert"].append(
                    f"{relative}/selected_predictions.csv:{row_number} task_id={task_id!r} "
                    f"has decision={selected!r}, selected_expert={copied_selected!r}, "
                    f"expert_name={source_expert!r}"
                )
        prediction_duplicates = sorted(
            task for task, values in predictions_by_task.items() if len(values) != 1
        )
        missing_predictions = sorted(set(decision_by_task) - set(predictions_by_task))
        extra_predictions = sorted(set(predictions_by_task) - set(decision_by_task))
        if prediction_duplicates or missing_predictions or extra_predictions:
            categories["selected_predictions_match_selected_expert"].append(
                f"{relative} prediction alignment failed: duplicates={prediction_duplicates[:5]}, "
                f"missing={missing_predictions[:5]}, extra={extra_predictions[:5]}"
            )

        failures = _read_json(run_dir / "failures.json", categories["failures_match_metrics"])
        metadata = _read_json(
            run_dir / "run_metadata.json", categories["reproducibility_artifacts"]
        )
        budget = _read_json(
            run_dir / "budget_summary.json", categories["failures_match_metrics"]
        )
        if failures is not None:
            failed_tasks = failures.get("failed_tasks")
            if not isinstance(failed_tasks, list):
                categories["failures_match_metrics"].append(
                    f"{relative}/failures.json must contain failed_tasks list"
                )
                failed_tasks = []
            actual_failed = len(failed_tasks) + (1 if failures.get("grid_failure") else 0)
            stats["num_failed_tasks"] += actual_failed
            reported_failed = _optional_int(failures.get("num_failed"))
            if reported_failed != actual_failed:
                categories["failures_match_metrics"].append(
                    f"{relative}/failures.json reports num_failed={reported_failed}, "
                    f"but records {actual_failed} failures"
                )
            if metadata is not None and _optional_int(metadata.get("num_failures")) != actual_failed:
                categories["failures_match_metrics"].append(
                    f"{relative}/run_metadata.json num_failures does not match failures.json"
                )
        if metadata is not None:
            for field in ("config", "seed", "git_commit", "environment"):
                if field not in metadata or metadata[field] in (None, "", {}):
                    categories["reproducibility_artifacts"].append(
                        f"{relative}/run_metadata.json lacks non-empty {field}"
                    )
            for field, actual in (
                ("num_decisions", len(decisions)),
                ("num_selected_predictions", len(predictions)),
            ):
                if _optional_int(metadata.get(field)) != actual:
                    categories["failures_match_metrics"].append(
                        f"{relative}/run_metadata.json {field} does not match CSV count {actual}"
                    )
            completed = _parse_timestamp(metadata.get("completed_at_utc"))
            if completed is None:
                categories["reproducibility_artifacts"].append(
                    f"{relative}/run_metadata.json lacks valid completed_at_utc"
                )
            else:
                stats["run_completed_at_utc"][str(run_dir / "selected_predictions.csv")] = (
                    completed.isoformat()
                )
        if budget is not None:
            if _optional_int(budget.get("num_decisions")) not in (None, len(decisions)):
                categories["failures_match_metrics"].append(
                    f"{relative}/budget_summary.json num_decisions does not match {len(decisions)}"
                )
            if _optional_int(budget.get("num_selected_predictions")) not in (
                None,
                len(predictions),
            ):
                categories["failures_match_metrics"].append(
                    f"{relative}/budget_summary.json num_selected_predictions does not match "
                    f"{len(predictions)}"
                )
            if budget.get("within_budget") is not True:
                categories["tool_calls_at_most_one"].append(
                    f"{relative}/budget_summary.json within_budget is not true"
                )

    return _run_checks(categories, stats), stats


def _run_checks(categories: Mapping[str, list[str]], stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries = {
        "route_decisions_one_expert_per_run": (
            f"validated exactly one expert decision for {stats['num_routing_tasks']} "
            f"routing tasks across {stats['num_run_directories']} run directories"
        ),
        "tool_calls_at_most_one": (
            f"validated tool_calls <= 1 across {stats['num_route_decisions']} decisions and "
            f"{stats['num_selected_predictions']} predictions"
        ),
        "selected_predictions_match_selected_expert": (
            f"aligned {stats['num_selected_predictions']} selected predictions to route decisions"
        ),
        "failures_match_metrics": (
            f"reconciled {stats['num_failed_tasks']} recorded failures with run counts"
        ),
        "reproducibility_artifacts": (
            f"validated required artifacts and provenance for {stats['num_run_directories']} runs"
        ),
    }
    public_stats = {key: value for key, value in stats.items() if key != "run_completed_at_utc"}
    return [
        _make_check(name, errors, summaries[name], dict(public_stats))
        for name, errors in categories.items()
    ]


def _check_evaluation_tree(
    *, evaluation_root: Path | None, run_context: Mapping[str, Any]
) -> list[dict[str, Any]]:
    count_errors: list[str] = []
    order_errors: list[str] = []
    stats = {
        "evaluation_root": str(evaluation_root) if evaluation_root else "",
        "num_metric_rows": 0,
        "num_evaluated_samples": 0,
        "num_evaluation_failures": 0,
        "num_summary_failure_cases": 0,
    }
    if evaluation_root is None or not evaluation_root.exists():
        message = f"Missing Stage 4 evaluation root {evaluation_root}"
        return [
            _make_check("failures_match_evaluation_metrics", [message], "evaluation unavailable", stats),
            _make_check("evaluator_after_route_decision", [message], "evaluation unavailable", stats),
        ]

    selected_dir = evaluation_root / "selected_predictions"
    metrics_path = selected_dir / "stage4_metrics.csv"
    metadata_path = selected_dir / "evaluation_metadata.json"
    failures_path = selected_dir / "evaluation_failures.json"
    config_path = selected_dir / "evaluation_config.json"
    metrics = _read_csv(metrics_path, count_errors) or []
    metadata = _read_json(metadata_path, count_errors)
    failures = _read_json(failures_path, count_errors)
    config = _read_json(config_path, order_errors)
    stats["num_metric_rows"] = len(metrics)
    for row_number, row in enumerate(metrics, start=2):
        count = _optional_int(row.get("num_samples"))
        if count is None or count < 0:
            count_errors.append(f"{metrics_path}:{row_number} has invalid num_samples")
        else:
            stats["num_evaluated_samples"] += count
    if metadata is not None:
        if _optional_int(metadata.get("num_metric_rows")) != stats["num_metric_rows"]:
            count_errors.append(f"{metadata_path} num_metric_rows does not match metrics CSV")
        if _optional_int(metadata.get("num_evaluated_samples")) != stats["num_evaluated_samples"]:
            count_errors.append(f"{metadata_path} num_evaluated_samples does not match metrics CSV")
    if failures is not None:
        failure_rows = failures.get("failures")
        if not isinstance(failure_rows, list):
            count_errors.append(f"{failures_path} must contain failures list")
        else:
            stats["num_evaluation_failures"] = len(failure_rows)

    summary_failures_path = evaluation_root / "stage4_failure_cases.json"
    summary_metadata_path = evaluation_root / "stage4_summary_metadata.json"
    summary_failures = _read_json(summary_failures_path, count_errors)
    summary_metadata = _read_json(summary_metadata_path, count_errors)
    if summary_failures is not None:
        cases = summary_failures.get("failure_cases")
        if not isinstance(cases, list):
            count_errors.append(f"{summary_failures_path} must contain failure_cases list")
        else:
            stats["num_summary_failure_cases"] = len(cases)
            if _optional_int(summary_failures.get("num_failure_cases")) != len(cases):
                count_errors.append(
                    f"{summary_failures_path} num_failure_cases does not match failure_cases"
                )
    if summary_metadata is not None:
        if _optional_int(summary_metadata.get("num_failure_cases")) != stats[
            "num_summary_failure_cases"
        ]:
            count_errors.append(
                f"{summary_metadata_path} num_failure_cases does not match failure cases"
            )
        for filename, field in (
            ("stage4_fold_metrics.csv", "num_fold_metric_rows"),
            ("stage4_policy_summary.csv", "num_policy_summary_rows"),
        ):
            rows = _read_csv(evaluation_root / filename, count_errors) or []
            if _optional_int(summary_metadata.get(field)) != len(rows):
                count_errors.append(
                    f"{summary_metadata_path} {field} does not match {filename} row count"
                )

    evaluation_completed = _parse_timestamp(metadata.get("completed_at_utc")) if metadata else None
    summary_completed = (
        _parse_timestamp(summary_metadata.get("completed_at_utc")) if summary_metadata else None
    )
    if evaluation_completed is None:
        order_errors.append(f"{metadata_path} lacks valid completed_at_utc")
    if summary_completed is None:
        order_errors.append(f"{summary_metadata_path} lacks valid completed_at_utc")
    configured_predictions = config.get("selected_predictions", []) if config else []
    if not isinstance(configured_predictions, list) or not configured_predictions:
        order_errors.append(f"{config_path} must list selected_predictions")
        configured_predictions = []
    completion_by_path = {
        _normalized_path(path): _parse_timestamp(value)
        for path, value in run_context.get("run_completed_at_utc", {}).items()
    }
    for raw_path in configured_predictions:
        normalized = _normalized_path(raw_path)
        run_completed = completion_by_path.get(normalized)
        if run_completed is None:
            order_errors.append(
                f"Evaluator input {raw_path} has no matching completed Stage 4 run metadata"
            )
            continue
        if evaluation_completed is not None and evaluation_completed < run_completed:
            order_errors.append(
                f"Evaluator completed before route run {raw_path}: "
                f"{evaluation_completed.isoformat()} < {run_completed.isoformat()}"
            )
        if summary_completed is not None and summary_completed < run_completed:
            order_errors.append(
                f"Summary completed before route run {raw_path}: "
                f"{summary_completed.isoformat()} < {run_completed.isoformat()}"
            )
    stats["num_evaluator_inputs"] = len(configured_predictions)
    stats["evaluator_completed_at_utc"] = (
        evaluation_completed.isoformat() if evaluation_completed else ""
    )
    stats["summary_completed_at_utc"] = summary_completed.isoformat() if summary_completed else ""
    return [
        _make_check(
            "failures_match_evaluation_metrics",
            count_errors,
            f"reconciled {stats['num_metric_rows']} metric rows and evaluator failure counts",
            stats,
        ),
        _make_check(
            "evaluator_after_route_decision",
            order_errors,
            f"validated evaluator ordering for {len(configured_predictions)} routed inputs",
            stats,
        ),
    ]


def _check_live_smoke(live_root: Path) -> dict[str, Any]:
    if not live_root.exists():
        return _fail_check("live_smoke", f"Missing live smoke root {live_root}")
    candidates = sorted(
        {path.parent for path in live_root.rglob("route_decisions.csv")},
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return _fail_check("live_smoke", f"No live route_decisions.csv found under {live_root}")
    run_dir = candidates[-1]
    errors: list[str] = []
    for name in REQUIRED_RUN_FILES:
        if not (run_dir / name).is_file():
            errors.append(f"{run_dir} is missing {name}")
    decisions = _read_csv(run_dir / "route_decisions.csv", errors) or []
    predictions = _read_csv(run_dir / "selected_predictions.csv", errors) or []
    failures = _read_json(run_dir / "failures.json", errors)
    metadata = _read_json(run_dir / "run_metadata.json", errors)
    budget = _read_json(run_dir / "budget_summary.json", errors)
    for index, row in enumerate([*decisions, *predictions], start=1):
        _check_tool_calls_value(row.get("tool_calls"), f"{run_dir}:row{index}", errors)
    if failures is not None:
        failed_tasks = failures.get("failed_tasks")
        if not isinstance(failed_tasks, list):
            errors.append(f"{run_dir}/failures.json must contain failed_tasks list")
        elif _optional_int(failures.get("num_failed")) != len(failed_tasks):
            errors.append(f"{run_dir}/failures.json count mismatch")
    if budget is not None:
        if budget.get("within_budget") is not True:
            errors.append(f"{run_dir}/budget_summary.json within_budget is not true")
        if _optional_int(budget.get("actual_tool_calls")) not in (None, len(predictions)):
            errors.append(f"{run_dir}/budget_summary.json actual_tool_calls count mismatch")
    evaluator_metadata = _read_json(run_dir / "evaluator_only" / "evaluation_metadata.json", errors)
    evaluator_failures = _read_json(run_dir / "evaluator_only" / "evaluation_failures.json", errors)
    metrics = _read_csv(run_dir / "evaluator_only" / "stage4_metrics.csv", errors) or []
    run_started = _parse_timestamp(metadata.get("started_at_utc")) if metadata else None
    run_completed = _parse_timestamp(metadata.get("completed_at_utc")) if metadata else None
    evaluation_completed = (
        _parse_timestamp(evaluator_metadata.get("completed_at_utc"))
        if evaluator_metadata
        else None
    )
    if (
        run_started is None
        or run_completed is None
        or evaluation_completed is None
        or not (run_started <= evaluation_completed <= run_completed)
    ):
        errors.append("Live evaluator completion is outside the live run execution window")
    if evaluator_metadata is not None:
        input_hashes = evaluator_metadata.get("input_sha256", {})
        if not isinstance(input_hashes, Mapping):
            errors.append("Live evaluation_metadata.json input_sha256 must be an object")
        else:
            prediction_hashes = [
                str(value)
                for key, value in input_hashes.items()
                if str(key).replace("\\", "/").endswith("/selected_predictions.csv")
            ]
            actual_hash = _sha256(run_dir / "selected_predictions.csv")
            if prediction_hashes != [actual_hash]:
                errors.append(
                    "Live evaluator input hash does not match the routed selected_predictions.csv"
                )
    if evaluator_failures is not None and not isinstance(evaluator_failures.get("failures"), list):
        errors.append("Live evaluation_failures.json must contain failures list")
    return _make_check(
        "live_smoke",
        errors,
        f"validated latest live smoke run with {len(predictions)} prediction(s)",
        {
            "live_run_dir": str(run_dir),
            "num_decisions": len(decisions),
            "num_selected_predictions": len(predictions),
            "num_metric_rows": len(metrics),
            "policy_name": decisions[0].get("policy_name", "") if decisions else "",
            "selected_expert": (
                decisions[0].get("selected_expert", "") if decisions else ""
            ),
            "dataset": decisions[0].get("dataset", "") if decisions else "",
            "category": decisions[0].get("category", "") if decisions else "",
            "k_shot": decisions[0].get("k_shot", "") if decisions else "",
            "seed": decisions[0].get("seed", "") if decisions else "",
            "actual_subprocess_runtime_ms": (
                budget.get("actual_subprocess_runtime_ms") if budget else None
            ),
            "run_started_at_utc": run_started.isoformat() if run_started else "",
            "run_completed_at_utc": run_completed.isoformat() if run_completed else "",
            "evaluator_completed_at_utc": (
                evaluation_completed.isoformat() if evaluation_completed else ""
            ),
        },
    )


def _discover_latest_grid(root: Path) -> Path | None:
    if (root / "grid_metadata.json").is_file():
        return root
    candidates = [path.parent for path in root.glob("*/grid_metadata.json")]
    return _latest_by_metadata(candidates, "grid_metadata.json")


def _discover_evaluation_root(root: Path, runs_root: Path | None) -> Path | None:
    if runs_root is not None:
        matched = root / runs_root.name
        if (matched / "stage4_summary_metadata.json").is_file():
            return matched
    if (root / "stage4_summary_metadata.json").is_file():
        return root
    candidates = [path.parent for path in root.glob("*/stage4_summary_metadata.json")]
    return _latest_by_metadata(candidates, "stage4_summary_metadata.json")


def _latest_by_metadata(candidates: Iterable[Path], filename: str) -> Path | None:
    ranked: list[tuple[datetime, Path]] = []
    for path in candidates:
        try:
            payload = json.loads((path / filename).read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        completed = _parse_timestamp(payload.get("completed_at_utc"))
        if completed is not None:
            ranked.append((completed, path))
    return max(ranked, default=(datetime.min.replace(tzinfo=timezone.utc), None))[1]


def _policy_root_from_grid(runs_root: Path | None, fallback: Path) -> Path:
    if runs_root is None:
        return fallback
    try:
        payload = json.loads((runs_root / "grid_metadata.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return fallback
    raw = payload.get("policy_artifact_root")
    if not raw:
        return fallback
    path = Path(str(raw))
    if path.is_absolute():
        return path
    project_relative = PROJECT_ROOT / path
    return project_relative if project_relative.exists() else path


def _forbidden_feature_name(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    segments = {normalized, *normalized.split("_")}
    if normalized in FORBIDDEN_FEATURE_NAMES:
        return True
    return any(
        token in normalized
        for token in (
            "support_set_id",
            "expert_score",
            "patchcore_score",
            "winclip_score",
            "anomalydino_score",
            "ground_truth",
        )
    ) or bool({"seed", "label", "labels"}.intersection(segments))


def _check_tool_calls_value(value: Any, context: str, errors: list[str]) -> None:
    parsed = _optional_int(value)
    if parsed is None or parsed < 0 or parsed > 1:
        errors.append(f"{context} has invalid tool_calls={value!r}; expected integer 0 or 1")


def _canonical_expert(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None or str(value).strip() == "":
        return None
    try:
        parsed_float = float(value)
        parsed_int = int(parsed_float)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed_float) or parsed_float != parsed_int:
        return None
    return parsed_int


def _read_csv(path: Path, errors: list[str]) -> list[dict[str, str]] | None:
    if not path.is_file():
        errors.append(f"Missing {path}")
        return None
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                errors.append(f"{path} has no CSV header")
                return None
            return [
                {str(key): (value or "").strip() for key, value in row.items()}
                for row in reader
            ]
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        errors.append(f"Could not read {path}: {exc}")
        return None


def _read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"Missing {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        errors.append(f"Could not read {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{path} root must be a JSON object")
        return None
    return payload


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalized_path(value: Any) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return os.path.normcase(os.path.normpath(str(path.resolve(strict=False))))


def _relative_display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _sortable_int(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def _make_check(
    name: str,
    errors: Sequence[str],
    summary: str,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if not errors else "FAIL",
        "summary": summary,
        "stats": dict(stats),
        "errors": list(errors),
        "num_errors": len(errors),
    }


def _fail_check(name: str, error: str) -> dict[str, Any]:
    return _make_check(name, [error], "required artifact unavailable", {})


def _skip_check(name: str, summary: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "SKIP",
        "summary": summary,
        "stats": {},
        "errors": [],
        "num_errors": 0,
    }


def _input_hashes(paths: Iterable[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            hashes[str(path)] = _sha256(path)
        elif path.is_dir():
            for name in (
                "grid_config.json",
                "grid_metadata.json",
                "stage4_summary_config.json",
                "stage4_summary_metadata.json",
            ):
                candidate = path / name
                if candidate.is_file():
                    hashes[str(candidate)] = _sha256(candidate)
    return hashes


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _environment() -> dict[str, str]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
