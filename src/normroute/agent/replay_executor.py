"""Replay Stage 4 route decisions from immutable Stage 2 predictions."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

from ..evaluation.export import PREDICTION_COLUMNS

from .policy import Policy, TrainRecord
from .protocol import (
    AgentTask,
    CANDIDATE_EXPERTS,
    FORBIDDEN_ROUTE_DECISION_FIELDS,
    ROUTE_DECISION_FIELDS,
    RouteDecision,
    validate_no_forbidden_fields,
    validate_route_decision,
)


REPLAY_PROTOCOL_VERSION = "stage4.replay.v1"
FOLD_MANIFEST_REQUIRED_COLUMNS = (
    "fold",
    "split",
    "task_id",
    "sample_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
)
ROUTE_DECISION_COLUMNS = (
    "task_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "policy_name",
    "selected_expert",
    "decision_reason",
    "estimated_cost_ms",
    "tool_calls",
    "selected_probability",
    "margin",
    "fold",
    "split",
)
SELECTED_PREDICTION_COLUMNS = (
    "task_id",
    "fold",
    "split",
    "policy_name",
    "selected_expert",
    "stage2_run_dir",
    "runtime_source",
    *PREDICTION_COLUMNS,
    "estimated_runtime_ms",
    "actual_runtime_ms",
)
ESTIMATED_RUNTIME_SOURCE = "estimated_runtime"
ALLOWED_SPLITS = frozenset({"train", "val", "test"})


class ReplayExecutionError(ValueError):
    """Raised when a replay cannot preserve its routing or leakage contract."""


@dataclass(frozen=True)
class ReplayResult:
    route_decisions_path: Path
    selected_predictions_path: Path
    failures_path: Path
    run_metadata_path: Path
    budget_summary_path: Path
    policy_state_path: Path
    num_tasks: int
    num_selected_predictions: int
    num_failures: int


class ReplayExecutor:
    """Select one expert per task and copy its existing Stage 2 prediction."""

    def __init__(
        self,
        *,
        stage2_root: str | Path,
        output_dir: str | Path,
        policy: Policy,
        tool_budget: int = 1,
        max_estimated_cost_ms: float | None = None,
    ) -> None:
        if isinstance(tool_budget, bool) or not isinstance(tool_budget, int) or tool_budget < 0:
            raise ReplayExecutionError("tool_budget must be an integer >= 0")
        if max_estimated_cost_ms is not None and (
            isinstance(max_estimated_cost_ms, bool)
            or not isinstance(max_estimated_cost_ms, (int, float))
            or not math.isfinite(float(max_estimated_cost_ms))
            or float(max_estimated_cost_ms) < 0
        ):
            raise ReplayExecutionError("max_estimated_cost_ms must be a finite number >= 0")
        self.stage2_root = Path(stage2_root)
        self.output_dir = Path(output_dir)
        self.policy = policy
        self.tool_budget = tool_budget
        self.max_estimated_cost_ms = (
            float(max_estimated_cost_ms) if max_estimated_cost_ms is not None else None
        )
        self._prediction_cache: dict[Path, list[dict[str, str]]] = {}

    def execute(
        self,
        *,
        tasks: Sequence[AgentTask | Mapping[str, Any]],
        manifest_rows: Sequence[Mapping[str, Any]],
        fold: str,
        split: str = "test",
        train_records: Sequence[TrainRecord] | None = None,
        tasks_path: str | Path | None = None,
        fold_manifest_path: str | Path | None = None,
    ) -> ReplayResult:
        """Run one fold/split replay and always write explicit per-task failures."""

        if not isinstance(fold, str) or not fold.strip():
            raise ReplayExecutionError("fold must be a non-empty string")
        if split not in ALLOWED_SPLITS:
            raise ReplayExecutionError(f"split must be one of {sorted(ALLOWED_SPLITS)!r}")
        if not self.stage2_root.is_dir():
            raise ReplayExecutionError(f"Stage 2 output root does not exist: {self.stage2_root}")

        typed_tasks = [_coerce_task(task) for task in tasks]
        task_by_id = _index_tasks(typed_tasks)
        assignments = _index_assignments(manifest_rows, task_by_id=task_by_id, fold=fold)
        selected_tasks = [
            task for task in typed_tasks if assignments[task.task_id]["split"] == split
        ]
        if not selected_tasks:
            raise ReplayExecutionError(f"No tasks assigned to fold={fold!r}, split={split!r}")

        safe_train_records: Sequence[TrainRecord]
        if train_records is None:
            safe_train_records = [
                task.policy_features
                for task in typed_tasks
                if assignments[task.task_id]["split"] == "train"
            ]
        else:
            safe_train_records = train_records
        _validate_policy_fold_binding(
            policy=self.policy,
            fold=fold,
            tool_budget=self.tool_budget,
            tasks=typed_tasks,
            assignments=assignments,
        )
        self.policy.fit(safe_train_records)

        started_at = _utc_now()
        decision_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        actual_tool_calls = 0

        for task in selected_tasks:
            decision: RouteDecision | None = None
            try:
                decision = self.policy.select(task)
                _validate_decision_for_task(decision, task=task, policy=self.policy)
                decision = replace(decision, fold=fold, split=split)
                validate_route_decision(decision.to_dict())
                decision_rows.append(decision.to_dict())

                budget_error = self._decision_budget_error(decision)
                if budget_error:
                    raise _TaskReplayFailure("budget_exceeded", budget_error)

                prediction, run_dir = self._selected_prediction(task, decision)
                prediction_tool_calls = _nonnegative_int(
                    prediction["tool_calls"], field="tool_calls", context=str(run_dir)
                )
                if prediction_tool_calls > self.tool_budget:
                    raise _TaskReplayFailure(
                        "budget_exceeded",
                        f"Stage 2 prediction uses {prediction_tool_calls} tool calls, "
                        f"exceeding per-task budget {self.tool_budget}",
                    )
                actual_tool_calls += prediction_tool_calls
                selected_rows.append(
                    {
                        "task_id": task.task_id,
                        "fold": fold,
                        "split": split,
                        "policy_name": self.policy.name,
                        "selected_expert": decision.selected_expert,
                        "stage2_run_dir": str(run_dir),
                        # This is historical Stage 2 runtime reused as a replay
                        # estimate.  It is not wall-clock time of this replay.
                        "runtime_source": ESTIMATED_RUNTIME_SOURCE,
                        **{column: prediction.get(column, "") for column in PREDICTION_COLUMNS},
                        "estimated_runtime_ms": prediction.get("runtime_ms", ""),
                        # Replay does not launch the expert, so there is no
                        # current wall-clock runtime to report.
                        "actual_runtime_ms": "",
                    }
                )
                if prediction["status"].lower() != "ok":
                    raise _TaskReplayFailure(
                        "stage2_prediction_failed",
                        prediction.get("error_message") or "Selected Stage 2 prediction is not ok",
                    )
            except _TaskReplayFailure as exc:
                failures.append(_failure_record(task, fold, split, exc.code, str(exc), decision))
            except Exception as exc:
                failures.append(
                    _failure_record(task, fold, split, "replay_error", str(exc), decision)
                )

        policy_state_path = self.output_dir / "policy.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save(policy_state_path)
        budget_summary = self._budget_summary(
            num_tasks=len(selected_tasks),
            decisions=decision_rows,
            selected_rows=selected_rows,
            failures=failures,
            actual_tool_calls=actual_tool_calls,
        )
        metadata = _run_metadata(
            policy=self.policy,
            stage2_root=self.stage2_root,
            output_dir=self.output_dir,
            fold=fold,
            split=split,
            tool_budget=self.tool_budget,
            max_estimated_cost_ms=self.max_estimated_cost_ms,
            tasks=selected_tasks,
            num_decisions=len(decision_rows),
            num_selected_predictions=len(selected_rows),
            num_failures=len(failures),
            started_at=started_at,
            tasks_path=Path(tasks_path) if tasks_path is not None else None,
            fold_manifest_path=(
                Path(fold_manifest_path) if fold_manifest_path is not None else None
            ),
            policy_state_path=policy_state_path,
        )

        route_path = self.output_dir / "route_decisions.csv"
        selected_path = self.output_dir / "selected_predictions.csv"
        failures_path = self.output_dir / "failures.json"
        metadata_path = self.output_dir / "run_metadata.json"
        budget_path = self.output_dir / "budget_summary.json"
        _write_csv_atomic(route_path, ROUTE_DECISION_COLUMNS, decision_rows)
        _write_csv_atomic(selected_path, SELECTED_PREDICTION_COLUMNS, selected_rows)
        _write_json_atomic(
            failures_path,
            {
                "protocol_version": REPLAY_PROTOCOL_VERSION,
                "num_failed": len(failures),
                "failed_tasks": failures,
            },
        )
        _write_json_atomic(metadata_path, metadata)
        _write_json_atomic(budget_path, budget_summary)
        return ReplayResult(
            route_decisions_path=route_path,
            selected_predictions_path=selected_path,
            failures_path=failures_path,
            run_metadata_path=metadata_path,
            budget_summary_path=budget_path,
            policy_state_path=policy_state_path,
            num_tasks=len(selected_tasks),
            num_selected_predictions=len(selected_rows),
            num_failures=len(failures),
        )

    run = execute

    def _decision_budget_error(self, decision: RouteDecision) -> str:
        if decision.tool_calls > self.tool_budget:
            return (
                f"Decision requests {decision.tool_calls} tool calls, exceeding per-task "
                f"budget {self.tool_budget}"
            )
        if (
            self.max_estimated_cost_ms is not None
            and decision.estimated_cost_ms > self.max_estimated_cost_ms
        ):
            return (
                f"Decision estimates {decision.estimated_cost_ms} ms, exceeding per-task "
                f"limit {self.max_estimated_cost_ms} ms"
            )
        return ""

    def _selected_prediction(
        self, task: AgentTask, decision: RouteDecision
    ) -> tuple[dict[str, str], Path]:
        expert_dir = _normalize_expert_name(decision.selected_expert)
        run_dir = (
            self.stage2_root
            / expert_dir
            / task.dataset
            / task.category
            / f"k{task.k_shot}"
            / f"seed{task.seed}"
            / task.support_set_id
        )
        predictions_path = run_dir / "predictions.csv"
        if not predictions_path.is_file():
            raise _TaskReplayFailure(
                "missing_stage2_run",
                f"Selected Stage 2 predictions do not exist: {predictions_path}",
            )
        rows = self._prediction_cache.get(predictions_path)
        if rows is None:
            rows = _read_stage2_predictions(predictions_path, decision.selected_expert)
            self._prediction_cache[predictions_path] = rows

        matches = [
            row
            for row in rows
            if row["image_id"] == task.sample_id
            and row["dataset"] == task.dataset
            and row["category"] == task.category
            and row["support_set_id"] == task.support_set_id
            and _same_int(row["k_shot"], task.k_shot)
            and _same_int(row["seed"], task.seed)
        ]
        if len(matches) != 1:
            raise _TaskReplayFailure(
                "prediction_match_error",
                f"Expected exactly one matching prediction for task_id={task.task_id!r} in "
                f"{predictions_path}; found {len(matches)}",
            )
        return matches[0], run_dir

    def _budget_summary(
        self,
        *,
        num_tasks: int,
        decisions: list[dict[str, Any]],
        selected_rows: list[dict[str, Any]],
        failures: list[dict[str, Any]],
        actual_tool_calls: int,
    ) -> dict[str, Any]:
        planned = sum(int(row["tool_calls"]) for row in decisions)
        total_budget = num_tasks * self.tool_budget
        budget_failures = sum(
            1 for failure in failures if failure["failure_type"] == "budget_exceeded"
        )
        return {
            "protocol_version": REPLAY_PROTOCOL_VERSION,
            "runtime_source": ESTIMATED_RUNTIME_SOURCE,
            "tool_budget_per_task": self.tool_budget,
            "max_estimated_cost_ms_per_task": self.max_estimated_cost_ms,
            "num_tasks": num_tasks,
            "num_decisions": len(decisions),
            "num_selected_predictions": len(selected_rows),
            "num_budget_failures": budget_failures,
            "total_tool_call_budget": total_budget,
            "planned_tool_calls": planned,
            "actual_replayed_tool_calls": actual_tool_calls,
            "planned_estimated_runtime_ms": sum(
                float(row["estimated_cost_ms"]) for row in decisions
            ),
            "planned_budget_remaining": max(total_budget - planned, 0),
            "within_budget": budget_failures == 0,
        }


class _TaskReplayFailure(ReplayExecutionError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def execute_replay(
    *,
    tasks: Sequence[AgentTask | Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    policy: Policy,
    stage2_root: str | Path,
    output_dir: str | Path,
    fold: str,
    split: str = "test",
    tool_budget: int = 1,
    max_estimated_cost_ms: float | None = None,
    train_records: Sequence[TrainRecord] | None = None,
    tasks_path: str | Path | None = None,
    fold_manifest_path: str | Path | None = None,
) -> ReplayResult:
    """Functional wrapper around :class:`ReplayExecutor`."""

    return ReplayExecutor(
        stage2_root=stage2_root,
        output_dir=output_dir,
        policy=policy,
        tool_budget=tool_budget,
        max_estimated_cost_ms=max_estimated_cost_ms,
    ).execute(
        tasks=tasks,
        manifest_rows=manifest_rows,
        fold=fold,
        split=split,
        train_records=train_records,
        tasks_path=tasks_path,
        fold_manifest_path=fold_manifest_path,
    )


def read_fold_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read the frozen Stage 4 fold manifest without silently skipping rows."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ReplayExecutionError(f"Fold manifest does not exist: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in FOLD_MANIFEST_REQUIRED_COLUMNS if column not in fieldnames]
        forbidden = sorted(FORBIDDEN_ROUTE_DECISION_FIELDS.intersection(fieldnames))
        if missing or forbidden:
            raise ReplayExecutionError(
                f"{manifest_path} has invalid columns; missing={missing}, forbidden={forbidden}"
            )
        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            clean = {key: (value or "").strip() for key, value in row.items()}
            missing_values = [
                column for column in FOLD_MANIFEST_REQUIRED_COLUMNS if not clean.get(column)
            ]
            if missing_values:
                raise ReplayExecutionError(
                    f"{manifest_path}:{line_number} has empty values: {missing_values}"
                )
            if clean["split"] not in ALLOWED_SPLITS:
                raise ReplayExecutionError(
                    f"{manifest_path}:{line_number} has invalid split={clean['split']!r}"
                )
            _nonnegative_int(clean["seed"], field="seed", context=str(manifest_path))
            _positive_int(clean["k_shot"], field="k_shot", context=str(manifest_path))
            rows.append(clean)
    if not rows:
        raise ReplayExecutionError(f"Fold manifest is empty: {manifest_path}")
    return rows


def _coerce_task(task: AgentTask | Mapping[str, Any]) -> AgentTask:
    if isinstance(task, AgentTask):
        return task
    return AgentTask.from_mapping(task)


def _index_tasks(tasks: Sequence[AgentTask]) -> dict[str, AgentTask]:
    if not tasks:
        raise ReplayExecutionError("Agent task list is empty")
    indexed: dict[str, AgentTask] = {}
    for task in tasks:
        if task.task_id in indexed:
            raise ReplayExecutionError(f"Duplicate Agent task_id: {task.task_id}")
        indexed[task.task_id] = task
    return indexed


def _index_assignments(
    rows: Sequence[Mapping[str, Any]], *, task_by_id: Mapping[str, AgentTask], fold: str
) -> dict[str, dict[str, str]]:
    assignments: dict[str, dict[str, str]] = {}
    for row in rows:
        try:
            validate_no_forbidden_fields(row, context="fold manifest row")
        except ValueError as exc:
            raise ReplayExecutionError(str(exc)) from exc
        if str(row.get("fold", "")).strip() != fold:
            continue
        task_id = str(row.get("task_id", "")).strip()
        if not task_id or task_id not in task_by_id:
            raise ReplayExecutionError(
                f"Fold {fold!r} manifest contains unknown or empty task_id={task_id!r}"
            )
        if task_id in assignments:
            raise ReplayExecutionError(
                f"Fold {fold!r} assigns task_id={task_id!r} more than once"
            )
        split = str(row.get("split", "")).strip()
        if split not in ALLOWED_SPLITS:
            raise ReplayExecutionError(
                f"Fold {fold!r} task_id={task_id!r} has invalid split={split!r}"
            )
        task = task_by_id[task_id]
        expected = {
            "sample_id": task.sample_id,
            "dataset": task.dataset,
            "category": task.category,
            "k_shot": str(task.k_shot),
            "seed": str(task.seed),
            "support_set_id": task.support_set_id,
        }
        mismatched = [
            key for key, value in expected.items() if str(row.get(key, "")).strip() != value
        ]
        if mismatched:
            raise ReplayExecutionError(
                f"Fold manifest disagrees with task_id={task_id!r}: {mismatched}"
            )
        assignments[task_id] = {key: str(value).strip() for key, value in row.items()}
    missing = sorted(set(task_by_id) - set(assignments))
    if missing:
        raise ReplayExecutionError(
            f"Fold {fold!r} does not assign every Agent task; missing task_ids={missing[:10]}"
        )
    return assignments


def _validate_decision_for_task(
    decision: RouteDecision, *, task: AgentTask, policy: Policy
) -> None:
    if not isinstance(decision, RouteDecision):
        raise ReplayExecutionError("Policy.select(task) must return RouteDecision")
    validate_route_decision(decision.to_dict())
    expected = {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "category": task.category,
        "k_shot": task.k_shot,
        "seed": task.seed,
        "support_set_id": task.support_set_id,
        "policy_name": policy.name,
    }
    mismatched = [field for field, value in expected.items() if getattr(decision, field) != value]
    if mismatched:
        raise ReplayExecutionError(f"RouteDecision disagrees with AgentTask: {mismatched}")
    if decision.fold or decision.split:
        raise ReplayExecutionError("Policy must not assign fold/split; executor owns split provenance")
    if decision.selected_expert not in task.candidate_experts:
        raise ReplayExecutionError(
            f"Policy selected non-candidate expert {decision.selected_expert!r}"
        )


def _validate_policy_fold_binding(
    *,
    policy: Policy,
    fold: str,
    tool_budget: int,
    tasks: Sequence[AgentTask],
    assignments: Mapping[str, Mapping[str, str]],
) -> None:
    """Reject a fold-calibrated artifact when its provenance does not match replay."""

    configuration = policy.configuration()
    artifact_fold = configuration.get("fold")
    if artifact_fold is not None and artifact_fold != fold:
        raise ReplayExecutionError(
            f"Policy artifact is calibrated for fold={artifact_fold!r}, not replay fold={fold!r}"
        )
    artifact_budget = configuration.get("budget")
    if artifact_budget is not None and artifact_budget != tool_budget:
        raise ReplayExecutionError(
            "Policy artifact budget does not match replay tool_budget; "
            f"artifact={artifact_budget!r}, replay={tool_budget!r}"
        )
    artifact_train_seeds = configuration.get("train_seeds")
    for split, artifact_seeds in (
        ("train", artifact_train_seeds),
        ("val", configuration.get("validation_seeds")),
    ):
        if artifact_seeds is None:
            continue
        manifest_seeds = sorted(
            {
                task.seed
                for task in tasks
                if assignments[task.task_id]["split"] == split
            }
        )
        if list(artifact_seeds) != manifest_seeds:
            label = "train_seeds" if split == "train" else "validation_seeds"
            raise ReplayExecutionError(
                f"Policy artifact {label} do not match the replay fold manifest; "
                f"artifact={list(artifact_seeds)}, manifest={manifest_seeds}"
            )


def _read_stage2_predictions(path: Path, selected_expert: str) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in PREDICTION_COLUMNS if column not in fieldnames]
        forbidden = sorted(FORBIDDEN_ROUTE_DECISION_FIELDS.intersection(fieldnames))
        if missing or forbidden:
            raise ReplayExecutionError(
                f"{path} has invalid columns; missing={missing}, forbidden={forbidden}"
            )
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ReplayExecutionError(f"Stage 2 predictions are empty: {path}")
    experts = {_normalize_expert_name(row.get("expert_name", "")) for row in rows}
    expected = _normalize_expert_name(selected_expert)
    if experts != {expected}:
        raise ReplayExecutionError(
            f"Each Stage 2 run must contain exactly one expert; {path} has {sorted(experts)!r}, "
            f"expected {[expected]!r}"
        )
    return rows


def _failure_record(
    task: AgentTask,
    fold: str,
    split: str,
    failure_type: str,
    message: str,
    decision: RouteDecision | None,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "category": task.category,
        "k_shot": task.k_shot,
        "seed": task.seed,
        "support_set_id": task.support_set_id,
        "fold": fold,
        "split": split,
        "selected_expert": decision.selected_expert if decision is not None else "",
        "failure_type": failure_type,
        "error_message": message,
    }


def _run_metadata(
    *,
    policy: Policy,
    stage2_root: Path,
    output_dir: Path,
    fold: str,
    split: str,
    tool_budget: int,
    max_estimated_cost_ms: float | None,
    tasks: Sequence[AgentTask],
    num_decisions: int,
    num_selected_predictions: int,
    num_failures: int,
    started_at: str,
    tasks_path: Path | None,
    fold_manifest_path: Path | None,
    policy_state_path: Path,
) -> dict[str, Any]:
    config = {
        "policy_name": policy.name,
        "policy_configuration": policy.configuration(),
        "fold": fold,
        "split": split,
        "tool_budget_per_task": tool_budget,
        "max_estimated_cost_ms_per_task": max_estimated_cost_ms,
        "stage2_root": str(stage2_root),
        "output_dir": str(output_dir),
        "runtime_source": ESTIMATED_RUNTIME_SOURCE,
    }
    seeds = sorted({task.seed for task in tasks})
    return {
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "stage": "stage4",
        "config": config,
        "seed": seeds[0] if len(seeds) == 1 else None,
        "seeds": seeds,
        "policy_name": policy.name,
        "fold": fold,
        "split": split,
        "candidate_experts": list(CANDIDATE_EXPERTS),
        "tasks_path": str(tasks_path or ""),
        "tasks_sha256": _optional_sha256(tasks_path),
        "fold_manifest_path": str(fold_manifest_path or ""),
        "fold_manifest_sha256": _optional_sha256(fold_manifest_path),
        "stage2_root": str(stage2_root),
        "output_dir": str(output_dir),
        "policy_state_path": str(policy_state_path),
        "num_tasks": len(tasks),
        "num_decisions": num_decisions,
        "num_selected_predictions": num_selected_predictions,
        "num_failures": num_failures,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "git_commit": _git_commit(),
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
        },
    }


def _write_csv_atomic(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalize_expert_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _same_int(value: str, expected: int) -> bool:
    try:
        return int(value) == expected
    except (TypeError, ValueError):
        return False


def _nonnegative_int(value: Any, *, field: str, context: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ReplayExecutionError(f"{context} field {field!r} must be an integer") from exc
    if parsed < 0 or isinstance(value, bool):
        raise ReplayExecutionError(f"{context} field {field!r} must be >= 0")
    return parsed


def _positive_int(value: Any, *, field: str, context: str) -> int:
    parsed = _nonnegative_int(value, field=field, context=context)
    if parsed < 1:
        raise ReplayExecutionError(f"{context} field {field!r} must be >= 1")
    return parsed


def _optional_sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if set(ROUTE_DECISION_COLUMNS) != ROUTE_DECISION_FIELDS:
    raise RuntimeError("Route decision CSV columns and RouteDecision schema diverged")
