"""Execute Stage 4 route decisions by launching exactly one live expert per task."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from ..evaluation import stage4 as stage4_evaluation
from ..evaluation.export import PREDICTION_COLUMNS

from .policy import Policy, TrainRecord
from .protocol import AgentTask, CANDIDATE_EXPERTS, RouteDecision, validate_route_decision
from .replay_executor import (
    ALLOWED_SPLITS,
    ROUTE_DECISION_COLUMNS,
    SELECTED_PREDICTION_COLUMNS,
    _coerce_task,
    _failure_record,
    _git_commit,
    _index_assignments,
    _index_tasks,
    _nonnegative_int,
    _normalize_expert_name,
    _optional_sha256,
    _read_stage2_predictions,
    _utc_now,
    _validate_decision_for_task,
    _validate_policy_fold_binding,
    _write_csv_atomic,
    _write_json_atomic,
)


LIVE_PROTOCOL_VERSION = "stage4.live.v1"
ACTUAL_RUNTIME_SOURCE = "actual_runtime"
AGENT_INPUT_COLUMNS = ("image_id", "dataset", "category", "split", "image_path")
# Live and replay deliberately share one exact selected-prediction schema.
# Runtime values differ by source, not by column layout.
LIVE_SELECTED_PREDICTION_COLUMNS = SELECTED_PREDICTION_COLUMNS
LIVE_EXECUTION_COLUMNS = (
    "task_id",
    "fold",
    "split",
    "policy_name",
    "selected_expert",
    "agent_input_csv",
    "task_agent_input_csv",
    "support_set_csv",
    "output_dir",
    "command",
    "returncode",
    "estimated_runtime_ms",
    "actual_runtime_ms",
    "stdout_log",
    "stderr_log",
    "status",
)


class LiveExecutionError(ValueError):
    """Raised when a live run cannot preserve routing or leakage guarantees."""


@dataclass(frozen=True)
class LiveResult:
    route_decisions_path: Path
    selected_predictions_path: Path
    live_executions_path: Path
    failures_path: Path
    run_metadata_path: Path
    budget_summary_path: Path
    policy_state_path: Path
    stdout_path: Path
    stderr_path: Path
    evaluation_paths: Mapping[str, Path]
    evaluation_failure_path: Path | None
    num_tasks: int
    num_selected_predictions: int
    num_failures: int
    num_evaluation_failures: int


Runner = Callable[..., subprocess.CompletedProcess[str]]
Clock = Callable[[], float]


class LiveExecutor:
    """Select and launch one expert subprocess for every selected routing task.

    The executor has no evaluator input on its decision path.  If an evaluator
    CSV is supplied, it is opened only after every expert subprocess has
    completed and all live artifacts have been written.
    """

    def __init__(
        self,
        *,
        data_root: str | Path,
        output_dir: str | Path,
        policy: Policy,
        tool_budget: int = 1,
        max_estimated_cost_ms: float | None = None,
        project_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        subprocess_runner: Runner | None = None,
        clock: Clock | None = None,
    ) -> None:
        if isinstance(tool_budget, bool) or not isinstance(tool_budget, int) or tool_budget < 0:
            raise LiveExecutionError("tool_budget must be an integer >= 0")
        if max_estimated_cost_ms is not None and (
            isinstance(max_estimated_cost_ms, bool)
            or not isinstance(max_estimated_cost_ms, (int, float))
            or not math.isfinite(float(max_estimated_cost_ms))
            or float(max_estimated_cost_ms) < 0
        ):
            raise LiveExecutionError("max_estimated_cost_ms must be a finite number >= 0")
        self.project_root = Path(project_root or Path(__file__).resolve().parents[3]).resolve()
        self.data_root = _resolve_path(data_root, base=self.project_root)
        self.output_dir = _resolve_path(output_dir, base=self.project_root)
        self.policy = policy
        self.tool_budget = tool_budget
        self.max_estimated_cost_ms = (
            float(max_estimated_cost_ms) if max_estimated_cost_ms is not None else None
        )
        self.python_executable = str(python_executable or sys.executable)
        self._run_subprocess = subprocess_runner or subprocess.run
        self._clock = clock or time.perf_counter

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
        evaluator_csv: str | Path | None = None,
        evaluation_output_dir: str | Path | None = None,
    ) -> LiveResult:
        """Run one fold/split live and persist every task outcome explicitly."""

        if not isinstance(fold, str) or not fold.strip():
            raise LiveExecutionError("fold must be a non-empty string")
        if split not in ALLOWED_SPLITS:
            raise LiveExecutionError(f"split must be one of {sorted(ALLOWED_SPLITS)!r}")

        typed_tasks = [_coerce_task(task) for task in tasks]
        task_by_id = _index_tasks(typed_tasks)
        assignments = _index_assignments(manifest_rows, task_by_id=task_by_id, fold=fold)
        selected_tasks = [
            task for task in typed_tasks if assignments[task.task_id]["split"] == split
        ]
        if not selected_tasks:
            raise LiveExecutionError(f"No tasks assigned to fold={fold!r}, split={split!r}")

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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        decision_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        execution_rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        attempted_task_ids: set[str] = set()
        actual_tool_calls = 0

        for task in selected_tasks:
            decision: RouteDecision | None = None
            invocation: _Invocation | None = None
            task_output_dir: Path | None = None
            stdout_log: Path | None = None
            stderr_log: Path | None = None
            try:
                # No evaluator path is inspected, opened, or hashed before this
                # policy decision (or before any other task's policy decision).
                decision = self.policy.select(task)
                _validate_decision_for_task(decision, task=task, policy=self.policy)
                decision = replace(decision, fold=fold, split=split)
                validate_route_decision(decision.to_dict())
                decision_rows.append(decision.to_dict())

                budget_error = self._decision_budget_error(decision)
                if budget_error:
                    raise _TaskLiveFailure("budget_exceeded", budget_error)
                if task.task_id in attempted_task_ids:
                    raise _TaskLiveFailure(
                        "duplicate_expert_call",
                        f"task_id={task.task_id!r} already attempted an expert call",
                    )

                agent_input_csv, support_set_csv = self._locate_inputs(task)
                task_output_dir = self.output_dir / "expert_runs" / _task_directory_name(
                    task.task_id
                )
                task_output_dir.mkdir(parents=True, exist_ok=False)
                task_agent_input_csv = task_output_dir / "agent_input.csv"
                _write_task_agent_input(
                    source=agent_input_csv,
                    destination=task_agent_input_csv,
                    task=task,
                )
                stdout_log = task_output_dir / "stdout.log"
                stderr_log = task_output_dir / "stderr.log"
                command = self._expert_command(
                    task=task,
                    decision=decision,
                    agent_input_csv=task_agent_input_csv,
                    support_set_csv=support_set_csv,
                    output_dir=task_output_dir,
                )

                attempted_task_ids.add(task.task_id)
                invocation = self._invoke_expert(
                    task=task,
                    decision=decision,
                    command=command,
                    agent_input_csv=agent_input_csv,
                    task_agent_input_csv=task_agent_input_csv,
                    support_set_csv=support_set_csv,
                    output_dir=task_output_dir,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    fold=fold,
                    split=split,
                )
                execution_rows.append(invocation.row)
                stdout_chunks.append(_aggregate_log_chunk(task.task_id, invocation.stdout))
                stderr_chunks.append(_aggregate_log_chunk(task.task_id, invocation.stderr))
                if invocation.returncode != 0:
                    raise _TaskLiveFailure(
                        "expert_subprocess_failed",
                        f"Expert subprocess returned non-zero exit code {invocation.returncode}",
                        returncode=invocation.returncode,
                    )

                prediction = _read_task_prediction(
                    task=task,
                    decision=decision,
                    predictions_path=task_output_dir / "predictions.csv",
                )
                prediction_tool_calls = _nonnegative_int(
                    prediction["tool_calls"],
                    field="tool_calls",
                    context=str(task_output_dir / "predictions.csv"),
                )
                if prediction_tool_calls > self.tool_budget:
                    raise _TaskLiveFailure(
                        "budget_exceeded",
                        f"Live prediction uses {prediction_tool_calls} tool calls, exceeding "
                        f"per-task budget {self.tool_budget}",
                    )
                actual_tool_calls += prediction_tool_calls
                selected_rows.append(
                    {
                        "task_id": task.task_id,
                        "fold": fold,
                        "split": split,
                        "policy_name": self.policy.name,
                        "selected_expert": decision.selected_expert,
                        "stage2_run_dir": str(task_output_dir),
                        "runtime_source": ACTUAL_RUNTIME_SOURCE,
                        **{
                            column: prediction.get(column, "")
                            for column in PREDICTION_COLUMNS
                        },
                        "estimated_runtime_ms": decision.estimated_cost_ms,
                        "actual_runtime_ms": invocation.actual_runtime_ms,
                    }
                )
                if prediction["status"].lower() != "ok":
                    raise _TaskLiveFailure(
                        "expert_prediction_failed",
                        prediction.get("error_message") or "Live expert prediction is not ok",
                    )
            except _TaskLiveFailure as exc:
                failure = _failure_record(task, fold, split, exc.code, str(exc), decision)
                failure.update(
                    {
                        "returncode": exc.returncode,
                        "estimated_runtime_ms": (
                            decision.estimated_cost_ms if decision is not None else None
                        ),
                        "actual_runtime_ms": (
                            invocation.actual_runtime_ms if invocation is not None else None
                        ),
                        "stdout_log": str(stdout_log or ""),
                        "stderr_log": str(stderr_log or ""),
                    }
                )
                failures.append(failure)
            except Exception as exc:
                failure = _failure_record(
                    task, fold, split, "live_execution_error", str(exc), decision
                )
                failure.update(
                    {
                        "returncode": None,
                        "estimated_runtime_ms": (
                            decision.estimated_cost_ms if decision is not None else None
                        ),
                        "actual_runtime_ms": (
                            invocation.actual_runtime_ms if invocation is not None else None
                        ),
                        "stdout_log": str(stdout_log or ""),
                        "stderr_log": str(stderr_log or ""),
                    }
                )
                failures.append(failure)

        policy_state_path = self.output_dir / "policy.json"
        self.policy.save(policy_state_path)
        route_path = self.output_dir / "route_decisions.csv"
        selected_path = self.output_dir / "selected_predictions.csv"
        executions_path = self.output_dir / "live_executions.csv"
        failures_path = self.output_dir / "failures.json"
        budget_path = self.output_dir / "budget_summary.json"
        stdout_path = self.output_dir / "stdout.log"
        stderr_path = self.output_dir / "stderr.log"
        metadata_path = self.output_dir / "run_metadata.json"

        _write_csv_atomic(route_path, ROUTE_DECISION_COLUMNS, decision_rows)
        _write_csv_atomic(
            selected_path, LIVE_SELECTED_PREDICTION_COLUMNS, selected_rows
        )
        _write_csv_atomic(executions_path, LIVE_EXECUTION_COLUMNS, execution_rows)
        _write_json_atomic(
            failures_path,
            {
                "protocol_version": LIVE_PROTOCOL_VERSION,
                "num_failed": len(failures),
                "failed_tasks": failures,
            },
        )
        _write_text_atomic(stdout_path, "".join(stdout_chunks))
        _write_text_atomic(stderr_path, "".join(stderr_chunks))
        _write_json_atomic(
            budget_path,
            self._budget_summary(
                num_tasks=len(selected_tasks),
                decisions=decision_rows,
                selected_rows=selected_rows,
                executions=execution_rows,
                failures=failures,
                actual_tool_calls=actual_tool_calls,
            ),
        )

        evaluation_paths: dict[str, Path] = {}
        evaluation_failure_path: Path | None = None
        num_evaluation_failures = 0
        resolved_evaluator = (
            _resolve_path(evaluator_csv, base=self.project_root)
            if evaluator_csv is not None
            else None
        )
        resolved_evaluation_output = (
            _resolve_path(evaluation_output_dir, base=self.project_root)
            if evaluation_output_dir is not None
            else self.output_dir / "evaluator_only"
        )
        if resolved_evaluator is not None:
            # This block is deliberately after every expert subprocess and after
            # the complete live predictions/failures artifacts are durable.
            if failures:
                error = stage4_evaluation.Stage4EvaluationError(
                    "Live expert failures prevent complete evaluation; failed tasks must not "
                    "be silently excluded"
                )
                evaluation_failure_path = stage4_evaluation.write_stage4_evaluation_failure(
                    output_dir=resolved_evaluation_output,
                    error=error,
                )
                num_evaluation_failures = 1
            else:
                try:
                    evaluation_result = stage4_evaluation.evaluate_selected_predictions(
                        selected_predictions=selected_path,
                        evaluator_csv=resolved_evaluator,
                    )
                    evaluation_paths = stage4_evaluation.write_stage4_evaluation(
                        result=evaluation_result,
                        output_dir=resolved_evaluation_output,
                        selected_predictions=selected_path,
                        evaluator_csv=resolved_evaluator,
                    )
                except (stage4_evaluation.Stage4EvaluationError, OSError, ValueError) as exc:
                    evaluation_failure_path = (
                        stage4_evaluation.write_stage4_evaluation_failure(
                            output_dir=resolved_evaluation_output,
                            error=exc,
                        )
                    )
                    num_evaluation_failures = 1

        metadata = _run_metadata(
            policy=self.policy,
            data_root=self.data_root,
            output_dir=self.output_dir,
            fold=fold,
            split=split,
            tool_budget=self.tool_budget,
            max_estimated_cost_ms=self.max_estimated_cost_ms,
            tasks=selected_tasks,
            num_decisions=len(decision_rows),
            num_selected_predictions=len(selected_rows),
            num_expert_calls=len(execution_rows),
            num_failures=len(failures),
            num_evaluation_failures=num_evaluation_failures,
            started_at=started_at,
            tasks_path=Path(tasks_path) if tasks_path is not None else None,
            fold_manifest_path=(
                Path(fold_manifest_path) if fold_manifest_path is not None else None
            ),
            policy_state_path=policy_state_path,
            evaluator_csv=resolved_evaluator,
            evaluation_output_dir=(
                resolved_evaluation_output if resolved_evaluator is not None else None
            ),
        )
        _write_json_atomic(metadata_path, metadata)

        return LiveResult(
            route_decisions_path=route_path,
            selected_predictions_path=selected_path,
            live_executions_path=executions_path,
            failures_path=failures_path,
            run_metadata_path=metadata_path,
            budget_summary_path=budget_path,
            policy_state_path=policy_state_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            evaluation_paths=evaluation_paths,
            evaluation_failure_path=evaluation_failure_path,
            num_tasks=len(selected_tasks),
            num_selected_predictions=len(selected_rows),
            num_failures=len(failures),
            num_evaluation_failures=num_evaluation_failures,
        )

    run = execute

    def _decision_budget_error(self, decision: RouteDecision) -> str:
        if decision.tool_calls != 1:
            return (
                "Live RouteDecision must request exactly one expert call; "
                f"got tool_calls={decision.tool_calls}"
            )
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

    def _locate_inputs(self, task: AgentTask) -> tuple[Path, Path]:
        agent_input_csv = (
            self.data_root / "manifests" / f"{task.dataset}_agent_input.csv"
        )
        support_set_csv = (
            self.data_root
            / "support_sets"
            / f"{task.dataset}_k{task.k_shot}_seed{task.seed}.csv"
        )
        if not agent_input_csv.is_file():
            raise _TaskLiveFailure(
                "missing_agent_input_csv",
                f"Could not locate agent_input_csv: {agent_input_csv}",
            )
        if not support_set_csv.is_file():
            raise _TaskLiveFailure(
                "missing_support_set_csv",
                f"Could not locate support_set_csv: {support_set_csv}",
            )
        return agent_input_csv.resolve(), support_set_csv.resolve()

    def _expert_command(
        self,
        *,
        task: AgentTask,
        decision: RouteDecision,
        agent_input_csv: Path,
        support_set_csv: Path,
        output_dir: Path,
    ) -> list[str]:
        return [
            self.python_executable,
            "-m",
            "normroute.cli.run_expert",
            "--expert",
            _normalize_expert_name(decision.selected_expert),
            "--agent-input-csv",
            str(agent_input_csv),
            "--support-set-csv",
            str(support_set_csv),
            "--dataset",
            task.dataset,
            "--category",
            task.category,
            "--k-shot",
            str(task.k_shot),
            "--seed",
            str(task.seed),
            "--support-set-id",
            task.support_set_id,
            "--output-dir",
            str(output_dir),
            "--budget",
            str(self.tool_budget),
            "--limit",
            "1",
        ]

    def _invoke_expert(
        self,
        *,
        task: AgentTask,
        decision: RouteDecision,
        command: list[str],
        agent_input_csv: Path,
        task_agent_input_csv: Path,
        support_set_csv: Path,
        output_dir: Path,
        stdout_log: Path,
        stderr_log: Path,
        fold: str,
        split: str,
    ) -> "_Invocation":
        environment = os.environ.copy()
        source_root = str(self.project_root / "src")
        current_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            source_root
            if not current_pythonpath
            else source_root + os.pathsep + current_pythonpath
        )
        started = self._clock()
        try:
            completed = self._run_subprocess(
                command,
                cwd=self.project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            returncode = int(completed.returncode)
            status = "ok" if returncode == 0 else "failed"
        except Exception as exc:
            stdout = ""
            stderr = str(exc) + "\n"
            returncode = -1
            status = "launch_error"
        actual_runtime_ms = max((self._clock() - started) * 1000.0, 0.0)
        _write_text_atomic(stdout_log, stdout)
        _write_text_atomic(stderr_log, stderr)
        row = {
            "task_id": task.task_id,
            "fold": fold,
            "split": split,
            "policy_name": self.policy.name,
            "selected_expert": decision.selected_expert,
            "agent_input_csv": str(agent_input_csv),
            "task_agent_input_csv": str(task_agent_input_csv),
            "support_set_csv": str(support_set_csv),
            "output_dir": str(output_dir),
            "command": json.dumps(command, ensure_ascii=True),
            "returncode": returncode,
            "estimated_runtime_ms": decision.estimated_cost_ms,
            "actual_runtime_ms": actual_runtime_ms,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "status": status,
        }
        return _Invocation(
            row=row,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            actual_runtime_ms=actual_runtime_ms,
        )

    def _budget_summary(
        self,
        *,
        num_tasks: int,
        decisions: Sequence[Mapping[str, Any]],
        selected_rows: Sequence[Mapping[str, Any]],
        executions: Sequence[Mapping[str, Any]],
        failures: Sequence[Mapping[str, Any]],
        actual_tool_calls: int,
    ) -> dict[str, Any]:
        planned = sum(int(row["tool_calls"]) for row in decisions)
        total_budget = num_tasks * self.tool_budget
        budget_failures = sum(
            1 for failure in failures if failure["failure_type"] == "budget_exceeded"
        )
        return {
            "protocol_version": LIVE_PROTOCOL_VERSION,
            "runtime_source": ACTUAL_RUNTIME_SOURCE,
            "estimated_runtime_source": "route_decision",
            "tool_budget_per_task": self.tool_budget,
            "max_estimated_cost_ms_per_task": self.max_estimated_cost_ms,
            "num_tasks": num_tasks,
            "num_decisions": len(decisions),
            "num_expert_calls": len(executions),
            "actual_expert_calls": len(executions),
            "num_selected_predictions": len(selected_rows),
            "num_budget_failures": budget_failures,
            "total_tool_call_budget": total_budget,
            "planned_tool_calls": planned,
            "actual_tool_calls": actual_tool_calls,
            "actual_prediction_tool_calls": actual_tool_calls,
            "planned_estimated_runtime_ms": sum(
                float(row["estimated_cost_ms"]) for row in decisions
            ),
            "actual_subprocess_runtime_ms": sum(
                float(row["actual_runtime_ms"]) for row in executions
            ),
            "planned_budget_remaining": max(total_budget - planned, 0),
            "within_budget": budget_failures == 0,
        }


@dataclass(frozen=True)
class _Invocation:
    row: dict[str, Any]
    stdout: str
    stderr: str
    returncode: int
    actual_runtime_ms: float


class _TaskLiveFailure(LiveExecutionError):
    def __init__(
        self, code: str, message: str, *, returncode: int | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.returncode = returncode


def execute_live(
    *,
    tasks: Sequence[AgentTask | Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    policy: Policy,
    data_root: str | Path,
    output_dir: str | Path,
    fold: str,
    split: str = "test",
    tool_budget: int = 1,
    max_estimated_cost_ms: float | None = None,
    train_records: Sequence[TrainRecord] | None = None,
    tasks_path: str | Path | None = None,
    fold_manifest_path: str | Path | None = None,
    evaluator_csv: str | Path | None = None,
    evaluation_output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    subprocess_runner: Runner | None = None,
    clock: Clock | None = None,
) -> LiveResult:
    """Functional wrapper around :class:`LiveExecutor`."""

    return LiveExecutor(
        data_root=data_root,
        output_dir=output_dir,
        policy=policy,
        tool_budget=tool_budget,
        max_estimated_cost_ms=max_estimated_cost_ms,
        project_root=project_root,
        python_executable=python_executable,
        subprocess_runner=subprocess_runner,
        clock=clock,
    ).execute(
        tasks=tasks,
        manifest_rows=manifest_rows,
        fold=fold,
        split=split,
        train_records=train_records,
        tasks_path=tasks_path,
        fold_manifest_path=fold_manifest_path,
        evaluator_csv=evaluator_csv,
        evaluation_output_dir=evaluation_output_dir,
    )


def _write_task_agent_input(
    *, source: Path, destination: Path, task: AgentTask
) -> None:
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in AGENT_INPUT_COLUMNS if column not in fieldnames]
        forbidden = sorted(
            {
                "label",
                "mask",
                "mask_path",
                "defect_type",
                "anomaly_type",
                "ground_truth",
            }.intersection(fieldnames)
        )
        if missing or forbidden:
            raise _TaskLiveFailure(
                "invalid_agent_input_csv",
                f"{source} has invalid columns; missing={missing}, forbidden={forbidden}",
            )
        matches = []
        for line_number, row in enumerate(reader, start=2):
            clean = {key: (value or "").strip() for key, value in row.items()}
            if clean.get("image_id") != task.sample_id:
                continue
            mismatched = [
                field
                for field, expected in (
                    ("dataset", task.dataset),
                    ("category", task.category),
                    ("split", "test"),
                )
                if clean.get(field) != expected
            ]
            if mismatched:
                raise _TaskLiveFailure(
                    "agent_input_task_mismatch",
                    f"{source}:{line_number} disagrees with task_id={task.task_id!r}: "
                    f"{mismatched}",
                )
            matches.append({column: clean[column] for column in AGENT_INPUT_COLUMNS})
    if len(matches) != 1:
        raise _TaskLiveFailure(
            "agent_input_match_error",
            f"Expected exactly one agent-input row for task_id={task.task_id!r}, "
            f"sample_id={task.sample_id!r} in {source}; found {len(matches)}",
        )
    _write_csv_atomic(destination, AGENT_INPUT_COLUMNS, matches)


def _read_task_prediction(
    *, task: AgentTask, decision: RouteDecision, predictions_path: Path
) -> dict[str, str]:
    if not predictions_path.is_file():
        raise _TaskLiveFailure(
            "missing_expert_predictions",
            f"Live expert did not write predictions.csv: {predictions_path}",
        )
    rows = _read_stage2_predictions(predictions_path, decision.selected_expert)
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
    if len(rows) != 1 or len(matches) != 1:
        raise _TaskLiveFailure(
            "expert_prediction_match_error",
            f"Expected exactly one matching live prediction for task_id={task.task_id!r} "
            f"in {predictions_path}; rows={len(rows)}, matches={len(matches)}",
        )
    return matches[0]


def _run_metadata(
    *,
    policy: Policy,
    data_root: Path,
    output_dir: Path,
    fold: str,
    split: str,
    tool_budget: int,
    max_estimated_cost_ms: float | None,
    tasks: Sequence[AgentTask],
    num_decisions: int,
    num_selected_predictions: int,
    num_expert_calls: int,
    num_failures: int,
    num_evaluation_failures: int,
    started_at: str,
    tasks_path: Path | None,
    fold_manifest_path: Path | None,
    policy_state_path: Path,
    evaluator_csv: Path | None,
    evaluation_output_dir: Path | None,
) -> dict[str, Any]:
    config = {
        "mode": "live",
        "policy_name": policy.name,
        "policy_configuration": policy.configuration(),
        "fold": fold,
        "split": split,
        "tool_budget_per_task": tool_budget,
        "max_estimated_cost_ms_per_task": max_estimated_cost_ms,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "evaluator_csv": str(evaluator_csv or ""),
        "evaluation_output_dir": str(evaluation_output_dir or ""),
        "runtime_source": ACTUAL_RUNTIME_SOURCE,
        "estimated_runtime_source": "route_decision",
    }
    seeds = sorted({task.seed for task in tasks})
    return {
        "protocol_version": LIVE_PROTOCOL_VERSION,
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
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "policy_state_path": str(policy_state_path),
        "num_tasks": len(tasks),
        "num_decisions": num_decisions,
        "num_expert_calls": num_expert_calls,
        "num_selected_predictions": num_selected_predictions,
        "num_failures": num_failures,
        "num_evaluation_failures": num_evaluation_failures,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "git_commit": _git_commit(),
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
        },
    }


def _resolve_path(path: str | Path, *, base: Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else base / value).resolve()


def _task_directory_name(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip(".")
    if safe == task_id and safe not in {"", ".", ".."}:
        return safe
    import hashlib

    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:10]
    return f"{(safe or 'task')[:80]}-{digest}"


def _aggregate_log_chunk(task_id: str, content: str) -> str:
    header = f"===== task_id={task_id} =====\n"
    if not content:
        return header
    return header + content + ("" if content.endswith("\n") else "\n")


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _same_int(value: Any, expected: int) -> bool:
    try:
        return int(value) == expected
    except (TypeError, ValueError):
        return False
