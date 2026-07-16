"""Replay multiple Stage 4 policies across multiple frozen seed-CV folds."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

from ..agent.policy import Policy
from ..agent.policy_registry import create_policy, list_policies
from ..agent.protocol import AgentTask, CANDIDATE_EXPERTS
from ..agent.replay_executor import (
    ROUTE_DECISION_COLUMNS,
    SELECTED_PREDICTION_COLUMNS,
    ReplayExecutionError,
    ReplayResult,
    execute_replay,
    read_fold_manifest,
)
from ..agent.task_builder import TaskBuildError, read_pre_route_tasks
from ..evaluation.export import PREDICTION_COLUMNS
from ..policies.learned import (
    LEARNED_METADATA_POLICY_NAMES,
    MODEL_ARTIFACT_FILENAME,
)


GRID_PROTOCOL_VERSION = "stage4.baseline_grid.v1"
DEFAULT_CONFIG_PATH = Path("configs/stage4/baselines.json")
GRID_RESULT_COLUMNS = (
    "policy_name",
    "fold",
    "split",
    "seed",
    "status",
    "num_tasks",
    "num_selected_predictions",
    "num_failures",
    "output_dir",
    "error_message",
)
_FORBIDDEN_TRAINING_COLUMNS = frozenset(
    {
        "label",
        "mask",
        "mask_path",
        "defect_type",
        "anomaly_type",
        "oracle_best_expert",
    }
)


class Stage4GridError(ValueError):
    """Raised when a Stage 4 baseline grid cannot be replayed safely."""


@dataclass(frozen=True)
class Stage4GridConfig:
    seed: int
    folds: tuple[str, ...]
    policies: tuple[str, ...]
    split: str
    tool_budget: int
    max_estimated_cost_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": GRID_PROTOCOL_VERSION,
            "seed": self.seed,
            "folds": list(self.folds),
            "policies": list(self.policies),
            "split": self.split,
            "tool_budget": self.tool_budget,
            "max_estimated_cost_ms": self.max_estimated_cost_ms,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Stage 4 fixed/random/runtime baselines over folds and policies."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--policies", nargs="+", choices=list_policies())
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--tool-budget", "--budget", dest="tool_budget", type=int)
    parser.add_argument("--max-estimated-cost-ms", type=float)
    parser.add_argument(
        "--tasks", default="outputs/stage4/tasks/pre_route_tasks.jsonl"
    )
    parser.add_argument(
        "--fold-manifest", default="outputs/stage4/splits/fold_manifest.csv"
    )
    parser.add_argument("--stage2-root", default="outputs/stage2")
    parser.add_argument("--output-root", default="outputs/stage4/runs")
    parser.add_argument("--expert-cost-card")
    parser.add_argument(
        "--policy-artifact-root",
        default="outputs/stage4/policies",
        help="Root containing <policy>/<fold>/policy_artifact.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        config = load_grid_config(args.config)
        config = Stage4GridConfig(
            seed=config.seed if args.seed is None else args.seed,
            folds=config.folds if args.folds is None else tuple(args.folds),
            policies=config.policies if args.policies is None else tuple(args.policies),
            split=config.split if args.split is None else args.split,
            tool_budget=config.tool_budget if args.tool_budget is None else args.tool_budget,
            max_estimated_cost_ms=(
                config.max_estimated_cost_ms
                if args.max_estimated_cost_ms is None
                else args.max_estimated_cost_ms
            ),
        )
        result = run_stage4_grid(
            config=config,
            tasks_path=args.tasks,
            fold_manifest_path=args.fold_manifest,
            stage2_root=args.stage2_root,
            output_root=args.output_root,
            expert_cost_card=args.expert_cost_card,
            policy_artifact_root=args.policy_artifact_root,
            config_path=args.config,
        )
    except (Stage4GridError, ReplayExecutionError, TaskBuildError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Wrote {result['results_path']}")
    print(f"Wrote {result['failures_path']}")
    print(f"Wrote {result['metadata_path']}")
    if result["num_failed"]:
        raise SystemExit(1)


def load_grid_config(path: str | Path) -> Stage4GridConfig:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage4GridError(f"Could not read Stage 4 grid config {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Stage4GridError("Stage 4 grid config root must be a JSON object")
    if payload.get("protocol_version") != GRID_PROTOCOL_VERSION:
        raise Stage4GridError(
            f"Stage 4 grid config protocol_version must be {GRID_PROTOCOL_VERSION!r}"
        )
    return _validate_grid_config(
        seed=payload.get("seed"),
        folds=payload.get("folds"),
        policies=payload.get("policies"),
        split=payload.get("split", "test"),
        tool_budget=payload.get("tool_budget", 1),
        max_estimated_cost_ms=payload.get("max_estimated_cost_ms"),
    )


def run_stage4_grid(
    *,
    config: Stage4GridConfig,
    tasks_path: str | Path,
    fold_manifest_path: str | Path,
    stage2_root: str | Path,
    output_root: str | Path,
    expert_cost_card: str | Path | Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    policy_artifact_root: str | Path = "outputs/stage4/policies",
) -> dict[str, Any]:
    """Run every requested policy/fold pair and record every failure explicitly."""

    config = _validate_grid_config(**{key: value for key, value in config.to_dict().items() if key != "protocol_version"})
    task_rows = read_pre_route_tasks(tasks_path)
    tasks = [AgentTask.from_mapping(row) for row in task_rows]
    manifest_rows = read_fold_manifest(fold_manifest_path)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    grid_failures: list[dict[str, Any]] = []
    runtime_cache: dict[str, list[dict[str, Any]]] = {}
    for policy_name, fold in itertools.product(config.policies, config.folds):
        combo_dir = output_path / policy_name / fold / config.split
        try:
            policy = _create_grid_policy(
                policy_name,
                seed=config.seed,
                expert_cost_card=expert_cost_card,
                fold=fold,
                policy_artifact_root=policy_artifact_root,
            )
            train_records: Sequence[Mapping[str, Any]] | None = None
            if policy_name == "fastest_expert" and expert_cost_card is None:
                if fold not in runtime_cache:
                    runtime_cache[fold] = build_training_runtime_records(
                        tasks=tasks,
                        manifest_rows=manifest_rows,
                        stage2_root=stage2_root,
                        fold=fold,
                    )
                train_records = runtime_cache[fold]
            result = execute_replay(
                tasks=tasks,
                manifest_rows=manifest_rows,
                policy=policy,
                stage2_root=stage2_root,
                output_dir=combo_dir,
                fold=fold,
                split=config.split,
                tool_budget=config.tool_budget,
                max_estimated_cost_ms=config.max_estimated_cost_ms,
                train_records=train_records,
                tasks_path=tasks_path,
                fold_manifest_path=fold_manifest_path,
            )
            record = _result_record(
                result=result,
                policy_name=policy_name,
                fold=fold,
                split=config.split,
                seed=config.seed,
                output_dir=combo_dir,
            )
            records.append(record)
            if result.num_failures:
                grid_failures.append(
                    {
                        "policy_name": policy_name,
                        "fold": fold,
                        "failure_type": "replay_task_failures",
                        "error_message": (
                            f"Replay recorded {result.num_failures} failed task(s); "
                            f"see {result.failures_path}"
                        ),
                        "traceback": "",
                    }
                )
        except Exception as exc:
            _write_failed_combo_artifacts(
                output_dir=combo_dir,
                policy_name=policy_name,
                fold=fold,
                split=config.split,
                seed=config.seed,
                config=config,
                error=exc,
            )
            records.append(
                {
                    "policy_name": policy_name,
                    "fold": fold,
                    "split": config.split,
                    "seed": config.seed,
                    "status": "error",
                    "num_tasks": 0,
                    "num_selected_predictions": 0,
                    "num_failures": 1,
                    "output_dir": str(combo_dir),
                    "error_message": str(exc),
                }
            )
            grid_failures.append(
                {
                    "policy_name": policy_name,
                    "fold": fold,
                    "failure_type": "grid_combo_error",
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    results_path = output_path / "grid_results.csv"
    failures_path = output_path / "grid_failures.json"
    metadata_path = output_path / "grid_metadata.json"
    config_output_path = output_path / "grid_config.json"
    _write_csv(results_path, GRID_RESULT_COLUMNS, records)
    _write_json(
        failures_path,
        {
            "protocol_version": GRID_PROTOCOL_VERSION,
            "num_failed": len(grid_failures),
            "failures": grid_failures,
        },
    )
    _write_json(config_output_path, config.to_dict())
    _write_json(
        metadata_path,
        {
            "protocol_version": GRID_PROTOCOL_VERSION,
            "config": config.to_dict(),
            "seed": config.seed,
            "git_commit": _git_commit(),
            "environment": _environment(),
            "tasks_path": str(tasks_path),
            "tasks_sha256": _sha256(Path(tasks_path)),
            "fold_manifest_path": str(fold_manifest_path),
            "fold_manifest_sha256": _sha256(Path(fold_manifest_path)),
            "config_path": str(config_path or ""),
            "config_sha256": _optional_sha256(Path(config_path)) if config_path else "",
            "expert_cost_card_path": (
                str(expert_cost_card) if isinstance(expert_cost_card, (str, Path)) else ""
            ),
            "policy_artifact_root": str(policy_artifact_root),
            "num_combinations": len(records),
            "num_failed": len(grid_failures),
            "completed_at_utc": _utc_now(),
        },
    )
    return {
        "records": records,
        "num_total": len(records),
        "num_failed": len(grid_failures),
        "results_path": results_path,
        "failures_path": failures_path,
        "metadata_path": metadata_path,
        "config_path": config_output_path,
    }


def build_training_runtime_records(
    *,
    tasks: Sequence[AgentTask],
    manifest_rows: Sequence[Mapping[str, Any]],
    stage2_root: str | Path,
    fold: str,
) -> list[dict[str, Any]]:
    """Read runtime only from the named fold's training Stage 2 runs."""

    task_by_id = {task.task_id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise Stage4GridError("Duplicate task_id while building training runtimes")
    training_ids: list[str] = []
    for row in manifest_rows:
        if str(row.get("fold", "")).strip() != fold:
            continue
        task_id = str(row.get("task_id", "")).strip()
        if task_id not in task_by_id:
            raise Stage4GridError(
                f"Fold {fold!r} runtime manifest contains unknown task_id={task_id!r}"
            )
        if str(row.get("split", "")).strip() == "train":
            training_ids.append(task_id)
    if not training_ids:
        raise Stage4GridError(f"Fold {fold!r} has no training tasks for runtime estimation")
    if len(training_ids) != len(set(training_ids)):
        raise Stage4GridError(f"Fold {fold!r} has duplicate training task assignments")

    root = Path(stage2_root)
    cache: dict[Path, list[dict[str, str]]] = {}
    records: list[dict[str, Any]] = []
    for task_id in training_ids:
        task = task_by_id[task_id]
        for expert in task.candidate_experts:
            run_dir = (
                root
                / _normalize_expert(expert)
                / task.dataset
                / task.category
                / f"k{task.k_shot}"
                / f"seed{task.seed}"
                / task.support_set_id
            )
            predictions_path = run_dir / "predictions.csv"
            rows = cache.get(predictions_path)
            if rows is None:
                rows = _read_training_predictions(predictions_path, expert)
                cache[predictions_path] = rows
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
                raise Stage4GridError(
                    f"Expected one training runtime prediction for task_id={task.task_id!r}, "
                    f"expert={expert}; found {len(matches)} in {predictions_path}"
                )
            prediction = matches[0]
            if prediction["status"].strip().lower() != "ok":
                raise Stage4GridError(
                    f"Training runtime source failed for task_id={task.task_id!r}, "
                    f"expert={expert}: {prediction.get('error_message', '')}"
                )
            runtime_ms = _finite_nonnegative(
                prediction["runtime_ms"], f"{predictions_path} runtime_ms"
            )
            records.append(
                {
                    "fold": fold,
                    "split": "train",
                    "runtime_source": "current_fold_training_run",
                    "dataset": task.dataset,
                    "category": task.category,
                    "k_shot": task.k_shot,
                    "expert_name": expert,
                    "runtime_ms": runtime_ms,
                }
            )
    return records


def _read_training_predictions(path: Path, expected_expert: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise Stage4GridError(f"Training Stage 2 predictions do not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in PREDICTION_COLUMNS if column not in fieldnames]
        forbidden = sorted(_FORBIDDEN_TRAINING_COLUMNS.intersection(fieldnames))
        if missing or forbidden:
            raise Stage4GridError(
                f"{path} is not a legal runtime source; missing={missing}, forbidden={forbidden}"
            )
        rows = [
            {key: (value or "").strip() for key, value in row.items()} for row in reader
        ]
    if not rows:
        raise Stage4GridError(f"Training Stage 2 predictions are empty: {path}")
    experts = {_normalize_expert(row.get("expert_name", "")) for row in rows}
    if experts != {_normalize_expert(expected_expert)}:
        raise Stage4GridError(
            f"Training runtime file {path} mixes experts or has wrong expert: {sorted(experts)}"
        )
    return rows


def _create_grid_policy(
    name: str,
    *,
    seed: int,
    expert_cost_card: str | Path | Mapping[str, Any] | None,
    fold: str,
    policy_artifact_root: str | Path,
) -> Policy:
    if name == "random_seeded":
        return create_policy(name, seed=seed)
    if name == "fastest_expert" and expert_cost_card is not None:
        return create_policy(name, cost_card=expert_cost_card)
    if name in {
        "category_prior",
        "category_shot_prior",
        "cost_aware",
        *LEARNED_METADATA_POLICY_NAMES,
    }:
        artifact_name = (
            MODEL_ARTIFACT_FILENAME
            if name in LEARNED_METADATA_POLICY_NAMES
            else "policy_artifact.json"
        )
        artifact = Path(policy_artifact_root) / name / fold / artifact_name
        if not artifact.is_file():
            raise Stage4GridError(
                f"Final policy artifact does not exist for policy={name!r}, "
                f"fold={fold!r}: {artifact}"
            )
        return create_policy(name, artifact=artifact)
    return create_policy(name)


def _validate_grid_config(
    *,
    seed: Any,
    folds: Any,
    policies: Any,
    split: Any,
    tool_budget: Any,
    max_estimated_cost_ms: Any,
) -> Stage4GridConfig:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Stage4GridError("Stage 4 grid seed must be an integer >= 0")
    folds_tuple = _nonempty_unique_strings(folds, "folds")
    policies_tuple = _nonempty_unique_strings(policies, "policies")
    unknown = sorted(set(policies_tuple) - set(list_policies()))
    if unknown:
        raise Stage4GridError(f"Unknown Stage 4 policies: {unknown}")
    if split not in {"train", "val", "test"}:
        raise Stage4GridError("Stage 4 grid split must be train, val, or test")
    if isinstance(tool_budget, bool) or not isinstance(tool_budget, int) or tool_budget < 0:
        raise Stage4GridError("Stage 4 grid tool_budget must be an integer >= 0")
    if max_estimated_cost_ms is not None:
        max_estimated_cost_ms = _finite_nonnegative(
            max_estimated_cost_ms, "max_estimated_cost_ms"
        )
    return Stage4GridConfig(
        seed=seed,
        folds=folds_tuple,
        policies=policies_tuple,
        split=split,
        tool_budget=tool_budget,
        max_estimated_cost_ms=max_estimated_cost_ms,
    )


def _nonempty_unique_strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise Stage4GridError(f"Stage 4 grid {field} must be a non-empty list")
    cleaned = tuple(str(item).strip() for item in value)
    if any(not item for item in cleaned) or len(cleaned) != len(set(cleaned)):
        raise Stage4GridError(f"Stage 4 grid {field} must contain unique non-empty strings")
    return cleaned


def _result_record(
    *,
    result: ReplayResult,
    policy_name: str,
    fold: str,
    split: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "policy_name": policy_name,
        "fold": fold,
        "split": split,
        "seed": seed,
        "status": "ok" if result.num_failures == 0 else "failed_tasks",
        "num_tasks": result.num_tasks,
        "num_selected_predictions": result.num_selected_predictions,
        "num_failures": result.num_failures,
        "output_dir": str(output_dir),
        "error_message": "",
    }


def _write_failed_combo_artifacts(
    *,
    output_dir: Path,
    policy_name: str,
    fold: str,
    split: str,
    seed: int,
    config: Stage4GridConfig,
    error: Exception,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "route_decisions.csv", ROUTE_DECISION_COLUMNS, [])
    _write_csv(output_dir / "selected_predictions.csv", SELECTED_PREDICTION_COLUMNS, [])
    _write_json(
        output_dir / "failures.json",
        {
            "protocol_version": GRID_PROTOCOL_VERSION,
            "num_failed": 1,
            "failed_tasks": [],
            "grid_failure": {
                "failure_type": "grid_combo_error",
                "error_message": str(error),
            },
        },
    )
    _write_json(
        output_dir / "run_metadata.json",
        {
            "protocol_version": GRID_PROTOCOL_VERSION,
            "stage": "stage4",
            "config": config.to_dict(),
            "seed": seed,
            "policy_name": policy_name,
            "fold": fold,
            "split": split,
            "git_commit": _git_commit(),
            "environment": _environment(),
            "num_failures": 1,
            "completed_at_utc": _utc_now(),
        },
    )
    _write_json(
        output_dir / "budget_summary.json",
        {
            "protocol_version": GRID_PROTOCOL_VERSION,
            "tool_budget_per_task": config.tool_budget,
            "num_tasks": 0,
            "num_budget_failures": 0,
            "within_budget": False,
        },
    )
    _write_json(
        output_dir / "policy.json",
        {
            "protocol_version": GRID_PROTOCOL_VERSION,
            "policy_name": policy_name,
            "status": "initialization_failed",
            "error_message": str(error),
        },
    )


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _normalize_expert(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _same_int(value: Any, expected: int) -> bool:
    try:
        return int(value) == expected
    except (TypeError, ValueError):
        return False


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise Stage4GridError(f"{field} must be a finite number >= 0")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage4GridError(f"{field} must be a finite number >= 0") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise Stage4GridError(f"{field} must be a finite number >= 0")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256(path: Path) -> str:
    return _sha256(path) if path.is_file() else ""


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
