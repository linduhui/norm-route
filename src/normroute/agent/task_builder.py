"""Build leakage-safe Stage 4 pre-route tasks from Stage 3 Agent tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .protocol import (
    CANDIDATE_EXPERTS,
    FORBIDDEN_PRE_ROUTE_FIELDS,
    PreRouteTask,
    Stage4ProtocolError,
    validate_no_forbidden_fields,
    validate_pre_route_task,
)


REQUIRED_STAGE3_FIELDS = (
    "task_id",
    "sample_id",
    "dataset",
    "category",
    "support_set_id",
    "k_shot",
    "seed",
)


class TaskBuildError(Stage4ProtocolError):
    """Raised when Stage 3 tasks cannot be safely converted to Stage 4."""


def build_pre_route_task(source: Mapping[str, Any], *, context: str = "Stage 3 task") -> dict[str, Any]:
    """Allowlist one Stage 3 task into the frozen Stage 4 pre-route schema."""

    if not isinstance(source, Mapping):
        raise TaskBuildError(f"{context} must be a JSON object")
    try:
        validate_no_forbidden_fields(source, context=context)
    except Stage4ProtocolError as exc:
        raise TaskBuildError(str(exc)) from exc

    missing = [field for field in REQUIRED_STAGE3_FIELDS if field not in source]
    if missing:
        raise TaskBuildError(f"{context} is missing required fields: {missing}")

    task = PreRouteTask(
        task_id=source["task_id"],
        sample_id=source["sample_id"],
        dataset=source["dataset"],
        category=source["category"],
        support_set_id=source["support_set_id"],
        k_shot=source["k_shot"],
        seed=source["seed"],
        candidate_experts=CANDIDATE_EXPERTS,
    )
    try:
        return task.to_dict()
    except Stage4ProtocolError as exc:
        raise TaskBuildError(f"{context}: {exc}") from exc


def read_stage3_agent_tasks(path: str | Path) -> list[dict[str, Any]]:
    """Read every Stage 3 JSONL task and fail on malformed, blank, or duplicate rows."""

    source_path = Path(path)
    if not source_path.is_file():
        raise TaskBuildError(f"Stage 3 task file does not exist: {source_path}")

    tasks: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    with source_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            context = f"{source_path}:{line_number}"
            if not line.strip():
                raise TaskBuildError(f"{context} is blank; failed tasks must not be skipped")
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TaskBuildError(f"{context} is invalid JSON: {exc.msg}") from exc
            if not isinstance(source, dict):
                raise TaskBuildError(f"{context} must contain a JSON object")
            task = build_pre_route_task(source, context=context)
            task_id = task["task_id"]
            if task_id in seen_task_ids:
                raise TaskBuildError(f"{context} has duplicate task_id={task_id!r}")
            seen_task_ids.add(task_id)
            tasks.append(task)

    if not tasks:
        raise TaskBuildError(f"Stage 3 task file is empty: {source_path}")
    return tasks


def build_pre_route_tasks(
    source_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Convert a complete Stage 3 JSONL file and atomically write Stage 4 tasks."""

    source = Path(source_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise TaskBuildError("Source and output task paths must be different")
    _ensure_agent_visible_path(destination)

    tasks = read_stage3_agent_tasks(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for task in tasks:
                validate_pre_route_task(task, context=str(destination))
                handle.write(json.dumps(task, sort_keys=True, ensure_ascii=True) + "\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def read_pre_route_tasks(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate a complete Stage 4 pre-route JSONL file."""

    task_path = Path(path)
    if not task_path.is_file():
        raise TaskBuildError(f"Pre-route task file does not exist: {task_path}")

    tasks: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    with task_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            context = f"{task_path}:{line_number}"
            if not line.strip():
                raise TaskBuildError(f"{context} is blank; failed tasks must not be skipped")
            try:
                task = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TaskBuildError(f"{context} is invalid JSON: {exc.msg}") from exc
            if not isinstance(task, dict):
                raise TaskBuildError(f"{context} must contain a JSON object")
            try:
                validate_pre_route_task(task, context=context)
            except Stage4ProtocolError as exc:
                raise TaskBuildError(str(exc)) from exc
            task_id = task["task_id"]
            if task_id in seen_task_ids:
                raise TaskBuildError(f"{context} has duplicate task_id={task_id!r}")
            seen_task_ids.add(task_id)
            tasks.append(task)

    if not tasks:
        raise TaskBuildError(f"Pre-route task file is empty: {task_path}")
    return tasks


def _ensure_agent_visible_path(path: Path) -> None:
    parts = {part.lower() for part in path.parts}
    if "evaluator_only" in parts or "oracle" in parts:
        raise TaskBuildError(
            "Pre-route tasks cannot be written under evaluator_only or oracle paths"
        )

    forbidden_name = FORBIDDEN_PRE_ROUTE_FIELDS.intersection({path.stem.lower()})
    if forbidden_name:
        raise TaskBuildError(f"Pre-route output path uses forbidden name: {path}")
