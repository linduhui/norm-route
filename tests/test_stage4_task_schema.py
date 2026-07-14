import json
from pathlib import Path

import pytest

from src.normroute.agent.protocol import (
    CANDIDATE_EXPERTS,
    POLICY_FEATURE_ALLOWLIST,
    PRE_ROUTE_TASK_FIELDS,
    STAGE4_TASK_PROTOCOL_VERSION,
    Stage4ProtocolError,
    validate_pre_route_task,
)
from src.normroute.agent.task_builder import (
    TaskBuildError,
    build_pre_route_task,
    build_pre_route_tasks,
    read_pre_route_tasks,
)


def _stage3_task(*, seed: int = 0) -> dict[str, object]:
    return {
        "task_id": f"task-{seed}",
        "sample_id": "sample-1",
        "dataset": "mvtec",
        "category": "bottle",
        "support_set_id": f"mvtec_bottle_k1_seed{seed}",
        "k_shot": 1,
        "seed": seed,
        "expert_scores": {
            "PatchCore": 0.1,
            "WinCLIP": 0.2,
            "AnomalyDINO": 0.3,
        },
    }


def test_pre_route_task_has_exact_schema_and_policy_features() -> None:
    task = build_pre_route_task(_stage3_task())

    assert set(task) == PRE_ROUTE_TASK_FIELDS
    assert task["protocol_version"] == STAGE4_TASK_PROTOCOL_VERSION
    assert task["candidate_experts"] == list(CANDIDATE_EXPERTS)
    assert task["policy_features"] == {
        "dataset": "mvtec",
        "category": "bottle",
        "k_shot": 1,
    }
    assert tuple(task["policy_features"]) == POLICY_FEATURE_ALLOWLIST
    assert "expert_scores" not in task
    validate_pre_route_task(task)


def test_build_pre_route_tasks_round_trips_jsonl_without_skips(tmp_path: Path) -> None:
    source = tmp_path / "agent_routing_tasks.jsonl"
    source.write_text(
        "".join(json.dumps(_stage3_task(seed=seed)) + "\n" for seed in range(2)),
        encoding="utf-8",
    )
    output = tmp_path / "stage4" / "pre_route_tasks.jsonl"

    written = build_pre_route_tasks(source, output)
    tasks = read_pre_route_tasks(written)

    assert written == output
    assert [task["task_id"] for task in tasks] == ["task-0", "task-1"]


def test_pre_route_schema_rejects_unrecognized_top_level_fields() -> None:
    task = build_pre_route_task(_stage3_task())
    task["expert_scores"] = {"PatchCore": 0.1}

    with pytest.raises(Stage4ProtocolError, match="unexpected fields"):
        validate_pre_route_task(task)


def test_task_builder_rejects_blank_rows_instead_of_skipping(tmp_path: Path) -> None:
    source = tmp_path / "agent_routing_tasks.jsonl"
    source.write_text(json.dumps(_stage3_task()) + "\n\n", encoding="utf-8")

    with pytest.raises(TaskBuildError, match="must not be skipped"):
        build_pre_route_tasks(source, tmp_path / "pre_route_tasks.jsonl")
