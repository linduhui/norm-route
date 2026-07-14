import copy

import pytest

from src.normroute.agent.protocol import (
    FORBIDDEN_PRE_ROUTE_FIELDS,
    POLICY_FEATURE_ALLOWLIST,
    PROVENANCE_FIELDS,
)
from src.normroute.agent.task_builder import TaskBuildError, build_pre_route_task


def _stage3_task() -> dict[str, object]:
    return {
        "task_id": "task-0",
        "sample_id": "sample-1",
        "dataset": "mvtec",
        "category": "bottle",
        "support_set_id": "mvtec_bottle_k1_seed0",
        "k_shot": 1,
        "seed": 0,
        "expert_scores": {
            "PatchCore": 0.1,
            "WinCLIP": 0.2,
            "AnomalyDINO": 0.3,
        },
    }


@pytest.mark.parametrize("forbidden_field", sorted(FORBIDDEN_PRE_ROUTE_FIELDS))
def test_task_builder_rejects_forbidden_fields_at_any_depth(forbidden_field: str) -> None:
    source = copy.deepcopy(_stage3_task())
    source["nested"] = {forbidden_field: "leak"}

    with pytest.raises(TaskBuildError, match="forbidden fields"):
        build_pre_route_task(source)


def test_seed_and_support_set_are_provenance_not_policy_features() -> None:
    task = build_pre_route_task(_stage3_task())

    assert set(PROVENANCE_FIELDS) == {"seed", "support_set_id"}
    assert set(PROVENANCE_FIELDS).isdisjoint(POLICY_FEATURE_ALLOWLIST)
    assert set(PROVENANCE_FIELDS).isdisjoint(task["policy_features"])
    assert task["seed"] == 0
    assert task["support_set_id"] == "mvtec_bottle_k1_seed0"


def test_realized_stage3_expert_scores_are_not_copied() -> None:
    task = build_pre_route_task(_stage3_task())
    serialized_keys = set(task)

    assert "expert_scores" not in serialized_keys
    assert FORBIDDEN_PRE_ROUTE_FIELDS.isdisjoint(serialized_keys)
