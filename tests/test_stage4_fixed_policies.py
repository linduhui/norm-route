from pathlib import Path

import pytest

from src.normroute.agent.policy_registry import create_policy, list_policies
from src.normroute.agent.protocol import AgentTask, CANDIDATE_EXPERTS
from src.normroute.policies.fixed import (
    AlwaysPatchCorePolicy,
    AlwaysWinCLIPPolicy,
    FastestExpertPolicy,
    RandomSeededPolicy,
)


def _task(index: int = 0) -> AgentTask:
    return AgentTask(
        task_id=f"task-{index}",
        sample_id=f"sample-{index}",
        dataset="mvtec",
        category="bottle",
        k_shot=1,
        seed=0,
        support_set_id="mvtec_bottle_k1_seed0",
    )


@pytest.mark.parametrize(
    ("policy_name", "expected_expert", "policy_type"),
    [
        ("always_anomalydino", "AnomalyDINO", object),
        ("always_patchcore", "PatchCore", AlwaysPatchCorePolicy),
        ("always_winclip", "WinCLIP", AlwaysWinCLIPPolicy),
    ],
)
def test_fixed_policies_select_exactly_the_named_expert(
    policy_name: str,
    expected_expert: str,
    policy_type: type,
    tmp_path: Path,
) -> None:
    assert policy_name in list_policies()
    policy = create_policy(policy_name)
    decision = policy.select(_task())

    assert isinstance(policy, policy_type)
    assert decision.selected_expert == expected_expert
    assert decision.selected_expert in CANDIDATE_EXPERTS
    assert decision.tool_calls == 1
    assert decision.estimated_cost_ms == 0.0
    assert {"label", "mask_path", "defect_type"}.isdisjoint(decision.to_dict())

    state = policy.save(tmp_path / policy_name / "policy.json")
    loaded = type(policy).load(state)
    assert loaded.select(_task()).selected_expert == expected_expert


def test_random_seeded_is_reproducible_and_seed_is_persisted(tmp_path: Path) -> None:
    first = RandomSeededPolicy(seed=73)
    second = RandomSeededPolicy(seed=73)
    first_choices = [first.select(_task(index)).selected_expert for index in range(20)]
    second_choices = [second.select(_task(index)).selected_expert for index in range(20)]

    assert first_choices == second_choices
    assert first.configuration() == {"seed": 73}

    state = first.save(tmp_path / "random" / "policy.json")
    loaded = RandomSeededPolicy.load(state)
    assert loaded.configuration() == {"seed": 73}
    assert loaded.select(_task(21)).selected_expert == first.select(_task(21)).selected_expert


def test_fastest_expert_uses_training_runtime_only_and_has_one_call() -> None:
    policy = FastestExpertPolicy()
    policy.fit(
        [
            {
                "fold": "fold0",
                "split": "train",
                "runtime_source": "current_fold_training_run",
                "dataset": "mvtec",
                "category": "bottle",
                "k_shot": 1,
                "expert_name": expert,
                "runtime_ms": runtime,
            }
            for expert, runtime in (
                ("PatchCore", 9.0),
                ("WinCLIP", 4.0),
                ("AnomalyDINO", 12.0),
            )
        ]
    )

    decision = policy.select(_task())
    assert decision.selected_expert == "WinCLIP"
    assert decision.estimated_cost_ms == pytest.approx(4.0)
    assert decision.tool_calls == 1
    assert "quality" in decision.decision_reason


@pytest.mark.parametrize(
    "bad_record",
    [
        {
            "split": "test",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 1,
            "expert_name": "PatchCore",
            "runtime_ms": 1.0,
        },
        {
            "split": "train",
            "dataset": "mvtec",
            "category": "bottle",
            "k_shot": 1,
            "expert_name": "PatchCore",
            "runtime_ms": 1.0,
            "auroc": 0.99,
        },
    ],
)
def test_fastest_expert_rejects_test_or_quality_inputs(
    bad_record: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        FastestExpertPolicy().fit([bad_record])


@pytest.mark.parametrize(
    "policy",
    [
        AlwaysPatchCorePolicy(),
        AlwaysWinCLIPPolicy(),
        RandomSeededPolicy(seed=3),
        FastestExpertPolicy(
            cost_card={
                "PatchCore": 3.0,
                "WinCLIP": 2.0,
                "AnomalyDINO": 1.0,
            }
        ),
    ],
)
def test_all_new_baselines_respect_one_call_budget(policy: object) -> None:
    policy.fit([])
    decision = policy.select(_task())
    assert decision.tool_calls == 1
    assert decision.selected_expert in CANDIDATE_EXPERTS
