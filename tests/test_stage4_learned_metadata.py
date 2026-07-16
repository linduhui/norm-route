import csv
import json
from pathlib import Path

import pytest

from src.normroute.agent.policy_registry import create_policy, list_policies
from src.normroute.agent.protocol import AgentTask
from src.normroute.features import (
    FEATURE_DENYLIST,
    METADATA_FEATURE_ALLOWLIST,
    MetadataFeatureError,
    build_metadata_feature_record,
    validate_metadata_feature_record,
)
from src.normroute.policies.learned import (
    DECISION_TREE_METADATA,
    DIAGNOSTIC_BASELINE_ROLE,
    MODEL_ARTIFACT_FILENAME,
    MULTINOMIAL_LOGISTIC_METADATA,
    DecisionTreeMetadataPolicy,
    MultinomialLogisticMetadataPolicy,
    calibrate_learned_metadata_policy,
)


MANIFEST_COLUMNS = [
    "fold",
    "split",
    "task_id",
    "sample_id",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
]
QUALITY_COLUMNS = [
    "expert",
    "dataset",
    "category",
    "k_shot",
    "seed",
    "support_set_id",
    "image_auroc",
    "image_ap",
    "query_path",
    "expert_scores",
]
EXPERTS = ("PatchCore", "WinCLIP", "AnomalyDINO")
STATIC_COSTS = {"PatchCore": 7.0, "WinCLIP": 11.0, "AnomalyDINO": 17.0}


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path: Path, *, test_metric: object = "poison-test") -> tuple[Path, Path]:
    specifications = [
        ("bottle", 1, 0, "train", 0),
        ("cable", 2, 0, "train", 1),
        ("capsule", 4, 0, "train", 2),
        ("bottle", 8, 1, "train", 0),
        ("cable", 1, 1, "train", 1),
        ("capsule", 2, 1, "train", 2),
        ("bottle", 1, 2, "val", 0),
        ("cable", 2, 2, "val", 1),
        ("capsule", 4, 2, "val", 2),
        ("bottle", 1, 3, "test", 2),
    ]
    manifest_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for index, (category, k_shot, seed, split, best_index) in enumerate(specifications):
        support_set_id = f"mvtec_{category}_k{k_shot}_seed{seed}"
        manifest_rows.append(
            {
                "fold": "fold0",
                "split": split,
                "task_id": f"task-{index}",
                "sample_id": f"sample-{index}",
                "dataset": "mvtec",
                "category": category,
                "k_shot": k_shot,
                "seed": seed,
                "support_set_id": support_set_id,
            }
        )
        for expert_index, expert in enumerate(EXPERTS):
            metric: object = 0.95 if expert_index == best_index else 0.1
            if split == "test":
                metric = test_metric
            quality_rows.append(
                {
                    "expert": expert,
                    "dataset": "mvtec",
                    "category": category,
                    "k_shot": k_shot,
                    "seed": seed,
                    "support_set_id": support_set_id,
                    "image_auroc": metric,
                    "image_ap": metric,
                    # These evaluator-side columns exist to prove that they are
                    # neither accepted nor copied into learned features.
                    "query_path": f"queries/{index}.png",
                    "expert_scores": "evaluator-only",
                }
            )
    manifest = tmp_path / "fold_manifest.csv"
    quality = tmp_path / "expert_quality_by_run.csv"
    _write_csv(manifest, MANIFEST_COLUMNS, manifest_rows)
    _write_csv(quality, QUALITY_COLUMNS, quality_rows)
    return manifest, quality


def _task(*, seed: int = 99, support: str = "support-99") -> AgentTask:
    return AgentTask(
        task_id=f"task-{seed}",
        sample_id=f"sample-{seed}",
        dataset="mvtec",
        category="bottle",
        k_shot=1,
        seed=seed,
        support_set_id=support,
    )


def test_metadata_feature_allowlist_and_denylist_are_exact() -> None:
    record = build_metadata_feature_record(
        category="bottle", k_shot=1, budget=1, static_cost=STATIC_COSTS
    )

    assert tuple(record) == METADATA_FEATURE_ALLOWLIST
    assert {"seed", "support_set_id", "query_path", "expert_scores"} <= FEATURE_DENYLIST
    for forbidden in ("seed", "support_set_id", "query_path", "expert_scores"):
        contaminated = dict(record)
        contaminated[forbidden] = "leak"
        with pytest.raises(MetadataFeatureError, match="denylisted"):
            validate_metadata_feature_record(contaminated)


@pytest.mark.parametrize(
    ("policy_name", "policy_type", "hyperparameter"),
    [
        (DECISION_TREE_METADATA, DecisionTreeMetadataPolicy, "max_depth"),
        (
            MULTINOMIAL_LOGISTIC_METADATA,
            MultinomialLogisticMetadataPolicy,
            "C",
        ),
    ],
)
def test_learned_models_serialize_load_frozen_and_report_confidence(
    tmp_path: Path,
    policy_name: str,
    policy_type: type,
    hyperparameter: str,
) -> None:
    manifest, quality = _inputs(tmp_path)
    output_dir = tmp_path / policy_name / "fold0"
    artifact = calibrate_learned_metadata_policy(
        expert_quality_by_run=quality,
        fold_manifest=manifest,
        fold="fold0",
        output_path=output_dir,
        policy_name=policy_name,
        budget=1,
        static_costs=STATIC_COSTS,
        static_cost_unit="ms",
        tree_max_depth_grid=(1, 2, 3),
        logistic_c_grid=(0.1, 1.0, 10.0),
        training_seed=73,
    )

    assert artifact == output_dir / MODEL_ARTIFACT_FILENAME
    assert {
        "model_artifact.json",
        "feature_manifest.json",
        "train_metadata.json",
        "validation_predictions.csv",
        "failures.json",
    } <= {path.name for path in output_dir.iterdir()}
    model_payload = json.loads(artifact.read_text(encoding="utf-8"))
    feature_manifest = json.loads(
        (output_dir / "feature_manifest.json").read_text(encoding="utf-8")
    )
    train_metadata = json.loads(
        (output_dir / "train_metadata.json").read_text(encoding="utf-8")
    )
    assert model_payload["frozen"] is True
    assert model_payload["diagnostic_baseline"] is True
    assert model_payload["baseline_role"] == DIAGNOSTIC_BASELINE_ROLE
    assert set(model_payload["selected_hyperparameter"]) == {hyperparameter}
    assert train_metadata["training_seed"] == 73
    assert train_metadata["seed"] == 73
    assert train_metadata["config"]["budget"] == 1
    assert train_metadata["artifacts"]["model"] == "model_artifact.json"
    assert train_metadata["label_source"] == "run_level_best_expert_from_split_outcomes"
    assert train_metadata["training_label_counts"] == {
        "AnomalyDINO": 2,
        "PatchCore": 2,
        "WinCLIP": 2,
    }
    encoded = " ".join(feature_manifest["encoded_feature_names"]).lower()
    assert "seed" not in encoded
    assert "support_set" not in encoded
    assert "query" not in encoded
    assert "score" not in encoded

    assert policy_name in list_policies()
    policy = create_policy(policy_name, artifact=artifact)
    assert isinstance(policy, policy_type)
    before = policy.select(_task(seed=91, support="support-a"))
    # Replay fit cannot refit or inspect even a deliberately contaminated record.
    policy.fit([{"seed": 999, "query_path": "forbidden", "expert_scores": {}}])
    after = policy.select(_task(seed=92, support="support-b"))
    assert after.selected_expert == before.selected_expert
    assert after.selected_probability == pytest.approx(before.selected_probability)
    assert after.margin == pytest.approx(before.margin)
    assert 0.0 <= before.selected_probability <= 1.0
    assert 0.0 <= before.margin <= before.selected_probability
    assert "selected_probability=" in before.decision_reason
    assert "margin=" in before.decision_reason
    assert "not final NORM-Route" in before.decision_reason

    saved = policy.save(tmp_path / f"{policy_name}-saved.json")
    loaded = policy_type.load(saved)
    round_trip = loaded.select(_task())
    assert round_trip.selected_expert == before.selected_expert
    assert round_trip.selected_probability == pytest.approx(before.selected_probability)
    assert round_trip.margin == pytest.approx(before.margin)


@pytest.mark.parametrize(
    "policy_name", [DECISION_TREE_METADATA, MULTINOMIAL_LOGISTIC_METADATA]
)
def test_test_fold_scores_cannot_change_learned_model_artifact(
    tmp_path: Path, policy_name: str
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    manifest_a, quality_a = _inputs(first_dir, test_metric="poison-a")
    manifest_b, quality_b = _inputs(second_dir, test_metric="poison-b")
    artifact_a = calibrate_learned_metadata_policy(
        expert_quality_by_run=quality_a,
        fold_manifest=manifest_a,
        fold="fold0",
        output_path=first_dir / "artifacts",
        policy_name=policy_name,
        static_costs=STATIC_COSTS,
        tree_max_depth_grid=(1, 2, 3),
        logistic_c_grid=(0.1, 1.0),
    )
    artifact_b = calibrate_learned_metadata_policy(
        expert_quality_by_run=quality_b,
        fold_manifest=manifest_b,
        fold="fold0",
        output_path=second_dir / "artifacts",
        policy_name=policy_name,
        static_costs=STATIC_COSTS,
        tree_max_depth_grid=(1, 2, 3),
        logistic_c_grid=(0.1, 1.0),
    )

    assert json.loads(artifact_a.read_text(encoding="utf-8")) == json.loads(
        artifact_b.read_text(encoding="utf-8")
    )
    metadata_a = json.loads(
        (artifact_a.parent / "train_metadata.json").read_text(encoding="utf-8")
    )
    metadata_b = json.loads(
        (artifact_b.parent / "train_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata_a == metadata_b
    assert "poison" not in json.dumps(metadata_a).lower()
